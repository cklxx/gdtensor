"""Format sessions (DTensorsView B.2.2.5 / B.2.2.6).

Sessions own the file lifecycle and ``get_file_offset`` only. Payload bytes
go through StorageDTensorsView.data_accessor.

Coordination model (one session per rank, all ranks share one root dir):
* begin_save: every rank creates the temp files it touches (O_CREAT, no
  truncate) and sets their final size (idempotent). The file leader rank
  writes the header. Header bytes are a pure function of
  (DTensorTable, PhysicalLayoutTable), so any rank computes the same bytes.
* commit (after a barrier): the leader rank of each file publishes it.
  Other ranks only close.

safetensors file = 8-byte LE header length N | JSON header (space-padded to
8 bytes) | payload. Tensor entries:
* ``tid``: the tensor's slots in this file concatenate to the full tensor in
  global row-major order (HF case: e.g. Shard(0) over storage columns).
* ``tid/slot<w>``: otherwise, one entry per slot; shape = owned box shape.
Both are valid safetensors; header order = DTensorTable order.

distcp file = raw payload, no header. ``.metadata`` (JSON in the lite
version; real DCP uses pickle) is written last; its presence marks the
checkpoint complete.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._dtensor import DTensorTable, global_tensor_meta_of, owned_box
from ._iobackend import FileIO
from ._layout import PayloadIndex, PhysicalLayoutTable, SlotPayload, TensorComponentPos, build_payload_index
from ._result import ResultError, catch

METADATA_NAME = ".metadata"
HF_INDEX_NAME = "model.safetensors.index.json"


# ------------------------------------------------------------------ headers


def _whole_tensor_files(table: DTensorTable, index: PayloadIndex) -> Dict[Tuple[str, int], bool]:
    """(path, tensor_order) -> True if the file's slots form the full tensor."""
    out = {}
    for path, fp in index.files.items():
        by_t: Dict[int, List[SlotPayload]] = {}
        for s in fp.slots:
            by_t.setdefault(s.tensor_order, []).append(s)
        for t, slots in by_t.items():
            numel = global_tensor_meta_of(table.named_d_tensors[t]).numel
            iv = np.concatenate([s.intervals for s in slots])
            ok = iv[0, 0] == 0 and iv[-1, 1] == numel and bool(np.all(iv[1:, 0] == iv[:-1, 1]))
            out[(path, t)] = ok
    return out


def safetensors_header(table: DTensorTable, index: PayloadIndex, path: str) -> bytes:
    fp = index.files[path]
    whole = _whole_tensor_files(table, index)
    header: Dict[str, dict] = {"__metadata__": {"format": "pt", "writer": "gdtensor"}}
    emitted = set()
    for s in fp.slots:
        nd = table.named_d_tensors[s.tensor_order]
        meta = global_tensor_meta_of(nd)
        if whole[(path, s.tensor_order)]:
            if s.tensor_order in emitted:
                continue
            emitted.add(s.tensor_order)
            begin = s.contiguous_payload_offset
            header[nd.global_tensor_id] = {
                "dtype": meta.dtype,
                "shape": list(meta.global_shape),
                "data_offsets": [begin, begin + meta.nbytes],
            }
        else:
            lo, hi = owned_box(nd, "storage", s.pos.rank)
            header[f"{nd.global_tensor_id}/slot{s.pos.rank}"] = {
                "dtype": meta.dtype,
                "shape": [h - l for l, h in zip(lo, hi)],
                "data_offsets": [s.contiguous_payload_offset, s.contiguous_payload_offset + s.nbytes],
            }
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def distcp_metadata(table: DTensorTable, layout: PhysicalLayoutTable, index: PayloadIndex) -> bytes:
    tensors = []
    for nd in table.named_d_tensors:
        m = global_tensor_meta_of(nd)
        tensors.append({"id": nd.global_tensor_id, "shape": list(m.global_shape), "dtype": m.dtype, "itemsize": m.itemsize})
    chunks = []
    for path in sorted(index.files):
        for s in index.files[path].slots:
            lo, hi = owned_box(table.named_d_tensors[s.tensor_order], "storage", s.pos.rank)
            chunks.append({
                "tensor_id": s.pos.tensor_id, "slot": s.pos.rank, "file": path,
                "offset": s.contiguous_payload_offset, "size": s.nbytes, "box_lo": lo, "box_hi": hi,
            })
    doc = {"format": "gdtensor-distcp", "tensors": tensors, "chunks": chunks}
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


# ------------------------------------------------------------------ sessions


@dataclass
class _Base:
    rank: int
    root: str
    table: DTensorTable
    layout: PhysicalLayoutTable
    index: PayloadIndex
    io: FileIO

    def abspath(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def my_files(self) -> List[str]:
        """Files this rank touches: files holding a slot in its column, plus
        files it leads."""
        out = {p for p, fp in self.index.files.items() if fp.leader.rank == self.rank}
        out |= {p for (p, s) in self.index.slots.values() if s.pos.rank == self.rank}
        return sorted(out)


class SaveSessionBase(_Base):
    tmp_suffix = ""

    def __init__(self, *a, coord=None, **kw) -> None:
        super().__init__(*a, **kw)
        self.coord = coord  # barrier + vote across ranks; None = single rank
        self.state = "open"  # open -> committing -> committed | unknown | aborted
        self.data_start: Dict[str, int] = {}

    def abspath(self, rel: str) -> str:  # writes go to the temp path
        return os.path.join(self.root, rel + self.tmp_suffix)

    def _header(self, path: str) -> bytes:
        return b""

    def _open_files(self) -> None:
        for path in self.my_files():
            full = self.abspath(path)
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            header = self._header(path)
            self.data_start[path] = len(header)
            fd = os.open(full, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                os.ftruncate(fd, len(header) + self.index.files[path].payload_size)
            finally:
                os.close(fd)
            if header and self.index.files[path].leader.rank == self.rank:
                self.io.pwrite(self.rank, full, 0, header)  # header first, data later

    @catch
    def get_file_offset(self, pos: TensorComponentPos) -> int:
        if self.state != "open":
            raise ResultError(f"get_file_offset on {self.state} session")
        hit = self.index.slots.get(TensorComponentPos(*pos))
        if hit is None:
            raise ResultError(f"{tuple(pos)} has no physical file")
        path, slot = hit
        if path not in self.data_start:
            self.data_start[path] = len(self._header(path))
        return self.data_start[path] + slot.contiguous_payload_offset

    def _publish(self) -> None:
        raise NotImplementedError

    def _after_barrier(self) -> None:
        """Hook that runs after every leader published (distcp .metadata)."""

    def _vote(self, ok: bool) -> bool:
        return ok if self.coord is None else self.coord.vote(ok)

    def _wait(self) -> None:
        if self.coord is not None:
            self.coord.wait()

    @catch
    def commit(self) -> None:
        if self.state != "open":
            raise ResultError(f"commit on {self.state} session")
        self.io.fds.close_rank(self.rank)
        if not self._vote(True):  # barrier: every writer is done (or failed)
            self.state = "aborted"
            self._discard()
            raise ResultError("another rank aborted; temp files discarded")
        self.state = "committing"
        err = None
        try:
            self._publish()
        except BaseException as exc:  # noqa: BLE001
            err = exc
        self._wait()
        if err is None:
            try:
                self._after_barrier()
            except BaseException as exc:  # noqa: BLE001
                err = exc
        if err is not None:
            self.state = "unknown"  # keep payloads; abort must not delete them
            raise ResultError(f"commit outcome unknown: {err}")
        self.state = "committed"

    @catch
    def abort(self) -> None:
        if self.state in ("committed", "aborted"):
            raise ResultError(f"abort on {self.state} session")
        self.io.fds.close_rank(self.rank)
        if self.state == "unknown":
            self.state = "aborted"  # final path may exist: delete nothing
            return
        self._vote(False)
        self.state = "aborted"
        self._discard()

    def _discard(self) -> None:
        for path in self.my_files():
            if self.index.files[path].leader.rank == self.rank:
                try:
                    os.remove(self.abspath(path))
                except FileNotFoundError:
                    pass


class SafeTensorsSaveSession(SaveSessionBase):
    tmp_suffix = ".tmp"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self._headers: Dict[str, bytes] = {}
        self._open_files()

    def _header(self, path: str) -> bytes:
        if path not in self._headers:
            self._headers[path] = safetensors_header(self.table, self.index, path)
        return self._headers[path]

    def _publish(self) -> None:
        for path in self.my_files():
            if self.index.files[path].leader.rank != self.rank:
                continue
            tmp, final = self.abspath(path), os.path.join(self.root, path)
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, final)  # readers see no file or the full new file

    def _after_barrier(self) -> None:
        """HF directory: the lowest leader writes model.safetensors.index.json
        when there are several files and every entry is a whole tensor."""
        leaders = sorted(fp.leader.rank for fp in self.index.files.values())
        if len(self.index.files) < 2 or self.rank != leaders[0]:
            return
        whole = _whole_tensor_files(self.table, self.index)
        if not all(whole.values()):
            return
        weight_map = {}
        for (path, t), _ in sorted(whole.items(), key=lambda kv: kv[0][1]):
            weight_map[self.table.named_d_tensors[t].global_tensor_id] = path
        total = sum(global_tensor_meta_of(nd).nbytes for nd in self.table.named_d_tensors)
        doc = {"metadata": {"total_size": total}, "weight_map": weight_map}
        tmp = os.path.join(self.root, HF_INDEX_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(self.root, HF_INDEX_NAME))


class DcpSaveSession(SaveSessionBase):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self._open_files()

    def _coordinator(self) -> int:
        return min(fp.leader.rank for fp in self.index.files.values())

    def _publish(self) -> None:
        for path in self.my_files():
            if self.index.files[path].leader.rank == self.rank:
                fd = os.open(self.abspath(path), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)

    def _after_barrier(self) -> None:
        if self.rank == self._coordinator():
            tmp = os.path.join(self.root, METADATA_NAME + ".tmp")
            data = distcp_metadata(self.table, self.layout, self.index)
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, os.path.join(self.root, METADATA_NAME))  # completion marker


class LoadSessionBase(_Base):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.state = "open"
        self.offsets: Dict[TensorComponentPos, int] = {}

    @catch
    def get_file_offset(self, pos: TensorComponentPos) -> int:
        if self.state != "open":
            raise ResultError(f"get_file_offset on {self.state} session")
        pos = TensorComponentPos(*pos)
        if pos not in self.offsets:
            raise ResultError(f"{tuple(pos)} not found in file header / metadata")
        return self.offsets[pos]

    @catch
    def finish(self) -> None:
        if self.state != "open":
            raise ResultError(f"finish on {self.state} session")
        self.io.fds.close_rank(self.rank, self.root)
        self.state = "finished"


class SafeTensorsLoadSession(LoadSessionBase):
    """Reads each needed header (8 B, then N B) before any get_file_offset."""

    def __init__(self, *a, needed_files: Optional[List[str]] = None, **kw) -> None:
        super().__init__(*a, **kw)
        files = needed_files if needed_files is not None else self.my_files()
        for path in files:
            full = self.abspath(path)
            n = struct.unpack("<Q", self.io.pread(self.rank, full, 0, 8))[0]
            header = json.loads(self.io.pread(self.rank, full, 8, n))
            data_start = 8 + n
            fp = self.index.files[path]
            seen: Dict[int, int] = {}
            for s in fp.slots:
                tid = s.pos.tensor_id
                if tid in header:
                    first = seen.setdefault(s.tensor_order, s.contiguous_payload_offset)
                    begin = header[tid]["data_offsets"][0] + (s.contiguous_payload_offset - first)
                elif f"{tid}/slot{s.pos.rank}" in header:
                    ent = header[f"{tid}/slot{s.pos.rank}"]
                    if ent["data_offsets"][1] - ent["data_offsets"][0] != s.nbytes:
                        raise ResultError(f"{path}: size mismatch for {tid}/slot{s.pos.rank}")
                    begin = ent["data_offsets"][0]
                else:
                    raise ResultError(f"{path}: header lacks {tid}")
                self.offsets[s.pos] = data_start + begin


class DcpLoadSession(LoadSessionBase):
    def __init__(self, *a, needed_files: Optional[List[str]] = None, **kw) -> None:
        super().__init__(*a, **kw)
        full = os.path.join(self.root, METADATA_NAME)
        if not os.path.exists(full):
            raise ResultError("distcp checkpoint is incomplete: .metadata is missing")
        doc = json.loads(self.io.pread(self.rank, full, 0, os.path.getsize(full)))
        for c in doc["chunks"]:
            self.offsets[TensorComponentPos(c["tensor_id"], c["slot"])] = c["offset"]
        for pos, (path, s) in self.index.slots.items():
            if self.offsets.get(pos) != s.contiguous_payload_offset:
                raise ResultError(f".metadata disagrees with layout at {tuple(pos)}")


# ------------------------------------------------------------------ factory


def make_storage_view(
    tensor_ids: List[str],
    table: DTensorTable,
    layout: PhysicalLayoutTable,
    root: str,
    io: Optional[FileIO] = None,
    coord=None,
):
    """Build a StorageDTensorsView whose sessions match layout.container_format."""
    from ._views import StorageDataAccessor, StorageDTensorsView

    io = io or FileIO()
    index = build_payload_index(table, layout)
    fmt = layout.container_format
    save_cls = {"safetensors": SafeTensorsSaveSession, "distcp": DcpSaveSession}[fmt]
    load_cls = {"safetensors": SafeTensorsLoadSession, "distcp": DcpLoadSession}[fmt]

    @catch
    def begin_save(view, rank: int):
        return save_cls(rank, root, view.d_tensor_table, view.physical_layout, index, io, coord=coord)

    @catch
    def begin_load(view, rank: int, needed_files: Optional[List[str]] = None):
        return load_cls(rank, root, view.d_tensor_table, view.physical_layout, index, io, needed_files=needed_files)

    accessor = StorageDataAccessor(index=index, io=io)
    view = StorageDTensorsView(list(tensor_ids), table, accessor, layout, begin_save, begin_load)
    view.root = root
    return view

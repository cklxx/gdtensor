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

distcp file = raw payload, no header. ``.metadata`` (JSON here; real DCP
uses pickle) is written last; its presence marks the checkpoint complete.

Load sessions read the headers of the files that hold this rank's storage
column plus every file it leads at begin_load; other headers are read on the
first get_file_offset that needs them.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from gdtensor.builder import (
    DTensorTable,
    PhysicalLayoutTable,
    TensorComponentPos,
    global_tensor_meta_of,
)
from ._geometry import nbytes_of, numel_of, owned_box
from ._layout import PayloadIndex, SlotPayload, build_payload_index, pos_key
from ._result import catch, invalid, io_error, not_found
from .iobackend import FileIO
from gdtensor.session import StorageLoadSession, StorageSaveSession

HF_INDEX_NAME = "model.safetensors.index.json"


# ------------------------------------------------------------------ headers


def _whole_tensor_files(
    table: DTensorTable, index: PayloadIndex
) -> Dict[Tuple[str, int], bool]:
    """(path, tensor_order) -> True if the file's slots form the full tensor."""
    out = {}
    for path, fp in index.files.items():
        by_t: Dict[int, List[SlotPayload]] = {}
        for s in fp.slots:
            by_t.setdefault(s.tensor_order, []).append(s)
        for t, slots in by_t.items():
            numel = numel_of(global_tensor_meta_of(table.named_d_tensors[t]))
            iv = np.concatenate([s.intervals for s in slots])
            ok = (
                iv[0, 0] == 0
                and iv[-1, 1] == numel
                and bool(np.all(iv[1:, 0] == iv[:-1, 1]))
            )
            out[(path, t)] = ok
    return out


def _storage_box(nd, rank):
    box = owned_box(nd, "storage", rank)
    if box is None:
        raise invalid("storage_mesh", "payload slot is not on the storage mesh")
    return box


def safetensors_header(table: DTensorTable, index: PayloadIndex, path: str) -> bytes:
    fp = index.files[path]
    whole = _whole_tensor_files(table, index)
    header: Dict[str, dict] = {"__metadata__": {"format": "pt", "writer": "gdtensor"}}
    emitted = set()
    for s in fp.slots:
        nd = table.named_d_tensors[s.tensor_order]
        meta = global_tensor_meta_of(nd)
        if whole[(path, s.tensor_order)] and s.tensor_order in emitted:
            continue
        if whole[(path, s.tensor_order)]:
            emitted.add(s.tensor_order)
            begin = s.contiguous_payload_offset
            header[nd.global_tensor_id] = {
                "dtype": meta.dtype,
                "shape": list(meta.global_shape),
                "data_offsets": [begin, begin + nbytes_of(meta)],
            }
            continue
        lo, hi = _storage_box(nd, s.pos.rank)
        header[f"{nd.global_tensor_id}/slot{s.pos.rank}"] = {
            "dtype": meta.dtype,
            "shape": [h - l for l, h in zip(lo, hi)],
            "data_offsets": [
                s.contiguous_payload_offset,
                s.contiguous_payload_offset + s.nbytes,
            ],
        }
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


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


def _size_file(path, size):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, size)
    finally:
        os.close(fd)


def _sync_file(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _finish_publication(session, published, error):
    if not published:
        return error
    try:
        session._after_barrier()
    except BaseException as exc:
        return exc
    return error


class SaveSessionBase(_Base, StorageSaveSession):
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
            self._open_file(path)

    def _open_file(self, path):
        full = self.abspath(path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        header = self._header(path)
        self.data_start[path] = len(header)
        _size_file(full, len(header) + self.index.files[path].payload_size)
        if header and self.index.files[path].leader.rank == self.rank:
            self.io.pwrite(self.rank, full, 0, header)

    def _leader_files(self):
        return [
            path
            for path in self.my_files()
            if self.index.files[path].leader.rank == self.rank
        ]

    @catch
    def init_data_accessor(self):
        if self.state != "open":
            raise invalid("session", f"init_data_accessor on {self.state} session")
        from .access import StorageDataAccessor

        return StorageDataAccessor(self.index, self.io, session=self)

    @catch
    def get_file_offset(self, pos: TensorComponentPos) -> int:
        if self.state != "open":
            raise invalid("session", f"get_file_offset on {self.state} session")
        hit = self.index.slots.get(pos)
        if hit is None:
            raise not_found(
                "tensor_entries", pos_key(pos), f"{pos_key(pos)} has no physical file"
            )
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

    @catch
    def commit(self) -> None:
        if self.state != "open":
            raise invalid("session", f"commit on {self.state} session")
        self.io.fds.close_rank(self.rank)
        if not self._vote(True):  # barrier: every writer is done (or failed)
            self.state = "aborted"
            self._discard()
            raise io_error("commit", "another rank aborted; temp files discarded")
        self.state = "committing"
        err = None
        try:
            self._publish()
        except BaseException as exc:  # noqa: BLE001
            err = exc
        published = self._vote(err is None)
        err = _finish_publication(self, published, err)
        completed = self._vote(published and err is None)
        if not completed:
            self.state = "unknown"  # keep payloads; abort must not delete them
            raise io_error(
                "commit",
                f"commit outcome unknown: {err or 'another rank failed to publish'}",
            )
        self.state = "committed"

    @catch
    def abort(self) -> None:
        if self.state in ("committed", "aborted"):
            raise invalid("session", f"abort on {self.state} session")
        self.io.fds.close_rank(self.rank)
        if self.state == "unknown":
            self.state = "aborted"  # final path may exist: delete nothing
            return
        self._vote(False)
        self.state = "aborted"
        self._discard()

    def _discard(self) -> None:
        for path in self._leader_files():
            Path(self.abspath(path)).unlink(missing_ok=True)


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
        for path in self._leader_files():
            tmp, final = self.abspath(path), os.path.join(self.root, path)
            _sync_file(tmp)
            os.replace(tmp, final)

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
        total = sum(
            nbytes_of(global_tensor_meta_of(nd)) for nd in self.table.named_d_tensors
        )
        doc = {"metadata": {"total_size": total}, "weight_map": weight_map}
        tmp = os.path.join(self.root, HF_INDEX_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(self.root, HF_INDEX_NAME))


class LoadSessionBase(_Base, StorageLoadSession):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.state = "open"
        self.offsets: Dict[TensorComponentPos, int] = {}

    def _read_file_offsets(self, path: str) -> None:
        """Fill self.offsets for every slot of one file."""

    @catch
    def init_data_accessor(self):
        if self.state != "open":
            raise invalid("session", f"init_data_accessor on {self.state} session")
        from .access import StorageDataAccessor

        return StorageDataAccessor(self.index, self.io, session=self)

    @catch
    def get_file_offset(self, pos: TensorComponentPos) -> int:
        if self.state != "open":
            raise invalid("session", f"get_file_offset on {self.state} session")
        if pos not in self.offsets:
            self._read_component(pos)
        if pos not in self.offsets:
            raise not_found(
                "tensor_entries",
                pos_key(pos),
                f"{pos_key(pos)} not found in header / metadata",
            )
        return self.offsets[pos]

    def _read_component(self, pos):
        hit = self.index.slots.get(pos)
        if hit is None:
            raise not_found(
                "tensor_entries", pos_key(pos), f"{pos_key(pos)} has no physical file"
            )
        self._read_file_offsets(hit[0])

    @catch
    def finish(self) -> None:
        if self.state != "open":
            raise invalid("session", f"finish on {self.state} session")
        self.io.fds.close_rank(self.rank, self.root)
        self.state = "finished"


class SafeTensorsLoadSession(LoadSessionBase):
    """Reads a header (8 B, then N B) before any get_file_offset in that file."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        for path in self.my_files():
            self._read_file_offsets(path)

    def _read_file_offsets(self, path: str) -> None:
        full = self.abspath(path)
        try:
            n = struct.unpack("<Q", self.io.pread(self.rank, full, 0, 8))[0]
            header = json.loads(self.io.pread(self.rank, full, 8, n))
        except OSError as exc:
            raise io_error("read", str(exc), full) from None
        data_start = 8 + n
        seen: Dict[int, int] = {}
        for slot in self.index.files[path].slots:
            self.offsets[slot.pos] = data_start + _safetensors_slot_offset(
                self.table, slot, header, seen, path, full
            )


def _safetensors_slot_offset(table, slot, header, seen, path, full):
    tid = slot.pos.tensor_id
    nd = table.named_d_tensors[slot.tensor_order]
    meta = global_tensor_meta_of(nd)
    if tid in header:
        entry = header[tid]
        expected_shape, expected_size = list(meta.global_shape), nbytes_of(meta)
        first = seen.setdefault(slot.tensor_order, slot.contiguous_payload_offset)
        begin = entry["data_offsets"][0] + slot.contiguous_payload_offset - first
    elif f"{tid}/slot{slot.pos.rank}" in header:
        entry = header[f"{tid}/slot{slot.pos.rank}"]
        lo, hi = _storage_box(nd, slot.pos.rank)
        expected_shape, expected_size = [h - l for l, h in zip(lo, hi)], slot.nbytes
        begin = entry["data_offsets"][0]
    else:
        raise not_found("tensor_id", tid, f"{path}: header lacks {tid}")
    offsets = entry["data_offsets"]
    if (
        entry.get("dtype") != meta.dtype
        or entry.get("shape") != expected_shape
        or offsets[0] < 0
        or offsets[1] - offsets[0] != expected_size
    ):
        raise io_error("read", f"{path}: metadata mismatch for {tid}", full)
    return begin


# ------------------------------------------------------------------ factory


def make_storage_view(
    tensor_ids: List[str],
    table: DTensorTable,
    layout: PhysicalLayoutTable,
    root: str,
    io: Optional[FileIO] = None,
    coord=None,
    dcp_mode: str = "thread",
):
    """Business constructor: a StorageDTensorsView over files in ``root``.

    ``dcp_mode`` selects the real-DCP simulation backend: "thread" assembles
    in shared process memory; "process" uses per-slot files under a stage dir.
    """
    from .access import StorageDataAccessor
    from .dcp_backend import DcpLoadSession, DcpSaveSession
    from gdtensor.builder import StorageDTensorsView

    io = io or FileIO()
    index = build_payload_index(table, layout)
    fmt = layout.container_format
    if fmt not in ("safetensors", "distcp"):
        raise invalid("container_format", f"unknown container_format {fmt!r}")
    save_cls = {"safetensors": SafeTensorsSaveSession}
    load_cls = {"safetensors": SafeTensorsLoadSession}

    def _open_session(cls, view, rank, **kwargs):
        try:
            return cls(
                rank,
                root,
                view.d_tensor_table,
                view.physical_layout,
                index,
                io,
                **kwargs,
            )
        except BaseException:
            io.fds.close_rank(rank, root)
            raise

    @catch
    def begin_save(view, rank: int):
        if fmt == "distcp":
            return DcpSaveSession(
                rank,
                root,
                view.d_tensor_table,
                view.physical_layout,
                index,
                io=io,
                coord=coord,
                mode=dcp_mode,
            )
        return _open_session(save_cls["safetensors"], view, rank, coord=coord)

    @catch
    def begin_load(view, rank: int):
        if fmt == "distcp":
            return _open_session(DcpLoadSession, view, rank)
        return _open_session(load_cls["safetensors"], view, rank)

    accessor = StorageDataAccessor(index=index, io=io)
    return StorageDTensorsView(
        list(tensor_ids), table, accessor, layout, begin_save, begin_load
    )

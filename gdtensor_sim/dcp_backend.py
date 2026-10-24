"""Real torch DCP backend for the simulation storage sessions.

The upper layer keeps the gdtensor copy plan and data accessor on the data
path. Ranks write their owned slots to a staging area. The coordinator
assembles the logical tensors and writes the checkpoint with the real
``torch.distributed.checkpoint`` API, so the on-disk artifact is genuine DCP:
a pickle ``.metadata`` (TensorStorageMetadata + storage_data) and torch_save
payload files (``__<rank>_<shard>.distcp``).

Dtype handling is byte-exact: staged slot bytes are the raw element bytes of
the tensor dtype. We never widen BF16 through float32 -- the 2-byte slot bytes
are interpreted directly as ``torch.bfloat16`` on save and recovered as raw
bytes through a ``uint8`` view on load. numpy has no bfloat16, so the torch
tensor is built straight from the raw buffer.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import (
    DTensorTable,
    TensorComponentPos,
    global_tensor_meta_of,
)
from gdtensor.builder import Ok
from gdtensor.session import StorageLoadSession, StorageSaveSession

from ._geometry import iv_total
from ._layout import PayloadIndex
from ._result import catch, invalid, io_error, not_found

STAGE_DIR = ".gdtensor_dcp_stage"

_TORCH_DTYPES = {
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "U8": "uint8",
    "I32": "int32",
    "I64": "int64",
    "BOOL": "bool",
}


def _torch_dtype(meta):
    import torch

    try:
        return getattr(torch, _TORCH_DTYPES[meta.dtype])
    except KeyError:
        raise invalid("dtype", f"unsupported dtype for DCP simulation: {meta.dtype!r}")


def _slot_key(pos: TensorComponentPos) -> str:
    safe = pos.tensor_id.replace("/", "_")
    return f"{safe}.slot{pos.rank}"


@dataclass
class _Stage:
    """Process-local staging area for one checkpoint root.

    Thread mode keeps full slot buffers in memory; process mode keeps only
    per-slot file paths and assembles by reading them back.
    """

    root: str
    slots: Dict[tuple, bytearray] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def stage_dir(self) -> str:
        return os.path.join(self.root, STAGE_DIR)

    def path(self, pos: TensorComponentPos) -> str:
        return os.path.join(self.stage_dir(), _slot_key(pos))

    def write(self, slot, rel: int, data: bytes) -> None:
        key = (slot.pos.tensor_id, slot.pos.rank)
        with self.lock:
            buf = self.slots.setdefault(key, bytearray(slot.nbytes))
            buf[rel : rel + len(data)] = data

    def bytes_of(self, pos: TensorComponentPos) -> bytes:
        with self.lock:
            return bytes(self.slots[(pos.tensor_id, pos.rank)])

    def ensure_capacity(self, slot, path: str) -> None:
        if not os.path.exists(path):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
            os.ftruncate(fd, slot.nbytes)
            os.close(fd)


_STAGES: Dict[str, _Stage] = {}
_STAGE_LOCK = threading.Lock()


def _stage_for(root: str) -> _Stage:
    with _STAGE_LOCK:
        stage = _STAGES.get(root)
        if stage is None:
            stage = _Stage(root)
            _STAGES[root] = stage
        return stage


def _slot_raw_bytes(nd, rank, intervals, stage: _Stage, on_disk: bool) -> bytes:
    """Raw element bytes of one rank's slot, in slot order."""
    from ._geometry import nonempty_columns  # noqa: F401  (kept for symmetry)

    pos = TensorComponentPos(nd.global_tensor_id, rank)
    if on_disk:
        path = stage.path(pos)
        if not os.path.exists(path):
            raise io_error("read", f"missing staged slot {_slot_key(pos)}", path)
        with open(path, "rb") as handle:
            return handle.read()
    try:
        return stage.bytes_of(pos)
    except KeyError:
        raise io_error("read", f"missing staged slot {_slot_key(pos)}") from None


def _assemble_tensor(nd, stage: _Stage, on_disk: bool):
    """Rebuild one full global tensor from its non-empty storage slots.

    Returns a torch tensor in the tensor's own dtype (BF16 stays BF16; it is
    never widened through float32). Verifies every expected slot is present and
    byte-complete and that every element of the global tensor was staged.
    """
    from ._geometry import nonempty_columns

    meta = global_tensor_meta_of(nd)
    n = int(np.prod(meta.global_shape))
    itemsize = meta.itemsize
    raw = bytearray(n * itemsize)
    filled = np.zeros(n, dtype=bool)
    for rank, intervals in nonempty_columns(nd).items():
        chunk = _slot_raw_bytes(nd, rank, intervals, stage, on_disk)
        expect = iv_total(intervals) * itemsize
        if len(chunk) != expect:
            raise io_error(
                "read",
                f"slot rank {rank} has {len(chunk)} bytes, expected {expect}",
            )
        cursor = 0
        for lo, hi in intervals:
            seg = (hi - lo) * itemsize
            raw[lo * itemsize : hi * itemsize] = chunk[cursor : cursor + seg]
            filled[lo:hi] = True
            cursor += seg
    if not filled.all():
        raise io_error(
            "read", f"tensor {nd.global_tensor_id} staged incompletely before publish"
        )
    return _torch_from_raw(bytes(raw), meta)


def _torch_from_raw(raw: bytes, meta):
    import torch

    shape = tuple(meta.global_shape)
    if int(np.prod(shape)) == 0:
        return torch.empty(shape, dtype=_torch_dtype(meta))
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    flat = torch.from_numpy(arr)
    return flat.view(_torch_dtype(meta)).reshape(shape)


def _real_dcp_save(
    table: DTensorTable, stage: _Stage, root: str, on_disk: bool
) -> None:
    from torch.distributed.checkpoint import save as dcp_save

    state = {
        nd.global_tensor_id: _assemble_tensor(nd, stage, on_disk)
        for nd in table.named_d_tensors
    }
    dcp_save(state, checkpoint_id=root, no_dist=True)


def _empty_torch_tensor(meta):
    import torch

    return torch.empty(tuple(meta.global_shape), dtype=_torch_dtype(meta))


def _real_dcp_load(table: DTensorTable, root: str) -> Dict[str, object]:
    from torch.distributed.checkpoint import load as dcp_load

    state: Dict[str, object] = {
        nd.global_tensor_id: _empty_torch_tensor(global_tensor_meta_of(nd))
        for nd in table.named_d_tensors
    }
    dcp_load(state, checkpoint_id=root, no_dist=True)
    return state


def _storage_offsets(root: str) -> Dict[str, int]:
    """Read ``storage_data[MetadataIndex(fqn)].offset`` from the real pickle
    ``.metadata``."""
    import pickle

    with open(os.path.join(root, ".metadata"), "rb") as handle:
        meta = pickle.loads(handle.read())
    return {idx.fqn: sd.offset for idx, sd in meta.storage_data.items()}


def _raw_payload_bases(root: str) -> Dict[str, List[tuple]]:
    """Resolve the real byte offset of each tensor's raw payload inside the
    torch-written ``.distcp`` archive, by parsing the archive structure (never by
    searching for raw byte content, never a hard-coded header constant).

    Each ``storage_data[MetadataIndex(fqn)]`` points at a self-contained sub-archive
    of ``length`` bytes (a torch zip). Inside it the raw tensor bytes are the
    ``archive/data/<n>`` zip entry (``data.pkl`` is the pickle, not the payload).
    The payload file offset = sub-archive start + (local-header offset of the
    data entry + 30 + filename length + extra-field length).
    """
    import pickle

    with open(os.path.join(root, ".metadata"), "rb") as handle:
        meta = pickle.loads(handle.read())
    # fqn -> list of (distcp_file, raw_payload_base, payload_nbytes). A tensor may
    # be sharded across several sub-archives (multiple storage_data entries for the
    # same fqn); keep ALL chunks instead of silently overwriting with the last one.
    out: Dict[str, List[tuple]] = {}
    by_file: Dict[str, List] = {}
    for idx, sd in meta.storage_data.items():
        by_file.setdefault(sd.relative_path, []).append((idx.fqn, sd))
    for rel, entries in by_file.items():
        raw = open(os.path.join(root, rel), "rb").read()
        for fqn, sd in entries:
            chunk = _resolve_one_subarchive(root, rel, fqn, sd, raw)
            out.setdefault(fqn, []).append(chunk)
    return out


def _resolve_one_subarchive(root, rel, fqn, sd, raw):
    import io
    import struct
    import zipfile

    sub = raw[sd.offset : sd.offset + sd.length]
    zf = zipfile.ZipFile(io.BytesIO(sub))
    zi = next(
        z
        for z in zf.infolist()
        if z.filename.startswith("archive/data/") and z.filename != "archive/data.pkl"
    )
    if zi.compress_type != 0:
        msg = f"{fqn}: unsupported DCP payload compression {zi.compress_type}"
        raise io_error("read", msg, os.path.join(root, rel))
    lh = sub[zi.header_offset : zi.header_offset + 30]
    nlen, elen = struct.unpack("<HH", lh[26:30])
    data_in_sub = zi.header_offset + 30 + nlen + elen
    return (rel, sd.offset + data_in_sub, zi.file_size)


def _resolve_slot(index: PayloadIndex, rng):
    """Map a file/payload-offset range back to its owning slot."""
    fp = index.files.get(rng.file_path)
    if fp is None:
        raise invalid("global_range", f"unknown file {rng.file_path}")
    starts = [x.contiguous_payload_offset for x in fp.slots]
    k = int(np.searchsorted(starts, rng.contiguous_payload_offset, side="right")) - 1
    if k < 0:
        raise invalid("global_range", "payload offset before first slot")
    slot = fp.slots[k]
    end = rng.contiguous_payload_offset + rng.size
    if end > slot.contiguous_payload_offset + slot.nbytes:
        raise invalid("global_range", "range crosses a slot boundary")
    rel = rng.contiguous_payload_offset - slot.contiguous_payload_offset
    return slot, rel


class _DcpAccessor(DTensorDataAccessor):
    """Routes gdtensor copy ranges to staged slots (save) or to full tensors
    loaded by real DCP (load)."""

    def __init__(
        self,
        mode: str,
        index: PayloadIndex,
        stage: _Stage,
        full: Dict[str, object] | None = None,
    ):
        self.mode = mode
        self.index = index
        self.stage = stage
        self.full: Dict[str, object] = full if full is not None else {}
        self._owned: Dict[TensorComponentPos, np.ndarray] = {}

    def _owned_linear(self, pos: TensorComponentPos) -> np.ndarray:
        if pos not in self._owned:
            _, slot = self.index.slots[pos]
            ranges = [np.arange(int(a), int(b)) for a, b in slot.intervals]
            self._owned[pos] = np.concatenate(ranges)
        return self._owned[pos]

    def _write_process(self, slot, rel: int, data: bytes) -> None:
        path = self.stage.path(slot.pos)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.stage.ensure_capacity(slot, path)
        fd = os.open(path, os.O_WRONLY)
        os.pwrite(fd, data, rel)
        os.close(fd)

    @catch
    def write_from_buffer(self, global_range, write_from_buffer) -> None:
        from .access import call_buffer_callback

        slot, rel = _resolve_slot(self.index, global_range)
        buf = bytearray(global_range.size)
        call_buffer_callback(write_from_buffer, buf)
        if self.mode == "thread":
            self.stage.write(slot, rel, bytes(buf))
        else:
            self._write_process(slot, rel, bytes(buf))

    @catch
    def read_to_buffer(self, global_range, read_to_buffer) -> None:
        import torch

        from .access import call_buffer_callback

        slot, rel = _resolve_slot(self.index, global_range)
        owned = self._owned_linear(slot.pos)
        n = global_range.size // slot.itemsize
        start = rel // slot.itemsize
        flat = torch.as_tensor(self.full[slot.pos.tensor_id]).reshape(-1)
        sel = flat[owned[start : start + n]].contiguous()
        data = sel.view(torch.uint8).numpy().tobytes()
        call_buffer_callback(read_to_buffer, bytearray(data))


def _vote(coord, ok: bool) -> bool:
    return ok if coord is None else coord.vote(ok)


class DcpSaveSession(StorageSaveSession):
    def __init__(
        self, rank, root, table, layout, index, io=None, coord=None, mode="thread"
    ):
        self.rank = rank
        self.root = root
        self.table = table
        self.layout = layout
        self.index = index
        self.coord = coord
        self.mode = mode
        self.state = "open"
        self.stage = _stage_for(root)
        if mode == "process":
            os.makedirs(self.stage.stage_dir(), exist_ok=True)

    @catch
    def init_data_accessor(self):
        if self.state != "open":
            raise invalid("session", f"init_data_accessor on {self.state} session")
        return _DcpAccessor(self.mode, self.index, self.stage)

    @catch
    def get_file_offset(self, pos):
        if self.state != "open":
            raise invalid("session", f"get_file_offset on {self.state} session")
        # Save-unpublished: staged slots live in the staging payload region with
        # no header, laid out contiguously in slot order. Map the slot's
        # contiguous_payload_offset directly; unknown pos is NotFound.
        hit = self.index.slots.get(pos)
        if hit is None:
            raise not_found(
                "tensor_entries",
                f"{pos.tensor_id}:{pos.rank}",
                f"{pos.tensor_id}:{pos.rank} has no physical file",
            )
        _path, slot = hit
        return Ok(slot.contiguous_payload_offset)

    @catch
    def commit(self):
        if self.state != "open":
            raise invalid("session", f"commit on {self.state} session")
        if not _vote(self.coord, True):
            self.state = "aborted"
            raise io_error("commit", "another rank aborted before DCP assembly")
        err = self._publish_if_coordinator()
        if not _vote(self.coord, err is None):
            self.state = "unknown"
            reason = err or "coordinator error"
            raise io_error("commit", f"DCP publish failed: {reason}")
        self.state = "committed"
        return Ok(None)

    def _publish_if_coordinator(self):
        if self.rank != 0:
            return None
        try:
            self._publish_real()
        except BaseException as exc:  # noqa: BLE001
            return exc
        return None

    def _publish_real(self) -> None:
        # No copied data: publish a valid real DCP checkpoint.
        if not self._any_slot_present():
            self._publish_empty()
            return
        on_disk = self.mode == "process"
        _real_dcp_save(self.table, self.stage, self.root, on_disk)
        self._clear_stage()

    def _clear_stage(self) -> None:
        # Reuse protection: drop staged bytes once published so a later save to
        # the same root cannot assemble stale slots from a previous cycle.
        if self.mode == "thread":
            self.stage.slots.clear()
        else:
            shutil.rmtree(self.stage.stage_dir(), ignore_errors=True)

    def _any_slot_present(self) -> bool:
        if self.mode == "thread":
            return bool(self.stage.slots)
        directory = self.stage.stage_dir()
        return os.path.isdir(directory) and bool(os.listdir(directory))

    def _publish_empty(self) -> None:
        from torch.distributed.checkpoint import save as dcp_save

        dcp_save({}, checkpoint_id=self.root, no_dist=True)

    @catch
    def abort(self):
        if self.state != "open":
            raise invalid("session", f"abort on {self.state} session")
        _vote(self.coord, False)
        self.state = "aborted"
        return Ok(None)


class DcpLoadSession(StorageLoadSession):
    def __init__(self, rank, root, table, layout, index, io=None):
        self.rank = rank
        self.root = root
        self.table = table
        self.layout = layout
        self.index = index
        self.state = "open"
        metadata = os.path.join(root, ".metadata")
        if not os.path.exists(metadata):
            msg = "distcp checkpoint is incomplete: .metadata missing"
            raise io_error("open", msg, root)
        self.stage = _stage_for(root)
        try:
            self.full = _real_dcp_load(table, root)
        except BaseException as exc:  # noqa: BLE001
            raise io_error("read", f"real DCP load failed: {exc}", root) from None
        # Real raw byte offset of each tensor's payload inside the torch-written
        # ``.distcp`` archive, resolved by parsing the sub-archive zip structure.
        self.offsets = _raw_payload_bases(root)

    @catch
    def init_data_accessor(self):
        if self.state != "open":
            raise invalid("session", f"init_data_accessor on {self.state} session")
        return _DcpAccessor("load", self.index, self.stage, self.full)

    @catch
    def get_file_offset(self, pos):
        if self.state != "open":
            raise invalid("session", f"get_file_offset on {self.state} session")
        hit = self.index.slots.get(pos)
        if hit is None:
            raise not_found(
                "tensor_entries",
                f"{pos.tensor_id}:{pos.rank}",
                f"{pos.tensor_id}:{pos.rank} has no physical file",
            )
        _path, slot = hit
        chunks = self.offsets.get(pos.tensor_id)
        if not chunks:
            raise not_found(
                "tensor_entries",
                f"{pos.tensor_id}:{pos.rank}",
                f"{pos.tensor_id}:{pos.rank} has no payload in the archive",
            )
        # Slot's global owned interval and the matched chunk's global origin.
        # local offset = (slot.global_first - chunk.global_origin) * itemsize.
        # no_dist: one chunk holds the whole tensor (origin 0) -> local = global_first.
        # sharded: each chunk holds exactly one rank's shard (origin = that shard's
        # global start) -> local = 0 for the owning rank.
        global_first = int(slot.intervals[0, 0]) * slot.itemsize
        # Total owned bytes = sum of interval lengths * itemsize, NOT
        # (last - first) * itemsize. The latter is wrong for non-contiguous
        # multi-interval slots.
        total_elems = int(sum(b - a for a, b in slot.intervals.tolist()))
        slot_bytes = total_elems * slot.itemsize
        # Select the chunk whose archive file matches the slot's authoritative path
        # (file_path_of(layout, pos)), compared as fully normalized paths against the
        # checkpoint root — never basename-only, never inferred from pos.rank.
        want = os.path.normpath(os.path.join(self.root, _path))
        matches = [
            (rel, base, nbytes)
            for rel, base, nbytes in chunks
            if os.path.normpath(os.path.join(self.root, rel)) == want
        ]
        if matches:
            chosen = matches[0]
        elif len(chunks) == 1:
            chosen = chunks[0]
        else:
            chosen = None
        if chosen is None:
            raise invalid(
                "file_offset",
                f"{pos.tensor_id}:{pos.rank}: no DCP chunk file matches slot path {want!r} "
                f"among {[os.path.normpath(os.path.join(self.root, c[0])) for c in chunks]}",
            )
        rel, base, nbytes = chosen
        # A dedicated shard chunk (one chunk per rank/slot) starts at the slot's own
        # global origin, so local offset is 0. A shared chunk (no_dist, whole tensor
        # for all slots) starts at global 0, so local offset = slot.global_first.
        chunk_origin = 0 if len(chunks) == 1 else global_first
        local = global_first - chunk_origin
        if local < 0 or local + slot_bytes > nbytes:
            raise invalid(
                "file_offset",
                f"{pos.tensor_id}:{pos.rank}: slot bytes [{local},{local + slot_bytes}) not contained "
                f"in chunk {rel} (size {nbytes}); a single base offset cannot represent a slot "
                "spanning multiple DCP chunk files",
            )
        return Ok(base + local)

    @catch
    def finish(self):
        if self.state != "open":
            raise invalid("session", f"finish on {self.state} session")
        self.state = "finished"
        return Ok(None)

"""Real torch DCP backend for the simulation storage sessions.

The upper layer keeps the gdtensor copy plan and data accessor on the data
path. Ranks write their owned slots to a staging area. The coordinator
assembles the logical tensors and writes the checkpoint with the real
``torch.distributed.checkpoint`` API, so the on-disk artifact is genuine DCP,
not a gdtensor-defined JSON format.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict

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
from ._result import catch, invalid, io_error

STAGE_DIR = ".gdtensor_dcp_stage"

_DTYPE_TO_NP = {
    "F16": np.float16,
    "BF16": np.float32,
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,
    "U8": np.uint8,
    "I32": np.int32,
    "I64": np.int64,
    "BOOL": np.bool_,
}

_DTYPE_TO_TORCH = {
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


def _np_dtype(meta):
    try:
        return _DTYPE_TO_NP[meta.dtype]
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


def _slot_bytes(nd, rank, intervals, stage: _Stage, on_disk: bool) -> np.ndarray:
    pos = TensorComponentPos(nd.global_tensor_id, rank)
    if on_disk:
        path = stage.path(pos)
        if not os.path.exists(path):
            raise io_error("read", f"missing staged slot {_slot_key(pos)}", path)
        with open(path, "rb") as handle:
            raw = handle.read()
    else:
        try:
            raw = stage.bytes_of(pos)
        except KeyError:
            raise io_error("read", f"missing staged slot {_slot_key(pos)}") from None
    return np.frombuffer(raw, dtype=_np_dtype(global_tensor_meta_of(nd)))


def _assemble_tensor(nd, stage: _Stage, on_disk: bool) -> np.ndarray:
    """Rebuild one full global tensor from its non-empty storage slots."""
    from ._geometry import nonempty_columns

    meta = global_tensor_meta_of(nd)
    out = np.zeros(int(np.prod(meta.global_shape)), dtype=_np_dtype(meta))
    for rank, intervals in nonempty_columns(nd).items():
        chunk = _slot_bytes(nd, rank, intervals, stage, on_disk)
        expect = iv_total(intervals)
        if chunk.size != expect:
            msg = f"slot rank {rank} has {chunk.size} elements, expected {expect}"
            raise io_error("read", msg)
        cursor = 0
        for lo, hi in intervals:
            n = int(hi - lo)
            out[lo:hi] = chunk[cursor : cursor + n]
            cursor += n
    return out.reshape(tuple(meta.global_shape))


def _torch_tensor(arr: np.ndarray, dtype_name: str):
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(arr))
    if dtype_name == "BF16":
        tensor = tensor.to(torch.bfloat16)
    return tensor


def _real_dcp_save(
    table: DTensorTable, stage: _Stage, root: str, on_disk: bool
) -> None:
    from torch.distributed.checkpoint import save as dcp_save

    state = {
        nd.global_tensor_id: _torch_tensor(
            _assemble_tensor(nd, stage, on_disk), global_tensor_meta_of(nd).dtype
        )
        for nd in table.named_d_tensors
    }
    dcp_save(state, checkpoint_id=root, no_dist=True)


def _empty_torch_tensor(meta):
    import torch

    return torch.empty(
        tuple(meta.global_shape), dtype=getattr(torch, _DTYPE_TO_TORCH[meta.dtype])
    )


def _real_dcp_load(table: DTensorTable, root: str) -> Dict[str, np.ndarray]:
    from torch.distributed.checkpoint import load as dcp_load

    state = {
        nd.global_tensor_id: _empty_torch_tensor(global_tensor_meta_of(nd))
        for nd in table.named_d_tensors
    }
    dcp_load(state, checkpoint_id=root, no_dist=True)
    return {key: value.numpy() for key, value in state.items()}


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

    def __init__(self, mode: str, index: PayloadIndex, stage: _Stage, full=None):
        self.mode = mode
        self.index = index
        self.stage = stage
        self.full = full
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
        from .access import call_buffer_callback

        slot, rel = _resolve_slot(self.index, global_range)
        owned = self._owned_linear(slot.pos)
        n = global_range.size // slot.itemsize
        start = rel // slot.itemsize
        flat = self.full[slot.pos.tensor_id].reshape(-1)
        data = np.ascontiguousarray(flat[owned[start : start + n]]).tobytes()
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
        return Ok(0)

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
        # No copied data: publish a valid empty real DCP checkpoint.
        if not self._any_slot_present():
            self._publish_empty()
            return
        on_disk = self.mode == "process"
        _real_dcp_save(self.table, self.stage, self.root, on_disk)

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

    @catch
    def init_data_accessor(self):
        if self.state != "open":
            raise invalid("session", f"init_data_accessor on {self.state} session")
        return _DcpAccessor("load", self.index, self.stage, self.full)

    @catch
    def get_file_offset(self, pos):
        if self.state != "open":
            raise invalid("session", f"get_file_offset on {self.state} session")
        return Ok(0)

    @catch
    def finish(self):
        if self.state != "open":
            raise invalid("session", f"finish on {self.state} session")
        self.state = "finished"
        return Ok(None)

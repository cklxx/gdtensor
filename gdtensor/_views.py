"""Views and accessors (DTensorsView B.2.1.1, B.2.2.1, B.2.3.1, B.2.3.2).

Lite deviation: the accessor callbacks get one ``memoryview`` instead of
``(buffer_ptr, buffer_len)``. The memoryview carries both values.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from ._dtensor import (
    DTensorTable,
    NamedDTensor,
    global_tensor_meta_of,
    iv_prefix,
    iv_total,
    named_of,
    owned_intervals,
    participants_of,
)
from ._iobackend import FileIO
from ._layout import PayloadIndex, PhysicalLayoutTable, build_payload_index
from ._ranges import RangeFromComputeView, RangeFromStorageView
from ._result import Result, ResultError, catch

Callback = Callable[[memoryview], object]


def _check_cb(out: object) -> None:
    if out is not None and getattr(out, "ok", True) is False:
        raise ResultError(out.error)


# ===================================================================== compute


class ComputeFabric:
    """Simulated device memory of every rank in one job.

    ``buffers[rank][tensor_id]`` is the local shard: owned elements in
    ascending global order (= row-major order of the owned box).
    Reads of ranges that a rank does not own go through the fabric; this
    stands for collective communication (compute -> compute reshard).
    """

    def __init__(self, table: DTensorTable) -> None:
        self.table = table
        self.buffers: Dict[int, Dict[str, np.ndarray]] = {}
        self._iv: Dict[tuple, np.ndarray] = {}
        self.comm_bytes = 0

    def intervals(self, rank: int, tensor_id: str) -> np.ndarray:
        key = (rank, tensor_id)
        if key not in self._iv:
            self._iv[key] = owned_intervals(named_of(self.table, tensor_id), "compute", rank)
        return self._iv[key]

    def allocate(self, ranks: Optional[List[int]] = None, fill: Optional[Callable] = None) -> "ComputeFabric":
        all_ranks = set()
        for nd in self.table.named_d_tensors:
            all_ranks.update(participants_of(nd, "compute"))
        for r in sorted(all_ranks if ranks is None else ranks):
            bufs = self.buffers.setdefault(r, {})
            for nd in self.table.named_d_tensors:
                iv = self.intervals(r, nd.global_tensor_id)
                n = iv_total(iv)
                if n == 0:
                    continue
                isz = global_tensor_meta_of(nd).itemsize
                buf = np.zeros(n * isz, dtype=np.uint8)
                if fill is not None:
                    buf[:] = fill(nd, iv)
                bufs[nd.global_tensor_id] = buf
        return self

    def locate(self, rank: int, rng: RangeFromComputeView, prefer_local: bool = True):
        nd = named_of(self.table, rng.global_tensor_id)
        isz = global_tensor_meta_of(nd).itemsize
        if rng.global_tensor_offset % isz or rng.size % isz:
            raise ResultError(f"compute range {rng} is not element aligned")
        e0 = rng.global_tensor_offset // isz
        e1 = e0 + rng.size // isz
        order = [rank] + [r for r in self.buffers if r != rank] if prefer_local else list(self.buffers)
        for r in order:
            buf = self.buffers.get(r, {}).get(rng.global_tensor_id)
            if buf is None:
                continue
            iv = self.intervals(r, rng.global_tensor_id)
            k = int(np.searchsorted(iv[:, 0], e0, side="right")) - 1
            if k >= 0 and iv[k, 1] >= e1:
                local = int(iv_prefix(iv, np.array([e0]))[0]) * isz
                return r, buf[local:local + rng.size]
        raise ResultError(f"no rank owns compute range {rng}")


@dataclass
class ComputeDataAccessor:
    """DTensorDataAccessor[RangeFromComputeView] bound to one rank."""

    rank: int
    fabric: ComputeFabric

    @catch
    def read_to_buffer(self, global_range: RangeFromComputeView, read_to_buffer: Callback) -> None:
        owner, view = self.fabric.locate(self.rank, global_range)
        if owner != self.rank:
            self.fabric.comm_bytes += global_range.size
        _check_cb(read_to_buffer(memoryview(view)))

    @catch
    def write_from_buffer(self, global_range: RangeFromComputeView, write_from_buffer: Callback) -> None:
        owner, view = self.fabric.locate(self.rank, global_range)
        if owner != self.rank:
            raise ResultError(f"rank {self.rank} writes compute range it does not own: {global_range}")
        _check_cb(write_from_buffer(memoryview(view)))


@dataclass
class ComputeDTensorsView:
    def check_constraints(self):
        from ._constraints import check_constraints as _check
        return _check(self)

    tensor_ids: List[str]
    d_tensor_table: DTensorTable
    data_accessor: ComputeDataAccessor


# ===================================================================== storage


class StorageDataAccessor:
    """DTensorDataAccessor[RangeFromStorageView].

    It only knows contiguous_payload_offset. The active session maps it to
    the real file offset through ``get_file_offset``. The session binding is
    thread-local, so one view serves many rank threads.
    """

    def __init__(self, index: PayloadIndex, io: FileIO) -> None:
        self.index = index
        self.io = io
        self._local = threading.local()

    @property
    def session(self):
        return getattr(self._local, "session", None)

    @session.setter
    def session(self, value) -> None:
        self._local.session = value

    def _file_offset(self, rng: RangeFromStorageView):
        if self.session is None:
            raise ResultError("storage accessor used outside a session")
        fp = self.index.files.get(rng.file_path)
        if fp is None:
            raise ResultError(f"unknown file {rng.file_path}")
        starts = [s.contiguous_payload_offset for s in fp.slots]
        k = int(np.searchsorted(starts, rng.contiguous_payload_offset, side="right")) - 1
        if k < 0:
            raise ResultError(f"offset {rng.contiguous_payload_offset} before first slot")
        slot = fp.slots[k]
        if rng.contiguous_payload_offset + rng.size > slot.contiguous_payload_offset + slot.nbytes:
            raise ResultError(f"storage range {rng} crosses a slot boundary")
        base = self.session.get_file_offset(slot.pos).unwrap()
        return self.session.abspath(rng.file_path), base + rng.contiguous_payload_offset - slot.contiguous_payload_offset

    @catch
    def read_to_buffer(self, global_range: RangeFromStorageView, read_to_buffer: Callback) -> None:
        path, off = self._file_offset(global_range)
        data = self.io.pread(self.session.rank, path, off, global_range.size)
        _check_cb(read_to_buffer(memoryview(data)))

    @catch
    def write_from_buffer(self, global_range: RangeFromStorageView, write_from_buffer: Callback) -> None:
        path, off = self._file_offset(global_range)
        buf = bytearray(global_range.size)
        _check_cb(write_from_buffer(memoryview(buf)))
        self.io.pwrite(self.session.rank, path, off, buf)


@dataclass
class StorageDTensorsView:
    def check_constraints(self):
        from ._constraints import check_constraints as _check
        return _check(self)

    tensor_ids: List[str]
    d_tensor_table: DTensorTable
    data_accessor: StorageDataAccessor
    physical_layout: PhysicalLayoutTable
    # Function fields, called as view.begin_save(view, rank) per the spec.
    begin_save: Callable[["StorageDTensorsView", int], Result] = field(repr=False, default=None)
    begin_load: Callable[["StorageDTensorsView", int], Result] = field(repr=False, default=None)


# DTensorsView = $compute ComputeDTensorsView | $storage StorageDTensorsView
DTensorsView = Union[ComputeDTensorsView, StorageDTensorsView]

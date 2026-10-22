"""Views and accessors (design B.2.1.1, B.2.2.1, B.2.3.1, B.2.3.2).

Accessor callbacks follow C.2: ``callback(buffer_ptr, buffer_len) ->
Result[None]``. ``buffer_ptr`` is the address of a host buffer that stays
valid during the call. ``read_to_buffer`` hands the range bytes to the
callback; ``write_from_buffer`` asks the callback to fill a buffer, then
writes it to the range.
"""
from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .builder_mixin import make_builder_mixin
from .dtensor import (
    DTensorTable,
    global_tensor_meta_of,
    iv_prefix,
    iv_total,
    named_of,
    owned_intervals,
    participants_of,
)
from .iobackend import FileIO
from .layout import PayloadIndex, PhysicalLayoutTable
from .ranges import RangeFromComputeView, RangeFromStorageView
from .result import Fail, catch, invalid, io_error
from ..access import DTensorDataAccessor
from ..tag import Err, Ok


def buffer_address(buf) -> int:
    """Address of a writable numpy array or bytearray."""
    if isinstance(buf, np.ndarray):
        return int(buf.ctypes.data)
    return ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))


def call_buffer_callback(callback, buf) -> None:
    out = callback(buffer_address(buf) if len(buf) else 0, len(buf))
    if type(out) is Err:
        raise Fail(out.value)
    if type(out) is not Ok:
        raise invalid("callback", f"buffer callback must return a Result, got {out!r}")


def copy_from_ptr(buffer_ptr: int, buffer_len: int) -> bytes:
    return ctypes.string_at(buffer_ptr, buffer_len) if buffer_len else b""


def copy_to_ptr(buffer_ptr: int, data) -> None:
    if len(data):
        ctypes.memmove(buffer_ptr, bytes(data), len(data))


# ===================================================================== compute


class ComputeFabric:
    """Simulated device memory of every rank in one job.

    ``buffers[rank][tensor_id]`` is the local shard: owned elements in
    ascending global order (= row-major order of the owned box). Reads of
    ranges a rank does not own go through the fabric; this stands for
    collective communication (compute -> compute reshard).
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
                buf = np.zeros(n * global_tensor_meta_of(nd).itemsize, dtype=np.uint8)
                if fill is not None:
                    buf[:] = fill(nd, iv)
                bufs[nd.global_tensor_id] = buf
        return self

    def locate(self, rank: int, rng: RangeFromComputeView):
        nd = named_of(self.table, rng.global_tensor_id)
        isz = global_tensor_meta_of(nd).itemsize
        if rng.global_tensor_offset % isz or rng.size % isz:
            raise invalid("global_range", f"compute range {rng} is not element aligned")
        e0 = rng.global_tensor_offset // isz
        e1 = e0 + rng.size // isz
        for r in [rank] + [r for r in self.buffers if r != rank]:
            buf = self.buffers.get(r, {}).get(rng.global_tensor_id)
            if buf is None:
                continue
            iv = self.intervals(r, rng.global_tensor_id)
            k = int(np.searchsorted(iv[:, 0], e0, side="right")) - 1
            if k >= 0 and iv[k, 1] >= e1:
                local = int(iv_prefix(iv, np.array([e0]))[0]) * isz
                return r, buf[local:local + rng.size]
        raise invalid("global_range", f"no rank owns compute range {rng}")


class ComputeDataAccessor(DTensorDataAccessor):
    """DTensorDataAccessor[RangeFromComputeView] bound to one rank."""

    def __init__(self, rank: int, fabric: ComputeFabric) -> None:
        self.rank = rank
        self.fabric = fabric

    @catch
    def read_to_buffer(self, global_range: RangeFromComputeView, read_to_buffer) -> None:
        owner, view = self.fabric.locate(self.rank, global_range)
        if owner != self.rank:
            self.fabric.comm_bytes += global_range.size
        call_buffer_callback(read_to_buffer, np.ascontiguousarray(view))

    @catch
    def write_from_buffer(self, global_range: RangeFromComputeView, write_from_buffer) -> None:
        owner, view = self.fabric.locate(self.rank, global_range)
        if owner != self.rank:
            raise invalid("global_range", f"rank {self.rank} writes compute range it does not own: {global_range}")
        call_buffer_callback(write_from_buffer, view)


ComputeDTensorsViewBuilderMixin = make_builder_mixin(
    "ComputeDTensorsView", [("tensor_ids", "list"), ("d_tensor_table", "object"), ("data_accessor", "scalar")])


@dataclass(eq=False)
class ComputeDTensorsView(ComputeDTensorsViewBuilderMixin):
    tensor_ids: List[str] = field(default_factory=list)
    d_tensor_table: DTensorTable = field(default_factory=DTensorTable)
    data_accessor: Any = None

    def __post_init__(self) -> None:
        self.tensor_ids = list(self.tensor_ids)

    def check_constraints(self):
        from .constraints import check_constraints
        return check_constraints(self)


# ===================================================================== storage


class StorageDataAccessor(DTensorDataAccessor):
    """DTensorDataAccessor[RangeFromStorageView].

    It knows ``contiguous_payload_offset`` only. A session maps it to the file
    offset through ``get_file_offset``. A view-level accessor uses the session
    that the executor binds for the current thread; a session-bound accessor
    (``init_data_accessor``) always uses its own session.
    """

    def __init__(self, index: PayloadIndex, io: FileIO, session=None) -> None:
        self.index = index
        self.io = io
        self._fixed = session
        self._local = threading.local()

    @property
    def session(self):
        return self._fixed if self._fixed is not None else getattr(self._local, "session", None)

    def bind(self, session) -> None:
        self._local.session = session

    def _file_offset(self, rng: RangeFromStorageView):
        s = self.session
        if s is None:
            raise invalid("session", "storage accessor used outside a session")
        fp = self.index.files.get(rng.file_path)
        if fp is None:
            raise invalid("global_range", f"unknown file {rng.file_path}")
        starts = [x.contiguous_payload_offset for x in fp.slots]
        k = int(np.searchsorted(starts, rng.contiguous_payload_offset, side="right")) - 1
        if k < 0:
            raise invalid("global_range", f"offset {rng.contiguous_payload_offset} before first slot")
        slot = fp.slots[k]
        if rng.contiguous_payload_offset + rng.size > slot.contiguous_payload_offset + slot.nbytes:
            raise invalid("global_range", f"storage range {rng} crosses a slot boundary")
        base = s.get_file_offset(slot.pos)
        if type(base) is Err:
            raise Fail(base.value)
        return s, s.abspath(rng.file_path), base.value + rng.contiguous_payload_offset - slot.contiguous_payload_offset

    @catch
    def read_to_buffer(self, global_range: RangeFromStorageView, read_to_buffer) -> None:
        s, path, off = self._file_offset(global_range)
        try:
            data = self.io.pread(s.rank, path, off, global_range.size)
        except OSError as exc:
            raise io_error("read", str(exc), path) from None
        call_buffer_callback(read_to_buffer, bytearray(data))

    @catch
    def write_from_buffer(self, global_range: RangeFromStorageView, write_from_buffer) -> None:
        s, path, off = self._file_offset(global_range)
        buf = bytearray(global_range.size)
        call_buffer_callback(write_from_buffer, buf)
        try:
            self.io.pwrite(s.rank, path, off, buf)
        except OSError as exc:
            raise io_error("write", str(exc), path) from None


StorageDTensorsViewBuilderMixin = make_builder_mixin(
    "StorageDTensorsView", [("tensor_ids", "list"), ("d_tensor_table", "object"), ("data_accessor", "scalar"),
                            ("physical_layout", "object"), ("begin_save", "scalar"), ("begin_load", "scalar")])


@dataclass(eq=False)
class StorageDTensorsView(StorageDTensorsViewBuilderMixin):
    tensor_ids: List[str] = field(default_factory=list)
    d_tensor_table: DTensorTable = field(default_factory=DTensorTable)
    data_accessor: Any = None
    physical_layout: PhysicalLayoutTable = field(default_factory=PhysicalLayoutTable)
    # Function fields, called as view.begin_save(view, rank) / view.begin_load(view, rank).
    begin_save: Optional[Callable] = field(default=None, repr=False)
    begin_load: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.tensor_ids = list(self.tensor_ids)

    def check_constraints(self):
        from .constraints import check_constraints
        return check_constraints(self)


DTensorsView = ComputeDTensorsView | StorageDTensorsView

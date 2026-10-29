"""In-memory rank simulation for the simulation CLI."""

from typing import Callable, Dict, List, Optional

import numpy as np

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import DTensorTable, global_tensor_meta_of
from ._geometry import (
    iv_prefix,
    iv_total,
    named_of,
    owned_intervals,
    participants_of,
)
from ._result import catch, invalid
from .access import call_buffer_callback


class ComputeFabric:
    """Simulated device memory of every rank in one job.

    ``buffers[rank][tensor_id]`` is the local shard: owned elements in
    ascending global order (= row-major order of the owned box). Reads of
    ranges a rank does not own go through the fabric; this stands for
    collective communication (reading another rank's shard).
    """

    def __init__(self, table: DTensorTable) -> None:
        self.table = table
        self.buffers: Dict[int, Dict[str, np.ndarray]] = {}
        self._iv: Dict[tuple, np.ndarray] = {}
        self.comm_bytes = 0

    def intervals(self, rank: int, tensor_id: str) -> np.ndarray:
        key = (rank, tensor_id)
        if key not in self._iv:
            self._iv[key] = owned_intervals(
                named_of(self.table, tensor_id), "compute", rank
            )
        return self._iv[key]

    def allocate(
        self, ranks: Optional[List[int]] = None, fill: Optional[Callable] = None
    ) -> "ComputeFabric":
        all_ranks = set()
        for nd in self.table.named_d_tensors:
            all_ranks.update(participants_of(nd, "compute"))
        for rank in sorted(all_ranks if ranks is None else ranks):
            self._allocate_rank(rank, fill)
        return self

    def _allocate_rank(self, rank, fill):
        self.buffers.setdefault(rank, {})
        for nd in self.table.named_d_tensors:
            self._allocate_tensor(rank, nd, fill)

    def _allocate_tensor(self, rank, nd, fill):
        intervals = self.intervals(rank, nd.global_tensor_id)
        count = iv_total(intervals)
        if count == 0:
            return
        data = np.zeros(count * global_tensor_meta_of(nd).itemsize, dtype=np.uint8)
        if fill is not None:
            data[:] = fill(nd, intervals)
        self.buffers[rank][nd.global_tensor_id] = data

    def locate(self, rank: int, rng):
        return _locate_range(self, rank, rng)


def _locate_range(fabric, rank, rng):
    nd = named_of(fabric.table, rng.global_tensor_id)
    isz = global_tensor_meta_of(nd).itemsize
    if rng.global_tensor_offset % isz or rng.size % isz:
        raise invalid("global_range", f"compute range {rng} is not element aligned")
    e0 = rng.global_tensor_offset // isz
    e1 = e0 + rng.size // isz
    for r in [rank] + [r for r in fabric.buffers if r != rank]:
        buf = fabric.buffers.get(r, {}).get(rng.global_tensor_id)
        if buf is None:
            continue
        iv = fabric.intervals(r, rng.global_tensor_id)
        k = int(np.searchsorted(iv[:, 0], e0, side="right")) - 1
        if k >= 0 and iv[k, 1] >= e1:
            local = int(iv_prefix(iv, np.array([e0]))[0]) * isz
            return r, buf[local : local + rng.size]
    raise invalid("global_range", f"no rank owns compute range {rng}")


class ComputeDataAccessor(DTensorDataAccessor):
    """DTensorDataAccessor[RangeFromComputeView] bound to one rank."""

    def __init__(self, rank: int, fabric: ComputeFabric) -> None:
        self.rank = rank
        self.fabric = fabric

    @catch
    def read_to_buffer(self, global_range, read_to_buffer) -> None:
        owner, view = self.fabric.locate(self.rank, global_range)
        if owner != self.rank:
            self.fabric.comm_bytes += global_range.size
        call_buffer_callback(read_to_buffer, np.ascontiguousarray(view))

    @catch
    def write_from_buffer(self, global_range, write_from_buffer) -> None:
        owner, view = self.fabric.locate(self.rank, global_range)
        if owner != self.rank:
            raise invalid(
                "global_range",
                f"rank {self.rank} writes compute range it does not own: {global_range}",
            )
        call_buffer_callback(write_from_buffer, view)

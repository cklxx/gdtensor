"""Main APIs (B.1.1): thin executors over the plan APIs.

They call the plan, open sessions, copy every (src, dst) pair through the
view accessors, and close sessions. They never re-select ranges.
"""
from __future__ import annotations

from typing import Optional

from .plan import (
    d_tensors_view_rank_plan_load,
    d_tensors_view_rank_plan_reshard,
    d_tensors_view_rank_plan_save,
)
from .ranges import CopyPlan, is_compute
from .result import ResultError, catch
from .views import ComputeDTensorsView, DTensorsView, StorageDTensorsView


def _accessor(view: DTensorsView):
    return view.data_accessor


def _copy(plan: CopyPlan, src_view: DTensorsView, dst_view: DTensorsView) -> None:
    src_acc, dst_acc = _accessor(src_view), _accessor(dst_view)
    for src, dst in plan:
        if src.size != dst.size:
            raise ResultError(f"size mismatch in plan pair {src} -> {dst}")
        box = {}

        def take(buf: memoryview) -> None:
            box["data"] = bytes(buf)

        src_acc.read_to_buffer(src, take).unwrap()

        def give(buf: memoryview) -> None:
            buf[:] = box["data"]

        dst_acc.write_from_buffer(dst, give).unwrap()


def _copy_storage_coalesced(plan: CopyPlan, src_view: DTensorsView, dst_view: DTensorsView) -> None:
    """Reshard into storage with aggregation.

    All destination pieces of one slot are assembled into the slot's full
    buffer first, so each slot is one pwrite. Read semantics stay on the
    plan ranges: this changes IO scheduling only, not range selection.
    """
    import numpy as np

    src_acc, dst_acc = _accessor(src_view), _accessor(dst_view)
    index = dst_view.data_accessor.index
    groups: dict = {}
    for src, dst in plan:
        fp = index.files[dst.file_path]
        starts = np.array([s.contiguous_payload_offset for s in fp.slots])
        k = int(np.searchsorted(starts, dst.contiguous_payload_offset, side="right")) - 1
        slot = fp.slots[k]
        groups.setdefault((dst.file_path, slot.pos), []).append(
            (dst.contiguous_payload_offset, dst.size, src))
    for (path, pos), entries in groups.items():
        slot = index.slots[pos][1]
        buf = bytearray(slot.nbytes)
        base = slot.contiguous_payload_offset
        for off, size, src in sorted(entries, key=lambda x: x[0]):
            piece = {}
            src_acc.read_to_buffer(src, lambda data: piece.update(data=bytes(data))).unwrap()
            buf[off - base: off - base + size] = piece["data"]
        from .ranges import RangeFromStorageView
        whole = RangeFromStorageView(path, base, slot.nbytes)
        dst_acc.write_from_buffer(whole, lambda data: data.__setitem__(slice(None), buf)).unwrap()


def _bind(view: DTensorsView, session) -> None:
    if isinstance(view, StorageDTensorsView):
        view.data_accessor.session = session


@catch
def d_tensors_view_rank_save(rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView) -> None:
    plan = d_tensors_view_rank_plan_save(rank, compute_view, storage_view).unwrap()
    session = storage_view.begin_save(storage_view, rank).unwrap()
    _bind(storage_view, session)
    try:
        _copy(plan, compute_view, storage_view)
    except BaseException:
        session.abort()
        raise
    finally:
        _bind(storage_view, None)
    session.commit().unwrap()


@catch
def d_tensors_view_rank_load(rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView) -> None:
    plan = d_tensors_view_rank_plan_load(rank, compute_view, storage_view).unwrap()
    files = sorted({src.file_path for src, _ in plan})
    session = storage_view.begin_load(storage_view, rank, files).unwrap()
    _bind(storage_view, session)
    try:
        _copy(plan, storage_view, compute_view)
    finally:
        _bind(storage_view, None)
        session.finish()


@catch
def d_tensors_view_rank_reshard(rank: int, src_view: DTensorsView, dst_view: DTensorsView,
                                coalesce_storage: bool = False) -> None:
    plan = d_tensors_view_rank_plan_reshard(rank, src_view, dst_view).unwrap()
    src_session = dst_session = None
    try:
        if isinstance(src_view, StorageDTensorsView):
            files = sorted({s.file_path for s, _ in plan if not is_compute(s)})
            src_session = src_view.begin_load(src_view, rank, files).unwrap()
            _bind(src_view, src_session)
        if isinstance(dst_view, StorageDTensorsView):
            dst_session = dst_view.begin_save(dst_view, rank).unwrap()
            _bind(dst_view, dst_session)
        copy = _copy_storage_coalesced if coalesce_storage and isinstance(dst_view, StorageDTensorsView) else _copy
        copy(plan, src_view, dst_view)
    except BaseException:
        if dst_session is not None:
            dst_session.abort()
        if src_session is not None:
            src_session.finish()
        raise
    finally:
        _bind(src_view, None)
        _bind(dst_view, None)
    if dst_session is not None:
        dst_session.commit().unwrap()
    if src_session is not None:
        src_session.finish().unwrap()

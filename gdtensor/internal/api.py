"""IO APIs (design B.1.1 / 5.4): thin executors over the plan APIs.

Each executor calls the plan, opens sessions, copies every (src, dst) pair
through the view accessors, and closes sessions. It never re-selects ranges.
"""
from __future__ import annotations

from .plan import (
    d_tensors_view_rank_plan_load,
    d_tensors_view_rank_plan_reshard,
    d_tensors_view_rank_plan_save,
)
from .ranges import CopyPlan
from .result import catch, invalid, must_ok
from .views import StorageDTensorsView, copy_from_ptr, copy_to_ptr
from ..tag import Ok


def _copy(plan: CopyPlan, src_view, dst_view) -> None:
    src_acc, dst_acc = src_view.data_accessor, dst_view.data_accessor
    for src, dst in plan:
        if src.size != dst.size:
            raise invalid("copy_plan", f"size mismatch in plan pair {src} -> {dst}")
        box = {}

        def take(buffer_ptr: int, buffer_len: int):
            box["data"] = copy_from_ptr(buffer_ptr, buffer_len)
            return Ok(None)

        must_ok(src_acc.read_to_buffer(src, take))

        def give(buffer_ptr: int, buffer_len: int):
            copy_to_ptr(buffer_ptr, box["data"])
            return Ok(None)

        must_ok(dst_acc.write_from_buffer(dst, give))


def _bind(view, session) -> None:
    if type(view) is StorageDTensorsView:
        view.data_accessor.bind(session)


@catch
def d_tensors_view_rank_save(rank: int, compute_view, storage_view) -> None:
    plan = must_ok(d_tensors_view_rank_plan_save(rank, compute_view, storage_view))
    session = must_ok(storage_view.begin_save(storage_view, rank))
    _bind(storage_view, session)
    try:
        _copy(plan, compute_view, storage_view)
    except BaseException:
        session.abort()
        raise
    finally:
        _bind(storage_view, None)
    must_ok(session.commit())


@catch
def d_tensors_view_rank_load(rank: int, compute_view, storage_view) -> None:
    plan = must_ok(d_tensors_view_rank_plan_load(rank, compute_view, storage_view))
    session = must_ok(storage_view.begin_load(storage_view, rank))
    _bind(storage_view, session)
    try:
        _copy(plan, storage_view, compute_view)
    finally:
        _bind(storage_view, None)
    must_ok(session.finish())


@catch
def d_tensors_view_rank_reshard(rank: int, src_view, dst_view) -> None:
    plan = must_ok(d_tensors_view_rank_plan_reshard(rank, src_view, dst_view))
    src_session = dst_session = None
    try:
        if type(src_view) is StorageDTensorsView:
            src_session = must_ok(src_view.begin_load(src_view, rank))
            _bind(src_view, src_session)
        if type(dst_view) is StorageDTensorsView:
            dst_session = must_ok(dst_view.begin_save(dst_view, rank))
            _bind(dst_view, dst_session)
        _copy(plan, src_view, dst_view)
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
        must_ok(dst_session.commit())
    if src_session is not None:
        must_ok(src_session.finish())

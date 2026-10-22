"""Public IO executors: plan, open sessions, copy, and close sessions."""

from __future__ import annotations

from dataclasses import replace
import ctypes
import logging

from .plan import (
    d_tensors_view_rank_plan_load,
    d_tensors_view_rank_plan_reshard,
    d_tensors_view_rank_plan_save,
)
from .ranges import CopyPlan
from .result import catch, invalid, must_ok
from .views import ComputeDTensorsView, DTensorsView, StorageDTensorsView
from .. import Result
from ..tag import Ok


_LOG = logging.getLogger(__name__)


def _read(src_acc, src) -> bytes:
    chunks = []
    read_size = 0

    def take(buffer_ptr: int, buffer_len: int):
        nonlocal read_size
        if buffer_len < 0 or read_size + buffer_len > src.size:
            raise invalid("buffer_len", "read chunks exceed range size")
        if buffer_len:
            chunks.append(ctypes.string_at(buffer_ptr, buffer_len))
        read_size += buffer_len
        return Ok(None)

    must_ok(src_acc.read_to_buffer(src, take))
    if read_size != src.size:
        raise invalid(
            "buffer_len", f"read chunks have {read_size} bytes, expected {src.size}"
        )
    return b"".join(chunks)


def _write(dst_acc, dst, data: bytes) -> None:
    written = 0

    def give(buffer_ptr: int, buffer_len: int):
        nonlocal written
        if buffer_len < 0 or written + buffer_len > dst.size:
            raise invalid("buffer_len", "write chunks exceed range size")
        if buffer_len:
            ctypes.memmove(buffer_ptr, data[written : written + buffer_len], buffer_len)
        written += buffer_len
        return Ok(None)

    must_ok(dst_acc.write_from_buffer(dst, give))
    if written != dst.size:
        raise invalid(
            "buffer_len", f"write chunks have {written} bytes, expected {dst.size}"
        )


def _copy(plan: CopyPlan, src_view, dst_view) -> None:
    for src, dst in plan:
        if src.size != dst.size:
            raise invalid("copy_plan", f"size mismatch in plan pair {src} -> {dst}")
        _write(dst_view.data_accessor, dst, _read(src_view.data_accessor, src))


def _log_plan(operation, rank, plan):
    _LOG.info(
        "%s rank=%d phase=plan pairs=%d bytes=%d",
        operation,
        rank,
        len(plan),
        sum(src.size for src, _ in plan),
    )


def _session_view(view, session):
    return replace(view, data_accessor=must_ok(session.init_data_accessor()))


def _cleanup(session, method):
    if session is None:
        return
    try:
        getattr(session, method)()
    except Exception as exc:
        _LOG.warning("session %s failed during error cleanup: %s", method, exc)


@catch
def d_tensors_view_rank_save(
    rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView
) -> Result[None]:
    if (
        type(compute_view) is not ComputeDTensorsView
        or type(storage_view) is not StorageDTensorsView
    ):
        raise invalid("views", "rank_save requires a compute view and a storage view")
    plan = must_ok(d_tensors_view_rank_plan_save(rank, compute_view, storage_view))
    _log_plan("save", rank, plan)
    assert storage_view.begin_save is not None
    session = must_ok(storage_view.begin_save(storage_view, rank))
    try:
        _copy(plan, compute_view, _session_view(storage_view, session))
    except BaseException:
        _cleanup(session, "abort")
        raise
    must_ok(session.commit())
    _LOG.info("save rank=%d phase=committed", rank)


@catch
def d_tensors_view_rank_load(
    rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView
) -> Result[None]:
    if (
        type(compute_view) is not ComputeDTensorsView
        or type(storage_view) is not StorageDTensorsView
    ):
        raise invalid("views", "rank_load requires a compute view and a storage view")
    plan = must_ok(d_tensors_view_rank_plan_load(rank, compute_view, storage_view))
    _log_plan("load", rank, plan)
    assert storage_view.begin_load is not None
    session = must_ok(storage_view.begin_load(storage_view, rank))
    try:
        _copy(plan, _session_view(storage_view, session), compute_view)
    except BaseException:
        _cleanup(session, "finish")
        raise
    must_ok(session.finish())
    _LOG.info("load rank=%d phase=finished", rank)


@catch
def d_tensors_view_rank_reshard(
    rank: int, src_view: DTensorsView, dst_view: DTensorsView
) -> Result[None]:
    if type(src_view) not in (ComputeDTensorsView, StorageDTensorsView) or type(
        dst_view
    ) not in (ComputeDTensorsView, StorageDTensorsView):
        raise invalid("views", "rank_reshard requires two DTensorsView values")
    plan = must_ok(d_tensors_view_rank_plan_reshard(rank, src_view, dst_view))
    _log_plan("reshard", rank, plan)
    src_session = dst_session = None
    try:
        if type(src_view) is StorageDTensorsView:
            assert src_view.begin_load is not None
            src_session = must_ok(src_view.begin_load(src_view, rank))
            src_view = _session_view(src_view, src_session)
        if type(dst_view) is StorageDTensorsView:
            assert dst_view.begin_save is not None
            dst_session = must_ok(dst_view.begin_save(dst_view, rank))
            dst_view = _session_view(dst_view, dst_session)
        _copy(plan, src_view, dst_view)
    except BaseException:
        _cleanup(dst_session, "abort")
        _cleanup(src_session, "finish")
        raise
    try:
        if dst_session is not None:
            must_ok(dst_session.commit())
    except BaseException:
        _cleanup(src_session, "finish")
        raise
    if src_session is not None:
        must_ok(src_session.finish())
    _LOG.info("reshard rank=%d phase=finished", rank)

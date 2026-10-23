"""Public IO executors: plan, open sessions, copy, and close sessions."""

from __future__ import annotations

from dataclasses import replace
import ctypes
import logging

from .plan import (
    d_tensors_view_rank_plan_load,
    d_tensors_view_rank_plan_save,
    validate_rank_load,
    validate_rank_save,
)
from .ranges import CopyPlan
from .result import catch, invalid, must_ok
from .views import ComputeDTensorsView, StorageDTensorsView
from .. import Result
from .products import Ok


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
    assert isinstance(plan, list)
    validate_rank_save(rank, compute_view, storage_view, plan)
    _log_plan("save", rank, plan)
    assert storage_view.begin_save is not None
    session = must_ok(storage_view.begin_save(storage_view, rank))
    assert session is not None
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
    assert isinstance(plan, list)
    validate_rank_load(rank, compute_view, storage_view, plan)
    _log_plan("load", rank, plan)
    assert storage_view.begin_load is not None
    session = must_ok(storage_view.begin_load(storage_view, rank))
    assert session is not None
    try:
        _copy(plan, _session_view(storage_view, session), compute_view)
    except BaseException:
        _cleanup(session, "finish")
        raise
    must_ok(session.finish())
    _LOG.info("load rank=%d phase=finished", rank)


# ---------------------------------------------------------------- B.1.1 chain load


def _finish_chain_sessions(opened):
    """finish load sessions; keep and return the first error."""
    primary = None
    for session in reversed(opened):
        primary = _finish_one_session(session, primary)
    return primary


def _finish_one_session(session, primary):
    try:
        must_ok(session.finish())
    except BaseException as exc:
        if primary is None:
            return exc
    return primary


def _collect_concrete_chains(read_chain, plan_chain, legs_for):
    """(read chain, plan chain, legs) tuples in data-flow order."""
    from .chain import (
        DirectDTensorsReadChain,
        MixedDTensorsReadChain,
        OneLayerCachedDTensorsReadChain,
        TwoLayerCachedDTensorsReadChain,
    )

    t = type(read_chain)
    if t in (
        DirectDTensorsReadChain,
        OneLayerCachedDTensorsReadChain,
        TwoLayerCachedDTensorsReadChain,
    ):
        return [(read_chain, plan_chain, legs_for(read_chain))]
    if t is MixedDTensorsReadChain:
        return [
            (seg.read_chain, seg_pc, legs_for(seg.read_chain))
            for seg, seg_pc in zip(
                read_chain.disjoint_tensor_ids_and_read_chain_list,
                plan_chain.disjoint_tensor_ids_and_copy_plan_chain_list,
            )
        ]
    raise invalid("read_chain", f"unknown read chain type {t.__name__}")


def _run_concrete_chain(rank, chain, pc, legs) -> None:
    """Open the src storage session once and execute every leg in order."""
    storage = chain.src
    assert type(storage) is StorageDTensorsView
    assert storage.begin_load is not None
    _LOG.info("chain_load rank=%d phase=begin", rank)
    session = must_ok(storage.begin_load(storage, rank))
    assert session is not None
    storage_session_view = _session_view(storage, session)
    try:
        _execute_chain_legs(pc, legs, storage, storage_session_view)
    except BaseException:
        primary = _finish_chain_sessions([session])
        if primary is not None:
            raise primary from None
        raise
    primary = _finish_chain_sessions([session])
    if primary is not None:
        raise primary
    _LOG.info("chain_load rank=%d phase=finished", rank)


def _execute_chain_legs(pc, legs, storage, storage_session_view) -> None:
    leg_plans = _chain_plans(pc)
    for (src, dst), plan in zip(legs, leg_plans):
        eff_src = storage_session_view if src is storage else src
        _copy(plan, eff_src, dst)


@catch
def d_tensors_view_rank_chain_load(rank: int, read_chain) -> Result[None]:
    from .chain import d_tensors_view_rank_plan_chain_load, legs_of

    plan_chain = must_ok(d_tensors_view_rank_plan_chain_load(rank, read_chain))
    chains = _collect_concrete_chains(read_chain, plan_chain, legs_of)
    for chain, pc, legs in chains:
        _run_concrete_chain(rank, chain, pc, legs)


def _chain_plans(plan_chain):
    from .chain import (
        DirectCopyPlanChain,
        DisjointTensorIdsAndCopyPlanChain,
        OneLayerCachedCopyPlanChain,
        TwoLayerCachedCopyPlanChain,
    )

    if type(plan_chain) is DisjointTensorIdsAndCopyPlanChain:
        plan_chain = plan_chain.copy_plan_chain
    t = type(plan_chain)
    if t is DirectCopyPlanChain:
        return [plan_chain.src_to_dst]
    if t is OneLayerCachedCopyPlanChain:
        return [plan_chain.src_to_cache, plan_chain.cache_to_dst]
    if t is TwoLayerCachedCopyPlanChain:
        return [
            plan_chain.src_to_small_cache,
            plan_chain.small_cache_to_big_cache,
            plan_chain.big_cache_to_dst,
        ]
    raise invalid("copy_plan_chain", f"unknown plan chain type {t.__name__}")

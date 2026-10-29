"""Public IO executors: plan, open sessions, copy, and close sessions."""

from __future__ import annotations

from dataclasses import replace
import ctypes
import logging
import os
import threading
import time
import weakref

from .plan import (
    d_tensors_view_rank_plan_load,
    d_tensors_view_rank_plan_save,
    validate_rank_load,
    validate_rank_save,
)
from .ranges import CopyPlan, RangeFromComputeView
from .result import catch, check_failed, invalid, must_ok
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
#
# Cached legs run from one node-level propagation plan (propagation.py). Every
# segment of a (layer, node) is filled once, stays resident until every
# consumer (a rank drain or a child segment fill) has ACKed its bytes, and is
# admitted only while the layer's resident bytes stay inside the plan budget.
# The stat function reports the same budget, so it is a hard bound.
#
# Coordination state lives in a _SlotTable per concrete chain. The default
# table is in-process (threads). Another process model replaces it through
# CACHE_COORDINATOR_FACTORY with the same method contract.

_WAIT_ENV = "GDTENSOR_CACHE_WAIT_SECONDS"
_DEFAULT_WAIT_SECONDS = 60.0


class _NodeState:
    """Residency of one (cache layer, node): segments, bytes, generation."""

    def __init__(self, budget: int, cohort: set) -> None:
        self.budget = budget
        self.cohort = set(cohort)
        self.slots: dict = {}  # key -> {"filled", "owed"}
        self.resident = 0
        self.gen = 0
        self.active = 0
        self.joined: set = set()
        self.error: BaseException | None = None

    def join(self, rank: int) -> int:
        """Join the current generation, or open a new one when the previous
        generation is idle and either complete or re-entered by this rank."""
        idle = self.active == 0 and self.joined
        if idle and (rank in self.joined or self.cohort <= self.joined):
            self.gen += 1
            self.slots, self.resident, self.error = {}, 0, None
            self.joined = set()
        self.active += 1
        self.joined.add(rank)
        return self.gen

    def fits(self, size: int) -> bool:
        return self.resident + size <= self.budget

    def admit(self, key, size: int) -> bool:
        if not self.fits(size):
            return False
        self.slots[key] = {"filled": False, "owed": None}
        self.resident += size
        return True

    def release(self, key) -> None:
        self.slots.pop(key, None)
        self.resident -= key[2]

    def check(self, gen: int) -> None:
        """Raise the error recorded for this generation, if any."""
        if self.error is not None and self.gen == gen:
            raise self.error

    def reserve(self, segs: list) -> bool:
        """Admit all segments or none; each starts unfilled with its owed set."""
        if any(s.key in self.slots for s in segs):
            return False
        if not self.fits(sum(s.size for s in segs)):
            return False
        for s in segs:
            self.admit(s.key, s.size)
            self.slots[s.key]["owed"] = s.owed()
        return True

    def ack(self, key, piece) -> None:
        slot = self.slots.get(key)
        if slot is None:
            raise check_failed("cache_ack", f"segment {key} released before {piece}")
        slot["owed"].discard(piece)
        if not slot["owed"]:
            self.release(key)

    def fail(self, exc: BaseException, key) -> None:
        if key is not None:
            self.release(key)
        self.error = self.error or exc

    def owed_to(self) -> list:
        owed = {c for slot in self.slots.values() for c, _o, _n in slot["owed"] or ()}
        return sorted(owed, key=str)


class _SlotTable:
    """In-process coordinator for one concrete chain: one lock, one version."""

    def __init__(self, plan) -> None:
        self.plan = plan
        self.cond = threading.Condition()
        self.version = 0
        self.nodes = {
            (layer, node): _NodeState(budget, plan.cohort[layer][node])
            for layer, budgets in enumerate(plan.budget)
            for node, budget in budgets.items()
        }

    def node(self, layer: int, node: int) -> _NodeState:
        return self.nodes[(layer, node)]

    def bump(self) -> None:
        self.version += 1
        self.cond.notify_all()


CACHE_COORDINATOR_FACTORY = _SlotTable
_view_slots: dict = {}
_table_lock = threading.RLock()


def _table_key(chain) -> tuple:
    from .propagation import cache_layers

    return (
        tuple(id(view) for view in cache_layers(chain)),
        chain.src.d_tensor_table,
        chain.src.physical_layout,
        chain.dst.d_tensor_table,
        chain.num_ranks_per_node,
    )


def _drop_table(key) -> None:
    # Pop under the lock, drop the table outside it: dropping the table can
    # run another view finalizer that takes the same lock.
    with _table_lock:
        table = _view_slots.pop(key, None)
    del table


def _slots(chain) -> _SlotTable:
    from .propagation import build_chain_plan, cache_layers

    key = _table_key(chain)
    with _table_lock:
        table = _view_slots.get(key)
    if table is not None:
        return table
    table = CACHE_COORDINATOR_FACTORY(build_chain_plan(chain))
    with _table_lock:
        table = _view_slots.setdefault(key, table)
    weakref.finalize(cache_layers(chain)[0], _drop_table, key)
    return table


def _wait_seconds() -> float:
    raw = os.environ.get(_WAIT_ENV, "")
    if not raw:
        return _DEFAULT_WAIT_SECONDS
    value = float(raw)
    if value <= 0:
        raise invalid(_WAIT_ENV, "must be a positive number of seconds")
    return value


class _RankRun:
    """One rank's pass over a cached concrete chain.

    Methods stay flat (no nested blocks) so that each step reads as one unit:
    drain what is filled, otherwise fill what fits, otherwise wait.
    """

    def __init__(self, table, rank: int, caches, storage, dst) -> None:
        from .propagation import rank_closure

        self.t, self.plan, self.rank = table, table.plan, rank
        self.caches, self.storage, self.dst = caches, storage, dst
        self.last = len(caches) - 1
        closure = rank_closure(self.plan, rank)
        self.gens = dict.fromkeys(sorted({(lay, n) for lay, n, _k in closure}), 0)
        self.wait = _wait_seconds()

    def run(self) -> None:
        with self.t.cond:
            self.gens = {ln: self.t.node(*ln).join(self.rank) for ln in self.gens}
        try:
            self._loop(list(self.plan.drains[self.rank]))
        finally:
            self._leave()

    def _leave(self) -> None:
        with self.t.cond:
            _ = [self._dec(ln) for ln in self.gens]
            self.t.bump()

    def _dec(self, ln) -> None:
        self.t.node(*ln).active -= 1

    def _loop(self, pending: list) -> None:
        """The wait limit restarts after every unit of progress."""
        deadline = time.monotonic() + self.wait
        while pending:
            version = self._version_checked()
            moved = self._step(pending)
            deadline = time.monotonic() + self.wait if moved else deadline
            _ = moved or self._block(version, deadline, pending)

    def _step(self, pending: list) -> bool:
        """Drain every filled piece; else fill one segment. True on progress."""
        done = [p for p in pending if self._filled(self.last, p[0], p[1])]
        for piece in done:
            self._drain(piece)
            pending.remove(piece)
        return bool(done) or any(
            self._ensure(self.last, n, k) for n, k, _o, _s in pending
        )

    def _version_checked(self) -> int:
        with self.t.cond:
            _ = [self.t.node(*ln).check(gen) for ln, gen in self.gens.items()]
            return self.t.version

    def _block(self, version: int, deadline: float, pending: list) -> None:
        with self.t.cond:
            left = deadline - time.monotonic()
            moved = left > 0 and self.t.cond.wait_for(
                lambda: self.t.version != version, timeout=left
            )
        if not moved:
            raise self._timeout(pending)

    def _timeout(self, pending: list):
        node, key = pending[0][0], pending[0][1]
        state = self.t.node(self.last, node)
        owed = state.owed_to()
        return check_failed(
            "cache_wait",
            f"rank {self.rank}: no progress for {self.wait}s on cache layer "
            f"{self.last} node {node} segment {key}; resident {state.resident} of "
            f"budget {state.budget} bytes owed to {owed}",
        )

    def _filled(self, layer: int, node: int, key) -> bool:
        with self.t.cond:
            slot = self.t.node(layer, node).slots.get(key)
            return slot is not None and slot["filled"]

    def _ensure(self, layer: int, node: int, key) -> bool:
        """Try to make one segment resident; True when this call filled it.

        Upstream segments are requested only while this layer has room, so a
        full layer does not pin upstream budget for a segment it cannot admit.
        """
        seg = self.plan.segments[layer][node][key]
        if not self._room(layer, node, seg):
            return False
        if layer > 0 and not self._parents_ready(layer, node, seg):
            return False
        if not self._reserve(layer, node, [seg]):
            return False
        self._fill(layer, node, seg)
        return True

    def _room(self, layer: int, node: int, seg) -> bool:
        with self.t.cond:
            state = self.t.node(layer, node)
            state.check(self.gens[(layer, node)])
            return seg.key not in state.slots and state.fits(seg.size)

    def _reserve(self, layer: int, node: int, segs: list) -> bool:
        """Admit all segments or none; each starts unfilled with its owed set."""
        with self.t.cond:
            state = self.t.node(layer, node)
            state.check(self.gens[(layer, node)])
            return state.reserve(segs)

    def _parents_ready(self, layer: int, node: int, seg) -> bool:
        """Fill the missing parent segments; True when all are filled.

        The plan gives every child segment exactly one parent segment, and a
        segment never exceeds the parent budget, so an empty parent layer
        always admits it.
        """
        pnode = self.plan.parent[layer][node]
        psegs = self.plan.segments[layer - 1][pnode]
        keys = sorted({pkey for pkey, _o, _n in seg.parents})
        if all(self._filled(layer - 1, pnode, k) for k in keys):
            return True
        missing = [psegs[k] for k in keys if not self._resident(layer - 1, pnode, k)]
        if not missing or not self._reserve(layer - 1, pnode, missing):
            return False
        for pseg in missing:
            self._fill(layer - 1, pnode, pseg)
        return True

    def _resident(self, layer: int, node: int, key) -> bool:
        with self.t.cond:
            return key in self.t.node(layer, node).slots

    def _fill(self, layer: int, node: int, seg) -> None:
        try:
            data = self._source_bytes(layer, seg)
            _write(self.caches[layer].data_accessor, _crange(seg.key), data)
        except BaseException as exc:
            self._fail(layer, node, exc, seg.key)
            raise
        with self.t.cond:
            self.t.node(layer, node).slots[seg.key]["filled"] = True
            self.t.bump()
        if layer > 0:
            self._ack_parents(layer, node, seg)

    def _ack_parents(self, layer: int, node: int, seg) -> None:
        pnode = self.plan.parent[layer][node]
        consumer = ("seg", layer, node, seg.key)
        for pkey, off, size in seg.parents:
            self._ack(layer - 1, pnode, pkey, (consumer, off, size))

    def _source_bytes(self, layer: int, seg) -> bytes:
        if layer == 0:
            acc = self.storage.data_accessor
            return b"".join(_read(acc, rng) for rng, _off in seg.storage)
        acc = self.caches[layer - 1].data_accessor
        return b"".join(
            _read(acc, RangeFromComputeView(seg.tid, off, size))
            for _k, off, size in seg.parents
        )

    def _drain(self, piece) -> None:
        node, key, off, size = piece
        rng = RangeFromComputeView(key[0], off, size)
        try:
            data = _read(self.caches[self.last].data_accessor, rng)
            _write(self.dst.data_accessor, rng, data)
        except BaseException as exc:
            self._fail(self.last, node, exc, None)
            raise
        self._ack(self.last, node, key, (("rank", self.rank), off, size))

    def _fail(self, layer: int, node: int, exc: BaseException, key) -> None:
        """Record one node-wide error for this generation; every rank of the
        generation that reads this node raises the same error object."""
        with self.t.cond:
            self.t.node(layer, node).fail(exc, key)
            self.t.bump()

    def _ack(self, layer: int, node: int, key, piece) -> None:
        with self.t.cond:
            self.t.node(layer, node).ack(key, piece)
            self.t.bump()


def _crange(key) -> RangeFromComputeView:
    return RangeFromComputeView(key[0], key[1], key[2])


def _run_cached(rank: int, chain, storage_session_view) -> None:
    from .propagation import cache_layers

    table = _slots(chain)
    _RankRun(table, rank, cache_layers(chain), storage_session_view, chain.dst).run()


def _run_one_concrete(rank: int, chain) -> None:
    """Open and finish one load session for one concrete chain (design 5.8:
    each Mixed segment has its own source session)."""
    from .chain import DirectDTensorsReadChain, rank_direct_plan

    storage = chain.src
    if type(storage) is not StorageDTensorsView or storage.begin_load is None:
        raise invalid("read_chain", "chain src must be a StorageDTensorsView")
    direct = type(chain) is DirectDTensorsReadChain
    plan = rank_direct_plan(rank, chain) if direct else None
    _LOG.info("chain_load rank=%d phase=begin concrete", rank)
    session = must_ok(storage.begin_load(storage, rank))
    try:
        view = _session_view(storage, session)
        if direct:
            _copy(plan, view, chain.dst)
        else:
            _run_cached(rank, chain, view)
    except BaseException:
        _cleanup(session, "finish")
        raise
    must_ok(session.finish())
    _LOG.info("chain_load rank=%d phase=finished concrete", rank)


@catch
def d_tensors_view_rank_chain_load(rank: int, read_chain) -> Result[None]:
    from .chain import concrete_chains

    for chain in concrete_chains(read_chain):
        _run_one_concrete(rank, chain)

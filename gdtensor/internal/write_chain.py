"""Thin write-chain executor (`chain_save` / `d_tensors_view_rank_chain_save`).

Mirrors the coordination contract of api.py for compute -> cache -> storage
write chains without altering api.py or propagation.py. Every write segment
tracks expected producer pieces; once all producers stage their bytes, an
eligible local (or designated service) rank reads the coalesced contiguous
segment and flushes it downstream.
"""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import logging
import os
import pickle
import tempfile
import threading
import time
import weakref

from .api import (
    _COORD_DIR_ENV,
    _COORD_ENV,
    _CacheRange,
    _cleanup,
    _copy,
    _error_cleanup,
    _new_owner,
    _open_process_file,
    _read,
    _session_view,
    _wait_seconds,
    _write,
    fcntl,
)
from .chain import (
    DirectDTensorsWriteChain,
    concrete_write_chains,
    legs_of,
    rank_direct_write_plan,
)
from .plan import _check_same_tensors
from .ranges import RangeFromComputeView
from .result import as_error, catch, check_failed, invalid, must_ok
from .views import ComputeDTensorsView, StorageDTensorsView
from .write_propagation import (
    WriteSegment,
    build_write_chain_plan,
    write_cache_layers,
    write_execution_closure,
)
from .. import Result
from ..access import _content_digest

_LOG = logging.getLogger(__name__)
_WMAGIC = "gdtensor-wcoord-v1"


class _WriteNodeState:
    """Residency of one (cache layer, node) during write coalescing."""

    def __init__(self, budget: int, cohort: set) -> None:
        self.budget = budget
        self.cohort = set(cohort)
        self.slots: dict = {}  # key -> {"owed": set, "ready": bool, "flushing": bool}
        self.completed: set = set()
        self.resident = 0
        self.gen = 0
        self.active = 0
        self.joined: set = set()
        self.entered: dict = {}
        self.call = 0
        self.error = None

    def join(self, rank: int) -> int:
        call = self.entered.get(rank, 0) + 1
        idle = self.active == 0 and self.joined
        if idle and (rank in self.joined or call > self.call):
            self.gen += 1
            self.slots, self.resident, self.error = {}, 0, None
            self.completed = set()
            self.joined = set()
        self.active += 1
        self.joined.add(rank)
        self.entered[rank] = call
        self.call = max(self.call, call)
        return self.gen

    def fits(self, size: int) -> bool:
        return self.resident + size <= self.budget

    def check(self, gen: int) -> None:
        if self.error is not None and self.gen == gen:
            from .result import Fail

            raise Fail(replace(self.error))

    def admit(self, seg: WriteSegment) -> bool:
        if seg.key in self.slots:
            return True
        if seg.key in self.completed or not self.fits(seg.size):
            return False
        self.slots[seg.key] = {
            "owed": seg.expected(),
            "ready": False,
            "flushing": False,
        }
        self.resident += seg.size
        return True

    def reserve_new(self, segs: list) -> bool:
        fresh = [s for s in segs if s.key not in self.slots]
        if any(s.key in self.completed for s in fresh):
            return False
        if not self.fits(sum(s.size for s in fresh)):
            return False
        _ = [self.admit(s) for s in fresh]
        return True

    def record_stage(self, key, piece) -> bool:
        slot = self.slots.get(key)
        if slot is None:
            raise check_failed(
                "cache_stage", f"segment {key} missing during stage of {piece}"
            )
        slot["owed"].discard(piece)
        if not slot["owed"]:
            slot["ready"] = True
            return True
        return False

    def claim_flush(self, key) -> bool:
        slot = self.slots.get(key)
        if slot is None or not slot["ready"] or slot["flushing"]:
            return False
        slot["flushing"] = True
        return True

    def complete_flush(self, key) -> None:
        if key in self.slots:
            self.slots.pop(key)
            self.resident -= key[2]
        self.completed.add(key)

    def release_unclaimed(self, key) -> None:
        slot = self.slots.get(key)
        if slot is not None and not slot["flushing"]:
            self.slots.pop(key)
            self.resident -= key[2]

    def fail(self, exc: BaseException, key=None) -> None:
        if key is not None:
            self.release_unclaimed(key)
        if self.error is None:
            self.error = replace(as_error(exc))

    def pending_producers(self) -> list:
        need = {p for slot in self.slots.values() for p, _o, _n in slot["owed"]}
        return sorted(need, key=str)


class _WriteSlotTable:
    """In-process write-chain coordinator."""

    in_memory = True

    def __init__(self, plan, key: str | None = None) -> None:
        self.plan = plan
        self.key = key
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.activity_lock = threading.Lock()
        self.version = 0
        self.inflight = 0
        self.nodes = {
            (layer, node): _WriteNodeState(budget, plan.cohort[layer][node])
            for layer, budgets in enumerate(plan.budget)
            for node, budget in budgets.items()
        }

    def node(self, layer: int, node: int) -> _WriteNodeState:
        return self.nodes[(layer, node)]

    def bump(self) -> None:
        self.version += 1
        self.cond.notify_all()

    def mark_touched(self, _key) -> None:
        """No-op: in-memory state is shared directly."""

    def participate(self, _rank: int):
        return nullcontext()

    def check_alive(self) -> None:
        """Thread callers share one process."""

    def activity(self, _owner, delta: int) -> None:
        with self.activity_lock:
            self.inflight += delta

    def finish(self, _owner) -> None:
        """No process registration in thread mode."""


class _WriteFileCond:
    """threading.Condition adapter over _FileWriteSlotTable."""

    def __init__(self, table) -> None:
        self.t = table

    def __enter__(self):
        self.t._enter()
        return self

    def __exit__(self, *_exc) -> None:
        self.t._exit()

    def notify_all(self) -> None:
        pass

    def wait_for(self, predicate, timeout=None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        result = None
        while result is None:
            result = self._step(predicate, deadline)
        return result

    def _step(self, predicate, deadline):
        self.t.check_alive()
        if predicate():
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        self.t._drop()
        self._poll()
        return None

    def _poll(self) -> None:
        try:
            time.sleep(self.t._POLL)
        finally:
            self.t._reacquire()


class _FileWriteSlotTable:
    """Process-safe flock coordinator for one concrete write chain."""

    _POLL = 0.005
    in_memory = False

    def __init__(self, plan, key: str) -> None:
        if fcntl is None:
            raise invalid("write_chain", "file coordinator needs POSIX fcntl")
        self.plan = plan
        self.key = key
        base = os.path.join(
            os.environ.get(_COORD_DIR_ENV, tempfile.gettempdir()),
            f"gdtensor-wcoord-{key}",
        )
        self._path = base + ".state"
        self._pid = os.getpid()
        self._fd = _open_process_file(base + ".lock", "a+b")
        self._thread_lock = threading.RLock()
        self._held: int | None = None
        self._dirty = False
        self.version = 0
        self.inflight = 0
        self.nodes = {}
        self.runs = {}
        self.error = None
        self._touched: set = set()
        self.cond = _WriteFileCond(self)
        self._reacquire()
        self._drop()

    def node(self, layer: int, node: int) -> _WriteNodeState:
        return self.nodes[(layer, node)]

    def bump(self) -> None:
        self._dirty = True

    def mark_touched(self, key) -> None:
        self._touched.add(key)

    @contextmanager
    def participate(self, rank: int):
        assert fcntl is not None
        self._check_process()
        owner, path = _new_owner(os.path.dirname(self._path))
        try:
            fcntl.flock(owner, fcntl.LOCK_EX)
            token = os.path.basename(path)
            self._register(token, rank)
            yield token
        finally:
            self._close_owner(owner, path)

    def _close_owner(self, owner, path) -> None:
        owner.close()
        if self._pid == os.getpid():
            os.unlink(path)

    def _register(self, path: str, rank: int) -> None:
        with self.cond:
            self.check_alive()
            self.runs[path] = {"rank": rank, "inflight": 0}
            self.bump()

    def activity(self, owner, delta: int) -> None:
        self.runs[owner]["inflight"] += delta
        self.inflight += delta

    def finish(self, owner) -> None:
        if self.runs[owner]["inflight"] == 0:
            self.runs.pop(owner)

    def check_alive(self) -> None:
        if self.error is not None:
            raise check_failed(*self.error)
        dead = next(
            (run["rank"] for path, run in self.runs.items() if self._unlocked(path)),
            None,
        )
        if dead is not None:
            self.error = (
                "cache_peer_down",
                f"rank {dead} exited during cache IO; use a new coordinator directory",
            )
            self.bump()
            raise check_failed(*self.error)

    def _unlocked(self, path: str) -> bool:
        assert fcntl is not None
        try:
            owner = _open_process_file(
                os.path.join(os.path.dirname(self._path), path), "r+b"
            )
        except FileNotFoundError:
            return True
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        finally:
            owner.close()

    def _check_process(self) -> None:
        if self._pid != os.getpid():
            raise check_failed(
                "cache_fork", "forked during cache IO; start a new chain_save"
            )

    def _unlock(self) -> None:
        assert fcntl is not None
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self._thread_lock.release()

    def _enter(self) -> None:
        self._check_process()
        if self._held == threading.get_ident():
            raise RuntimeError("_FileWriteSlotTable.cond is not reentrant")
        self._thread_lock.acquire()
        self._lock_refresh()

    def _reacquire(self) -> None:
        self._check_process()
        self._thread_lock.acquire()
        self._lock_refresh()

    def _lock_refresh(self) -> None:
        assert fcntl is not None
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            self._refresh()
            self._held = threading.get_ident()
            self._dirty = False
        except BaseException:
            self._unlock()
            raise

    def _exit(self) -> None:
        if self._held != threading.get_ident() or self._pid != os.getpid():
            return
        try:
            self._commit()
        finally:
            self._drop()

    def _commit(self) -> None:
        if not self._dirty:
            return
        self.version += 1
        latest = self._read_state()
        latest["nodes"].update({k: self.nodes[k] for k in self._touched})
        latest.update(
            version=self.version,
            inflight=self.inflight,
            runs=self.runs,
            error=self.error,
        )
        self._write_state(latest)

    def _drop(self) -> None:
        self._held = None
        self._unlock()

    def _refresh(self) -> None:
        state = self._read_state()
        self.nodes = state["nodes"]
        self.version = state["version"]
        self.inflight = state["inflight"]
        self.runs = state["runs"]
        self.error = state["error"]
        self._touched = set()

    @staticmethod
    def _read_pickle(path: str):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _read_state(self) -> dict:
        try:
            state = self._read_pickle(self._path)
        except FileNotFoundError:
            return self._empty_state()
        except (EOFError, pickle.UnpicklingError, ValueError) as exc:
            raise check_failed(
                "cache_state", "invalid coordinator state; use a new directory"
            ) from exc
        if not isinstance(state, dict) or state.get("magic") != _WMAGIC:
            raise check_failed(
                "cache_state", "incompatible coordinator state; use a new directory"
            )
        return state

    def _empty_state(self) -> dict:
        return {
            "magic": _WMAGIC,
            "version": 0,
            "inflight": 0,
            "nodes": {
                (layer, node): _WriteNodeState(budget, self.plan.cohort[layer][node])
                for layer, budgets in enumerate(self.plan.budget)
                for node, budget in budgets.items()
            },
            "runs": {},
            "error": None,
        }

    def _write_state(self, state: dict) -> None:
        blob = pickle.dumps(state, protocol=5)
        tmp = self._path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, self._path)


def _write_coordinator_factory():
    mode = os.environ.get(_COORD_ENV, "file")
    if mode == "thread":
        return _WriteSlotTable
    if mode == "file":
        return _FileWriteSlotTable
    raise invalid(_COORD_ENV, "must be 'file' or 'thread'")


_write_view_slots: dict = {}
_write_view_users: dict = {}
_write_table_lock = threading.RLock()
_WRITE_FINGERPRINT_LOCK = threading.Lock()
_WRITE_FINGERPRINTS: dict = {}


@dataclass
class _WriteFingerprint:
    chain: object
    future: Future
    users: int = 0


def _write_after_fork() -> None:
    global _write_table_lock, _WRITE_FINGERPRINT_LOCK
    _write_table_lock = threading.RLock()
    _WRITE_FINGERPRINT_LOCK = threading.Lock()
    _WRITE_FINGERPRINTS.clear()
    _write_view_slots.clear()
    _write_view_users.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_write_after_fork)


def _write_chain_fingerprint(chain) -> str:
    tables = [view.d_tensor_table for view in write_cache_layers(chain)]
    tables += [chain.src.d_tensor_table, chain.dst.d_tensor_table]
    payload = ("write", tables, chain.dst.physical_layout, chain.num_ranks_per_node)
    return _content_digest(payload)[:20]


def _fp_entry(chain):
    key = id(chain)
    with _WRITE_FINGERPRINT_LOCK:
        entry = _WRITE_FINGERPRINTS.get(key)
        leader = entry is None
        if entry is None:
            entry = _WriteFingerprint(chain, Future())
            _WRITE_FINGERPRINTS[key] = entry
        entry.users += 1
    return entry, leader


def _fp_release(key, entry) -> None:
    with _WRITE_FINGERPRINT_LOCK:
        entry.users -= 1
        if entry.users == 0 and _WRITE_FINGERPRINTS.get(key) is entry:
            _WRITE_FINGERPRINTS.pop(key, None)


def _resolve_fp(chain, entry, leader):
    future = entry.future
    if leader:
        try:
            future.set_result(_write_chain_fingerprint(chain))
        except BaseException as exc:
            future.set_exception(exc)
            raise
    return future.result()


def _drop_write_table(key, view_id, expected) -> None:
    with _write_table_lock:
        table = _write_view_slots.get(key)
        users = _write_view_users.get(key)
        if users is not None:
            users.pop(view_id, None)
        if table is expected() and not users:
            _write_view_slots.pop(key, None)
            _write_view_users.pop(key, None)
    del table


def _track_write_view(users, view, cache_key, table) -> None:
    view_id = id(view)
    if view_id in users:
        return
    users[view_id] = weakref.ref(view)
    weakref.finalize(view, _drop_write_table, cache_key, view_id, weakref.ref(table))


def _write_slots(chain, active_fp):
    views = write_cache_layers(chain)
    namespaces = []
    for view in views:
        acc = view.data_accessor
        ns = getattr(acc, "cache_namespace", lambda: None)()
        namespaces.append(("object", os.getpid(), id(acc)) if ns is None else ns)
    fingerprint = _resolve_fp(chain, *active_fp)
    key = _content_digest((fingerprint, tuple(namespaces)))[:20]
    directory = os.path.realpath(os.environ.get(_COORD_DIR_ENV, tempfile.gettempdir()))
    cache_key = (os.getpid(), os.environ.get(_COORD_ENV, "file"), directory, key)
    with _write_table_lock:
        table = _write_view_slots.get(cache_key)
        if table is None:
            table = _write_coordinator_factory()(build_write_chain_plan(chain), key)
            _write_view_slots[cache_key] = table
            _write_view_users[cache_key] = {}
        users = _write_view_users[cache_key]
        _ = [_track_write_view(users, view, cache_key, table) for view in views]
    return table


class _WriteRankRun:
    """One rank's pass over a cached concrete write chain."""

    def __init__(self, table, rank: int, caches, src, storage) -> None:
        self.t, self.plan, self.rank = table, table.plan, rank
        self.caches, self.src, self.storage = caches, src, storage
        self.last = len(caches) - 1
        self.closure = write_execution_closure(self.plan, rank)
        self.services = list(self.plan.services.get(rank, ()))
        self.gens = dict.fromkeys(sorted({(lay, n) for lay, n, _k in self.closure}), 0)
        self.wait = _wait_seconds()
        self.flush_items = self._initial_flush_items()

    def _initial_flush_items(self) -> dict:
        items = sorted(self.closure, key=self._sort_item)
        return {
            layer: [
                (node, key)
                for lay, node, key in items
                if lay == layer and self.rank in self.plan.flush_ranks[(lay, node, key)]
            ]
            for layer in range(len(self.caches))
        }

    def _sort_item(self, item):
        layer, node, key = item
        seg = self.plan.segments[layer][node][key]
        return (layer, seg.cell[1], seg.cell[2], seg.tid, seg.off, node)

    def _sort_stage(self, piece):
        node, key, off, _size = piece
        seg = self.plan.segments[0][node][key]
        return (seg.cell[1], seg.cell[2], seg.tid, off, node)

    def _range(self, layer: int, node: int, tid: str, off: int, size: int):
        gen = self.gens[(layer, node)]
        identity = ("write", self.t.key, layer, gen)
        return _CacheRange(tid, off, size, node, identity, _content_digest(identity))

    def run(self) -> None:
        with self.t.participate(self.rank) as self.owner:
            self._run()

    def _run(self) -> None:
        with self.t.cond:
            self.t.check_alive()
            self.gens = {ln: self.t.node(*ln).join(self.rank) for ln in self.gens}
            _ = [self.t.mark_touched(ln) for ln in self.gens]
            self.t.bump()
        pending = sorted(self.plan.stages[self.rank], key=self._sort_stage)
        try:
            self._wait_cohort()
            self._loop(pending)
            self._version_checked()
        finally:
            self._leave()

    def _leave(self) -> None:
        with self.t.cond:
            _ = [self._dec(ln) for ln in self.gens]
            self.t.finish(self.owner)
            self.t.bump()

    def _dec(self, ln) -> None:
        self.t.node(*ln).active -= 1
        self.t.mark_touched(ln)

    def _wait_cohort(self) -> None:
        deadline = time.monotonic() + self.wait
        while not self._cohort_round(deadline):
            pass

    def _cohort_round(self, deadline: float) -> bool:
        with self.t.cond:
            self.t.check_alive()
            _ = [self.t.node(*ln).check(gen) for ln, gen in self.gens.items()]
            gaps = [(lay, n) for lay, n in self.gens if self._cohort_needs_wait(lay, n)]
        if not gaps:
            return True
        if time.monotonic() >= deadline:
            raise self._cohort_timeout(gaps[0])
        with self.t.cond:
            version = self.t.version
            left = max(0.0, deadline - time.monotonic())
            self.t.cond.wait_for(
                lambda: self.t.version != version, timeout=min(left, 0.05)
            )
        return False

    def _cohort_needs_wait(self, layer: int, node: int) -> bool:
        state = self.t.node(layer, node)
        if not (state.cohort - state.joined):
            return False
        if self.plan.supply[layer][node] <= state.budget:
            return False
        return len(state.cohort) > 1

    def _cohort_timeout(self, gap):
        layer, node = gap
        state = self.t.node(layer, node)
        missing = [("rank", r) for r in sorted(state.cohort - state.joined)]
        return check_failed(
            "cache_wait",
            f"rank {self.rank}: cache layer {layer} node {node} supplies "
            f"{self.plan.supply[layer][node]} bytes but budget is {state.budget}; "
            f"missing producers {missing} must call chain_save concurrently",
        )

    def _version_checked(self) -> int:
        with self.t.cond:
            self.t.check_alive()
            _ = [self.t.node(*ln).check(gen) for ln, gen in self.gens.items()]
            return self.t.version

    def _has_work(self, pending: list) -> bool:
        return bool(pending or self.services or self._has_ready_flush())

    def _loop(self, pending: list) -> None:
        deadline = time.monotonic() + self.wait
        while self._has_work(pending):
            deadline = self._loop_tick(pending, deadline)

    def _loop_tick(self, pending: list, deadline: float) -> float:
        version = self._version_checked()
        moved = self._step(pending)
        if moved:
            return time.monotonic() + self.wait
        if self._has_work(pending):
            self._block(version, deadline, pending)
        return deadline

    def _has_ready_flush(self) -> bool:
        with self.t.cond:
            return any(
                key in self.t.node(layer, node).slots
                for layer, items in self.flush_items.items()
                for node, key in items
            )

    def _step(self, pending: list) -> bool:
        if self._flush_step():
            self._prune_services()
            return True
        if self._stage_step(pending):
            self._prune_services()
            return True
        self._prune_services()
        return False

    def _prune_services(self) -> None:
        if not self.services:
            return
        with self.t.cond:
            self.services = [
                item
                for item in self.services
                if item[2] not in self.t.node(item[0], item[1]).completed
            ]

    def _flush_step(self) -> bool:
        return any(self._flush_layer(layer) for layer in range(self.last, -1, -1))

    def _flush_layer(self, layer: int) -> bool:
        items = self.flush_items[layer]
        self._drop_completed_flushes(layer, items)
        if layer == self.last:
            return any(self._try_flush_storage(node, key) for node, key in list(items))
        return self._try_flush_interlayer(layer, items)

    def _drop_completed_flushes(self, layer: int, items: list) -> None:
        with self.t.cond:
            items[:] = [
                (n, k) for n, k in items if k not in self.t.node(layer, n).completed
            ]

    def _try_flush_storage(self, node: int, key) -> bool:
        if not self._claim_storage(node, key):
            return False
        seg = self.plan.segments[self.last][node][key]
        self._execute_storage_flush(node, seg)
        return True

    def _claim_storage(self, node: int, key) -> bool:
        with self.t.cond:
            state = self.t.node(self.last, node)
            state.check(self.gens[(self.last, node)])
            claimed = state.claim_flush(key)
            (self.t.mark_touched((self.last, node)), self.t.bump()) if claimed else None
            return claimed

    def _execute_storage_flush(self, node: int, seg: WriteSegment) -> None:
        try:
            self._inflight(1)
        except BaseException as exc:
            _error_cleanup(exc, self._fail, self.last, node, exc, None)
            raise
        try:
            self._copy_to_storage(node, seg)
        except BaseException as exc:
            _error_cleanup(exc, self._flush_failed, self.last, node, exc)
            raise
        self._finish_storage_flush(node, seg.key)

    def _copy_to_storage(self, node: int, seg: WriteSegment) -> None:
        data = _read(
            self.caches[self.last].data_accessor,
            self._range(self.last, node, seg.tid, seg.off, seg.size),
        )
        dst_acc = self.storage.data_accessor
        for dst_rng, rel in seg.storage:
            _write(dst_acc, dst_rng, data[rel : rel + dst_rng.size])

    def _finish_storage_flush(self, node: int, key) -> None:
        with self.t.cond:
            self.t.node(self.last, node).complete_flush(key)
            self.t.activity(self.owner, -1)
            self.t.mark_touched((self.last, node))
            self.t.bump()

    def _try_flush_interlayer(self, layer: int, items: list) -> bool:
        ready = self._ready_interlayer(layer, items)
        return any(
            self._flush_interlayer_pass(layer, ready, hot_first)
            for hot_first in (True, False)
        )

    def _ready_interlayer(self, layer: int, items: list) -> list:
        with self.t.cond:
            return [
                (n, k)
                for n, k in items
                if self._is_flushable(self.t.node(layer, n), k, layer, n)
            ]

    def _is_flushable(self, state, key, layer: int, node: int) -> bool:
        state.check(self.gens[(layer, node)])
        slot = state.slots.get(key)
        return bool(slot and slot["ready"] and not slot["flushing"])

    def _flush_interlayer_pass(self, layer: int, ready: list, hot_first: bool) -> bool:
        return any(
            self._try_interlayer_item(layer, node, key, hot_first)
            for node, key in ready
        )

    def _try_interlayer_item(self, layer: int, node: int, key, hot_first: bool) -> bool:
        tnode = self.plan.target[layer][node]
        seg = self.plan.segments[layer][node][key]
        tsegs = [
            self.plan.segments[layer + 1][tnode][tk] for tk, _o, _s in seg.children
        ]
        if not self._claim_interlayer(layer, node, seg, tnode, tsegs, hot_first):
            return False
        self._execute_interlayer_flush(layer, node, seg, tnode)
        return True

    def _claim_interlayer(
        self, layer: int, node: int, seg, tnode: int, tsegs: list, hot_first: bool
    ) -> bool:
        with self.t.cond:
            return self._claim_interlayer_locked(
                layer, node, seg, tnode, tsegs, hot_first
            )

    def _claim_interlayer_locked(
        self, layer: int, node: int, seg, tnode: int, tsegs: list, hot_first: bool
    ) -> bool:
        state = self.t.node(layer, node)
        tstate = self.t.node(layer + 1, tnode)
        state.check(self.gens[(layer, node)])
        tstate.check(self.gens[(layer + 1, tnode)])
        if hot_first and not any(ts.key in tstate.slots for ts in tsegs):
            return False
        if not self._is_flushable(state, seg.key, layer, node):
            return False
        if not tstate.reserve_new(tsegs):
            return False
        state.slots[seg.key]["flushing"] = True
        self.t.mark_touched((layer + 1, tnode))
        self.t.mark_touched((layer, node))
        self.t.bump()
        return True

    def _execute_interlayer_flush(
        self, layer: int, node: int, seg: WriteSegment, tnode: int
    ) -> None:
        try:
            self._inflight(1)
        except BaseException as exc:
            _error_cleanup(exc, self._fail_pair, layer, node, layer + 1, tnode, exc)
            raise
        try:
            self._copy_interlayer(layer, node, seg, tnode)
        except BaseException as exc:
            _error_cleanup(exc, self._interlayer_failed, layer, node, tnode, exc)
            raise
        self._finish_interlayer_flush(layer, node, seg, tnode)

    def _copy_interlayer(
        self, layer: int, node: int, seg: WriteSegment, tnode: int
    ) -> None:
        data = _read(
            self.caches[layer].data_accessor,
            self._range(layer, node, seg.tid, seg.off, seg.size),
        )
        t_acc = self.caches[layer + 1].data_accessor
        for _tkey, off, size in seg.children:
            _write(
                t_acc,
                self._range(layer + 1, tnode, seg.tid, off, size),
                data[off - seg.off : off - seg.off + size],
            )

    def _finish_interlayer_flush(
        self, layer: int, node: int, seg: WriteSegment, tnode: int
    ) -> None:
        producer = ("seg", layer, node, seg.key)
        with self.t.cond:
            tstate = self.t.node(layer + 1, tnode)
            _ = [
                tstate.record_stage(tk, (producer, off, size))
                for tk, off, size in seg.children
            ]
            self.t.node(layer, node).complete_flush(seg.key)
            self.t.activity(self.owner, -1)
            self.t.mark_touched((layer + 1, tnode))
            self.t.mark_touched((layer, node))
            self.t.bump()

    def _stage_step(self, pending: list) -> bool:
        if not pending:
            return False
        piece = self._pick_hot_stage(pending)
        if piece is not None:
            self._run_stage(pending, piece)
            return True
        piece = self._admit_cold_stage(pending)
        if piece is not None:
            self._run_stage(pending, piece)
            return True
        return False

    def _pick_hot_stage(self, pending: list):
        with self.t.cond:
            return next(
                (p for p in pending if self._is_hot_piece_locked(p)),
                None,
            )

    def _is_hot_piece_locked(self, piece) -> bool:
        node, key, _off, _size = piece
        state = self.t.node(0, node)
        state.check(self.gens[(0, node)])
        return key in state.slots

    def _admit_cold_stage(self, pending: list):
        passes = (True, False) if self.last > 0 else (False,)
        with self.t.cond:
            return next(
                (hit for h in passes if (hit := self._pick_cold_locked(pending, h))),
                None,
            )

    def _pick_cold_locked(self, pending: list, hot_l1: bool):
        return next(
            (p for p in pending if self._try_admit_piece_locked(p, hot_l1)),
            None,
        )

    def _try_admit_piece_locked(self, piece, hot_l1: bool) -> bool:
        node, key, _off, _size = piece
        state = self.t.node(0, node)
        state.check(self.gens[(0, node)])
        if key in state.slots:
            return True
        seg = self.plan.segments[0][node][key]
        if key in state.completed or not state.fits(seg.size):
            return False
        if self.last > 0 and not self._reserve_l1_for_stage(node, seg, hot_l1):
            return False
        if not state.admit(seg):
            return False
        self.t.mark_touched((0, node))
        self.t.bump()
        return True

    def _reserve_l1_for_stage(self, node: int, seg: WriteSegment, hot_l1: bool) -> bool:
        tnode = self.plan.target[0][node]
        tstate = self.t.node(1, tnode)
        tstate.check(self.gens[(1, tnode)])
        tsegs = [self.plan.segments[1][tnode][tk] for tk, _o, _s in seg.children]
        if hot_l1 and not any(ts.key in tstate.slots for ts in tsegs):
            return False
        if not tstate.reserve_new(tsegs):
            return False
        self.t.mark_touched((1, tnode))
        return True

    def _run_stage(self, pending: list, piece) -> None:
        node, key, off, size = piece
        try:
            self._inflight(1)
        except BaseException as exc:
            _error_cleanup(exc, self._fail, 0, node, exc, None)
            raise
        try:
            data = _read(
                self.src.data_accessor, RangeFromComputeView(key[0], off, size)
            )
            _write(
                self.caches[0].data_accessor,
                self._range(0, node, key[0], off, size),
                data,
            )
        except BaseException as exc:
            _error_cleanup(exc, self._flush_failed, 0, node, exc)
            raise
        with self.t.cond:
            self.t.node(0, node).record_stage(key, (("rank", self.rank), off, size))
            self.t.activity(self.owner, -1)
            self.t.mark_touched((0, node))
            self.t.bump()
        pending.remove(piece)

    def _block(self, version: int, deadline: float, pending: list) -> None:
        with self.t.cond:
            done = self._block_locked(version, deadline)
        if not done:
            raise self._timeout(pending)

    def _block_locked(self, version: int, deadline: float) -> bool:
        return self._wait_until_deadline(version, deadline) or self._drain_inflight(
            version
        )

    def _wait_until_deadline(self, version: int, deadline: float) -> bool:
        moved = False
        active = True
        while not moved and active and time.monotonic() < deadline:
            left = deadline - time.monotonic()
            moved = self.t.cond.wait_for(
                lambda: self.t.version != version, timeout=left
            )
            active = moved or self.t.inflight > 0
        return moved

    def _drain_inflight(self, version: int) -> bool:
        moved = False
        while not moved and self.t.inflight > 0:
            moved = self.t.cond.wait_for(
                lambda: self.t.version != version or self.t.inflight == 0,
                timeout=0.05,
            )
        return moved

    def _timeout(self, pending: list):
        fallback = next(
            (
                (layer_, n, k)
                for layer_, items in self.flush_items.items()
                for n, k in items
            ),
            (0, 0, ("", 0, 0)),
        )
        layer, node, key = (
            (0, pending[0][0], pending[0][1])
            if pending
            else (self.services[0] if self.services else fallback)
        )
        state = self.t.node(layer, node)
        need = state.pending_producers()
        return check_failed(
            "cache_wait",
            f"rank {self.rank}: no progress for {self.wait}s on cache layer "
            f"{layer} node {node} segment {key}; resident {state.resident} of "
            f"budget {state.budget} bytes waiting for producers {need}",
        )

    def _inflight(self, delta: int) -> None:
        if self.t.in_memory:
            self.t.activity(self.owner, delta)
            return
        with self.t.cond:
            self.t.activity(self.owner, delta)
            self.t.bump()

    def _fail(self, layer: int, node: int, exc: BaseException, key) -> None:
        downstream = self._fail_targets(layer, node)
        with self.t.cond:
            self._fail_node(layer, node, exc, key)
            _ = [self._fail_node(lay, n, exc, None) for lay, n in downstream]
            self.t.bump()

    def _fail_targets(self, layer: int, node: int) -> list:
        out, cur = [], node
        for lay in range(layer + 1, self.last + 1):
            cur = self.plan.target[lay - 1][cur]
            out.append((lay, cur))
        return out

    def _fail_node(self, layer: int, node: int, exc: BaseException, key) -> None:
        state = self.t.nodes.get((layer, node))
        if state is None:
            return
        state.fail(exc, key)
        self.t.mark_touched((layer, node))

    def _fail_pair(
        self, l0: int, n0: int, l1: int, n1: int, exc: BaseException
    ) -> None:
        self._fail(l0, n0, exc, None)

    def _flush_failed(self, layer: int, node: int, exc: BaseException) -> None:
        self._fail(layer, node, exc, None)
        self._inflight(-1)

    def _interlayer_failed(
        self, layer: int, node: int, tnode: int, exc: BaseException
    ) -> None:
        self._fail_pair(layer, node, layer + 1, tnode, exc)
        self._inflight(-1)


def _validate_concrete_write_legs(chain) -> None:
    for src, dst in legs_of(chain):
        _check_same_tensors(src, dst)


def _run_save_session(dst, rank: int, mode: str, body) -> None:
    session = must_ok(dst.begin_save(dst, rank))
    assert session is not None
    try:
        body(_session_view(dst, session))
    except BaseException:
        _cleanup(session, "abort")
        raise
    must_ok(session.commit())
    _LOG.info("chain_save rank=%d phase=committed %s", rank, mode)


def _run_cached_write(rank: int, chain) -> None:
    active = _fp_entry(chain)
    try:
        table = _write_slots(chain, active)
        if rank not in table.plan.ranks:
            raise invalid(
                "rank", f"rank {rank} not in write-chain ranks {table.plan.ranks}"
            )
        _run_save_session(
            chain.dst,
            rank,
            "concrete",
            lambda view: _WriteRankRun(
                table, rank, write_cache_layers(chain), chain.src, view
            ).run(),
        )
    finally:
        _fp_release(id(chain), active[0])


def _run_one_concrete_write(rank: int, chain) -> None:
    if (
        type(chain.src) is not ComputeDTensorsView
        or type(chain.dst) is not StorageDTensorsView
        or chain.dst.begin_save is None
    ):
        raise invalid(
            "write_chain",
            "chain must connect ComputeDTensorsView to StorageDTensorsView",
        )
    _validate_concrete_write_legs(chain)
    if type(chain) is DirectDTensorsWriteChain:
        plan = rank_direct_write_plan(rank, chain)
        _run_save_session(
            chain.dst, rank, "direct", lambda view: _copy(plan, chain.src, view)
        )
        return
    _run_cached_write(rank, chain)


@catch
def d_tensors_view_rank_chain_save(rank: int, write_chain) -> Result[None]:
    for chain in concrete_write_chains(write_chain):
        _run_one_concrete_write(rank, chain)

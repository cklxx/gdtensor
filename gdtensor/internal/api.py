"""Public IO executors: plan, open sessions, copy, and close sessions."""

from __future__ import annotations

from dataclasses import replace
import ctypes
import hashlib
import logging
import os
import pickle
import tempfile
import threading
import time
import weakref
from typing import Any

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

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl: Any = None


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
        self.error: tuple | None = None  # (check_name, msg), pickle-safe

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
            name, msg = self.error
            raise check_failed(name, msg)

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
        if self.error is None:
            name = getattr(exc, "check_name", None) or type(exc).__name__
            self.error = (name, str(exc) or name)

    def owed_to(self) -> list:
        owed = {c for slot in self.slots.values() for c, _o, _n in slot["owed"] or ()}
        return sorted(owed, key=str)


class _SlotTable:
    """In-process coordinator for one concrete chain: one lock, one version."""

    def __init__(self, plan, key: str | None = None) -> None:
        self.plan = plan
        self.key = key
        self.cond = threading.Condition()
        self.version = 0
        self.inflight = 0
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

    def mark_touched(self, key) -> None:
        """No-op: this table shares one in-memory state, so there is no
        snapshot merge to protect."""


# ---------------------------------------------------------------- process-safe table
#
# The real deployment shape is one process per rank. _SlotTable cannot see
# another process's state, so the default coordinator is _FileSlotTable: the
# plan state lives in one pickle file guarded by fcntl.flock, and any process
# that builds the same concrete chain (same content fingerprint) shares it.
# Threads of one process serialize on an RLock before taking the flock; the
# cond contract (enter/exit/wait_for) is duck-typed by _FileCond.

_COORD_ENV = "GDTENSOR_CACHE_COORDINATOR"
_COORD_DIR_ENV = "GDTENSOR_CACHE_DIR"
_MAGIC = "gdtensor-coord-v1"


def _DBG(*parts) -> None:
    if _DBG_ENABLED:
        print("[coord]", " ".join(map(str, parts)), flush=True)


_DBG_ENABLED = os.environ.get("GDTENSOR_CACHE_COORD_DEBUG") == "1"


class _FileCond:
    """threading.Condition contract over _FileSlotTable.

    `with cond:` takes the thread lock and the file lock, then refreshes the
    local state. wait_for drops both locks, sleeps briefly, re-locks and
    re-reads, so writers in other processes can make progress.
    """

    def __init__(self, table) -> None:
        self.t = table

    def __enter__(self):
        _DBG("enter", threading.current_thread().name)
        self.t._enter()
        return self

    def __exit__(self, *_exc) -> None:
        _DBG("exit", threading.current_thread().name)
        self.t._exit()

    def notify_all(self) -> None:
        pass

    def wait_for(self, predicate, timeout=None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        result = None
        while result is None:
            result = self._wait_step(predicate, deadline)
        return result

    def _wait_step(self, predicate, deadline):
        action = self._wait_ready(predicate, deadline)
        if action == "poll":
            _DBG("wfdrop", threading.current_thread().name)
            self.t._drop()
            self._poll()
            return None
        return action == "ready"

    def _wait_ready(self, predicate, deadline) -> str:
        if predicate():
            return "ready"
        if deadline is not None and time.monotonic() >= deadline:
            return "timeout"
        return "poll"

    def _poll(self) -> None:
        try:
            time.sleep(self.t._POLL)
        finally:
            _DBG("wfreq", threading.current_thread().name)
            self.t._reacquire()


class _FileSlotTable:
    """Process-safe coordinator for one concrete chain (content fingerprint).

    The state is {magic, version, inflight, nodes: {(layer, node): _NodeState}}.
    Every mutation runs under a process RLock and an exclusive flock on the
    state file, then bumps the version and writes the file atomically
    (temp file + os.replace). Waits poll the file version.
    """

    _POLL = 0.005

    def __init__(self, plan, key: str) -> None:
        if fcntl is None:
            raise invalid("read_chain", "file coordinator needs POSIX fcntl")
        self.plan = plan
        self.key = key
        base = os.path.join(
            os.environ.get(_COORD_DIR_ENV, tempfile.gettempdir()),
            f"gdtensor-coord-{key}",
        )
        self._path = base + ".state"
        self._fd = open(base + ".lock", "a+b")
        self._thread_lock = threading.RLock()
        self._held = False
        self._dirty = False
        self.version = 0
        self.inflight = 0
        self.nodes = {}
        self._touched: set = set()
        self.cond = _FileCond(self)
        self._reacquire()
        self._drop()  # load the initial state, then release both locks

    def node(self, layer: int, node: int) -> _NodeState:
        return self.nodes[(layer, node)]

    def bump(self) -> None:
        """Mark this critical section as a write: __exit__ persists the new
        version and state. Read-only sections (e.g. wait_for polling) do not
        bump, so a peer's version change is the only wake-up signal."""
        self._dirty = True

    def mark_touched(self, key) -> None:
        """Record one (layer, node) state this critical section changed; the
        writer merges exactly these states into the latest on-disk version,
        avoiding a stale snapshot from rolling back a peer's update."""
        self._touched.add(key)

    def _enter(self) -> None:
        _DBG("t_enter", threading.current_thread().name, "held", self._held)
        self._thread_lock.acquire()
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)  # type: ignore[possibly-missing-attribute]
            self._refresh()
            self._check_reentrant()
            self._held = True
            self._dirty = False
        except BaseException:
            self._unlock()
            raise

    def _check_reentrant(self) -> None:
        if self._held:
            raise RuntimeError("_FileSlotTable.cond is not reentrant")

    def _unlock(self) -> None:
        self._thread_lock.release()

    def _exit(self) -> None:
        _DBG(
            "t_exit",
            threading.current_thread().name,
            "v",
            self.version,
            "d",
            self._dirty,
        )
        if self._dirty:
            self.version += 1
            latest = self._read_state()
            latest["nodes"].update({k: self.nodes[k] for k in self._touched})
            latest["version"] = self.version
            latest["inflight"] = self.inflight
            self._write_state(latest)
        self._held = False
        fcntl.flock(self._fd, fcntl.LOCK_UN)  # type: ignore[possibly-missing-attribute]
        self._thread_lock.release()

    def _drop(self) -> None:
        _DBG("t_drop", threading.current_thread().name)
        self._held = False
        fcntl.flock(self._fd, fcntl.LOCK_UN)  # type: ignore[possibly-missing-attribute]
        self._thread_lock.release()

    def _reacquire(self) -> None:
        _DBG("t_reacq", threading.current_thread().name)
        self._thread_lock.acquire()
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)  # type: ignore[possibly-missing-attribute]
            self._refresh()
            self._held = True
        except BaseException:
            self._thread_lock.release()
            raise

    def _refresh(self) -> None:
        state = self._read_state()
        self.nodes = state["nodes"]
        self.version = state["version"]
        self.inflight = state["inflight"]
        self._touched = set()

    def _read_state(self) -> dict:
        try:
            state = self._read_raw()
        except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError):
            state = None
        if not state or state.get("magic") != _MAGIC:
            return {
                "magic": _MAGIC,
                "version": 0,
                "inflight": 0,
                "nodes": self._initial_nodes(),
            }
        return state

    def _read_raw(self) -> dict:
        size = os.path.getsize(self._path)
        if size > 1 << 20:
            _LOG.error("coordinator state oversized: %d bytes at %s", size, self._path)
        with open(self._path, "rb") as fh:
            return pickle.load(fh)

    def _initial_nodes(self) -> dict:
        return {
            (layer, node): _NodeState(budget, self.plan.cohort[layer][node])
            for layer, budgets in enumerate(self.plan.budget)
            for node, budget in budgets.items()
        }

    def _write_state(self, state: dict) -> None:
        try:
            blob = pickle.dumps(state, protocol=5)
        except BaseException:
            _LOG.error(
                "coordinator state not picklable: %r", state.get("nodes", {}).keys()
            )
            raise
        tmp = self._path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, self._path)


def _coordinator_factory():
    mode = os.environ.get(_COORD_ENV, "file")
    if mode == "thread":
        return _SlotTable
    if mode == "file":
        return _FileSlotTable
    raise invalid(_COORD_ENV, "must be 'file' or 'thread'")


_view_slots: dict = {}
_table_lock = threading.RLock()


def _chain_fingerprint(chain) -> str:
    """Content hash of a concrete chain: identical chains in different
    processes share one coordinator file. Views (which hold accessors) are
    excluded, so the key never retains them."""
    from .propagation import cache_layers

    tables = [view.d_tensor_table for view in cache_layers(chain)]
    tables += [chain.src.d_tensor_table, chain.dst.d_tensor_table]
    payload = (tables, chain.src.physical_layout, chain.num_ranks_per_node)
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()[:20]


def _drop_table(key) -> None:
    # Pop under the lock, drop the table outside it: dropping the table can
    # run another view finalizer that takes the same lock.
    with _table_lock:
        table = _view_slots.pop(key, None)
    del table


def _slots(chain):
    from .propagation import build_chain_plan, cache_layers

    key = _chain_fingerprint(chain)
    with _table_lock:
        table = _view_slots.get(key)
    if table is not None:
        return table
    table = _coordinator_factory()(build_chain_plan(chain), key)
    with _table_lock:
        table = _view_slots.setdefault(key, table)
    for view in cache_layers(chain):
        weakref.finalize(view, _drop_table, key)
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
            [self.t.mark_touched(ln) for ln in self.gens]
            self.t.bump()
        try:
            self._wait_cohort()
            self._loop(list(self.plan.drains[self.rank]))
        finally:
            self._leave()

    def _wait_cohort(self) -> None:
        """Wait for missing consumers of a node whose budget cannot hold the
        whole demand. After the wait limit this is a clear error, not a hang."""
        deadline = time.monotonic() + self.wait
        while not self._cohort_round(deadline):
            pass

    def _cohort_round(self, deadline: float) -> bool:
        """One wait round; True when the cohort is complete."""
        with self.t.cond:
            gaps = self._cohort_gaps()
        if gaps:
            return self._cohort_wait_or_timeout(deadline, gaps)
        return True

    def _cohort_wait_or_timeout(self, deadline: float, gaps: list) -> bool:
        if self._cohort_expired(deadline, gaps):
            raise self._cohort_timeout(gaps)
        with self.t.cond:
            version = self.t.version
            left = deadline - time.monotonic()
            self.t.cond.wait_for(
                lambda: self.t.version != version, timeout=min(left, 0.05)
            )
        return False

    def _cohort_expired(self, deadline: float, gaps: list) -> bool:
        return time.monotonic() >= deadline

    def _cohort_timeout(self, gaps: list):
        layer, node = gaps[0]
        state = self.t.node(layer, node)
        missing = [("rank", r) for r in sorted(state.cohort - state.joined)]
        return check_failed(
            "cache_wait",
            f"rank {self.rank}: cache layer {layer} node {node} "
            f"demands {self.plan.demand[layer][node]} bytes but budget "
            f"is {state.budget}; missing consumers {missing} must call "
            f"chain_load concurrently (one thread or process per rank)",
        )

    def _cohort_gaps(self) -> list:
        """(layer, node) pairs that miss a consumer while over their budget.
        A node needs its whole cohort online only when some segment is
        consumed by more than one rank: a missing consumer would keep that
        segment resident forever. Purely sharded segments make serial callers
        safe."""
        return [
            (layer, node)
            for layer, node in self.gens
            if self._cohort_needs_wait(layer, node)
        ]

    def _cohort_needs_wait(self, layer: int, node: int) -> bool:
        state = self.t.node(layer, node)
        missing = state.cohort - state.joined
        if not missing or self.plan.demand[layer][node] <= state.budget:
            return False
        mine = {key for _ln, key, _o, _s in self.plan.drains[self.rank]}
        return any(
            c[0][0] == "rank" and c[0][1] != self.rank
            for seg in self.plan.segments[layer][node].values()
            if seg.key in mine
            for c in seg.owed()
        )

    def _leave(self) -> None:
        with self.t.cond:
            _ = [self._dec(ln) for ln in self.gens]
            self.t.bump()

    def _dec(self, ln) -> None:
        self.t.node(*ln).active -= 1
        self.t.mark_touched(ln)

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
        """Wait for a version change or for the in-flight work to finish. A
        pass with no own progress times out only when nothing is in flight:
        a slow fill or drain counts as progress, so healthy peers never hit
        the wait limit."""
        with self.t.cond:
            moved = self._wait_deadline(version, deadline)
            done = moved or self._wait_inflight(version)
        if not done:
            raise self._timeout(pending)

    def _wait_deadline(self, version: int, deadline: float) -> bool:
        ready = False
        while not ready and time.monotonic() < deadline:
            ready = self._deadline_once(version, deadline) is not None
        return ready

    def _deadline_once(self, version: int, deadline: float):
        left = deadline - time.monotonic()
        ready = self._wait_version(version, left)
        if self.t.inflight == 0:
            return ready
        return ready or None

    def _wait_version(self, version: int, timeout: float) -> bool:
        return timeout > 0 and self.t.cond.wait_for(
            lambda: self.t.version != version, timeout=timeout
        )

    def _wait_inflight(self, version: int) -> bool:
        done = False
        while not done and self.t.inflight > 0:
            done = self.t.cond.wait_for(
                lambda: self.t.version != version or self.t.inflight == 0,
                timeout=0.05,
            )
        return done

    def _timeout(self, pending: list):
        node, key = pending[0][0], pending[0][1]
        state = self.t.node(self.last, node)
        owed = state.owed_to()
        msg = (
            f"rank {self.rank}: no progress for {self.wait}s on cache layer "
            f"{self.last} node {node} segment {key}; resident {state.resident} of "
            f"budget {state.budget} bytes owed to {owed}"
        )
        return check_failed("cache_wait", msg)

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
            ok = state.reserve(segs)
            self.t.mark_touched((layer, node))
            self.t.bump() if ok else None
        return ok

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
        self._inflight(1)
        try:
            data = self._source_bytes(layer, seg)
            _write(self.caches[layer].data_accessor, _crange(seg.key), data)
        except BaseException as exc:
            self._inflight(-1)
            self._fail(layer, node, exc, seg.key)
            raise
        with self.t.cond:
            self.t.node(layer, node).slots[seg.key]["filled"] = True
            self.t.inflight -= 1
            self.t.mark_touched((layer, node))
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
        self._inflight(1)
        try:
            data = _read(self.caches[self.last].data_accessor, rng)
            _write(self.dst.data_accessor, rng, data)
        except BaseException as exc:
            self._inflight(-1)
            self._fail(self.last, node, exc, None)
            raise
        self._inflight(-1)
        self._ack(self.last, node, key, (("rank", self.rank), off, size))

    def _inflight(self, delta: int) -> None:
        """Mark a fill or drain as in flight: waiting peers treat in-flight
        work as progress and do not time out. The delta is persisted so a peer
        in another process observes it."""
        with self.t.cond:
            self.t.inflight += delta
            self.t.bump()

    def _fail(self, layer: int, node: int, exc: BaseException, key) -> None:
        """Record one node-wide error for this generation; every rank of the
        generation that reads this node raises the same error object."""
        with self.t.cond:
            self.t.node(layer, node).fail(exc, key)
            self.t.mark_touched((layer, node))
            self.t.bump()

    def _ack(self, layer: int, node: int, key, piece) -> None:
        with self.t.cond:
            self.t.node(layer, node).ack(key, piece)
            self.t.mark_touched((layer, node))
            self.t.bump()


def _crange(key) -> RangeFromComputeView:
    return RangeFromComputeView(key[0], key[1], key[2])


def _run_cached(rank: int, chain, storage_session_view) -> None:
    from .propagation import cache_layers

    table = _slots(chain)
    if rank not in table.plan.ranks:
        raise invalid("rank", f"rank {rank} not in read-chain ranks {table.plan.ranks}")
    _RankRun(table, rank, cache_layers(chain), storage_session_view, chain.dst).run()


def _run_one_concrete(rank: int, chain) -> None:
    """Open and finish one load session for one concrete chain (design 5.8:
    each Mixed segment has its own source session)."""
    from .chain import DirectDTensorsReadChain, rank_direct_plan

    storage = chain.src
    if type(storage) is not StorageDTensorsView or storage.begin_load is None:
        raise invalid("read_chain", "chain src must be a StorageDTensorsView")
    plan = (
        rank_direct_plan(rank, chain)
        if type(chain) is DirectDTensorsReadChain
        else None
    )
    _LOG.info("chain_load rank=%d phase=begin concrete", rank)
    session = must_ok(storage.begin_load(storage, rank))
    assert session is not None
    try:
        view = _session_view(storage, session)
        if plan is not None:
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

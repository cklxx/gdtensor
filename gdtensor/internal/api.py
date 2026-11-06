"""Public IO executors: plan, open sessions, copy, and close sessions."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
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
_PROCESS_FILES: weakref.WeakSet = weakref.WeakSet()
_PROCESS_FILES_LOCK = threading.Lock()


def _open_process_file(path, mode):
    with _PROCESS_FILES_LOCK:
        file = open(path, mode)
        _PROCESS_FILES.add(file)
        return file


def _new_owner(directory):
    with _PROCESS_FILES_LOCK:
        fd, path = tempfile.mkstemp(prefix="gdtensor-run-", dir=directory)
        owner = os.fdopen(fd, "a+b")
        _PROCESS_FILES.add(owner)
        return owner, path


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
        self.completed: set = set()  # filled this generation, including released slots
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
            self.completed = set()
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
        if any(s.key in self.slots or s.key in self.completed for s in segs):
            return False
        if not self.fits(sum(s.size for s in segs)):
            return False
        for s in segs:
            self.admit(s.key, s.size)
            self.slots[s.key]["owed"] = s.owed()
        return True

    def ack(self, key, piece) -> bool:
        slot = self.slots.get(key)
        if slot is None:
            raise check_failed("cache_ack", f"segment {key} released before {piece}")
        slot["owed"].discard(piece)
        if not slot["owed"]:
            self.release(key)
            return True
        return False

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

    # One process's threads share this table; the file table sets this False so
    # every state change is persisted. Threads can skip a notify when a change
    # makes no waiter predicate true (see bump's whitelist in the call sites).
    in_memory = True

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

    @property
    def _in_mem(self) -> bool:
        return getattr(self, "in_memory", False)

    def bump(self) -> None:
        self.version += 1
        self.cond.notify_all()

    def mark_touched(self, key) -> None:
        """No-op: this table shares one in-memory state, so there is no
        snapshot merge to protect."""

    def participate(self, rank: int):
        return nullcontext()

    def check_alive(self) -> None:
        """Thread callers cannot outlive a killed process."""

    def activity(self, owner, delta: int) -> None:
        self.inflight += delta

    def finish(self, owner) -> None:
        """No process registration in the thread coordinator."""


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
_MAGIC = "gdtensor-coord-v3"


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
        self.t.check_alive()
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

    State records node residency, active owners, in-flight IO and terminal error.
    Mutations hold a thread RLock and a file flock, then atomically replace state.
    Per-call flocks remain held outside these short transactions. An abandoned
    owner fails this chain; restarting safely requires a new coordinator directory.
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
        self._pid = os.getpid()
        self._fd = _open_process_file(base + ".lock", "a+b")
        self._thread_lock = threading.RLock()
        self._held: int | None = None
        self._dirty = False
        self.version = 0
        self.inflight = 0
        self.nodes = {}
        self._touched: set = set()
        self.cond = _FileCond(self)
        self._reacquire()
        self._drop()  # load the initial state, then release both locks

    # File state is persisted on bump, so no notify may be skipped.
    in_memory = False

    @property
    def _in_mem(self) -> bool:
        return getattr(self, "in_memory", False)

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

    @contextmanager
    def participate(self, rank: int):
        assert fcntl is not None
        self._check_process()
        # Kernel-held locks have no expiry: slow living callers remain valid.
        owner, path = _new_owner(os.path.dirname(self._path))
        try:
            fcntl.flock(owner, fcntl.LOCK_EX)
            token = os.path.basename(path)
            self._register(token, rank)
            yield token
        finally:
            self._close_owner(owner, path)

    def _close_owner(self, owner, path):
        owner.close()
        if self._pid == os.getpid():
            os.unlink(path)

    def _register(self, path, rank):
        with self.cond:
            self.check_alive()
            self.runs[path] = {"rank": rank, "inflight": 0}
            self.bump()

    def activity(self, owner, delta: int) -> None:
        self.runs[owner]["inflight"] += delta
        self.inflight += delta

    def finish(self, owner) -> None:
        # An interrupted commit leaves its owner registered, even after exit.
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

    def _unlocked(self, path) -> bool:
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

    def _enter(self) -> None:
        _DBG("t_enter", threading.current_thread().name, "held", self._held)
        self._check_process()
        self._check_reentrant()
        self._thread_lock.acquire()
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)  # type: ignore[possibly-missing-attribute]
            self._refresh()
            self._held = threading.get_ident()
            self._dirty = False
        except BaseException:
            self._unlock()
            raise

    def _check_process(self) -> None:
        if self._pid != os.getpid():
            raise check_failed(
                "cache_fork", "forked during cache IO; start a new chain_load"
            )

    def _check_reentrant(self) -> None:
        if self._held == threading.get_ident():
            raise RuntimeError("_FileSlotTable.cond is not reentrant")

    def _unlock(self) -> None:
        assert fcntl is not None
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self._thread_lock.release()

    def _exit(self) -> None:
        if self._held != threading.get_ident() or self._pid != os.getpid():
            return
        _DBG(
            "t_exit",
            threading.current_thread().name,
            "v",
            self.version,
            "d",
            self._dirty,
        )
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
        _DBG("t_drop", threading.current_thread().name)
        self._held = None
        self._unlock()

    def _reacquire(self) -> None:
        _DBG("t_reacq", threading.current_thread().name)
        self._check_process()
        self._thread_lock.acquire()
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)  # type: ignore[possibly-missing-attribute]
            self._refresh()
            self._held = threading.get_ident()
        except BaseException:
            self._unlock()
            raise

    def _refresh(self) -> None:
        state = self._read_state()
        self.nodes = state["nodes"]
        self.version = state["version"]
        self.inflight = state["inflight"]
        self.runs = state["runs"]
        self.error = state["error"]
        self._touched = set()

    def _read_state(self) -> dict:
        try:
            state = self._read_raw()
        except FileNotFoundError:
            return {
                "magic": _MAGIC,
                "version": 0,
                "inflight": 0,
                "nodes": self._initial_nodes(),
                "runs": {},
                "error": None,
            }
        except (EOFError, pickle.UnpicklingError, ValueError) as exc:
            raise check_failed(
                "cache_state", "invalid coordinator state; use a new directory"
            ) from exc
        if not isinstance(state, dict) or state.get("magic") != _MAGIC:
            raise check_failed(
                "cache_state", "incompatible coordinator state; use a new directory"
            )
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


def _after_fork() -> None:
    global _table_lock
    # Close, never unlock: fork shares the parent's open-file descriptions.
    for file in list(_PROCESS_FILES):
        file.close()
    _PROCESS_FILES.clear()
    _table_lock = threading.RLock()
    _view_slots.clear()
    _PROCESS_FILES_LOCK.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_PROCESS_FILES_LOCK.acquire,
        after_in_parent=_PROCESS_FILES_LOCK.release,
        after_in_child=_after_fork,
    )


def _chain_fingerprint(chain) -> str:
    """Content hash of a concrete chain: identical chains in different
    processes share one coordinator file. Views (which hold accessors) are
    excluded, so the key never retains them."""
    from .propagation import cache_layers

    tables = [view.d_tensor_table for view in cache_layers(chain)]
    tables += [chain.src.d_tensor_table, chain.dst.d_tensor_table]
    payload = (tables, chain.src.physical_layout, chain.num_ranks_per_node)
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()[:20]


def _drop_table(key, expected) -> None:
    # A finalizer for an older table must not remove its replacement.
    with _table_lock:
        table = _view_slots.get(key)
        if table is expected():
            _view_slots.pop(key, None)
    del table


def _slots(chain, source_identity=None):
    from .propagation import build_chain_plan, cache_layers

    identity = (
        source_identity
        if source_identity is not None
        else chain.src.data_accessor.cache_identity()
    )
    key = _chain_fingerprint(chain)
    if identity is not None:
        key = hashlib.sha256(pickle.dumps((key, identity), protocol=5)).hexdigest()[:20]
    directory = os.path.realpath(os.environ.get(_COORD_DIR_ENV, tempfile.gettempdir()))
    cache_key = (os.getpid(), os.environ.get(_COORD_ENV, "file"), directory, key)
    with _table_lock:
        table = _view_slots.get(cache_key)
        if table is not None:
            return table
        table = _coordinator_factory()(build_chain_plan(chain), key)
        _view_slots[cache_key] = table
    for view in cache_layers(chain):
        weakref.finalize(view, _drop_table, cache_key, weakref.ref(table))
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
        from .propagation import execution_closure

        self.t, self.plan, self.rank = table, table.plan, rank
        self.caches, self.storage, self.dst = caches, storage, dst
        self.source_identity = storage.data_accessor.cache_identity()
        self.last = len(caches) - 1
        closure = execution_closure(self.plan, rank)
        self.closure = closure
        self._absent: set = set()
        self.services = list(self.plan.services.get(rank, ()))
        self.gens = dict.fromkeys(sorted({(lay, n) for lay, n, _k in closure}), 0)
        self.wait = _wait_seconds()

    def run(self) -> None:
        with self.t.participate(self.rank) as self.owner:
            self._run()

    def _run(self) -> None:
        with self.t.cond:
            self.t.check_alive()
            self.gens = {ln: self.t.node(*ln).join(self.rank) for ln in self.gens}
            [self.t.mark_touched(ln) for ln in self.gens]
            self.t.bump()
        try:
            self._wait_cohort()
            self._loop(list(self.plan.drains[self.rank]))
            self._version_checked()
            self._check_source()
        finally:
            self._leave()

    def _check_source(self):
        if self.source_identity != self.storage.data_accessor.cache_identity():
            raise check_failed(
                "cache_source_changed", "source file changed during load"
            )

    def _adopt(self, layer: int, node: int, key) -> bool:
        """Adopt a segment the cache view already reports as present.

        Adoption asks the view for metadata only, so a cold cache costs one
        look-up and a warm cache skips the storage read.
        """
        if (layer, node, key) in self._absent:
            return False
        if self._adopted(layer, node, key):
            return True
        if self.source_identity is None:
            return False
        seg = self.plan.segments[layer][node][key]
        acc = self.caches[layer].data_accessor
        if self.source_identity != self.storage.data_accessor.cache_identity():
            raise check_failed(
                "cache_source_changed", "source file changed during load"
            )
        if not acc.has_bytes(self._range(seg.key, node)):
            self._absent.add((layer, node, key))
            return False
        return self._mark_adopted(layer, node, seg)

    def _range(self, key, node):
        return _CacheRange(key[0], key[1], key[2], node, self.source_identity)

    def _adopted(self, layer: int, node: int, key) -> bool:
        with self.t.cond:
            return self._slot_filled(layer, node, key)

    def _slot_filled(self, layer: int, node: int, key) -> bool:
        state = self.t.node(layer, node)
        if key in state.completed:
            return True
        slot = state.slots.get(key)
        return bool(slot["filled"]) if slot is not None else False

    def _mark_adopted(self, layer: int, node: int, seg) -> bool:
        with self.t.cond:
            return self._seal_node(layer, node, seg)

    def _seal_node(self, layer: int, node: int, seg) -> bool:
        state = self.t.node(layer, node)
        if seg.key in state.completed:
            return True
        if seg.key in state.slots:
            return bool(state.slots[seg.key]["filled"])
        if not state.reserve([seg]):
            return False
        state.slots[seg.key]["filled"] = True
        state.completed.add(seg.key)
        self.t.mark_touched((layer, node))
        self.t.bump()
        return True

    def _classify_drains(self, pending) -> list:
        """Return the drainable pieces of pending, in pending order.

        One lock pass finds pieces already filled; cache probes run outside
        the lock; one more lock pass seals any probed hits. This replaces a
        per-piece lock probe for every pending piece of the rank.
        """
        with self.t.cond:
            ready, probe_keys = self._drain_scan(pending)
        segs = self._probe_drain_bytes(probe_keys)
        if not segs:
            return [piece for piece in pending if piece in ready]
        with self.t.cond:
            sealed = self._seal_drains(segs)
        return [
            piece
            for piece in pending
            if piece in ready or (piece[0], piece[1]) in sealed
        ]

    def _drain_scan(self, pending):
        """Split pending into filled pieces and (node, key) probe candidates.

        The caller holds the table lock. Each (node, key) is probed at most
        once per scan, and keys already known absent are not probed again.
        """
        ready = {
            piece
            for piece in pending
            if self._slot_filled(self.last, piece[0], piece[1])
        }
        seen = set()
        probe_keys = []
        for piece in pending:
            self._collect_probe(piece, ready, seen, probe_keys)
        return ready, probe_keys

    def _collect_probe(self, piece, ready, seen, probe_keys) -> None:
        node_key = (piece[0], piece[1])
        if piece in ready or node_key in seen:
            return
        seen.add(node_key)
        if (self.last, piece[0], piece[1]) not in self._absent:
            probe_keys.append(node_key)

    def _probe_drain_bytes(self, keys) -> list:
        """Probe the last cache layer for (node, key) pairs, lock-free."""
        if self.source_identity is None:
            return []
        acc = self.caches[self.last].data_accessor
        segs = []
        for node, key in keys:
            self._probe_one(node, key, acc, segs)
        return segs

    def _probe_one(self, node, key, acc, segs) -> None:
        seg = self.plan.segments[self.last][node][key]
        if self.source_identity != self.storage.data_accessor.cache_identity():
            raise check_failed(
                "cache_source_changed", "source file changed during load"
            )
        if not acc.has_bytes(self._range(seg.key, node)):
            self._absent.add((self.last, node, key))
            return
        segs.append(seg)

    def _seal_drains(self, segs) -> set:
        """Seal probed segments; the caller holds the table lock.

        Re-checks each segment under the lock, because another rank may have
        reserved it between the probe and this pass. Returns the (node, key)
        pairs that are now drainable; bumps once if anything changed.
        """
        sealed = set()
        changed = []
        for seg in segs:
            self._seal_one(seg, sealed, changed)
        for node in dict.fromkeys(changed):
            self.t.mark_touched((self.last, node))
        if changed:
            # A sealed segment makes drain predicates true: always bump.
            self.t.bump()
        return sealed

    def _seal_one(self, seg, sealed, changed) -> None:
        state = self.t.node(self.last, seg.node)
        slot = state.slots.get(seg.key)
        filled = seg.key in state.completed or bool(slot and slot["filled"])
        if filled:
            sealed.add((seg.node, seg.key))
            return
        if slot is not None or not state.reserve([seg]):
            return
        state.slots[seg.key]["filled"] = True
        state.completed.add(seg.key)
        sealed.add((seg.node, seg.key))
        changed.append(seg.node)

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
            self.t.finish(self.owner)
            # Leaving changes no waiter predicate for threads (join bumps).
            self.t.bump() if not self.t._in_mem else None

    def _dec(self, ln) -> None:
        self.t.node(*ln).active -= 1
        self.t.mark_touched(ln)

    def _loop(self, pending: list) -> None:
        """The wait limit restarts after every unit of progress."""
        deadline = time.monotonic() + self.wait
        while pending or self.services:
            version = self._version_checked()
            moved = self._step(pending)
            deadline = time.monotonic() + self.wait if moved else deadline
            _ = moved or self._block(version, deadline, pending)

    def _step(self, pending: list) -> bool:
        """Drain every filled piece; else fill one segment. True on progress."""
        done = self._classify_drains(pending)
        for piece in done:
            self._drain(piece)
            pending.remove(piece)
        return (
            bool(done)
            or self._service_step()
            or any(self._ensure(self.last, n, k) for n, k, _o, _s in pending)
        )

    def _service_step(self) -> bool:
        ready = next(
            (
                item
                for item in self.services
                if self._completed(*item) or self._ensure(*item)
            ),
            None,
        )
        if ready is None:
            return False
        self.services.remove(ready)
        return True

    def _version_checked(self) -> int:
        with self.t.cond:
            self.t.check_alive()
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
        layer, node, key = (
            (self.last, pending[0][0], pending[0][1]) if pending else self.services[0]
        )
        state = self.t.node(layer, node)
        owed = state.owed_to()
        msg = (
            f"rank {self.rank}: no progress for {self.wait}s on cache layer "
            f"{layer} node {node} segment {key}; resident {state.resident} of "
            f"budget {state.budget} bytes owed to {owed}"
        )
        return check_failed("cache_wait", msg)

    def _completed(self, layer: int, node: int, key) -> bool:
        with self.t.cond:
            return key in self.t.node(layer, node).completed

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
        if not self._can_fill(layer, node, key):
            return False
        if not self._room(layer, node, seg):
            return False
        if self._adopt(layer, node, key):
            return False
        if layer > 0 and not self._parents_ready(layer, node, seg):
            return False
        if not self._reserve(layer, node, [seg]):
            return False
        self._fill(layer, node, seg)
        return True

    def _can_fill(self, layer: int, node: int, key) -> bool:
        return self.rank in self.plan.fill_ranks[(layer, node, key)]

    def _room(self, layer: int, node: int, seg) -> bool:
        with self.t.cond:
            state = self.t.node(layer, node)
            state.check(self.gens[(layer, node)])
            return (
                seg.key not in state.completed
                and seg.key not in state.slots
                and state.fits(seg.size)
            )

    def _reserve(self, layer: int, node: int, segs: list) -> bool:
        """Admit all segments or none; each starts unfilled with its owed set."""
        with self.t.cond:
            state = self.t.node(layer, node)
            state.check(self.gens[(layer, node)])
            ok = state.reserve(segs)
            self.t.mark_touched((layer, node))
            # A reserve makes no waiter predicate true for threads; the
            # fill completion that follows it bumps.
            self.t.bump() if ok and not self.t._in_mem else None
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
        for k in keys:
            self._adopt(layer - 1, pnode, k)
        if all(self._filled(layer - 1, pnode, k) for k in keys):
            return True
        missing = [psegs[k] for k in keys if not self._resident(layer - 1, pnode, k)]
        if any(not self._can_fill(layer - 1, pnode, s.key) for s in missing):
            return False
        if not missing or not self._reserve(layer - 1, pnode, missing):
            return False
        for pseg in missing:
            self._fill(layer - 1, pnode, pseg)
        return True

    def _resident(self, layer: int, node: int, key) -> bool:
        with self.t.cond:
            return key in self.t.node(layer, node).slots

    def _fill(self, layer: int, node: int, seg) -> None:
        if not self._can_fill(layer, node, seg.key):
            raise check_failed(
                "cache_fill_owner", f"rank {self.rank} cannot fill node {node}"
            )
        self._inflight(1)
        try:
            data = self._source_bytes(layer, seg)
            _write(self.caches[layer].data_accessor, self._range(seg.key, node), data)
        except BaseException as exc:
            self._inflight(-1)
            self._fail(layer, node, exc, seg.key)
            raise
        with self.t.cond:
            state = self.t.node(layer, node)
            state.slots[seg.key]["filled"] = True
            state.completed.add(seg.key)
            self.t.activity(self.owner, -1)
            self.t.mark_touched((layer, node))
            self._ack_parents_locked(layer, node, seg) if layer > 0 else None
            # Fill completion makes fill/drain predicates true: always bump.
            self.t.bump()

    def _ack_parents_locked(self, layer: int, node: int, seg) -> None:
        """ACK this fill's parent pieces; the caller holds the table lock."""
        pnode = self.plan.parent[layer][node]
        pstate = self.t.node(layer - 1, pnode)
        consumer = ("seg", layer, node, seg.key)
        for pkey, off, size in seg.parents:
            pstate.ack(pkey, (consumer, off, size))
        self.t.mark_touched((layer - 1, pnode))

    def _source_bytes(self, layer: int, seg) -> bytes:
        if layer == 0:
            acc = self.storage.data_accessor
            return b"".join(_read(acc, rng) for rng, _off in seg.storage)
        acc = self.caches[layer - 1].data_accessor
        return b"".join(
            _read(
                acc,
                self._range((seg.tid, off, size), self.plan.parent[layer][seg.node]),
            )
            for _k, off, size in seg.parents
        )

    def _drain(self, piece) -> None:
        node, key, off, size = piece
        rng = RangeFromComputeView(key[0], off, size)
        self._inflight(1)
        try:
            data = _read(
                self.caches[self.last].data_accessor,
                self._range((key[0], off, size), node),
            )
            _write(self.dst.data_accessor, rng, data)
        except BaseException as exc:
            self._inflight(-1)
            self._fail(self.last, node, exc, None)
            raise
        with self.t.cond:
            self.t.activity(self.owner, -1)
            released = self.t.node(self.last, node).ack(
                key, (("rank", self.rank), off, size)
            )
            self.t.mark_touched((self.last, node))
            # Bump only when a waiter predicate can change: a slot release,
            # or the last in-flight unit finishing. Other ACKs change no
            # predicate (waiters read version and inflight only).
            gated = not self.t._in_mem or released or self.t.inflight == 0
            self.t.bump() if gated else None

    def _inflight(self, delta: int) -> None:
        """Mark a fill or drain as in flight: waiting peers treat in-flight
        work as progress and do not time out. The delta is persisted so a peer
        in another process observes it."""
        with self.t.cond:
            self.t.activity(self.owner, delta)
            # +1 makes no waiter predicate true; a -1 to zero is observed by
            # the inflight poll, and error-path -1 is followed by _fail.
            self.t.bump() if not self.t._in_mem else None

    def _fail(self, layer: int, node: int, exc: BaseException, key) -> None:
        """Record one node-wide error for this generation; every rank of the
        generation that reads this node raises the same error object."""
        with self.t.cond:
            self.t.node(layer, node).fail(exc, key)
            self.t.mark_touched((layer, node))
            self.t.bump()


@dataclass(eq=True)
class _CacheRange(RangeFromComputeView):
    """Internal cache address; public byte-range fields remain unchanged."""

    cache_node: int = 0
    source_identity: Any = None


def _cache_range(key, node: int) -> RangeFromComputeView:
    return _CacheRange(key[0], key[1], key[2], node)


def _crange(key) -> RangeFromComputeView:
    return RangeFromComputeView(key[0], key[1], key[2])


def _run_cached(rank: int, chain, storage_session_view) -> None:
    from .propagation import cache_layers

    identity = storage_session_view.data_accessor.cache_identity()
    table = _slots(chain, identity)
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
        identity_of = getattr(view.data_accessor, "cache_identity", lambda: None)
        identity = identity_of()
        if plan is not None:
            _copy(plan, view, chain.dst)
        else:
            _run_cached(rank, chain, view)
        if identity != identity_of():
            raise check_failed(
                "cache_source_changed", "source file changed during load"
            )
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

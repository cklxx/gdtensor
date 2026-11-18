"""Public IO executors: plan, open sessions, copy, and close sessions."""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
import ctypes
import logging
import os
import pickle
import tempfile
import threading
import time
import weakref
from typing import Any

import numpy as np

from .plan import (
    d_tensors_view_rank_plan_load,
    d_tensors_view_rank_plan_save,
    validate_rank_load,
    validate_rank_save,
)
from .ranges import CopyPlan, RangeFromComputeView
from .result import Fail, as_error, catch, check_failed, invalid, must_ok
from .views import ComputeDTensorsView, StorageDTensorsView
from .. import Result
from ..access import _content_digest
from .products import Ok

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl: Any = None


_LOG = logging.getLogger(__name__)
_PROCESS_FILES: weakref.WeakSet = weakref.WeakSet()
_PROCESS_FILES_LOCK = threading.Lock()

# Bound requested bytes held by one cache batch. Ranges stay intact; an
# oversized range forms one batch.
_DRAIN_BATCH_BYTES = 4 << 20


def _bounded_batches(items, size_of):
    batch, nbytes = [], 0
    for item in items:
        size = size_of(item)
        if batch and nbytes + size > _DRAIN_BATCH_BYTES:
            yield batch
            batch, nbytes = [], 0
        batch.append(item)
        nbytes += size
    if batch:
        yield batch


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
    return _join_bytes(chunks)


def _join_bytes(chunks) -> bytes:
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _write(dst_acc, dst, data: bytes) -> None:
    written = 0
    view = np.frombuffer(data, dtype=np.uint8)
    source = view.ctypes.data

    def give(buffer_ptr: int, buffer_len: int):
        nonlocal written
        if buffer_len < 0 or written + buffer_len > dst.size:
            raise invalid("buffer_len", "write chunks exceed range size")
        if buffer_len:
            ctypes.memmove(buffer_ptr, source + written, buffer_len)
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


def _error_cleanup(exc, cleanup, *args):
    try:
        cleanup(*args)
    except BaseException as cleanup_error:
        raise exc from cleanup_error


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
        self.completed_at: dict = {}  # key -> table version when completed
        self.resident = 0
        self.gen = 0
        self.started = 0  # table version when this generation opened
        self.active = 0
        self.joined: set = set()
        self.entered: dict = {}  # rank -> number of calls joined at this node
        self.call = 0  # greatest rank call admitted to the current generation
        self.error = None  # public Error value, pickle-safe

    def join(self, rank: int, version: int = 0) -> int:
        """Join this rank's next call, opening a generation only when it is newer."""
        call = self.entered.get(rank, 0) + 1
        idle = self.active == 0 and self.joined
        if idle and (rank in self.joined or call > self.call):
            self.gen += 1
            self.started = version
            self.slots, self.resident, self.error = {}, 0, None
            self.completed = set()
            self.completed_at = {}
            self.joined = set()
        self.active += 1
        self.joined.add(rank)
        self.entered[rank] = call
        self.call = max(self.call, call)
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
            raise Fail(replace(self.error))

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
        if piece not in slot["owed"]:
            return False
        slot["owed"].remove(piece)
        if not slot["owed"]:
            self.release(key)
            return True
        return False

    def _fail_slot(self, key) -> None:
        slot = self.slots.get(key)
        if slot is not None and not slot["filled"]:
            self.release(key)

    def fail(self, exc: BaseException, key) -> None:
        if key is not None:
            self._fail_slot(key)
        if self.error is None:
            self.error = replace(as_error(exc))

    def owed_to(self) -> list:
        owed = {c for slot in self.slots.values() for c, _o, _n in slot["owed"] or ()}
        return sorted(owed, key=str)


class _SlotTable:
    """In-process coordinator: one lock, one version, targeted all-fit waits."""

    # One process's threads share this table; the file table sets this False so
    # every state change is persisted. Threads can skip a notify when a change
    # makes no waiter predicate true (see bump's whitelist in the call sites).
    in_memory = True

    def __init__(self, plan, key: str | None = None) -> None:
        self.plan = plan
        self.key = key
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.activity_lock = threading.Lock()
        self.rank_conds = {rank: threading.Condition(self.lock) for rank in plan.ranks}
        self.waiters = {}
        last = len(plan.budget) - 1
        for rank in plan.ranks:
            _ = [
                self.waiters.setdefault((last, node, key), set()).add(rank)
                for node, key, _off, _size in plan.drains.get(rank, ())
            ]
            _ = [
                self._add_waiter(rank, layer, node, key)
                for layer, node, key in plan.services.get(rank, ())
            ]
        self.version = 0
        self.inflight = 0
        self.nodes = {
            (layer, node): _NodeState(budget, plan.cohort[layer][node])
            for layer, budgets in enumerate(plan.budget)
            for node, budget in budgets.items()
        }
        # Shared absent-probe memo: (layer, node, key) -> node generation at
        # probe time. Within one generation, bytes reach a node only through
        # a fill or a seal, both visible in completed/slots before the memo
        # is consulted, so a hit never hides new bytes.
        self._absent_memo: dict = {}

    def node(self, layer: int, node: int) -> _NodeState:
        return self.nodes[(layer, node)]

    def _add_waiter(self, rank, layer, node, key) -> None:
        self.waiters.setdefault((layer, node, key), set()).add(rank)
        if layer == 0:
            return
        seg = self.plan.segments[layer][node][key]
        pnode = self.plan.parent[layer][node]
        for pkey, _off, _size in seg.parents:
            self.waiters.setdefault((layer - 1, pnode, pkey), set()).add(rank)

    @property
    def _in_mem(self) -> bool:
        return getattr(self, "in_memory", False)

    def bump(self, ranks=None) -> None:
        self.version += 1
        if ranks is None:
            self.cond.notify_all()
            _ = [cond.notify_all() for cond in self.rank_conds.values()]
            return
        for rank in ranks:
            self.rank_conds[rank].notify_all()

    def mark_touched(self, key) -> None:
        """No-op: this table shares one in-memory state, so there is no
        snapshot merge to protect."""

    def participate(self, rank: int):
        return nullcontext()

    def check_alive(self) -> None:
        """Thread callers cannot outlive a killed process."""

    def activity(self, owner, delta: int) -> None:
        with self.activity_lock:
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
_MAGIC = "gdtensor-coord-v5"


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
_view_users: dict = {}
_table_lock = threading.RLock()


def _after_fork() -> None:
    global _table_lock, _FINGERPRINT_LOCK
    # Close, never unlock: fork shares the parent's open-file descriptions.
    for file in list(_PROCESS_FILES):
        file.close()
    _PROCESS_FILES.clear()
    _table_lock = threading.RLock()
    _FINGERPRINT_LOCK = threading.Lock()
    _FINGERPRINTS.clear()
    _view_slots.clear()
    _view_users.clear()
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
    return _content_digest(payload)[:20]


@dataclass
class _Fingerprint:
    chain: object
    future: Future
    users: int = 0


_FINGERPRINT_LOCK = threading.Lock()
_FINGERPRINTS: dict[int, _Fingerprint] = {}


def _fingerprint_entry(chain):
    key = id(chain)
    with _FINGERPRINT_LOCK:
        entry = _FINGERPRINTS.get(key)
        leader = entry is None
        if entry is None:
            entry = _Fingerprint(chain, Future())
            _FINGERPRINTS[key] = entry
        entry.users += 1
    return entry, leader


def _fingerprint_release(key, entry) -> None:
    with _FINGERPRINT_LOCK:
        entry.users -= 1
        if entry.users == 0 and _FINGERPRINTS.get(key) is entry:
            _FINGERPRINTS.pop(key, None)


def _resolve_fingerprint(chain, entry, leader):
    future = entry.future
    if leader:
        try:
            future.set_result(_chain_fingerprint(chain))
        except BaseException as exc:
            future.set_exception(exc)
            raise
    return future.result()


@contextmanager
def _shared_cache_key(chain, identity, namespaces, active=None):
    """Share one active fingerprint; expire it after the last caller."""
    key = id(chain)
    entry, leader = _fingerprint_entry(chain) if active is None else active
    try:
        try:
            fingerprint = _resolve_fingerprint(chain, entry, leader)
            cache_key = _content_digest((fingerprint, identity, tuple(namespaces)))[:20]
        except ValueError as exc:
            raise invalid("cache_identity", str(exc)) from None
        yield cache_key
    finally:
        if active is None:
            _fingerprint_release(key, entry)


def _drop_table(key, view_id, expected) -> None:
    # A finalizer for an older table must not remove its replacement.
    with _table_lock:
        table = _view_slots.get(key)
        users = _view_users.get(key)
        if users is not None:
            users.pop(view_id, None)
        if table is expected() and not users:
            _view_slots.pop(key, None)
            _view_users.pop(key, None)
    del table


def _track_view(users, view, cache_key, table) -> None:
    view_id = id(view)
    if view_id in users:
        return
    users[view_id] = weakref.ref(view)
    weakref.finalize(view, _drop_table, cache_key, view_id, weakref.ref(table))


def _slot_table(cache_key, chain, views):
    from .propagation import build_chain_plan

    with _table_lock:
        table = _view_slots.get(cache_key)
        if table is None:
            table = _coordinator_factory()(build_chain_plan(chain), cache_key[-1])
            _view_slots[cache_key] = table
            _view_users[cache_key] = {}
        users = _view_users[cache_key]
        for view in views:
            _track_view(users, view, cache_key, table)
    return table


def _slots(chain, source_identity=None, active_fingerprint=None):
    from .propagation import cache_layers

    identity = (
        source_identity
        if source_identity is not None
        else chain.src.data_accessor.cache_identity()
    )
    views = cache_layers(chain)
    namespaces = []
    for view in views:
        accessor = view.data_accessor
        namespace = getattr(accessor, "cache_namespace", lambda: None)()
        namespaces.append(
            ("object", os.getpid(), id(accessor)) if namespace is None else namespace
        )
    with _shared_cache_key(chain, identity, namespaces, active_fingerprint) as key:
        directory = os.path.realpath(
            os.environ.get(_COORD_DIR_ENV, tempfile.gettempdir())
        )
        cache_key = (os.getpid(), os.environ.get(_COORD_ENV, "file"), directory, key)
        return _slot_table(cache_key, chain, views)


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

    def __init__(self, table, rank: int, caches, storage, dst, source_identity) -> None:
        from .propagation import execution_closure

        self.t, self.plan, self.rank = table, table.plan, rank
        self.caches, self.storage, self.dst = caches, storage, dst
        self.source_identity = source_identity
        self.source_digest = _content_digest(source_identity)
        self.last = len(caches) - 1
        closure = execution_closure(self.plan, rank)
        self.closure = closure
        self._absent: set = set()
        self._parent_blocks = {}
        self.services = list(self.plan.services.get(rank, ()))
        self.gens = dict.fromkeys(sorted({(lay, n) for lay, n, _k in closure}), 0)
        self.wait = _wait_seconds()
        # All-fit hot plans never evict or reuse a segment during the load, so
        # the deterministic preflight can skip consumer obligations. Cold plans
        # use layer-ordered fills and targeted dependency waits.
        # File mode keeps the legacy loop (its state must persist every transition).
        self._all_fit = table._in_mem and all(
            budget >= demand
            for layer, demands in enumerate(self.plan.demand)
            for node, demand in demands.items()
            for budget in (self.plan.budget[layer][node],)
        )
        self.wait_cond = table.rank_conds[rank] if self._all_fit else table.cond

    def run(self) -> None:
        with self.t.participate(self.rank) as self.owner:
            self._run()

    def _run(self) -> None:
        with self.t.cond:
            self.t.check_alive()
            version = self.t.version
            self.gens = {
                ln: self.t.node(*ln).join(self.rank, version) for ln in self.gens
            }
            [self.t.mark_touched(ln) for ln in self.gens]
            self.t.bump()
        try:
            self._wait_cohort()
            self._loop(list(self.plan.drains[self.rank])) if not self._all_fit else None
            self._run_all_fit() if self._all_fit else None
            self._version_checked()
        except BaseException as exc:
            self._fail_all_fit(exc)
            raise
        finally:
            self._leave()

    def _fail_all_fit(self, exc) -> None:
        if not self._all_fit:
            return
        for layer, node in self.gens:
            self._fail(layer, node, exc, None)

    def _run_all_fit(self) -> None:
        """Skip fills when this rank's reads and fill duties are present."""
        pending = list(self.plan.drains[self.rank])
        services = [
            (node, key, key[1], key[2])
            for layer, node, key in self.services
            if layer == self.last
        ]
        preflight = pending + services
        ready = (
            self._classify_drains(preflight, self.last)
            if self.source_identity is not None
            else []
        )
        hot = self.source_identity is not None and len(ready) == len(preflight)
        layer = 0
        while hot and len(self.plan.ranks) > 1 and layer < self.last:
            owned = [
                (node, key, key[1], key[2])
                for lay, node, key in self.closure
                if lay == layer and self._can_fill(layer, node, key)
            ]
            hot = len(self._classify_drains(owned, layer, seal=False)) == len(owned)
            layer += 1
        if not hot:
            self._run_all_fit_cold(pending)
            return
        self._drain_hot(pending)

    def _run_all_fit_cold(self, pending: list) -> None:
        """Fill owned work once in layer order, then drain the rank's reads."""
        owned = sorted(item for item in self.closure if self._can_fill(*item))
        by_node = {}
        for item in owned:
            by_node.setdefault(item[:2], []).append(item)
        with self.t.cond:
            _ = [self.t._add_waiter(self.rank, *item) for item in owned]
        for (layer, node), items in sorted(by_node.items()):
            self._fill_cold_node(layer, node, items)
        for node, key in dict.fromkeys((p[0], p[1]) for p in pending):
            self._wait_cold_slot(self.last, node, key)
        self._drain_hot(pending)

    def _fill_cold_node(self, layer: int, node: int, items: list) -> None:
        pieces = [(node, key, key[1], key[2]) for _layer, _node, key in items]
        self._classify_drains(pieces, layer)
        with self.t.cond:
            state = self.t.node(layer, node)
            items = [
                item
                for item in items
                if item[2] not in state.completed and item[2] not in state.slots
            ]
        if not items:
            return
        if layer == 0:
            _ = [self._ensure(*item) for item in items]
            return
        pnode = self.plan.parent[layer][node]
        parents = {
            pkey
            for _layer, _node, key in items
            for pkey, _off, _size in self.plan.segments[layer][node][key].parents
        }
        for pkey in sorted(parents):
            self._wait_cold_slot(layer - 1, pnode, pkey)
        if self._has_parent_batch(layer):
            self._ensure_items(items)
            return
        for item in items:
            self._ensure(*item)

    def _wait_cold_slot(self, layer: int, node: int, key) -> None:
        """Wait for one published dependency without rescanning other segments."""
        deadline = time.monotonic() + self.wait
        with self.wait_cond:
            self._wait_cold_locked(layer, node, key, deadline)

    def _wait_cold_locked(self, layer: int, node: int, key, deadline: float) -> None:
        state = self.t.node(layer, node)
        while not self._slot_filled(layer, node, key):
            deadline = self._cold_wait_once(state, layer, node, key, deadline)

    def _cold_wait_once(self, state, layer: int, node: int, key, deadline: float):
        self.t.check_alive()
        state.check(self.gens[layer, node])
        deadline = time.monotonic() + self.wait if self.t.inflight > 0 else deadline
        left = deadline - time.monotonic()
        if left <= 0:
            raise self._timeout_at(layer, node, key)
        version = self.t.version
        self.wait_cond.wait_for(
            lambda: self._slot_filled(layer, node, key) or self.t.version != version,
            timeout=left,
        )
        return time.monotonic() + self.wait if self.t.version != version else deadline

    def _fill_node_batches(self, layer, node, segs, batch_read) -> bool:
        moved = False
        batches = _bounded_batches(
            segs, lambda seg: sum(size for _key, _off, size in seg.parents)
        )
        for batch in batches:
            moved = self._fill_reserved_batch(layer, node, batch, batch_read) or moved
        return moved

    def _fill_reserved_batch(self, layer, node, batch, batch_read) -> bool:
        if self._reserve(layer, node, batch):
            self._fill_parent_batch(layer, node, batch, batch_read)
            return True
        moved = False
        for seg in batch:
            moved = self._fill_one(layer, node, seg.key, batch_read) or moved
        return moved

    def _fill_one(self, layer: int, node: int, key, batch_read=None) -> bool:
        if self._adopt(layer, node, key):
            return False
        seg = self.plan.segments[layer][node][key]
        if not self._reserve(layer, node, [seg]):
            return False
        if batch_read is None:
            self._fill(layer, node, seg)
        else:
            self._fill_parent_batch(layer, node, [seg], batch_read)
        return True

    def _fill_parent_batch(self, layer, node, segs, batch_read) -> None:
        """Fill one reserved child-node batch from its parent cache node."""
        try:
            self._inflight(1)
        except BaseException as exc:
            _error_cleanup(exc, self._abort_parent_batch, layer, node, segs, exc)
            raise
        try:
            self._read_fill_parent_batch(layer, node, segs, batch_read)
        except BaseException as exc:
            _error_cleanup(exc, self._abort_parent_io, layer, node, segs, exc)
            raise
        self._inflight(-1)

    def _abort_parent_io(self, layer, node, segs, exc) -> None:
        self._abort_parent_batch(layer, node, segs, exc)
        self._inflight(-1)

    def _read_fill_parent_batch(self, layer, node, segs, batch_read) -> None:
        payloads = self._parent_payloads(layer, segs, batch_read)
        for seg, payload in zip(segs, payloads):
            self._fill(layer, node, seg, payload)

    def _parent_payloads(self, layer, segs, batch_read) -> list:
        ranges = [rng for seg in segs for rng in self._source_ranges(layer, seg)]
        sizes = [size for seg in segs for _key, _off, size in seg.parents]
        data = batch_read(ranges)
        if len(data) != len(ranges) or any(
            len(raw) != size for raw, size in zip(data, sizes)
        ):
            raise invalid("buffer_len", "cache batch returned wrong byte counts")
        payloads, start = [], 0
        for seg in segs:
            payload, start = self._parent_payload(seg, data, start)
            payloads.append(payload)
        return payloads

    def _parent_payload(self, seg, data, start):
        if sum(size for _key, _off, size in seg.parents) != seg.size:
            raise invalid("cache_segment", "parent bytes do not match segment")
        end = start + len(seg.parents)
        payload = _join_bytes(data[start:end])
        if len(payload) != seg.size:
            raise invalid("buffer_len", "parent payload has wrong byte count")
        return payload, end

    def _release_parent_slot(self, state, key) -> None:
        slot = state.slots.get(key)
        if slot is not None and not slot["filled"]:
            state.release(key)

    def _release_parent_slots(self, state, segs) -> None:
        for seg in segs:
            self._release_parent_slot(state, seg.key)

    def _fail_parent_state(self, state, exc) -> None:
        if state.error is None:
            state.fail(exc, None)

    def _abort_parent_batch(self, layer, node, segs, exc) -> None:
        with self.t.cond:
            state = self.t.node(layer, node)
            self._release_parent_slots(state, segs)
            self._fail_parent_state(state, exc)
            self.t.mark_touched((layer, node))
            self.t.bump()

    def _adopt(self, layer: int, node: int, key) -> bool:
        """Adopt a segment the cache view already reports as present.

        Adoption asks the view for metadata only, so a cold cache costs one
        look-up and a warm cache skips the storage read.
        """
        if (layer, node, key) in self._absent:
            return False
        if self.source_identity is None:
            return False
        piece = (node, key, key[1], key[2])
        return bool(self._classify_drains([piece], layer))

    def _absent_memo_fp(self, layer: int, node: int, key):
        """(generation, hit) for the shared absent memo; None for file mode.

        The memo is scoped to the node's current generation (see the table
        comment), so a hit is exact within a generation, not a timeout.
        """
        if not self.t._in_mem:
            return None
        with self.t.cond:
            state = self.t.node(layer, node)
            fp = state.gen
            memo = getattr(self.t, "_absent_memo", {})
            return fp, memo.get((layer, node, key)) == fp

    def _absent_memo_store(self, layer: int, node: int, key, memo) -> None:
        if memo is None:
            return
        with self.t.cond:
            self.t._absent_memo[(layer, node, key)] = memo[0]

    def _range(self, key, node):
        return _CacheRange(
            key[0], key[1], key[2], node, self.source_identity, self.source_digest
        )

    def _slot_filled(self, layer: int, node: int, key) -> bool:
        state = self.t.node(layer, node)
        if key in state.completed:
            return True
        slot = state.slots.get(key)
        return bool(slot["filled"]) if slot is not None else False

    def _fail_nodes(self, layer, nodes, exc, release: bool) -> None:
        for node in nodes:
            self._fail(layer, node, exc, None)
        if release:
            self._inflight(-1)

    def _classify_drains(self, pending, layer: int, seal: bool = True) -> list:
        """Return present pieces of one cache layer, in pending order.

        One lock pass finds pieces already filled; cache probes run outside
        the lock; one more lock pass seals any probed hits. This replaces a
        per-piece lock probe for every pending piece of the rank. A presence-
        only preflight uses seal=False to avoid creating consumer obligations.
        """
        with self.t.cond:
            ready, probe_keys = self._drain_scan(pending, layer)
        if not probe_keys:
            return [piece for piece in pending if piece in ready]
        nodes = dict.fromkeys(node for node, _key in probe_keys)
        try:
            self._inflight(1)
        except BaseException as exc:
            _error_cleanup(exc, self._fail_nodes, layer, nodes, exc, False)
            raise
        try:
            result = self._classify_probe(pending, ready, probe_keys, layer, seal)
        except BaseException as exc:
            _error_cleanup(exc, self._fail_nodes, layer, nodes, exc, True)
            raise
        self._inflight(-1)
        return result

    def _classify_probe(self, pending, ready, probe_keys, layer, seal):
        segs = self._probe_drain_bytes(probe_keys, layer)
        if not segs:
            return [piece for piece in pending if piece in ready]
        with self.t.cond:
            sealed = self._seal_drains(segs, layer, reserve=seal)
        return [
            piece
            for piece in pending
            if piece in ready or (piece[0], piece[1]) in sealed
        ]

    def _drain_scan(self, pending, layer: int):
        """Split pending into filled pieces and (node, key) probe candidates.

        The caller holds the table lock. Each (node, key) is probed at most
        once per scan, and keys already known absent are not probed again.
        """
        ready = {
            piece for piece in pending if self._slot_filled(layer, piece[0], piece[1])
        }
        seen = set()
        probe_keys = []
        for piece in pending:
            self._collect_probe(piece, ready, seen, probe_keys, layer)
        return ready, probe_keys

    def _collect_probe(self, piece, ready, seen, probe_keys, layer: int) -> None:
        node_key = (piece[0], piece[1])
        if piece in ready or node_key in seen:
            return
        seen.add(node_key)
        item = (layer, piece[0], piece[1])
        if item in self._absent:
            return
        memo_hit = (
            self.t._in_mem
            and self.t._absent_memo.get(item) == self.t.node(layer, piece[0]).gen
        )
        if memo_hit:
            self._absent.add(item)
            return
        probe_keys.append(node_key)

    def _probe_drain_bytes(self, keys, layer: int) -> list:
        """Probe one cache layer outside the coordinator lock."""
        if self.source_identity is None:
            return []
        acc = self.caches[layer].data_accessor
        segs = []
        batch_probe = getattr(acc, "has_bytes_batch", None)
        if batch_probe is not None:
            self._probe_batches(keys, acc, segs, layer, batch_probe)
            return segs
        for node, key in keys:
            self._probe_one(node, key, acc, segs, layer)
        return segs

    def _probe_batches(self, keys, acc, segs, layer, batch_probe) -> None:
        by_node = {}
        for node, key in keys:
            by_node.setdefault(node, []).append((node, key))
        for node_keys in by_node.values():
            self._probe_node_batches(node_keys, acc, segs, layer, batch_probe)

    def _probe_node_batches(self, node_keys, acc, segs, layer, batch_probe) -> None:
        for batch in _bounded_batches(node_keys, lambda item: item[1][2]):
            self._probe_node_batch(batch, acc, segs, layer, batch_probe)

    def _probe_node_batch(self, batch, acc, segs, layer, batch_probe) -> None:
        ranges = [self._range(key, node) for node, key in batch]
        if batch_probe(ranges):
            segs.extend(self.plan.segments[layer][node][key] for node, key in batch)
            return
        if len(batch) == 1:
            node, key = batch[0]
            memo = self._absent_memo_fp(layer, node, key)
            self._absent.add((layer, node, key))
            self._absent_memo_store(layer, node, key, memo)
            return
        for node, key in batch:
            self._probe_one(node, key, acc, segs, layer)

    def _probe_one(self, node, key, acc, segs, layer: int) -> None:
        seg = self.plan.segments[layer][node][key]
        memo = self._absent_memo_fp(layer, node, key)
        if memo is not None and memo[1]:
            return
        if not acc.has_bytes(self._range(seg.key, node)):
            self._absent.add((layer, node, key))
            self._absent_memo_store(layer, node, key, memo)
            return
        segs.append(seg)

    def _seal_drains(self, segs, layer: int, reserve: bool = True) -> set:
        """Seal probed segments; the caller holds the table lock.

        Re-checks each segment under the lock, because another rank may have
        reserved it between the probe and this pass. Returns the (node, key)
        pairs that are now drainable; bumps once if anything changed.
        """
        sealed = set()
        changed = []
        for seg in segs:
            self._seal_one(seg, sealed, changed, layer, reserve)
        for node in dict.fromkeys(changed):
            self.t.mark_touched((layer, node))
        if changed:
            # A sealed segment makes drain predicates true: always bump.
            self.t.bump()
        return sealed

    def _seal_one(self, seg, sealed, changed, layer: int, reserve: bool) -> None:
        state = self.t.node(layer, seg.node)
        slot = state.slots.get(seg.key)
        filled = seg.key in state.completed or bool(slot and slot["filled"])
        if filled:
            sealed.add((seg.node, seg.key))
            return
        if slot is not None or (reserve and not state.reserve([seg])):
            return
        if reserve:
            state.slots[seg.key]["filled"] = True
        state.completed.add(seg.key)
        state.completed_at[seg.key] = self.t.version
        self._settle_completed_children(layer, seg.node, seg)
        if reserve and layer > 0:
            self._ack_parents_locked(layer, seg.node, seg)
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
            self.t.bump() if not self.t._in_mem or self._all_fit else None

    def _dec(self, ln) -> None:
        self.t.node(*ln).active -= 1
        self.t.mark_touched(ln)

    def _loop(self, pending: list) -> None:
        """The wait limit restarts after local or peer progress."""
        deadline = time.monotonic() + self.wait
        observed = None
        while pending or self.services:
            version = self._version_checked()
            # In-memory probes do not bump version. File-mode probe commits
            # also change version, but they are not evidence of cache progress.
            deadline = (
                time.monotonic() + self.wait
                if self.t._in_mem and (observed is None or version != observed)
                else deadline
            )
            observed = version
            moved = self._step(pending)
            deadline = time.monotonic() + self.wait if moved else deadline
            # A peer can release budget during _step. Keep the pre-step version
            # so _block cannot miss that last notification.
            self._block(version, deadline, pending) if not moved else None

    def _step(self, pending: list) -> bool:
        """Drain every filled piece; else fill ready segments. True on progress."""
        if self._all_fit and self._service_step():
            # All-fit has no capacity release dependency on local drains. Do
            # not rescan every pending piece before each assigned service fill.
            return True
        done = self._classify_drains(pending, self.last)
        self._drain_hot(done)
        for piece in done:
            pending.remove(piece)
        items = [(self.last, n, k) for n, k, _o, _s in pending]
        return (
            bool(done)
            or (not self._all_fit and self._service_step())
            or self._ensure_items(items)
        )

    def _service_step(self) -> bool:
        if self._all_fit:
            return self._service_all_fit()
        ready = next((item for item in self.services if self._completed(*item)), None)
        moved = False if ready is not None else self._ensure_items(self.services)
        if ready is None:
            ready = next(
                (item for item in self.services if self._completed(*item)), None
            )
        if ready is None:
            return moved
        self.services.remove(ready)
        return True

    def _service_all_fit(self) -> bool:
        if not self.services:
            return False
        item = self.services[0]
        if self._completed(*item):
            self.services.pop(0)
            return True
        moved = self._ensure_items([item])
        if self._completed(*item):
            self.services.pop(0)
            return True
        return moved

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
        with self.wait_cond:
            done = self._block_locked(version, deadline, pending)
        if not done:
            raise self._timeout(pending)

    def _all_fit_ready(self, pending) -> bool:
        services = self.services[:1] if self._all_fit else self.services
        return any(
            self._slot_filled(self.last, node, key)
            for node, key, _off, _size in pending
        ) or any(self._service_ready(*item) for item in services)

    def _service_ready(self, layer, node, key) -> bool:
        state = self.t.node(layer, node)
        if key in state.completed:
            return True
        if not self._all_fit or layer == 0 or key in state.slots:
            return False
        seg = self.plan.segments[layer][node][key]
        pnode = self.plan.parent[layer][node]
        return all(
            self._slot_filled(layer - 1, pnode, pkey)
            for pkey, _off, _size in seg.parents
        )

    def _block_locked(self, version, deadline, pending) -> bool:
        # A peer can publish between _step and this wait.
        if self._all_fit_ready(pending):
            return True
        if self._all_fit and self.t._in_mem:
            return self._block_all_fit(version, deadline, pending)
        moved = self._wait_deadline(version, deadline)
        return moved or self._wait_inflight(version)

    def _block_all_fit(self, version, deadline, pending) -> bool:
        # Unrelated fills must not wake every rank into a full pending scan.
        # A bounded poll still discovers parents needed by local fills.
        left = deadline - time.monotonic()
        if left <= 0:
            return self._wait_inflight(version)
        self.wait_cond.wait_for(
            lambda: self._all_fit_ready(pending), timeout=min(left, 0.05)
        )
        return True

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
            done = self.wait_cond.wait_for(
                lambda: self.t.version != version or self.t.inflight == 0,
                timeout=0.05,
            )
        return done

    def _timeout(self, pending: list):
        layer, node, key = (
            (self.last, pending[0][0], pending[0][1]) if pending else self.services[0]
        )
        return self._timeout_at(layer, node, key)

    def _timeout_at(self, layer: int, node: int, key):
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

    def _ensure_items(self, items: list) -> bool:
        items = list(dict.fromkeys(items))
        # Service lists contain lower-layer fills followed by child fills. Do
        # not scan every child dependency before making immediately available
        # lower-layer progress; large fan-outs can otherwise exceed cache_wait
        # in one scan while their parent owners have done no useful work.
        batched = []
        moved = any(self._ensure_or_batch(item, batched) for item in items)
        return moved or self._ensure_parent_batches(batched)

    def _ensure_or_batch(self, item, batched) -> bool:
        if not self._has_parent_batch(item[0]):
            return self._ensure(*item)
        batched.append(item)
        return False

    def _has_parent_batch(self, layer: int) -> bool:
        return (
            layer > 0
            and getattr(self.caches[layer - 1].data_accessor, "read_bytes_batch", None)
            is not None
        )

    def _ensure_parent_batches(self, items: list) -> bool:
        by_node = {}
        for layer, node, key in items:
            seg = self._prepare_fill(layer, node, key)
            by_node.setdefault((layer, node), []).append(
                seg
            ) if seg is not None else None
        moved = False
        for (layer, node), segs in sorted(by_node.items()):
            batch_read = self.caches[layer - 1].data_accessor.read_bytes_batch
            moved = self._fill_node_batches(layer, node, segs, batch_read) or moved
        return moved

    def _ensure(self, layer: int, node: int, key) -> bool:
        """Try to make one segment resident; True when this call filled it."""
        seg = self._prepare_fill(layer, node, key)
        if seg is None or not self._reserve(layer, node, [seg]):
            return False
        self._fill(layer, node, seg)
        return True

    def _prepare_fill(self, layer: int, node: int, key):
        """Adopt hot bytes before checking ownership for a cold fill."""
        seg = self.plan.segments[layer][node][key]
        if not self._room(layer, node, seg):
            return None
        if self._adopt(layer, node, key):
            return None
        if not self._can_fill(layer, node, key):
            return None
        if layer > 0 and not self._parents_ready(layer, node, seg):
            return None
        return seg

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

    def _parent_blocked(self, layer, pnode, child) -> bool:
        blocked = self._parent_blocks.get(child)
        if blocked is None:
            return False
        with self.t.cond:
            # completed is monotonic within this run's generation. An unchanged
            # size proves that no blocked parent became ready.
            state = self.t.node(layer - 1, pnode)
            return len(state.completed) == blocked[1]

    def _parents_ready(self, layer: int, node: int, seg) -> bool:
        """Fill missing parents; skip an unchanged non-owned dependency block."""
        child = layer, node, seg.key
        pnode = self.plan.parent[layer][node]
        if self._parent_blocked(layer, pnode, child):
            return False
        psegs = self.plan.segments[layer - 1][pnode]
        keys = sorted({pkey for pkey, _o, _n in seg.parents})
        with self.t.cond:
            missing = [
                psegs[k] for k in keys if not self._slot_filled(layer - 1, pnode, k)
            ]
        blocked_keys = tuple(s.key for s in missing)
        if not missing:
            return True
        previous = self._parent_blocks.get(child)
        if self._all_fit and previous is not None and previous[0] == blocked_keys:
            return False
        for pseg in missing:
            self._adopt(layer - 1, pnode, pseg.key)
        with self.t.cond:
            missing = [
                pseg
                for pseg in missing
                if not self._slot_filled(layer - 1, pnode, pseg.key)
            ]
            completed_mark = len(self.t.node(layer - 1, pnode).completed)
        if not missing:
            return True
        if self._parent_unowned(layer, pnode, child, missing, completed_mark):
            return False
        if not self._reserve(layer - 1, pnode, missing):
            return False
        for pseg in missing:
            self._fill(layer - 1, pnode, pseg)
        return True

    def _parent_unowned(self, layer, pnode, child, missing, completed_mark) -> bool:
        blocked = any(not self._can_fill(layer - 1, pnode, s.key) for s in missing)
        if blocked and self._all_fit:
            self._parent_blocks[child] = (
                tuple(s.key for s in missing),
                completed_mark,
            )
        return blocked

    def _fill(self, layer: int, node: int, seg, data=None) -> None:
        if not self._can_fill(layer, node, seg.key):
            raise check_failed(
                "cache_fill_owner", f"rank {self.rank} cannot fill node {node}"
            )
        try:
            self._inflight(1)
        except BaseException as exc:
            # Reservation is already durable; an entry-commit failure must not
            # leave an unfilled slot without an owner or an error.
            _error_cleanup(exc, self._fail, layer, node, exc, seg.key)
            raise
        try:
            self._write_fill(layer, node, seg, data)
        except BaseException as exc:
            _error_cleanup(exc, self._fill_failed, layer, node, seg, exc)
            raise
        # An interrupted file commit keeps its owner registered. A later
        # caller must report cache_peer_down instead of trusting that commit.
        with self.t.cond:
            state = self.t.node(layer, node)
            state.slots[seg.key]["filled"] = True
            state.completed.add(seg.key)
            state.completed_at[seg.key] = self.t.version
            self._settle_completed_children(layer, node, seg)
            self.t.activity(self.owner, -1)
            self.t.mark_touched((layer, node))
            self._ack_parents_locked(layer, node, seg) if layer > 0 else None
            # Fill completion makes fill/drain predicates true: always bump.
            waiters = (
                self.t.waiters.get((layer, node, seg.key), ())
                if self._all_fit
                else None
            )
            self.t.bump(waiters) if self._all_fit else self.t.bump()

    def _fill_failed(self, layer, node, seg, exc) -> None:
        self._fail(layer, node, exc, seg.key)
        self._inflight(-1)

    def _write_fill(self, layer, node, seg, data) -> None:
        if data is None:
            data = self._source_bytes(layer, seg)
        _write(self.caches[layer].data_accessor, self._range(seg.key, node), data)

    def _ack_parent(self, pstate, pkey, consumer, off, size) -> None:
        if pkey not in pstate.slots:
            # A hot child can satisfy its parent obligation without reserving
            # the parent in this generation. A released parent has none left.
            return
        pstate.ack(pkey, (consumer, off, size))

    def _settle_completed_children(self, layer: int, node: int, seg) -> None:
        """Discard obligations to child segments completed before admission."""
        state = self.t.node(layer, node)
        if seg.key not in state.slots:
            return
        for consumer, pieces in seg.consumers.items():
            self._settle_completed_child(state, seg.key, consumer, pieces)

    def _settle_completed_child(self, state, key, consumer, pieces) -> None:
        if consumer[0] != "seg":
            return
        _, child_layer, child_node, child_key = consumer
        child = self.t.node(child_layer, child_node)
        if child_key not in child.completed:
            return
        if child.completed_at.get(child_key, state.started) < state.started:
            return
        for off, size in pieces:
            state.ack(key, (consumer, off, size))

    def _ack_parents_locked(self, layer: int, node: int, seg) -> None:
        """ACK this fill's parent pieces; the caller holds the table lock."""
        pnode = self.plan.parent[layer][node]
        pstate = self.t.node(layer - 1, pnode)
        consumer = ("seg", layer, node, seg.key)
        for pkey, off, size in seg.parents:
            self._ack_parent(pstate, pkey, consumer, off, size)
        self.t.mark_touched((layer - 1, pnode))

    def _source_ranges(self, layer: int, seg) -> list:
        parent = self.plan.parent[layer][seg.node]
        return [
            self._range((seg.tid, off, size), parent) for _key, off, size in seg.parents
        ]

    def _source_bytes(self, layer: int, seg) -> bytes:
        if layer == 0:
            acc = self.storage.data_accessor
            return _join_bytes([_read(acc, rng) for rng, _off in seg.storage])
        acc = self.caches[layer - 1].data_accessor
        return _join_bytes([_read(acc, rng) for rng in self._source_ranges(layer, seg)])

    def _drain_hot(self, pending) -> None:
        acc = self.caches[self.last].data_accessor
        batch_read = getattr(acc, "read_bytes_batch", None)
        if batch_read is None:
            self._drain_hot_scalar(pending)
            return
        for batch in _bounded_batches(pending, lambda piece: piece[3]):
            self._drain_hot_batch(batch, batch_read)

    def _drain_hot_scalar(self, pending) -> None:
        for piece in pending:
            self._drain(piece)

    def _drain_hot_batch(self, batch, batch_read) -> None:
        ranges = [
            self._range((key[0], off, size), node) for node, key, off, size in batch
        ]
        nodes = dict.fromkeys(piece[0] for piece in batch)
        try:
            self._inflight(1)
        except BaseException as exc:
            _error_cleanup(exc, self._fail_nodes, self.last, nodes, exc, False)
            raise
        try:
            self._read_drain_batch(batch, ranges, batch_read)
        except BaseException as exc:
            _error_cleanup(exc, self._fail_nodes, self.last, nodes, exc, True)
            raise
        self._inflight(-1)

    def _read_drain_batch(self, batch, ranges, batch_read) -> None:
        data = batch_read(ranges)
        if len(data) != len(batch) or any(
            len(raw) != piece[3] for raw, piece in zip(data, batch)
        ):
            raise invalid("buffer_len", "cache batch returned wrong byte counts")
        for piece, raw in zip(batch, data):
            self._drain(piece, raw)

    def _drain(self, piece, data=None) -> None:
        node, key, off, size = piece
        rng = RangeFromComputeView(key[0], off, size)
        try:
            self._inflight(1)
        except BaseException as exc:
            _error_cleanup(exc, self._fail, self.last, node, exc, None)
            raise
        try:
            self._write_drain(piece, rng, data)
        except BaseException as exc:
            _error_cleanup(exc, self._drain_failed, node, exc)
            raise
        with self.t.cond:
            self.t.activity(self.owner, -1)
            released = self.t.node(self.last, node).ack(
                key, (("rank", self.rank), off, size)
            )
            self.t.mark_touched((self.last, node))
            # Capacity-limited waits can depend on a slot release or the last
            # in-flight unit. All-fit readiness ignores ACK releases and uses
            # bounded in-flight polls, so its ACKs do not broadcast.
            gated = not self.t._in_mem or (
                not self._all_fit and (released or self.t.inflight == 0)
            )
            self.t.bump() if gated else None

    def _drain_failed(self, node, exc) -> None:
        self._fail(self.last, node, exc, None)
        self._inflight(-1)

    def _write_drain(self, piece, rng, data) -> None:
        node, key, off, size = piece
        if data is None:
            data = _read(
                self.caches[self.last].data_accessor,
                self._range((key[0], off, size), node),
            )
        _write(self.dst.data_accessor, rng, data)

    def _inflight(self, delta: int) -> None:
        """Mark a fill or drain as in flight: waiting peers treat in-flight
        work as progress and do not time out. The delta is persisted so a peer
        in another process observes it."""
        if self.t._in_mem:
            # Do not serialize every I/O probe behind the global state lock.
            self.t.activity(self.owner, delta)
            return
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
    source_digest: str | None = field(default=None, compare=False, repr=False)


def _run_cached(rank: int, chain, storage_session_view, identity) -> None:
    from .propagation import cache_layers

    active = _fingerprint_entry(chain)
    try:
        table = _slots(chain, identity, active)
        if rank not in table.plan.ranks:
            raise invalid(
                "rank", f"rank {rank} not in read-chain ranks {table.plan.ranks}"
            )
        _RankRun(
            table,
            rank,
            cache_layers(chain),
            storage_session_view,
            chain.dst,
            identity,
        ).run()
    finally:
        _fingerprint_release(id(chain), active[0])


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
            _run_cached(rank, chain, view, identity)
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

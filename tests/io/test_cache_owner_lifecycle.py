"""Owner cleanup through public IO and injected filesystem failures."""

import errno
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.cache_oracle import cache_view, compute_table, storage_schema
from tests.support.cache_shapes import _build as _build_case
from tests.support.cluster_sim import Cluster, NodeCache, RankMem, StorageMem

internal_api = sys.modules[gio.chain_load.__module__]


pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX file locks")


class _Session:
    def __init__(self, accessor):
        self.accessor = accessor

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        return gb.Ok(self.accessor)

    def finish(self):
        return gb.Ok(None)


class _Cache(NodeCache):
    def __init__(self, cluster, size):
        super().__init__(cluster, {"w": size}, lambda _rank: 0)
        self.after_write = lambda: None

    def write_from_buffer(self, global_range, write_from_buffer):
        result = super().write_from_buffer(global_range, write_from_buffer)
        if type(result) is gb.Ok:
            self.after_write()
        return result


def _build():
    values = np.array([31, 0, 247, 83, 19, 206, 55, 132], dtype=np.uint8)
    cluster = Cluster(1, 2)
    source = StorageMem(cluster, values)
    table, layout = storage_schema(["w"], size=len(values))
    session_factory = lambda _view, _rank: gb.Ok(_Session(source))
    storage = gb.StorageDTensorsView(
        ["w"],
        table,
        source,
        layout,
        begin_save=session_factory,
        begin_load=session_factory,
    )
    cache = _Cache(cluster, len(values))
    dst = RankMem(cluster, len(values), 2)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=storage,
        cache=cache_view(["w"], cache, size=len(values)),
        dst=gb.ComputeDTensorsView(["w"], compute_table(["w"], size=len(values)), dst),
        num_ranks_per_node=2,
    )
    result = chain.check_constraints()
    assert type(result) is gb.Ok, result
    return chain, cluster, cache, dst, values


def _load_round(chain, cluster, dst, values):
    for rank in range(2):
        cluster.rank = rank
        dst.buffers[rank].fill(0xA5)
        result = gio.chain_load(rank, chain)
        assert type(result) is gb.Ok, result
        expected = np.full(len(values), 0xA5, dtype=np.uint8)
        start, stop = rank * 4, (rank + 1) * 4
        expected[start:stop] = values[start:stop]
        np.testing.assert_array_equal(dst.buffers[rank], expected)


def _report(function, args, sender):
    try:
        function(*args)
        sender.send(None)
    except BaseException:
        sender.send(traceback.format_exc()[-6000:])
    finally:
        sender.close()


def _stop(process):
    if process.pid is None:
        return
    if process.is_alive():
        process.kill()
    process.join(5)
    assert not process.is_alive(), "worker survived kill"
    process.close()


def _run_bounded(function, *args):
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_report, args=(function, args, sender))
    try:
        process.start()
        sender.close()
        process.join(25)
        assert not process.is_alive(), "public cache IO did not finish within 25s"
        assert process.exitcode == 0, process.exitcode
        assert receiver.poll(0), "worker did not report its assertions"
        error = receiver.recv()
        assert error is None, error
    finally:
        _stop(process)
        receiver.close()
        sender.close()


@pytest.fixture(autouse=True)
def isolated_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "file")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0.2")


class _PauseReplace:
    """Pause a commit while the caller holds the coordinator's thread lock."""

    def __init__(self):
        self.replace = os.replace
        self.pid = os.getpid()
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, src, dst, *args, **kwargs):
        if os.getpid() == self.pid and str(dst).endswith(".state"):
            self.entered.set()
            assert self.release.wait(5), "parent commit was not released"
        return self.replace(src, dst, *args, **kwargs)


def _thread_round(args, errors):
    try:
        _load_round(*args)
    except BaseException:
        errors.append(traceback.format_exc()[-6000:])


def _child_round(proceed, args):
    assert proceed.wait(10), "parent did not finish its generation"
    _load_round(*args)


def _fork_during_commit(args, pause):
    ctx = mp.get_context("fork")
    proceed = ctx.Event()
    receiver, sender = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_report, args=(_child_round, (proceed, args), sender))
    errors = []
    thread = threading.Thread(target=_thread_round, args=(args, errors), daemon=True)
    thread.start()
    try:
        assert pause.entered.wait(5), "parent did not enter a coordinator commit"
        child.start()
        sender.close()
        pause.release.set()
        thread.join(5)
        assert not thread.is_alive(), "parent cache IO hung"
        assert not errors, errors
        proceed.set()
        child.join(5)
        assert not child.is_alive(), "child inherited the locked coordinator table"
        assert child.exitcode == 0, child.exitcode
        assert receiver.poll(0), "child did not report its assertions"
        error = receiver.recv()
        assert error is None, error
    finally:
        pause.release.set()
        proceed.set()
        _stop(child)
        thread.join(5)
        receiver.close()
        sender.close()


def _fork_case():
    chain, cluster, cache, dst, values = _build()
    args = (chain, cluster, dst, values)
    _load_round(*args)
    assert cache.writes and cache.reads, "warm-up did not create a cache table"
    pause = _PauseReplace()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", pause)
        _fork_during_commit(args, pause)
    _load_round(*args)


def test_fork_after_table_creation_uses_fresh_child_coordinator():
    _run_bounded(_fork_case)


def _slots_in_thread(chain, errors):
    try:
        internal_api._slots(chain)
    except BaseException:
        errors.append(traceback.format_exc()[-6000:])


def _slots_in_child(chain):
    assert internal_api._slots(chain) is not None
    assert not internal_api._FINGERPRINTS


def _hold_fingerprint_lock(entered, release):
    with internal_api._FINGERPRINT_LOCK:
        entered.set()
        assert release.wait(5), "parent fingerprint lock was not released"


def _fork_fingerprint_case(hold_lock):
    os.environ["GDTENSOR_CACHE_COORDINATOR"] = "thread"
    chain = _build()[0]
    original = internal_api._chain_fingerprint
    pid = os.getpid()
    fingerprint_entered = threading.Event()
    release_fingerprint = threading.Event()
    lock_entered = threading.Event()
    release_lock = threading.Event()

    def paused_fingerprint(value):
        if os.getpid() == pid:
            fingerprint_entered.set()
            assert release_fingerprint.wait(5), "parent fingerprint was not released"
        return original(value)

    ctx = mp.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_report, args=(_slots_in_child, (chain,), sender))
    errors = []
    thread = threading.Thread(
        target=_slots_in_thread, args=(chain, errors), daemon=True
    )
    locker = None
    patch = pytest.MonkeyPatch()
    patch.setattr(internal_api, "_chain_fingerprint", paused_fingerprint)
    try:
        thread.start()
        assert fingerprint_entered.wait(5), "parent fingerprint did not start"
        if hold_lock:
            locker = threading.Thread(
                target=_hold_fingerprint_lock,
                args=(lock_entered, release_lock),
                daemon=True,
            )
            locker.start()
            assert lock_entered.wait(5), "parent fingerprint lock was not acquired"
        child.start()
        sender.close()
        release_fingerprint.set()
        release_lock.set()
        thread.join(10)
        assert not thread.is_alive(), "parent fingerprint call hung"
        assert not errors, errors
        if locker is not None:
            locker.join(5)
            assert not locker.is_alive(), "parent fingerprint lock call hung"
        child.join(10)
        assert not child.is_alive(), "child inherited active fingerprint state"
        assert child.exitcode == 0, child.exitcode
        assert receiver.poll(0), "child did not report its assertions"
        error = receiver.recv()
        assert error is None, error
        assert not internal_api._FINGERPRINTS
    finally:
        patch.undo()
        release_fingerprint.set()
        release_lock.set()
        _stop(child)
        thread.join(5)
        if locker is not None:
            locker.join(5)
        receiver.close()
        sender.close()


def test_fork_with_pending_fingerprint_starts_clean():
    _run_bounded(_fork_fingerprint_case, False)


def test_fork_with_locked_fingerprint_state_starts_clean():
    _run_bounded(_fork_fingerprint_case, True)


class _PauseOwnerCreation:
    def __init__(self):
        self.mkstemp = tempfile.mkstemp
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args, **kwargs):
        fd, path = self.mkstemp(*args, **kwargs)
        if Path(path).name.startswith("gdtensor-run-"):
            self.entered.set()
            assert self.release.wait(5), "owner fd creation was not released"
        return fd, path


def _fd_target(path):
    try:
        return os.readlink(path)
    except FileNotFoundError:
        return ""


def _assert_no_owner_fds():
    inherited = [
        target
        for path in Path("/proc/self/fd").iterdir()
        if "gdtensor-run-" in (target := _fd_target(path))
    ]
    assert not inherited, f"child inherited owner descriptors: {inherited}"


def _start_child(child, returned, errors):
    try:
        child.start()
    except BaseException:
        errors.append(traceback.format_exc()[-6000:])
    finally:
        returned.set()


def _fork_during_owner_creation(args, pause):
    ctx = mp.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_report, args=(_assert_no_owner_fds, (), sender))
    attempting, returned = threading.Event(), threading.Event()
    # Before-fork hooks run in reverse registration order, before the library lock.
    os.register_at_fork(before=attempting.set)
    errors = []
    io_thread = threading.Thread(target=_thread_round, args=(args, errors), daemon=True)
    fork_thread = threading.Thread(
        target=_start_child, args=(child, returned, errors), daemon=True
    )
    io_thread.start()
    try:
        assert pause.entered.wait(5), "owner raw fd was not created"
        fork_thread.start()
        assert attempting.wait(5), "fork did not reach its before-fork hooks"
        assert not returned.wait(0.2), "fork passed an unregistered owner fd"
        pause.release.set()
        fork_thread.join(5)
        io_thread.join(5)
        assert not fork_thread.is_alive(), "fork did not resume after registration"
        assert not io_thread.is_alive(), "parent cache IO did not complete"
        assert not errors, errors
        sender.close()
        child.join(5)
        assert not child.is_alive(), "child descriptor check hung"
        assert child.exitcode == 0, child.exitcode
        assert receiver.poll(0), "child did not report its descriptor check"
        error = receiver.recv()
        assert error is None, error
    finally:
        pause.release.set()
        if fork_thread.ident is not None:
            fork_thread.join(5)
        _stop(child)
        io_thread.join(5)
        receiver.close()
        sender.close()


def _owner_creation_case():
    chain, cluster, _cache, dst, values = _build()
    args = (chain, cluster, dst, values)
    pause = _PauseOwnerCreation()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tempfile, "mkstemp", pause)
        _fork_during_owner_creation(args, pause)
    _load_round(*args)


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="requires Linux procfs")
def test_fork_waits_for_owner_fd_registration():
    _run_bounded(_owner_creation_case)


class _FailFilledCommit:
    def __init__(self):
        self.replace = os.replace
        self.armed = False
        self.failures = 0

    def arm(self):
        self.armed = True

    def __call__(self, src, dst, *args, **kwargs):
        if self.armed and not self.failures and str(dst).endswith(".state"):
            self.failures += 1
            raise OSError(errno.EIO, "injected filled-slot commit failure")
        return self.replace(src, dst, *args, **kwargs)


def _assert_peer_down(chain, cluster, rank):
    cluster.rank = rank
    result = gio.chain_load(rank, chain)
    assert type(result) is gb.Err, result
    assert getattr(result.value, "check_name", None) == "cache_peer_down", result


def _commit_failure_case(root, fresh_directory):
    chain, cluster, cache, dst, values = _build()
    fault = _FailFilledCommit()
    cache.after_write = fault.arm
    cluster.rank = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", fault)
        result = gio.chain_load(0, chain)
    assert fault.failures == 1, "filled-slot commit fault was not injected"
    assert type(result) is gb.Err, result
    assert cache.writes and not cache.reads, "failure was not between fill and drain"
    _assert_peer_down(chain, cluster, 1)
    _assert_peer_down(chain, cluster, 0)
    if fresh_directory:
        fresh = Path(root) / "fresh"
        fresh.mkdir()
        os.environ["GDTENSOR_CACHE_DIR"] = str(fresh)
        _load_round(chain, cluster, dst, values)
        assert list(fresh.glob("*.state")), "new directory was not used"


@pytest.mark.parametrize("fresh_directory", [False, True], ids=["same-dir", "new-dir"])
def test_filled_commit_failure_does_not_leave_unowned_inflight(
    tmp_path, fresh_directory
):
    _run_bounded(_commit_failure_case, str(tmp_path), fresh_directory)


def test_fill_entry_commit_failure_releases_reservation():
    chain, cluster, cache, dst, values = _build()
    fault = _FailFilledCommit()
    original = internal_api._RankRun._fill
    armed = []

    def fail_entry(self, layer, node, seg, data=None):
        if self.rank == 0 and not armed:
            armed.append(True)
            fault.arm()
        return original(self, layer, node, seg, data)

    cluster.rank = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", fault)
        patch.setattr(internal_api._RankRun, "_fill", fail_entry)
        result = gio.chain_load(0, chain)
    assert fault.failures == 1 and type(result) is gb.Err
    assert not cache.writes
    table = internal_api._slots(chain)
    assert table.inflight == 0
    assert all(not state.slots for state in table.nodes.values())
    _load_round(chain, cluster, dst, values)


def test_drain_entry_commit_failure_publishes_error():
    chain, cluster, _cache, _dst, _values = _build()
    fault = _FailFilledCommit()
    original = internal_api._RankRun._drain
    armed = []

    def fail_entry(self, piece, data=None):
        if self.rank == 0 and not armed:
            armed.append(True)
            fault.arm()
        return original(self, piece, data)

    cluster.rank = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", fault)
        patch.setattr(internal_api._RankRun, "_drain", fail_entry)
        result = gio.chain_load(0, chain)
    assert fault.failures == 1 and type(result) is gb.Err
    assert "injected filled-slot commit failure" in result.value.msg
    table = internal_api._slots(chain)
    assert table.inflight == 0
    assert table.node(0, 0).error is not None


@pytest.mark.parametrize("operation", ["fill", "probe", "drain"])
def test_interrupted_error_publication_keeps_owner(operation):
    chain, cluster, cache, dst, _values = _build()
    fault = _FailFilledCommit()
    original_fail = internal_api._RankRun._fail

    def fail(self, *args):
        fault.arm()
        return original_fail(self, *args)

    def source_error(*_args):
        raise OSError(errno.EIO, "original fill failure")

    def probe_error(*_args):
        raise OSError(errno.EIO, "original probe failure")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", fault)
        patch.setattr(internal_api._RankRun, "_fail", fail)
        if operation == "fill":
            patch.setattr(internal_api._RankRun, "_source_bytes", source_error)
        elif operation == "probe":
            # Keep the identity stable for owner inspection after this patch context.
            cast(Any, chain).src.data_accessor.cache_identity = lambda: "probe-source"
            patch.setattr(cache, "has_bytes", probe_error)
        else:
            patch.setattr(
                dst,
                "write_from_buffer",
                lambda *_args: gb.Err(
                    gb.GDTensorIOError(operation="write", msg="original drain failure")
                ),
            )
        cluster.rank = 0
        result = gio.chain_load(0, chain)

    assert fault.failures == 1
    assert type(result) is gb.Err, result
    assert f"original {operation} failure" in result.value.msg, result
    table = internal_api._slots(chain)
    assert table.inflight == 1
    assert isinstance(table, internal_api._FileSlotTable)
    assert len(table.runs) == 1
    assert table.node(0, 0).error is None
    _assert_peer_down(chain, cluster, 1)


@pytest.mark.parametrize("coordinator", ["file", "thread"])
def test_probe_failure_reaches_peer_without_sharing_error_object(
    coordinator, monkeypatch
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", coordinator)
    chain, cluster, cache, _dst, _values = _build()
    monkeypatch.setattr(
        cast(Any, chain).src.data_accessor,
        "cache_identity",
        lambda: "owner-test-source",
    )
    original = cache.has_bytes
    entered, release = threading.Event(), threading.Event()
    failures = []

    def probe(global_range):
        rank = cluster.rank
        if rank == 1:
            entered.set()
            assert release.wait(5), "failing rank did not probe"
        elif not failures:
            assert entered.wait(5), "peer did not join the probe generation"
            failures.append(rank)
            release.set()
            raise OSError(errno.EIO, "injected cache probe failure")
        return original(global_range)

    cache.has_bytes = probe
    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, chain)

    threads = [threading.Thread(target=one, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not any(thread.is_alive() for thread in threads)
    assert failures == [0]
    assert all(type(result) is gb.Err for result in results.values()), results
    assert results[0].value == results[1].value
    state = internal_api._slots(chain).node(0, 0)
    assert results[0].value is not state.error
    assert results[1].value is not state.error
    incoming = internal_api._NodeState(0, set())
    incoming.fail(internal_api.Fail(results[0].value), None)
    assert incoming.error == results[0].value
    assert incoming.error is not results[0].value
    message = results[1].value.msg
    results[0].value.msg = "mutated by caller"
    assert results[1].value.msg == message
    assert state.error is not None
    assert state.error.msg == message
    results[1].value.msg = "mutated by peer"
    assert state.error.msg == message


def test_fill_error_is_published_before_inflight_release(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "5")
    chain, cluster, _cache, _dst, _values = _build()
    original_source = internal_api._RankRun._source_bytes
    original_fail = internal_api._RankRun._fail
    original_block = internal_api._RankRun._block
    failing, release, waiting = threading.Event(), threading.Event(), threading.Event()

    def source_bytes(self, layer, seg):
        if self.rank == 0:
            raise OSError(errno.EIO, "injected fill failure")
        return original_source(self, layer, seg)

    def fail(self, layer, node, exc, key):
        if self.rank == 0 and not failing.is_set():
            failing.set()
            assert release.wait(5), "fill failure was not released"
        return original_fail(self, layer, node, exc, key)

    def block(self, version, deadline, pending):
        if self.rank == 1:
            waiting.set()
        return original_block(self, version, deadline, pending)

    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, chain)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(internal_api._RankRun, "_source_bytes", source_bytes)
        patch.setattr(internal_api._RankRun, "_fail", fail)
        patch.setattr(internal_api._RankRun, "_block", block)
        first = threading.Thread(target=one, args=(0,))
        second = threading.Thread(target=one, args=(1,))
        first.start()
        try:
            assert failing.wait(5), "rank 0 did not reach the fill failure"
            second.start()
            assert waiting.wait(5), "rank 1 did not wait for the fill"
            time.sleep(0.4)
        finally:
            release.set()
        first.join(10)
        second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert all(type(result) is gb.Err for result in results.values()), results
    assert results[0].value == results[1].value
    assert "injected fill failure" in results[1].value.msg


def test_parent_probe_failure_reaches_waiting_peer():
    case = dict(
        seed=0,
        k=1,
        nodes=2,
        layers=[[0], [0, 1]],
        steps=[1, 1],
        width=[1, 1],
        sizes=[8],
        replicate=[False],
        coordinator="file",
        bandwidth=0.0,
        repeat=1,
    )
    built = _build_case(case)
    chain, cluster = built["chain"], built["cluster"]
    parent = built["caches"][0]
    original = parent.has_bytes
    entered, release = threading.Event(), threading.Event()

    def probe(global_range):
        rank = cluster.rank
        if rank == 1:
            entered.set()
            assert release.wait(5), "failing parent probe was not released"
            return original(global_range)
        assert entered.wait(5), "peer did not reach the parent probe"
        release.set()
        raise OSError(errno.EIO, "injected parent probe failure")

    parent.has_bytes = probe
    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, chain)

    threads = [threading.Thread(target=one, args=(rank,)) for rank in (1, 0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not any(thread.is_alive() for thread in threads)
    assert all(type(result) is gb.Err for result in results.values()), results
    assert results[0].value == results[1].value
    assert "injected parent probe failure" in results[1].value.msg


def test_v3_coordinator_state_is_rejected(tmp_path):
    chain, cluster, _cache, dst, values = _build()
    _load_round(chain, cluster, dst, values)
    path = next(tmp_path.glob("*.state"))
    state = pickle.loads(path.read_bytes())
    state["magic"] = "gdtensor-coord-v3"
    path.write_bytes(pickle.dumps(state))
    cluster.rank = 0
    result = gio.chain_load(0, chain)
    assert type(result) is gb.Err, result
    assert result.value.check_name == "cache_state", result

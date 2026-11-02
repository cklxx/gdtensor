"""Owner cleanup through public IO and injected filesystem failures."""

import errno
import multiprocessing as mp
import os
import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.cache_oracle import cache_view, compute_table, storage_schema
from tests.support.cluster_sim import Cluster, NodeCache, RankMem, StorageMem


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

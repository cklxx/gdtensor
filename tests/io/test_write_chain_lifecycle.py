"""Process lifecycle, crash recovery, tamper, and fault propagation for write chains."""

from __future__ import annotations

import errno
import fcntl
import gc
import json
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import safetensors.numpy

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_conn.file_cache import DiskNodeCache
from gdtensor_conn.formats import COMMIT_MARKER_NAME, make_storage_view
from gdtensor_conn.iobackend import FileIO, SdkFileIO
from tests.support.cluster_sim import Cluster, NodeCache
from tests.support.crc import crc_of
from tests.support.sdk_client import MemorySdkClient

internal_wc = sys.modules[gio.chain_save.__module__]


@pytest.fixture(autouse=True)
def _isolated_write_coord(tmp_path: Path, monkeypatch):
    coord = tmp_path / "wcoord"
    coord.mkdir()
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "file")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(coord))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "1.5")


class _DummyDevice:
    def charge(self, _node: int, _nbytes: int) -> None:
        pass


class _FileVote:
    """Picklable POSIX file barrier/vote for fork & spawn worker processes."""

    def __init__(self, vote_dir: str, world: int) -> None:
        self.vote_dir = vote_dir
        self.world = world
        self.round = 0

    def _bump(self, state_path: Path, lock_path: Path, ok: bool) -> None:
        with lock_path.open("a+b") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            cur = (
                json.loads(state_path.read_text())
                if state_path.exists()
                else {"n": 0, "fail": 0}
            )
            cur["n"] += 1
            cur["fail"] += int(not ok)
            state_path.write_text(json.dumps(cur))

    def vote(self, ok: bool) -> bool:
        rid = self.round
        self.round += 1
        state_path = Path(self.vote_dir) / f"round-{rid}.json"
        lock_path = Path(self.vote_dir) / f"round-{rid}.lock"
        self._bump(state_path, lock_path, ok)
        return _poll_vote_result(state_path, self.world, rid)

    def wait(self) -> None:
        self.vote(True)


def _poll_vote_result(state_path: Path, world: int, rid: int) -> bool:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        cur = json.loads(state_path.read_text())
        if cur["n"] >= world:
            return cur["fail"] == 0
        time.sleep(0.005)
    raise TimeoutError(f"vote round {rid} timed out")


class _ProcessWriteCache(ga.DTensorDataAccessor):
    """Cross-process file-backed cache accessor supporting implicit/explicit ns."""

    def __init__(self, root: str, sizes: dict[str, int], explicit: bool) -> None:
        self.root = Path(root)
        self.sizes = sizes
        self.explicit = explicit

    def cache_namespace(self):
        return "mp-write-shared" if self.explicit else None

    def _slot_path(self, rng) -> Path:
        identity = getattr(rng, "source_digest", "local")
        sub = self.root / str(identity)
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{rng.global_tensor_id}-{getattr(rng, 'cache_node', 0)}.bin"

    def _ensure_size(self, fd: int, size: int) -> None:
        if os.fstat(fd).st_size < size:
            os.ftruncate(fd, size)

    def write_from_buffer(self, global_range, write_from_buffer):
        buf = np.empty(global_range.size, dtype=np.uint8)
        res = write_from_buffer(buf.ctypes.data, buf.nbytes)
        if type(res) is not gb.Ok:
            return res
        path = self._slot_path(global_range)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            self._ensure_size(fd, self.sizes[global_range.global_tensor_id])
            os.pwrite(fd, buf.tobytes(), global_range.global_tensor_offset)
        finally:
            os.close(fd)
        return res

    def read_to_buffer(self, global_range, read_to_buffer):
        path = self._slot_path(global_range)
        fd = os.open(path, os.O_RDONLY)
        try:
            data = os.pread(fd, global_range.size, global_range.global_tensor_offset)
        finally:
            os.close(fd)
        arr = np.frombuffer(data, dtype=np.uint8)
        return read_to_buffer(arr.ctypes.data, arr.nbytes)


class _ArrayCompute(ga.DTensorDataAccessor):
    def __init__(self, data: dict[str, np.ndarray], on_read=None) -> None:
        self.data = data
        self.on_read = on_read

    def read_to_buffer(self, global_range, read_to_buffer):
        if self.on_read is not None:
            self.on_read(global_range)
        off, size = global_range.global_tensor_offset, global_range.size
        buf = self.data[global_range.global_tensor_id][off : off + size].copy()
        return read_to_buffer(buf.ctypes.data, buf.nbytes)

    def write_from_buffer(self, global_range, write_from_buffer):
        return gb.Err(gb.GDTensorIOError(operation="write", msg="read-only"))


class _WriteSession:
    def __init__(self, acc) -> None:
        self.acc = acc

    def get_file_offset(self, _pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        return gb.Ok(self.acc)

    def commit(self):
        return gb.Ok(None)

    def abort(self):
        return gb.Ok(None)


def _compute_view(acc, tids, shapes, world, placements=None):
    mesh = gb.ArrangedMesh(gt.WorldSize(world))
    place = [gb.Shard(0)] if placements is None else placements
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(gb.GlobalTensorMeta(shapes[tid], "U8", 1), place, mesh),
            )
            for tid in tids
        ]
    )
    return gb.ComputeDTensorsView(list(tids), table, acc)


def _cache_view(acc, tids, shapes, steps=1, nodes=None, width=1):
    nodes = [0] if nodes is None else nodes
    mesh = gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=gb.LiteralMesh(list(nodes), gt.NumNodes(max(nodes) + 1)),
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta(shapes[tid], "U8", 1),
                    [gb.Shard(0), gb.Shard(0), gb.Shard(0)],
                    mesh,
                ),
            )
            for tid in tids
        ]
    )
    return gb.CacheDTensorsView(list(tids), table, acc)


def _single_file_safetensors_storage(tids, shapes, world, root: str, io_backend, coord):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.LiteralMesh([0], gt.WorldSize(world))
    )
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta(shapes[tid], "U8", 1),
                    [gb.Replicate(), gb.Replicate()],
                    mesh,
                ),
            )
            for tid in tids
        ]
    )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(len(tids))),
                gb.LiteralMesh([0], gt.WorldSize(world)),
            )
        ],
        list(tids),
    )
    leader = gb.TensorComponentPos(tids[0], 0)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {leader: group}, {leader: "model.safetensors"}
    )
    return make_storage_view(
        list(tids), table, layout, root, io=io_backend, coord=coord
    )


def _mem_storage_view(acc, tids, shapes, world, single_slot: bool = False):
    slot_mesh = (
        gb.LiteralMesh([0], gt.WorldSize(world))
        if single_slot
        else gb.ArrangedMesh(gt.WorldSize(world))
    )
    place = [gb.Replicate(), gb.Replicate() if single_slot else gb.Shard(0)]
    mesh = gb.SlotOuterProductMesh(gb.ArrangedMesh(gt.ConstantOne(1)), slot_mesh)
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(gb.GlobalTensorMeta(shapes[tid], "U8", 1), place, mesh),
            )
            for tid in tids
        ]
    )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(len(tids))), slot_mesh
            )
        ],
        list(tids),
    )
    ranks = [0] if single_slot else list(range(world))
    entries = {gb.TensorComponentPos(t, r): group for t in tids for r in ranks}
    layout = gb.PhysicalLayoutTable(
        "safetensors", entries, {gb.TensorComponentPos(tids[0], 0): "model.bin"}
    )
    cb = lambda _v, _r: gb.Ok(_WriteSession(acc))
    return gb.StorageDTensorsView(
        list(tids), table, acc, layout, begin_save=cb, begin_load=cb
    )


class _MemStorage(ga.DTensorDataAccessor):
    def __init__(self, size: int) -> None:
        self.buf = np.zeros(size, dtype=np.uint8)

    def read_to_buffer(self, global_range, read_to_buffer):
        return gb.Err(gb.GDTensorIOError(operation="read", msg="write-only"))

    def write_from_buffer(self, global_range, write_from_buffer):
        off, size = global_range.contiguous_payload_offset, global_range.size
        tmp = np.empty(size, dtype=np.uint8)
        res = write_from_buffer(tmp.ctypes.data, size)
        if type(res) is gb.Ok:
            self.buf[off : off + size] = tmp
        return res


def _table_of(chain):
    active = internal_wc._fp_entry(chain)
    try:
        return internal_wc._write_slots(chain, active)
    finally:
        internal_wc._fp_release(id(chain), active[0])


def _stop_proc(proc) -> None:
    if proc.pid is None:
        return
    if proc.is_alive():
        proc.kill()
    proc.join(5)
    proc.close()


def _run_report(fn, args, sender) -> None:
    try:
        fn(*args)
        sender.send(None)
    except BaseException:
        sender.send(traceback.format_exc()[-6000:])
    finally:
        sender.close()


# ---------------------------------------------------------------- multiprocess


def _mp_worker(
    rank: int,
    world: int,
    ckpt_dir: str,
    cache_dir: str,
    coord_dir: str,
    vote_dir: str,
    raw_bytes: bytes,
    sender,
) -> None:
    try:
        os.environ["GDTENSOR_CACHE_COORDINATOR"] = "file"
        os.environ["GDTENSOR_CACHE_DIR"] = coord_dir
        os.environ["GDTENSOR_CACHE_WAIT_SECONDS"] = "10"
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
        shapes = {"w": [4, 4]}
        cv = _compute_view(
            _ArrayCompute({"w": arr}), ["w"], shapes, world, [gb.Shard(1)]
        )
        cache = _ProcessWriteCache(cache_dir, {"w": 16}, explicit=True)
        sv = _single_file_safetensors_storage(
            ["w"], shapes, world, ckpt_dir, FileIO(), _FileVote(vote_dir, world)
        )
        chain = gb.OneLayerCachedDTensorsRWChain(
            src=cv,
            cache=_cache_view(cache, ["w"], shapes),
            dst=sv,
            num_ranks_per_node=world,
        )
        res = gio.chain_save(rank, chain)
        assert type(res) is gb.Ok, res
        sender.send(None)
    except BaseException:
        sender.send(traceback.format_exc()[-6000:])
    finally:
        sender.close()


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_multiprocess_safetensors_chain_save(tmp_path: Path, start_method: str):
    ckpt_dir, cache_dir, coord_dir, vote_dir = [
        tmp_path / name for name in ("ckpt", "cache", "coord", "vote")
    ]
    for p in (ckpt_dir, cache_dir, coord_dir, vote_dir):
        p.mkdir(parents=True, exist_ok=True)
    raw = np.arange(16, dtype=np.uint8) + 17
    world = 2
    ctx = cast(Any, mp.get_context(start_method))
    procs, pipes = [], []
    for r in range(world):
        rx, tx = ctx.Pipe(duplex=False)
        args = (
            r,
            world,
            str(ckpt_dir),
            str(cache_dir),
            str(coord_dir),
            str(vote_dir),
            raw.tobytes(),
            tx,
        )
        proc = ctx.Process(target=_mp_worker, args=args)
        proc.start()
        tx.close()
        procs.append(proc)
        pipes.append(rx)
    try:
        for proc, rx in zip(procs, pipes):
            proc.join(25)
            assert not proc.is_alive() and proc.exitcode == 0, proc.exitcode
            err = rx.recv()
            assert err is None, err
    finally:
        for proc, rx in zip(procs, pipes):
            _stop_proc(proc)
            rx.close()
    assert (ckpt_dir / COMMIT_MARKER_NAME).exists()
    got = safetensors.numpy.load_file(ckpt_dir / "model.safetensors")["w"].reshape(-1)
    assert crc_of(got.tobytes()) == crc_of(raw.tobytes())
    np.testing.assert_array_equal(got, raw)


def _send_write_key(chain, sender) -> None:
    try:
        sender.send(_table_of(chain).key)
    except BaseException:
        sender.send(traceback.format_exc()[-6000:])
    finally:
        sender.close()


@pytest.mark.parametrize("explicit", [False, True], ids=["implicit", "explicit"])
def test_implicit_vs_explicit_namespace_across_fork(tmp_path: Path, explicit: bool):
    shapes = {"w": [8]}
    cache = _ProcessWriteCache(str(tmp_path / "c"), {"w": 8}, explicit=explicit)
    src = _ArrayCompute({"w": np.arange(8, dtype=np.uint8)})
    dst = _MemStorage(8)
    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(src, ["w"], shapes, 2),
        cache=_cache_view(cache, ["w"], shapes),
        dst=_mem_storage_view(dst, ["w"], shapes, 2),
        num_ranks_per_node=2,
    )
    parent_key = _table_of(chain).key
    ctx = mp.get_context("fork")
    rx, tx = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_send_write_key, args=(chain, tx))
    try:
        child.start()
        tx.close()
        child.join(10)
        assert not child.is_alive() and child.exitcode == 0, child.exitcode
        child_key = rx.recv()
        assert isinstance(child_key, str), child_key
        assert (child_key == parent_key) is explicit
    finally:
        _stop_proc(child)
        rx.close()


# ---------------------------------------------------------------- crash & fork


def _crash_in_stage(chain, cache, ready_tx, die_rx) -> None:
    def aborting_write(_rng, _cb):
        ready_tx.send("in_stage")
        ready_tx.close()
        die_rx.recv()
        os._exit(1)

    cache.write_from_buffer = aborting_write
    gio.chain_save(0, chain)


def test_peer_crash_during_stage_raises_cache_peer_down(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "1.0")
    cluster = Cluster(1, 2)
    shapes = {"w": [8]}
    cache = NodeCache(cluster, {"w": 8}, lambda _r: 0)
    cache.namespace = "peer-crash-stage"
    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(
            _ArrayCompute({"w": np.arange(8, dtype=np.uint8)}), ["w"], shapes, 2
        ),
        cache=_cache_view(cache, ["w"], shapes),
        dst=_mem_storage_view(_MemStorage(8), ["w"], shapes, 2, single_slot=True),
        num_ranks_per_node=2,
    )
    in_step, orig_step = threading.Event(), internal_wc._WriteFileCond._step
    parent_pid = os.getpid()

    def watch_step(cond, pred, deadline):
        if os.getpid() == parent_pid:
            in_step.set()
        return orig_step(cond, pred, deadline)

    monkeypatch.setattr(internal_wc._WriteFileCond, "_step", watch_step)
    ctx = mp.get_context("fork")
    r_rx, r_tx = ctx.Pipe(duplex=False)
    d_rx, d_tx = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_crash_in_stage, args=(chain, cache, r_tx, d_rx))
    peer_out: dict[int, Any] = {}

    def run_peer() -> None:
        cluster.rank = 1
        peer_out[1] = gio.chain_save(1, chain)

    peer = threading.Thread(target=run_peer, daemon=True)
    try:
        child.start()
        r_tx.close()
        d_rx.close()
        assert r_rx.poll(5) and r_rx.recv() == "in_stage"
        peer.start()
        assert in_step.wait(5), "peer did not enter _WriteFileCond._step"
        d_tx.send("die")
        child.join(5)
        peer.join(5)
    finally:
        d_tx.close()
        r_rx.close()
        _stop_proc(child)
    assert not peer.is_alive()
    res = peer_out[1]
    assert (
        type(res) is gb.Err
        and getattr(res.value, "check_name", None) == "cache_peer_down"
    ), res


def _crash_in_cohort(chain, ready_tx, die_rx) -> None:
    orig_round = internal_wc._WriteRankRun._cohort_round
    signaled = False

    def watch_round(self, deadline):
        nonlocal signaled
        if not signaled:
            signaled = True
            ready_tx.send("in_cohort")
            ready_tx.close()
            die_rx.recv()
            os._exit(1)
        return orig_round(self, deadline)

    internal_wc._WriteRankRun._cohort_round = watch_round
    gio.chain_save(0, chain)


def test_peer_crash_during_cohort_wait_raises_cache_peer_down(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "2.0")
    cluster = Cluster(1, 3)
    shapes = {"w": [12]}
    cache = NodeCache(cluster, {"w": 12}, lambda _r: 0)
    cache.namespace = "peer-crash-cohort"
    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(
            _ArrayCompute({"w": np.arange(12, dtype=np.uint8)}), ["w"], shapes, 3
        ),
        cache=_cache_view(cache, ["w"], shapes, steps=2, nodes=[0], width=1),
        dst=_mem_storage_view(_MemStorage(12), ["w"], shapes, 3, single_slot=True),
        num_ranks_per_node=3,
    )
    in_step, orig_step = threading.Event(), internal_wc._WriteFileCond._step
    parent_pid = os.getpid()

    def watch_step(cond, pred, deadline):
        if os.getpid() == parent_pid:
            in_step.set()
        return orig_step(cond, pred, deadline)

    monkeypatch.setattr(internal_wc._WriteFileCond, "_step", watch_step)
    ctx = mp.get_context("fork")
    r_rx, r_tx = ctx.Pipe(duplex=False)
    d_rx, d_tx = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_crash_in_cohort, args=(chain, r_tx, d_rx))
    peer_out: dict[int, Any] = {}

    def run_peer() -> None:
        cluster.rank = 1
        peer_out[1] = gio.chain_save(1, chain)

    peer = threading.Thread(target=run_peer, daemon=True)
    t0 = time.monotonic()
    try:
        child.start()
        r_tx.close()
        d_rx.close()
        assert r_rx.poll(5) and r_rx.recv() == "in_cohort"
        peer.start()
        assert in_step.wait(5), "peer did not wait in cohort"
        d_tx.send("die")
        child.join(5)
        peer.join(5)
        elapsed = time.monotonic() - t0
    finally:
        d_tx.close()
        r_rx.close()
        _stop_proc(child)
    assert not peer.is_alive() and elapsed < 1.0, elapsed
    res = peer_out[1]
    assert (
        type(res) is gb.Err
        and getattr(res.value, "check_name", None) == "cache_peer_down"
    ), res


def _fork_fp_worker(chain) -> None:
    assert not internal_wc._WRITE_FINGERPRINTS
    assert _table_of(chain) is not None


def _hold_fp_lock(entered, release) -> None:
    with internal_wc._WRITE_FINGERPRINT_LOCK:
        entered.set()
        assert release.wait(5), "parent fp lock not released"


def test_fork_safety_with_held_locks_and_pending_fingerprint(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    shapes = {"w": [8]}
    cache = NodeCache(Cluster(1, 2), {"w": 8}, lambda _r: 0)
    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(
            _ArrayCompute({"w": np.arange(8, dtype=np.uint8)}), ["w"], shapes, 2
        ),
        cache=_cache_view(cache, ["w"], shapes),
        dst=_mem_storage_view(_MemStorage(8), ["w"], shapes, 2, single_slot=True),
        num_ranks_per_node=2,
    )
    orig_fp = internal_wc._write_chain_fingerprint
    pid = os.getpid()
    fp_entered, fp_release = threading.Event(), threading.Event()
    lk_entered, lk_release = threading.Event(), threading.Event()

    def paused_fp(val):
        if os.getpid() == pid:
            fp_entered.set()
            assert fp_release.wait(5)
        return orig_fp(val)

    monkeypatch.setattr(internal_wc, "_write_chain_fingerprint", paused_fp)
    errs: list[str] = []

    def parent_slots() -> None:
        try:
            _table_of(chain)
        except BaseException:
            errs.append(traceback.format_exc())

    t_fp = threading.Thread(target=parent_slots, daemon=True)
    t_lk = threading.Thread(
        target=_hold_fp_lock, args=(lk_entered, lk_release), daemon=True
    )
    ctx = mp.get_context("fork")
    rx, tx = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_run_report, args=(_fork_fp_worker, (chain,), tx))
    try:
        t_fp.start()
        assert fp_entered.wait(5)
        t_lk.start()
        assert lk_entered.wait(5)
        child.start()
        tx.close()
        fp_release.set()
        lk_release.set()
        t_fp.join(5)
        t_lk.join(5)
        child.join(10)
        assert not child.is_alive() and child.exitcode == 0 and rx.recv() is None
        assert not errs and not internal_wc._WRITE_FINGERPRINTS
    finally:
        fp_release.set()
        lk_release.set()
        _stop_proc(child)
        rx.close()


# ---------------------------------------------------------------- state + commit + tamper


def _run_two_rank_threads(chain, cluster=None) -> dict[int, Any]:
    out: dict[int, Any] = {}

    def one(r: int) -> None:
        if cluster is not None:
            cluster.rank = r
        out[r] = gio.chain_save(r, chain)

    threads = [threading.Thread(target=one, args=(r,), daemon=True) for r in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(10)
    assert not any(th.is_alive() for th in threads)
    return out


@pytest.mark.parametrize("mode", ["truncated", "bad_magic"])
def test_corrupt_write_coordinator_state_rejected(tmp_path: Path, mode: str):
    cluster = Cluster(1, 2)
    shapes = {"w": [8]}
    cache = NodeCache(cluster, {"w": 8}, lambda _r: 0)
    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(
            _ArrayCompute({"w": np.arange(8, dtype=np.uint8)}), ["w"], shapes, 2
        ),
        cache=_cache_view(cache, ["w"], shapes),
        dst=_mem_storage_view(_MemStorage(8), ["w"], shapes, 2, single_slot=True),
        num_ranks_per_node=2,
    )
    res = _run_two_rank_threads(chain, cluster)
    assert all(type(v) is gb.Ok for v in res.values()), res
    state_file = next((tmp_path / "wcoord").glob("gdtensor-wcoord-*.state"))
    if mode == "truncated":
        state_file.write_bytes(b"corrupt")
    else:
        data = pickle.loads(state_file.read_bytes())
        data["magic"] = "gdtensor-wcoord-v0"
        state_file.write_bytes(pickle.dumps(data))
    cluster.rank = 0
    err = gio.chain_save(0, chain)
    assert (
        type(err) is gb.Err and getattr(err.value, "check_name", None) == "cache_state"
    ), err


class _FailCommitReplace:
    def __init__(self) -> None:
        self.orig = os.replace
        self.armed = False
        self.failures = 0

    def __call__(self, src, dst, *args, **kwargs):
        if self.armed and not self.failures and str(dst).endswith(".state"):
            self.failures += 1
            raise OSError(errno.EIO, "injected state commit failure")
        return self.orig(src, dst, *args, **kwargs)


def test_interrupted_error_commit_keeps_owner_and_peer_gets_peer_down():
    cluster = Cluster(1, 2)
    shapes = {"w": [8]}
    fault = _FailCommitReplace()
    orig_fail = internal_wc._WriteRankRun._fail

    def armed_fail(self, *args):
        fault.armed = True
        return orig_fail(self, *args)

    def bad_src(*_args):
        raise OSError(errno.EIO, "original stage failure")

    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(
            _ArrayCompute({"w": np.arange(8, dtype=np.uint8)}, bad_src),
            ["w"],
            shapes,
            2,
        ),
        cache=_cache_view(NodeCache(cluster, {"w": 8}, lambda _r: 0), ["w"], shapes),
        dst=_mem_storage_view(_MemStorage(8), ["w"], shapes, 2),
        num_ranks_per_node=2,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", fault)
        patch.setattr(internal_wc._WriteRankRun, "_fail", armed_fail)
        cluster.rank = 0
        res = gio.chain_save(0, chain)
    assert fault.failures == 1 and type(res) is gb.Err
    assert "original stage failure" in res.value.msg
    table = _table_of(chain)
    assert table.inflight == 1 and len(table.runs) == 1
    assert table.node(0, 0).error is None
    cluster.rank = 1
    peer_err = gio.chain_save(1, chain)
    assert (
        type(peer_err) is gb.Err
        and getattr(peer_err.value, "check_name", None) == "cache_peer_down"
    )


class _CoordVote:
    def __init__(self, world: int) -> None:
        self.world = world
        self.cond = threading.Condition()
        self.votes: list[bool] = []

    def vote(self, ok: bool) -> bool:
        with self.cond:
            self.votes.append(ok)
            self.cond.notify_all()
            self.cond.wait_for(
                lambda: not all(self.votes) or len(self.votes) >= self.world,
                timeout=5.0,
            )
            return all(self.votes) and len(self.votes) >= self.world


def _tamper_cache(cache: DiskNodeCache, mode: str) -> None:
    if mode == "checksum":
        cache.integrity[next(iter(cache.integrity))] = b"bad-digest"
        return
    path = Path(next(iter(cache.stamps.values()))[0])
    if mode == "truncate":
        os.truncate(path, 1)
    elif mode == "replace":
        rep = path.with_name("replaced.bin")
        rep.write_bytes(b"X" * path.stat().st_size)
        os.replace(rep, path)
    else:
        with path.open("r+b") as fh:
            fh.seek(0)
            fh.write(b"\xff\xff\xff\xff")


@pytest.mark.parametrize("mode", ["checksum", "truncate", "byte_flip", "replace"])
def test_disk_node_cache_tamper_aborts_chain_save(
    tmp_path: Path, monkeypatch, mode: str
):
    cluster = Cluster(1, 2)
    shapes = {"w": [4, 4]}
    ckpt_dir = tmp_path / "ckpt"
    cache = DiskNodeCache(
        cluster, {"w": 16}, lambda _r: 0, str(tmp_path / "dcache"), _DummyDevice()
    )
    orig_read = cache.read_to_buffer
    tampered = False

    def tamper_once(rng, cb):
        nonlocal tampered
        if not tampered:
            tampered = True
            _tamper_cache(cache, mode)
        return orig_read(rng, cb)

    monkeypatch.setattr(cache, "read_to_buffer", tamper_once)
    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(
            _ArrayCompute({"w": np.arange(16, dtype=np.uint8)}),
            ["w"],
            shapes,
            2,
            [gb.Shard(1)],
        ),
        cache=_cache_view(cache, ["w"], shapes),
        dst=_single_file_safetensors_storage(
            ["w"], shapes, 2, str(ckpt_dir), FileIO(), _CoordVote(2)
        ),
        num_ranks_per_node=2,
    )
    try:
        res = _run_two_rank_threads(chain, cluster)
    finally:
        cache.close()
    assert any(type(v) is gb.Err for v in res.values()), res
    assert not (ckpt_dir / "model.safetensors").exists()
    assert not (ckpt_dir / COMMIT_MARKER_NAME).exists()


# ---------------------------------------------------------------- propagation + budget + sdk


def test_two_layer_upstream_failure_propagates_to_downstream_waiter(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0.6")
    cluster = Cluster(2, 1)
    shapes = {"w": [8]}
    l0 = NodeCache(cluster, {"w": 8}, lambda r: r)
    l1 = NodeCache(cluster, {"w": 8}, lambda _r: 1)

    def read_or_fail(rng):
        if cluster.rank == 0:
            raise OSError(errno.EIO, "injected L0 stage error")

    chain = gb.TwoLayerCachedDTensorsRWChain(
        src=_compute_view(
            _ArrayCompute({"w": np.arange(8, dtype=np.uint8)}, read_or_fail),
            ["w"],
            shapes,
            2,
        ),
        first_cache=_cache_view(l0, ["w"], shapes, steps=1, nodes=[0, 1], width=1),
        second_cache=_cache_view(l1, ["w"], shapes, steps=1, nodes=[1], width=1),
        dst=_mem_storage_view(_MemStorage(8), ["w"], shapes, 2),
        num_ranks_per_node=1,
    )
    t0 = time.monotonic()
    res = _run_two_rank_threads(chain, cluster)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.4, (
        f"downstream rank waited until timeout instead of failing fast ({elapsed:.2f}s): {res}"
    )
    assert all(type(v) is gb.Err for v in res.values()), res
    assert "injected L0 stage error" in res[1].value.msg, res[1]


def test_multi_tensor_cell_budget_aggregation_and_weakref_cleanup(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    cluster = Cluster(1, 2)
    shapes = {"a": [4], "b": [8]}
    data = {"a": np.arange(4, dtype=np.uint8), "b": np.arange(8, dtype=np.uint8) + 10}
    cache = NodeCache(cluster, {"a": 4, "b": 8}, lambda _r: 0)
    orig_admit = internal_wc._WriteNodeState.admit
    orig_write = cache.write_from_buffer
    r0_staged, r0_count, peak = threading.Event(), 0, 0

    def record_admit(state, seg):
        nonlocal peak
        ok = orig_admit(state, seg)
        if ok:
            peak = max(peak, state.resident)
        return ok

    def track_write(rng, cb):
        nonlocal r0_count
        res = orig_write(rng, cb)
        if type(res) is not gb.Ok or cluster.rank != 0:
            return res
        r0_count += 1
        if r0_count >= 2:
            r0_staged.set()
        return res

    def hold_rank1(_rng):
        if cluster.rank == 1:
            assert r0_staged.wait(5), "rank 0 did not stage both tensors"

    monkeypatch.setattr(internal_wc._WriteNodeState, "admit", record_admit)
    monkeypatch.setattr(cache, "write_from_buffer", track_write)
    dst = _MemStorage(12)
    cview = _cache_view(cache, ["a", "b"], shapes)
    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(_ArrayCompute(data, hold_rank1), ["a", "b"], shapes, 2),
        cache=cview,
        dst=_mem_storage_view(dst, ["a", "b"], shapes, 2, single_slot=True),
        num_ranks_per_node=2,
    )
    assert list(gio.chain_stat_mem_overhead(chain)) == [12]
    res = _run_two_rank_threads(chain, cluster)
    assert all(type(v) is gb.Ok for v in res.values()), res
    assert peak == 12
    assert any(
        v.key == _table_of(chain).key for v in internal_wc._write_view_slots.values()
    )
    key = _table_of(chain).key
    del chain, cview
    gc.collect()
    assert all(v.key != key for v in internal_wc._write_view_slots.values())


def test_sdk_delayed_leader_cross_rank_staging_notorch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    world, k = 2, 2
    mesh_s = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.LiteralMesh([1], gt.WorldSize(world))
    )
    st = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([8], "U8", 1),
                    [gb.Replicate(), gb.Replicate()],
                    mesh_s,
                ),
            )
            for tid in ("a", "b")
        ]
    )
    grp = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(2)),
                gb.LiteralMesh([1], gt.WorldSize(world)),
            )
        ],
        ["a", "b"],
    )
    layout = gb.PhysicalLayoutTable(
        "safetensors",
        {gb.TensorComponentPos("a", 1): grp, gb.TensorComponentPos("b", 1): grp},
        {gb.TensorComponentPos("a", 1): "remote.safetensors"},
    )
    client = MemorySdkClient()
    root = str(tmp_path / "sdk_ckpt")
    cluster = Cluster(1, k)
    data = {"a": np.arange(8, dtype=np.uint8), "b": np.arange(8, dtype=np.uint8) + 20}
    cache = DiskNodeCache(
        cluster, {"a": 8, "b": 8}, lambda _r: 0, str(tmp_path / "c"), _DummyDevice()
    )
    chain = gb.OneLayerCachedDTensorsRWChain(
        src=_compute_view(
            _ArrayCompute(data),
            ["a", "b"],
            {"a": [8], "b": [8]},
            world,
            [gb.Replicate()],
        ),
        cache=_cache_view(cache, ["a", "b"], {"a": [8], "b": [8]}),
        dst=make_storage_view(
            ["a", "b"], st, layout, root, io=SdkFileIO(client), coord=_CoordVote(world)
        ),
        num_ranks_per_node=k,
    )
    out: dict[int, Any] = {}

    def one(r: int) -> None:
        if r == 1:
            time.sleep(0.05)
        cluster.rank = r
        out[r] = gio.chain_save(r, chain)

    threads = [
        threading.Thread(target=one, args=(r,), daemon=True) for r in range(world)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(10)
    cache.close()
    assert all(type(v) is gb.Ok for v in out.values()), out
    loaded = safetensors.numpy.load(bytes(client.objects[f"{root}/remote.safetensors"]))
    np.testing.assert_array_equal(loaded["a"], data["a"])
    np.testing.assert_array_equal(loaded["b"], data["b"])


def test_two_layer_tight_budget_no_circular_deadlock(monkeypatch):
    """L0 admission must jointly reserve L1 so hot L1 slots cannot starve behind cold L0."""
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0.6")
    world = 4
    shapes = {"w": [16]}
    data = {"w": np.arange(16, dtype=np.uint8)}
    cluster = Cluster(1, world)
    l0 = NodeCache(cluster, {"w": 16}, lambda _r: 0)
    l1 = NodeCache(cluster, {"w": 16}, lambda _r: 0)
    l1_a1_flushed, b_tried = threading.Event(), threading.Event()
    tried_count, orig_l1_write = [0], l1.write_from_buffer
    orig_admit_cold = internal_wc._WriteRankRun._admit_cold_stage

    def _peer_admit(run, pending):
        assert l1_a1_flushed.wait(5)
        piece = orig_admit_cold(run, pending)
        tried_count[0] += 1
        if tried_count[0] >= 2:
            b_tried.set()
        return piece

    def gate_admit(run, pending):
        if run.rank in (2, 3):
            return _peer_admit(run, pending)
        if run.rank == 1:
            assert l1_a1_flushed.wait(5) and b_tried.wait(5)
        return orig_admit_cold(run, pending)

    def watch_l1_write(rng, cb):
        res = orig_l1_write(rng, cb)
        if type(res) is gb.Ok and rng.global_tensor_offset == 0:
            l1_a1_flushed.set()
        return res

    monkeypatch.setattr(internal_wc._WriteRankRun, "_admit_cold_stage", gate_admit)
    monkeypatch.setattr(l1, "write_from_buffer", watch_l1_write)
    dst = _MemStorage(16)
    chain = gb.TwoLayerCachedDTensorsRWChain(
        src=_compute_view(_ArrayCompute(data), ["w"], shapes, world),
        first_cache=_cache_view(l0, ["w"], shapes, steps=4, nodes=[0], width=2),
        second_cache=_cache_view(l1, ["w"], shapes, steps=2, nodes=[0], width=1),
        dst=_mem_storage_view(dst, ["w"], shapes, world, single_slot=True),
        num_ranks_per_node=world,
    )
    out = _run_threads(world, cluster, chain)
    assert all(type(v) is gb.Ok for v in out.values()), out
    np.testing.assert_array_equal(dst.buf, data["w"])


def _run_threads(world: int, cluster: Cluster, chain) -> dict[int, Any]:
    out: dict[int, Any] = {}

    def run_rank(r: int) -> None:
        cluster.rank = r
        out[r] = gio.chain_save(r, chain)

    threads = [
        threading.Thread(target=run_rank, args=(r,), daemon=True) for r in range(world)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(5)
        assert not th.is_alive()
    return out

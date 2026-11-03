"""Co-located deployment simulation: cache DN and ranks share each node.

Each node has its own cache memory and NIC (tests/support/cluster_sim.py). The
tests check that (1) every rank gets correct bytes, (2) every cache access
hits the node the plan says, (3) the stat traffic functions equal the bytes
the simulated NICs move, and (4) a slow NIC does not stall the executor.
"""

import multiprocessing as mp
import os
import threading
import time
from typing import cast

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.cluster_sim import (
    Cluster,
    NodeCache,
    RankMem,
    StorageMem,
    layer_targets,
)


@pytest.fixture(autouse=True)
def isolated_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "3")


def _storage(acc, size, world):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh
    )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(world))
            )
        ],
        ["w"],
    )
    entries = {gb.TensorComponentPos("w", r): group for r in range(world)}
    layout = gb.PhysicalLayoutTable(
        "safetensors", entries, {gb.TensorComponentPos("w", 0): "mem"}
    )

    class _Load:
        def get_file_offset(self, pos):
            return gb.Ok(0)

        def init_data_accessor(self):
            return gb.Ok(acc)

        def finish(self):
            return gb.Ok(None)

    return gb.StorageDTensorsView(
        ["w"],
        gb.DTensorTable([gb.NamedDTensor("w", dt)]),
        acc,
        layout,
        lambda _v, _r: gb.Ok(_Load()),
        lambda _v, _r: gb.Ok(_Load()),
    )


def _cache(acc, size, steps, nodes, width):
    spatial = gb.LiteralMesh(list(nodes), gt.NumNodes(max(nodes) + 1))
    mesh = gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=spatial,
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1),
        [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
        mesh,
    )
    return gb.CacheDTensorsView(["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), acc)


def _compute(acc, size, world):
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1),
        [gb.Shard(0)],
        gb.ArrangedMesh(gt.WorldSize(world)),
    )
    return gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), acc
    )


def _build(layers, size=16, world=8, k=2, steps=1, width=1, bandwidth=0.0):
    """layers: node tuples in data-flow order (one or two cache layers)."""
    cluster = Cluster(max(world // k, max(max(n) for n in layers) + 1), k, bandwidth)
    targets = layer_targets(cluster, [list(n) for n in layers])
    caches = [NodeCache(cluster, {"w": size}, t) for t in targets]
    src = StorageMem(cluster, np.arange(size, dtype=np.uint8))
    dst = RankMem(cluster, size, world)
    views = [_cache(c, size, steps, n, width) for c, n in zip(caches, layers)]
    common = dict(
        src=_storage(src, size, world),
        dst=_compute(dst, size, world),
        num_ranks_per_node=k,
    )
    if len(views) == 1:
        chain = gb.OneLayerCachedDTensorsReadChain(cache=views[0], **common)
    else:
        chain = gb.TwoLayerCachedDTensorsReadChain(
            small_cache=views[0], big_cache=views[1], **common
        )
    return chain, cluster, caches, dst


def _run(chain, cluster, world):
    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, chain)

    threads = [threading.Thread(target=one, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "chain_load hung"
    assert set(results) == set(range(world)), results
    bad = {r: v.value.msg for r, v in results.items() if type(v) is not gb.Ok}
    assert not bad, bad


def _check(chain, cluster, caches, dst, size, world):
    """Correct bytes on every rank, and every cache access on the planned node."""
    assert type(chain.check_constraints()) is gb.Ok
    _run(chain, cluster, world)
    expected = np.arange(size, dtype=np.uint8)
    for rank, data in dst.buffers.items():
        start = rank * ((size + world - 1) // world)
        stop = min(size, start + (size + world - 1) // world)
        want = np.full(size, 0xA5, dtype=np.uint8)
        want[start:stop] = expected[start:stop]
        np.testing.assert_array_equal(data, want, err_msg=f"rank {rank}")
    assert [c.wrong_node for c in caches] == [0] * len(caches)
    for cache in caches:
        assert all(
            actor == target or target >= world // cluster.k
            for actor, target, _off, _size in cache.writes
        )
    return cluster.nic_bytes()


def _stat_vs_nic(chain, cluster):
    rx, tx = cluster.nic_bytes()
    nodes = sorted(cluster.nics)
    # Storage has no node id: the stat books a storage read as inbound of the
    # cache node it fills; the NIC books cache traffic between nodes.
    observed_in = [rx[n] + cluster.storage_in[n] for n in nodes]
    return (
        (gio.chain_stat_inbound_traffic(chain), observed_in),
        (gio.chain_stat_outbound_traffic(chain), tx),
    )


CASES = {
    "full_cover_one_layer": [(0, 1, 2, 3)],
    "partial_one_layer": [(0, 1)],
    "single_dn_one_layer": [(0,)],
    "two_layer_fanout": [(0,), (0, 1, 2, 3)],
    "two_layer_partial": [(0, 1), (0, 1, 2)],
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("steps,width", [(1, 1), (2, 2)])
def test_colocated_delivery_hits_planned_nodes(name, steps, width):
    size, world = 16, 8
    chain, cluster, caches, dst = _build(CASES[name], size, world, 2, steps, width)
    _check(chain, cluster, caches, dst, size, world)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("steps,width", [(1, 1), (2, 2), (8, 2)])
def test_colocated_nic_bytes_match_stats(name, steps, width):
    chain, cluster, caches, dst = _build(CASES[name], steps=steps, width=width)
    _check(chain, cluster, caches, dst, 16, 8)
    (stat_in, obs_in), (stat_out, tx) = _stat_vs_nic(chain, cluster)
    assert stat_out == tx, (stat_out, tx)
    assert stat_in == obs_in, (stat_in, obs_in)


def test_full_cover_moves_no_network_bytes():
    chain, cluster, caches, dst = _build([(0, 1, 2, 3)])
    rx, tx = _check(chain, cluster, caches, dst, 16, 8)
    assert rx == tx == [0, 0, 0, 0]
    # Each node reads only its own 4 B from storage, once for both local ranks.
    assert [cluster.storage_in[n] for n in range(4)] == [4, 4, 4, 4]


def test_slow_nic_does_not_time_out(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "5")
    # 64 B/s link: one 4 B remote piece takes about 0.06 s.
    chain, cluster, caches, dst = _build([(0,)], bandwidth=64.0)
    _check(chain, cluster, caches, dst, 16, 8)


@pytest.mark.parametrize("layers", [[(4,)], [(4,), (0, 1, 2, 3)]])
def test_dn_outside_rank_nodes_is_rejected(layers):
    chain, cluster, caches, _dst = _build(layers)
    result = chain.check_constraints()
    assert type(result) is gb.Err, result
    assert result.value.constraint_name == "cache_subset"
    assert sum(cluster.storage_in.values()) == 0
    assert all(not cache.writes for cache in caches)


def test_missing_local_filler_returns_error_without_remote_write(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0.2")
    chain, cluster, caches, _dst = _build([(0,)])
    cluster.rank = 2
    result = gio.chain_load(2, chain)
    assert type(result) is gb.Err, result
    assert result.value.check_name == "cache_wait"
    assert not caches[0].writes
    assert sum(cluster.storage_in.values()) == 0


@pytest.mark.parametrize("layers", [[(0, 1, 2, 3)], [(0,)], [(0,), (0, 1, 2, 3)]])
def test_serial_calls_preserve_completed_fills(layers):
    chain, cluster, caches, _dst = _build(layers)
    for rank in range(8):
        cluster.rank = rank
        assert type(gio.chain_load(rank, chain)) is gb.Ok
    pairs = _stat_vs_nic(chain, cluster)
    assert all(stat == observed for stat, observed in pairs), pairs
    assert all(actor == target for c in caches for actor, target, _o, _s in c.writes)


def _shared_cache_worker(rank, buffers, lock, queue):
    chain, cluster, caches, dst = _build([(0,)], size=8, world=4, steps=2)
    data, valid = buffers
    caches[0].mem[(0, "w")] = (
        np.frombuffer(data, np.uint8),
        np.frombuffer(valid, np.uint8),
    )
    caches[0].lock = lock
    cluster.rank = rank
    result = gio.chain_load(rank, chain)
    want = np.full(8, 0xA5, np.uint8)
    want[rank * 2 : (rank + 1) * 2] = np.arange(8, dtype=np.uint8)[
        rank * 2 : (rank + 1) * 2
    ]
    ok = type(result) is gb.Ok and np.array_equal(dst.buffers[rank], want)
    queue.put((rank, ok, cluster.nic_bytes(), cluster.storage_in, caches[0].writes))


def _finish_process(process):
    if process.pid is None:
        return
    if process.is_alive():
        process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    assert not process.is_alive(), f"worker {process.pid} survived cleanup"


def test_one_process_per_rank_uses_local_dn_and_shared_cache():
    ctx = mp.get_context("spawn")
    queue, lock = ctx.Queue(), ctx.RLock()
    buffers = (ctx.RawArray("B", 8), ctx.RawArray("B", 8))
    processes = [
        ctx.Process(target=_shared_cache_worker, args=(r, buffers, lock, queue))
        for r in range(4)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(10)
        assert all(p.exitcode == 0 for p in processes)
    finally:
        for process in processes:
            _finish_process(process)
        queue.close()
    assert sorted((r, ok) for r, ok, *_ in results) == [(r, True) for r in range(4)]
    rx = [sum(item[2][0][n] for item in results) for n in range(2)]
    tx = [sum(item[2][1][n] for item in results) for n in range(2)]
    storage = [sum(item[3][n] for item in results) for n in range(2)]
    assert rx == [0, 4] and tx == [4, 0] and storage == [8, 0]
    assert all(
        actor == target == 0 for item in results for actor, target, _o, _s in item[4]
    )


def _death_worker(rank, buffers, lock, entered, release, ready, pipe):
    try:
        _death_load(rank, buffers, lock, entered, release, ready, pipe)
    finally:
        pipe.close()


def _death_load(rank, buffers, lock, entered, release, ready, pipe):
    chain, cluster, caches, dst = _build([(0,)], size=8, world=2, k=1)
    caches[0].mem[(0, "w")] = tuple(np.frombuffer(b, np.uint8) for b in buffers)
    caches[0].lock = lock
    src = cast(gb.StorageDTensorsView, chain.src)
    read = src.data_accessor.read_to_buffer

    def gated_read(rng, callback):
        entered.set()
        assert release.wait(15), "parent did not release source read"
        return read(rng, callback)

    src.data_accessor.read_to_buffer = gated_read
    cluster.rank = rank
    ready.set()
    result = gio.chain_load(rank, chain)
    want = np.full(8, 0xA5, np.uint8)
    want[rank * 4 : (rank + 1) * 4] = np.arange(8, dtype=np.uint8)[
        rank * 4 : (rank + 1) * 4
    ]
    pipe.send(
        dict(
            rank=rank,
            ok=type(result) is gb.Ok,
            correct=np.array_equal(dst.buffers[rank], want),
            error=repr(result.value) if type(result) is gb.Err else None,
            check_name=getattr(result.value, "check_name", None)
            if type(result) is gb.Err
            else None,
        )
    )


def _death_pair(ctx):
    buffers = (ctx.RawArray("B", 8), ctx.RawArray("B", 8))
    lock, entered, release = ctx.RLock(), ctx.Event(), ctx.Event()
    workers, receivers, senders, ready = [], [], [], []
    for rank in range(2):
        recv, send = ctx.Pipe(duplex=False)
        event = ctx.Event()
        args = (rank, buffers, lock, entered, release, event, send)
        worker = ctx.Process(target=_death_worker, args=args)
        workers.append(worker)
        receivers.append(recv)
        senders.append(send)
        ready.append(event)
    return workers, receivers, senders, ready, entered, release, (buffers, lock)


def _confirm_survivor_waiting(worker, receiver, grace=1.0):
    # The survivor reached chain_load but has produced no result, so it is
    # blocked on the in-flight fill. A periodic faulthandler dump is not usable
    # here: repeated dumps from the watchdog thread kill the spawned child
    # (exitcode -SIGSEGV) on some runs.
    end = time.monotonic() + grace
    while time.monotonic() < end:
        assert worker.is_alive(), (
            f"survivor exited before fault injection, exitcode={worker.exitcode}"
        )
        assert not receiver.poll(0), "survivor finished before fault injection"
        time.sleep(0.05)


def _death_observe(workers, receivers, deadline):
    rows = []
    for rank, worker in enumerate(workers):
        worker.join(max(0, deadline - time.monotonic()))
        result = receivers[rank].recv() if receivers[rank].poll(0) else None
        rows.append(
            dict(
                rank=rank,
                alive=worker.is_alive(),
                exitcode=worker.exitcode,
                result=result,
            )
        )
    return rows


def _death_round(tmp_path, kill, restart=False, wait="6"):
    # The wait budget must outlive the blocked-survivor probe: a short budget
    # would let the survivor return before the host confirms it is blocked.
    os.environ["GDTENSOR_CACHE_WAIT_SECONDS"] = wait
    ctx = mp.get_context("spawn")
    # Keep parent references until spawn has unpickled every semaphore and buffer.
    workers, receivers, senders, ready, entered, release, _shared = _death_pair(ctx)
    if restart:
        release.set()
    try:
        workers[0].start()
        stage = ready[0] if restart else entered
        assert stage.wait(10), "filler never reached the required stage"
        workers[1].start()
        assert ready[1].wait(10), "survivor never reached chain_load"
        if not restart:
            _confirm_survivor_waiting(workers[1], receivers[1])
        deadline = time.monotonic() + 4
        if kill:
            workers[0].kill()
            workers[0].join(5)
            assert workers[0].exitcode == -9, workers[0].exitcode
        else:
            release.set()
        return _death_observe(workers, receivers, deadline)
    finally:
        for worker in workers:
            _finish_process(worker)
        for pipe in receivers + senders:
            pipe.close()


@pytest.mark.parametrize("kill", [False, True], ids=["release", "kill-filler"])
def test_filler_process_exit_is_bounded(tmp_path, monkeypatch, kill):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "file")
    rows = _death_round(tmp_path, kill)
    survivor = rows[1]
    assert not survivor["alive"], ("survivor did not exit within 4s", rows)
    assert survivor["exitcode"] == 0 and survivor["result"] is not None, rows
    if kill:
        assert not survivor["result"]["ok"], rows
        assert survivor["result"]["check_name"] == "cache_peer_down", rows
    else:
        assert all(not row["alive"] and row["exitcode"] == 0 for row in rows), rows
        assert all(row["result"]["ok"] and row["result"]["correct"] for row in rows), (
            rows
        )


@pytest.mark.parametrize(
    "reuse_state", [True, False], ids=["same-directory", "fresh-directory"]
)
def test_restart_after_filler_death_is_bounded(tmp_path, monkeypatch, reuse_state):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "file")
    _death_round(tmp_path, kill=True)
    # Both old processes are reaped before any change of coordinator directory.
    if not reuse_state:
        tmp_path = tmp_path / "fresh"
        tmp_path.mkdir()
        monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    rows = _death_round(tmp_path, kill=False, restart=True, wait="0.2")
    assert not any(row["alive"] for row in rows), ("restart reused stuck state", rows)
    assert all(row["exitcode"] == 0 and row["result"] is not None for row in rows), rows
    assert all(not row["result"]["ok"] or row["result"]["correct"] for row in rows), (
        rows
    )
    if not reuse_state:
        assert all(row["result"]["ok"] and row["result"]["correct"] for row in rows), (
            rows
        )
    else:
        assert all(row["result"]["check_name"] == "cache_peer_down" for row in rows), (
            rows
        )

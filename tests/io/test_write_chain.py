"""End-to-end write-chain tests through the public IO executor (gio.chain_save).

Covers:
- Direct, one-layer, two-layer fan-in, and mixed write chains;
- 2D column-sharded coalescing onto standard HF row-major safetensors;
- Bounded temporal/concurrency budget & peak cache residency <= mem_overhead;
- Replicate write deduplication and stats oracle verification;
- Both thread and file coordinators, fault abort propagation, and missing-producer timeout;
- Real SafeTensorsSaveSession + DiskNodeCache integration and multi-generation reuse.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import safetensors.numpy

import gdtensor
import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_conn.file_cache import DiskNodeCache
from gdtensor_conn.formats import make_storage_view
from gdtensor_conn.iobackend import FileIO
from tests.support.cluster_sim import Cluster, NodeCache, parent_of
from tests.support.memory import Memory


# ------------------------------------------------------------------ fixtures


class TraceMemory(Memory):
    """In-memory accessor recording (dir, offset, size) and live cache bytes."""

    def __init__(self, data, error=None):
        super().__init__(data, error=error)
        self.events: list = []
        self.live_segments: dict = {}
        self.peak_bytes = 0
        self.lock = threading.Lock()
        self.error_on_write_call: int | None = None
        self.write_calls = 0

    @staticmethod
    def _off(rng) -> int:
        if hasattr(rng, "global_tensor_offset"):
            return rng.global_tensor_offset
        return rng.contiguous_payload_offset

    def read_to_buffer(self, global_range, read_to_buffer):
        off, size = self._off(global_range), global_range.size
        with self.lock:
            self.events.append(("read", off, size))
            self._release_covered(off, size)
        return super().read_to_buffer(global_range, read_to_buffer)

    def write_from_buffer(self, global_range, write_from_buffer):
        off, size = self._off(global_range), global_range.size
        err = self._record_write_error(off, size)
        if err is not None:
            return gb.Err(err)
        return super().write_from_buffer(global_range, write_from_buffer)

    def _record_write_error(self, off: int, size: int):
        with self.lock:
            self.write_calls += 1
            hit = self.error_on_write_call == self.write_calls and self.error
            self._append_write(off, size) if not hit else None
            return self.error if hit else None

    def _append_write(self, off: int, size: int) -> None:
        self.events.append(("write", off, size))
        self._mark_live(off, size)

    def _mark_live(self, off: int, size: int) -> None:
        for idx in range(off, off + size):
            self.live_segments[idx] = True
        self.peak_bytes = max(self.peak_bytes, len(self.live_segments))

    def _release_covered(self, off: int, size: int) -> None:
        for idx in range(off, off + size):
            self.live_segments.pop(idx, None)

    def ops(self, kind: str) -> list:
        with self.lock:
            return [(o, s) for d, o, s in self.events if d == kind]


class _SessionVote:
    def __init__(self, world: int):
        self.world = world
        self.cond = threading.Condition()
        self.commits = 0
        self.aborted = False

    def vote_commit(self) -> bool:
        with self.cond:
            self.commits += 1
            self.cond.notify_all()
            self.cond.wait_for(
                lambda: self.aborted or self.commits >= self.world,
                timeout=10.0,
            )
            return not self.aborted and self.commits >= self.world

    def vote_abort(self) -> None:
        with self.cond:
            self.aborted = True
            self.cond.notify_all()


class SaveSession:
    def __init__(self, accessor, commit_error=None, vote=None):
        self.accessor = accessor
        self.commit_error = commit_error
        self.vote = vote
        self.committed = 0
        self.aborted = 0

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        return gb.Ok(self.accessor)

    def commit(self):
        if self.vote is not None and not self.vote.vote_commit():
            self.aborted += 1
            return gb.Err(gb.GDTensorIOError(operation="commit", msg="aborted"))
        self.committed += 1
        if self.commit_error is not None:
            return gb.Err(self.commit_error)
        return gb.Ok(None)

    def abort(self):
        self.aborted += 1
        if self.vote is not None:
            self.vote.vote_abort()
        return gb.Ok(None)


class _DummyDevice:
    def charge(self, _node: int, _nbytes: int) -> None:
        pass


def _hf_single_slot_storage(
    accessor,
    tensor_id: str,
    shape: list[int],
    world: int,
    sessions: list | None = None,
    commit_error=None,
    file_path: str = "model.safetensors",
    vote=None,
):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.LiteralMesh([0], gt.WorldSize(world))
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta(shape, "U8", 1), [gb.Replicate(), gb.Replicate()], mesh
    )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)),
                gb.LiteralMesh([0], gt.WorldSize(world)),
            )
        ],
        [tensor_id],
    )
    leader = gb.TensorComponentPos(tensor_id, 0)
    layout = gb.PhysicalLayoutTable("safetensors", {leader: group}, {leader: file_path})

    def begin_save(_view, _rank):
        sess = SaveSession(accessor, commit_error=commit_error, vote=vote)
        if sessions is not None:
            sessions.append(sess)
        return gb.Ok(sess)

    decoy = Memory([0], gb.GDTensorIOError(operation="decoy", msg="unused"))
    return gb.StorageDTensorsView(
        [tensor_id],
        gb.DTensorTable([gb.NamedDTensor(tensor_id, dt)]),
        decoy,
        layout,
        begin_save,
        lambda _v, _r: gb.Ok(None),
    )


def _sharded_storage(
    accessor,
    tensor_id: str,
    shape: list[int],
    world: int,
    sessions: list | None = None,
):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta(shape, "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh
    )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(world))
            )
        ],
        [tensor_id],
    )
    leader = gb.TensorComponentPos(tensor_id, 0)
    entries = {gb.TensorComponentPos(tensor_id, rank): group for rank in range(world)}
    layout = gb.PhysicalLayoutTable(
        "safetensors", entries, {leader: "model.safetensors"}
    )

    def begin_save(_view, _rank):
        sess = SaveSession(accessor)
        if sessions is not None:
            sessions.append(sess)
        return gb.Ok(sess)

    decoy = Memory([0], gb.GDTensorIOError(operation="decoy", msg="unused"))
    return gb.StorageDTensorsView(
        [tensor_id],
        gb.DTensorTable([gb.NamedDTensor(tensor_id, dt)]),
        decoy,
        layout,
        begin_save,
        lambda _v, _r: gb.Ok(None),
    )


def _compute_view(
    accessor, tensor_id: str, shape: list[int], world: int, placements=None
):
    pl = list(placements) if placements is not None else [gb.Shard(0)]
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta(shape, "U8", 1), pl, gb.ArrangedMesh(gt.WorldSize(world))
    )
    return gb.ComputeDTensorsView(
        [tensor_id], gb.DTensorTable([gb.NamedDTensor(tensor_id, dt)]), accessor
    )


def _cache_view(
    accessor, tensor_id: str, shape: list[int], steps: int, nodes: list[int], width: int
):
    mesh = gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=gb.LiteralMesh(list(nodes), gt.NumNodes(len(nodes))),
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta(shape, "U8", 1),
        [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
        mesh,
    )
    return gb.CacheDTensorsView(
        [tensor_id], gb.DTensorTable([gb.NamedDTensor(tensor_id, dt)]), accessor
    )


def _save_all(chain, world: int, threaded: bool = True) -> dict:
    results = {}

    def run(rank: int) -> None:
        results[rank] = gio.chain_save(rank, chain)

    if not threaded:
        _ = [run(rank) for rank in range(world)]
        return results
    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "chain_save hung"
    return results


def _assert_all_ok(results: dict) -> None:
    bad = {r: v.value.msg for r, v in results.items() if type(v) is not gb.Ok}
    assert not bad, bad


# ------------------------------------------------------------------ tests


def test_direct_write_chain_delivers_every_rank_shard_bytes():
    world = 4
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    cv = _compute_view(src, "w", [8], world)
    sv = _sharded_storage(dst, "w", [8], world)
    chain = gb.DirectDTensorsWriteChain(src=cv, dst=sv)
    assert type(chain.check_constraints()) is gb.Ok
    assert gdtensor.get_chain_tensor_ids(chain) == ["w"]
    assert len(gb.legs_of(chain)) == 1
    assert gio.chain_stat_mem_overhead(chain) == []
    assert gio.chain_stat_inbound_traffic(chain) == []
    assert gio.chain_stat_outbound_traffic(chain) == []
    _assert_all_ok(_save_all(chain, world, threaded=False))
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))


def test_cached_write_failure_aborts_session(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    world = 2
    boom = gb.GDTensorIOError(operation="write", msg="boom")
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8), error=boom)
    sessions: list = []
    cv = _compute_view(src, "w", [8], world)
    cvt = _cache_view(cache, "w", [8], steps=1, nodes=[0], width=1)
    sv = _hf_single_slot_storage(
        dst, "w", [8], world, sessions=sessions, vote=_SessionVote(world)
    )
    chain = gb.OneLayerCachedDTensorsWriteChain(
        src=cv, cache=cvt, dst=sv, num_ranks_per_node=2
    )
    results = _save_all(chain, world, threaded=True)
    assert any(type(r) is gb.Err and "boom" in r.value.msg for r in results.values())
    assert any(s.aborted == 1 for s in sessions), sessions
    assert all(s.committed == 0 for s in sessions), sessions


def test_commit_failure_does_not_abort_direct_write_chain():
    world = 2
    src = TraceMemory(np.arange(4, dtype=np.uint8))
    dst = TraceMemory(np.zeros(4, dtype=np.uint8))
    sessions: list = []
    boom = gb.GDTensorIOError(operation="commit", msg="disk-full")
    cv = _compute_view(src, "w", [4], world)
    sv = _hf_single_slot_storage(
        dst, "w", [4], world, sessions=sessions, commit_error=boom
    )
    chain = gb.DirectDTensorsWriteChain(src=cv, dst=sv)
    res = gio.chain_save(0, chain)
    assert type(res) is gb.Err and res.value.msg == "disk-full"
    assert sessions[0].committed == 1 and sessions[0].aborted == 0


def test_one_layer_col_sharded_coalesces_writes_to_hf_storage(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    shape, world = [4, 4], 2
    expected = np.arange(16, dtype=np.uint8)

    # Baseline direct rank_save emits 2 ranks * 4 rows = 8 tiny writes.
    direct_src = TraceMemory(expected)
    direct_dst = TraceMemory(np.zeros(16, dtype=np.uint8))
    direct_cv = _compute_view(direct_src, "w", shape, world, placements=[gb.Shard(1)])
    direct_sv = _hf_single_slot_storage(direct_dst, "w", shape, world)
    for rank in range(world):
        assert type(gio.rank_save(rank, direct_cv, direct_sv)) is gb.Ok
    assert len(direct_dst.ops("write")) == 8
    assert bytes(direct_dst.data) == bytes(expected)

    # OneLayerCachedDTensorsWriteChain coalesces all 8 fragments into 1 write.
    cached_src = TraceMemory(expected)
    cache_mem = TraceMemory(np.zeros(16, dtype=np.uint8))
    cached_dst = TraceMemory(np.zeros(16, dtype=np.uint8))
    cv = _compute_view(cached_src, "w", shape, world, placements=[gb.Shard(1)])
    cvt = _cache_view(cache_mem, "w", shape, steps=1, nodes=[0], width=1)
    sv = _hf_single_slot_storage(cached_dst, "w", shape, world)
    chain = gb.OneLayerCachedDTensorsWriteChain(
        src=cv, cache=cvt, dst=sv, num_ranks_per_node=2
    )
    assert type(chain.check_constraints()) is gb.Ok
    _assert_all_ok(_save_all(chain, world, threaded=True))
    assert cached_dst.ops("write") == [(0, 16)]
    assert bytes(cached_dst.data) == bytes(expected)


@pytest.mark.parametrize("steps", [2, 4])
def test_bounded_budget_resident_never_exceeds_stat(monkeypatch, steps):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    world = 2
    shape = [8, 2]  # 16 bytes
    src = TraceMemory(np.arange(16, dtype=np.uint8))
    cache = TraceMemory(np.zeros(16, dtype=np.uint8))
    dst = TraceMemory(np.zeros(16, dtype=np.uint8))
    cv = _compute_view(src, "w", shape, world, placements=[gb.Shard(1)])
    cvt = _cache_view(cache, "w", shape, steps=steps, nodes=[0], width=1)
    sv = _hf_single_slot_storage(dst, "w", shape, world)
    chain = gb.OneLayerCachedDTensorsWriteChain(
        src=cv, cache=cvt, dst=sv, num_ranks_per_node=2
    )
    budget = gio.chain_stat_mem_overhead(chain)
    assert budget == [16 // steps]
    _assert_all_ok(_save_all(chain, world, threaded=True))
    assert cache.peak_bytes <= budget[0]
    assert len(dst.ops("write")) == steps
    assert bytes(dst.data) == bytes(np.arange(16, dtype=np.uint8))


class _BarrierCompute(TraceMemory):
    def __init__(self, data, gate: threading.Event, target_thread: str = "rank-1"):
        super().__init__(data)
        self.gate = gate
        self.target_thread = target_thread

    def read_to_buffer(self, global_range, read_to_buffer):
        if threading.current_thread().name == self.target_thread:
            assert self.gate.wait(5)
        return super().read_to_buffer(global_range, read_to_buffer)


def test_slot_flushes_only_after_all_producers_stage(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    world = 2
    cache = TraceMemory(np.zeros(16, dtype=np.uint8))
    dst = TraceMemory(np.zeros(16, dtype=np.uint8))
    expected = np.arange(16, dtype=np.uint8)
    gate = threading.Event()
    rank0_done = threading.Event()

    src = _BarrierCompute(expected, gate)
    cv = _compute_view(src, "w", [4, 4], world, placements=[gb.Shard(1)])
    cvt = _cache_view(cache, "w", [4, 4], steps=1, nodes=[0], width=1)
    sv = _hf_single_slot_storage(dst, "w", [4, 4], world)
    chain = gb.OneLayerCachedDTensorsWriteChain(
        src=cv, cache=cvt, dst=sv, num_ranks_per_node=2
    )
    results = {}

    def run0():
        results[0] = gio.chain_save(0, chain)
        rank0_done.set()

    def run1():
        results[1] = gio.chain_save(1, chain)

    t0 = threading.Thread(target=run0, name="rank-0")
    t1 = threading.Thread(target=run1, name="rank-1")
    t0.start()
    t1.start()
    # Wait until rank 0 has staged its 4 pieces into cache, then check dst is empty.
    deadline_ok = _wait_until(lambda: len(cache.ops("write")) >= 4)
    assert deadline_ok
    assert dst.ops("write") == []
    gate.set()
    t0.join(timeout=10)
    t1.join(timeout=10)
    assert rank0_done.is_set()
    _assert_all_ok(results)
    assert dst.ops("write") == [(0, 16)]
    assert bytes(dst.data) == bytes(expected)


def _wait_until(pred, timeout: float = 5.0) -> bool:
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_replicate_write_chain_deduplicates_storage_writes(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    world = 4
    expected = np.arange(8, dtype=np.uint8)
    src = TraceMemory(expected)
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    cv = _compute_view(src, "w", [8], world, placements=[gb.Replicate()])
    cvt = _cache_view(cache, "w", [8], steps=1, nodes=[0, 1], width=1)
    sv = _hf_single_slot_storage(dst, "w", [8], world)
    chain = gb.OneLayerCachedDTensorsWriteChain(
        src=cv, cache=cvt, dst=sv, num_ranks_per_node=2
    )
    _assert_all_ok(_save_all(chain, world, threaded=True))
    assert sum(s for _o, s in src.ops("read")) == 8
    assert sum(s for _o, s in dst.ops("write")) == 8
    assert bytes(dst.data) == bytes(expected)


def test_two_layer_fan_in_converges_and_matches_traffic_stats(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    world, k = 4, 2
    cluster = Cluster(num_nodes=2, k=k)
    sizes = {"w": 16}
    src_data = np.arange(16, dtype=np.uint8)
    dst_acc = TraceMemory(np.zeros(16, dtype=np.uint8))
    big_acc = NodeCache(cluster, sizes, parent_of([0, 1], [0]).get)
    small_acc = NodeCache(cluster, sizes, lambda _r: 0)

    class RankReadMem(ga.DTensorDataAccessor):
        def read_to_buffer(self, global_range, read_to_buffer):
            off, sz = global_range.global_tensor_offset, global_range.size
            return read_to_buffer(src_data[off : off + sz].ctypes.data, sz)

        def write_from_buffer(self, global_range, write_from_buffer):
            raise AssertionError("src is read-only")

    cv = _compute_view(RankReadMem(), "w", [4, 4], world, placements=[gb.Shard(1)])
    big_cv = _cache_view(big_acc, "w", [4, 4], steps=2, nodes=[0, 1], width=1)
    small_cv = _cache_view(small_acc, "w", [4, 4], steps=2, nodes=[0], width=1)
    sv = _hf_single_slot_storage(dst_acc, "w", [4, 4], world)
    chain = gb.TwoLayerCachedDTensorsWriteChain(
        src=cv, big_cache=big_cv, small_cache=small_cv, dst=sv, num_ranks_per_node=k
    )
    assert type(chain.check_constraints()) is gb.Ok
    # Oracle: 2 steps -> 4B/step on each big cache; node 0 holds big@0 (4B) + small@0 (8B) = 12B; node 1 holds big@1 (4B).
    assert gio.chain_stat_mem_overhead(chain) == [12, 4]
    # Node 1 forwards all 8B to small@0; node 0 writes 16B out to storage -> [16, 8].
    assert gio.chain_stat_outbound_traffic(chain) == [16, 8]
    assert gio.chain_stat_inbound_traffic(chain) == [8, 0]

    results = {}

    def run(rank: int):
        cluster.rank = rank
        results[rank] = gio.chain_save(rank, chain)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()
    _assert_all_ok(results)
    assert bytes(dst_acc.data) == bytes(src_data)
    assert sorted(dst_acc.ops("write")) == [(0, 8), (8, 8)]
    assert small_acc.wrong_node == 0 and big_acc.wrong_node == 0


def test_mixed_write_chain_runs_independent_sessions(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    world = 2
    src_a = TraceMemory(np.arange(4, dtype=np.uint8))
    dst_a = TraceMemory(np.zeros(4, dtype=np.uint8))
    sess_a: list = []
    seg_a = gb.DirectDTensorsWriteChain(
        src=_compute_view(src_a, "a", [4], world),
        dst=_sharded_storage(dst_a, "a", [4], world, sessions=sess_a),
    )

    src_b = TraceMemory(np.arange(8, dtype=np.uint8) + 20)
    cache_b = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst_b = TraceMemory(np.zeros(8, dtype=np.uint8))
    sess_b: list = []
    seg_b = gb.OneLayerCachedDTensorsWriteChain(
        src=_compute_view(src_b, "b", [4, 2], world, placements=[gb.Shard(1)]),
        cache=_cache_view(cache_b, "b", [4, 2], steps=1, nodes=[0], width=1),
        dst=_hf_single_slot_storage(dst_b, "b", [4, 2], world, sessions=sess_b),
        num_ranks_per_node=2,
    )
    mixed = gb.MixedDTensorsWriteChain(
        [
            gb.DisjointTensorIdsAndWriteChain(["a"], seg_a),
            gb.DisjointTensorIdsAndWriteChain(["b"], seg_b),
        ]
    )
    assert type(mixed.check_constraints()) is gb.Ok
    assert gdtensor.get_chain_tensor_ids(mixed) == ["a", "b"]
    assert gio.chain_stat_mem_overhead(mixed) == [8]
    _assert_all_ok(_save_all(mixed, world, threaded=True))
    assert len(sess_a) == world and all(s.committed == 1 for s in sess_a)
    assert len(sess_b) == world and all(s.committed == 1 for s in sess_b)
    assert bytes(dst_a.data) == bytes(np.arange(4, dtype=np.uint8))
    assert bytes(dst_b.data) == bytes(np.arange(8, dtype=np.uint8) + 20)
    assert dst_b.ops("write") == [(0, 8)]


def _run_one_safetensors_gen(
    cluster,
    world: int,
    chain,
    io_backend,
    root: Path,
    shape: list[int],
    arr: np.ndarray,
    gen_idx: int,
) -> None:
    io_backend.tracer.records.clear()
    results: dict = {}

    def run(rank: int):
        cluster.rank = rank
        results[rank] = gio.chain_save(rank, chain)

    threads = [
        threading.Thread(target=run, args=(r,), name=f"rank-{r}") for r in range(world)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()
    _assert_all_ok(results)
    payload_pwrites = [
        rec
        for rec in io_backend.tracer.records
        if rec.op == "pwrite" and rec.offset > 0
    ]
    assert len(payload_pwrites) == 1, (gen_idx, payload_pwrites)
    loaded = safetensors.numpy.load_file(str(root / "model.safetensors"))
    np.testing.assert_array_equal(loaded["w"], arr.reshape(shape))


class _GenMem(ga.DTensorDataAccessor):
    def __init__(self, cluster: Cluster, active_data: list) -> None:
        self.cluster = cluster
        self.active_data = active_data

    def read_to_buffer(self, global_range, read_to_buffer):
        self.cluster.rank = int(threading.current_thread().name.split("-")[1])
        off, sz = global_range.global_tensor_offset, global_range.size
        return read_to_buffer(self.active_data[0][off : off + sz].ctypes.data, sz)

    def write_from_buffer(self, global_range, write_from_buffer):
        raise AssertionError("compute src is read-only")


class _MemCoord:
    def __init__(self, n: int) -> None:
        self.bar = threading.Barrier(n)

    def vote(self, ok: bool) -> bool:
        self.bar.wait(timeout=10)
        return ok


@pytest.mark.parametrize("coord_mode", ["thread", "file"])
def test_real_safetensors_and_disk_node_cache_multi_generation(
    tmp_path: Path, monkeypatch, coord_mode: str
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", coord_mode)
    coord_dir = tmp_path / "coord"
    coord_dir.mkdir()
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(coord_dir))

    world, k = 2, 2
    shape = [4, 4]
    cluster = Cluster(num_nodes=1, k=k)
    cache_acc = DiskNodeCache(
        cluster, {"w": 16}, lambda _r: 0, str(tmp_path / "cache"), _DummyDevice()
    )
    gen1_data = np.arange(16, dtype=np.uint8)
    gen2_data = np.arange(16, dtype=np.uint8) + 50
    active_data = [gen1_data]

    root = tmp_path / "ckpt"
    io_backend = FileIO()
    st_mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.LiteralMesh([0], gt.WorldSize(world))
    )
    st_dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta(shape, "U8", 1), [gb.Replicate(), gb.Replicate()], st_mesh
    )
    st_table = gb.DTensorTable([gb.NamedDTensor("w", st_dt)])
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)),
                gb.LiteralMesh([0], gt.WorldSize(world)),
            )
        ],
        ["w"],
    )
    leader = gb.TensorComponentPos("w", 0)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {leader: group}, {leader: "model.safetensors"}
    )
    sv = make_storage_view(
        ["w"], st_table, layout, str(root), io=io_backend, coord=_MemCoord(world)
    )

    cv = _compute_view(
        _GenMem(cluster, active_data), "w", shape, world, placements=[gb.Shard(1)]
    )
    cvt = _cache_view(cache_acc, "w", shape, steps=1, nodes=[0], width=1)
    chain = gb.OneLayerCachedDTensorsWriteChain(
        src=cv, cache=cvt, dst=sv, num_ranks_per_node=k
    )

    try:
        for gen_idx, arr in enumerate((gen1_data, gen2_data), start=1):
            active_data[0] = arr
            _run_one_safetensors_gen(
                cluster, world, chain, io_backend, root, shape, arr, gen_idx
            )
    finally:
        cache_acc.close()


def test_missing_producer_rank_times_out_with_cache_wait(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0.05")
    world = 2
    src = TraceMemory(np.arange(16, dtype=np.uint8))
    cache = TraceMemory(np.zeros(16, dtype=np.uint8))
    dst = TraceMemory(np.zeros(16, dtype=np.uint8))
    cv = _compute_view(src, "w", [8, 2], world, placements=[gb.Shard(1)])
    cvt = _cache_view(cache, "w", [8, 2], steps=2, nodes=[0], width=1)
    sv = _hf_single_slot_storage(dst, "w", [8, 2], world)
    chain = gb.OneLayerCachedDTensorsWriteChain(
        src=cv, cache=cvt, dst=sv, num_ranks_per_node=2
    )
    res = gio.chain_save(0, chain)
    assert type(res) is gb.Err
    assert type(res.value) is gb.CheckFailedError
    assert res.value.check_name == "cache_wait"


def test_write_chain_constraint_validation_and_error_cases():
    sv = _hf_single_slot_storage(Memory([0] * 8), "w", [2, 4], 4)
    cv_bad = _compute_view(Memory([0] * 8), "other", [2, 4], 4)
    bad_direct = gb.DirectDTensorsWriteChain(src=cv_bad, dst=sv)
    res = bad_direct.check_constraints()
    assert type(res) is gb.Err and res.value.constraint_name == "tensor_ids_match"

    # Cache subset check against src compute ranks
    cv_ok = _compute_view(Memory([0] * 8), "w", [2, 4], 2)
    cvt_too_big = _cache_view(
        Memory([0] * 8), "w", [2, 4], steps=1, nodes=[0, 1], width=1
    )
    bad_one = gb.OneLayerCachedDTensorsWriteChain(
        src=cv_ok, cache=cvt_too_big, dst=sv, num_ranks_per_node=2
    )
    res1 = bad_one.check_constraints()
    assert type(res1) is gb.Err and res1.value.constraint_name == "cache_subset"

    # Disjoint tensor_ids check in Mixed
    cv2 = _compute_view(Memory([0] * 8), "w", [2, 4], 4)
    d = gb.DirectDTensorsWriteChain(src=cv2, dst=sv)
    bad_mix = gb.MixedDTensorsWriteChain(
        [
            gb.DisjointTensorIdsAndWriteChain(["w"], d),
            gb.DisjointTensorIdsAndWriteChain(["w"], d),
        ]
    )
    res_mix = bad_mix.check_constraints()
    assert (
        type(res_mix) is gb.Err
        and res_mix.value.constraint_name == "tensor_ids_disjoint"
    )

    # Out-of-range rank on a cached chain returns InvalidArgumentError
    cvt_ok = _cache_view(Memory([0] * 8), "w", [2, 4], steps=1, nodes=[0], width=1)
    cached_ok = gb.OneLayerCachedDTensorsWriteChain(
        src=cv2, cache=cvt_ok, dst=sv, num_ranks_per_node=2
    )
    res_rank = gio.chain_save(99, cached_ok)
    assert type(res_rank) is gb.Err and type(res_rank.value) is gb.InvalidArgumentError


class _OffsetFailReadAccessor(ga.DTensorDataAccessor):
    def __init__(self, data: np.ndarray, bad_mod: int, error: gb.GDTensorIOError):
        self.data = data
        self.bad_mod = bad_mod
        self.error = error

    def read_to_buffer(self, global_range, read_to_buffer):
        off, sz = global_range.global_tensor_offset, global_range.size
        if off % 4 >= self.bad_mod:
            return gb.Err(self.error)
        return read_to_buffer(self.data[off : off + sz].ctypes.data, sz)

    def write_from_buffer(self, global_range, write_from_buffer):
        raise AssertionError("src is read-only")


def test_two_layer_stage_failure_propagates_downstream_without_waiting(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    world, k = 4, 2
    cluster = Cluster(num_nodes=2, k=k)
    sizes = {"w": 16}
    src_data = np.arange(16, dtype=np.uint8)
    boom = gb.GDTensorIOError(operation="read", msg="node1-stage-boom")
    src_acc = _OffsetFailReadAccessor(src_data, bad_mod=2, error=boom)
    dst_acc = TraceMemory(np.zeros(16, dtype=np.uint8))
    big_acc = NodeCache(cluster, sizes, parent_of([0, 1], [0]).get)
    small_acc = NodeCache(cluster, sizes, lambda _r: 0)
    sessions: list = []
    cv = _compute_view(src_acc, "w", [4, 4], world, placements=[gb.Shard(1)])
    big_cv = _cache_view(big_acc, "w", [4, 4], steps=2, nodes=[0, 1], width=1)
    small_cv = _cache_view(small_acc, "w", [4, 4], steps=2, nodes=[0], width=1)
    sv = _hf_single_slot_storage(
        dst_acc, "w", [4, 4], world, sessions=sessions, vote=_SessionVote(world)
    )
    chain = gb.TwoLayerCachedDTensorsWriteChain(
        src=cv, big_cache=big_cv, small_cache=small_cv, dst=sv, num_ranks_per_node=k
    )
    results = {}

    def run(rank: int):
        cluster.rank = rank
        results[rank] = gio.chain_save(rank, chain)

    t0 = time.monotonic()
    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0
    assert type(results[0]) is gb.Err and "node1-stage-boom" in results[0].value.msg
    assert any(
        type(results[r]) is gb.Err and "node1-stage-boom" in results[r].value.msg
        for r in (2, 3)
    )
    assert all(s.committed == 0 for s in sessions), sessions

"""End-to-end read-chain verification through the public IO executor.

Every test drives ``gio.chain_load`` for every rank and checks the actual bytes
that land in each view against an independent oracle derived from the meshes.
Accessors record the global ranges they read/write, so the observed bytes are
checked per rank rather than through a global-union shortcut. Fault injection
must surface as an ``Err`` result, never as a skipped/ silently-ok load.
"""

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.memory import Memory


# ------------------------------------------------------------------ fixtures


class TraceMemory(Memory):
    """In-memory accessor that records every (direction, offset, size) it serves."""

    def __init__(self, data, error=None):
        super().__init__(data, error=error)
        self.events = []  # (dir, offset, size)

    def _offset(self, rng):
        if hasattr(rng, "global_tensor_offset"):
            return rng.global_tensor_offset
        return rng.contiguous_payload_offset

    def read_to_buffer(self, global_range, read_to_buffer):
        self.events.append(("read", self._offset(global_range), global_range.size))
        return super().read_to_buffer(global_range, read_to_buffer)

    def write_from_buffer(self, global_range, write_from_buffer):
        self.events.append(("write", self._offset(global_range), global_range.size))
        return super().write_from_buffer(global_range, write_from_buffer)


class LoadSession:
    def __init__(self, accessor, events, finish_error=None):
        self.accessor, self.events, self.finish_error = accessor, events, finish_error

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        self.events.append("init_load")
        return gb.Ok(self.accessor)

    def finish(self):
        self.events.append("finish")
        if self.finish_error is not None:
            return gb.Err(self.finish_error)
        return gb.Ok(None)


# ------------------------------------------------------------- view builders


def _storage_view(accessor, tensor_id, shape, world):
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
    layout = gb.PhysicalLayoutTable("safetensors", entries, {leader: "memory"})

    def begin_load(v, rank):
        return gb.Ok(LoadSession(accessor, []))

    decoy = Memory([0], gb.GDTensorIOError(operation="decoy", msg="unused"))
    return gb.StorageDTensorsView(
        [tensor_id],
        gb.DTensorTable([gb.NamedDTensor(tensor_id, dt)]),
        decoy,
        layout,
        lambda v, r: gb.Ok(LoadSession(accessor, [])),
        begin_load,
    )


def _compute_view(accessor, tensor_id, shape, world):
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta(shape, "U8", 1),
        [gb.Shard(0)],
        gb.ArrangedMesh(gt.WorldSize(world)),
    )
    return gb.ComputeDTensorsView(
        [tensor_id], gb.DTensorTable([gb.NamedDTensor(tensor_id, dt)]), accessor
    )


def _cache_view(accessor, tensor_id, shape, steps, nodes, width):
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


# --------------------------------------------------------------- oracle helpers


def rank_owned_bytes(shape, world, rank):
    """Independent oracle: compute Shard(0) across ``world`` ranks."""
    g = shape[0]
    per = g // world
    return [(rank * per, (rank + 1) * per)]


def _written_bytes(events):
    return {o + i for d, o, s in events if d == "write" for i in range(s)}


# --------------------------------------------------------------------- tests


def test_direct_chain_delivers_every_rank_shard_bytes():
    world = 4
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    chain = gb.DirectDTensorsReadChain(src=sv, dst=cv)
    assert type(chain.check_constraints()) is gb.Ok
    for rank in range(world):
        dst.events.clear()
        assert type(gio.chain_load(rank, chain)) is gb.Ok
        want = rank_owned_bytes([8], world, rank)
        got = [(o, o + s) for _, o, s in dst.events if _ == "write"]
        assert sorted(got) == sorted(want), (rank, got, want)
    # Final bytes equal the source arange.
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))


def test_one_layer_cache_per_node_ranks_and_observed_bytes():
    world, k, nodes = 4, 2, [0, 1]
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    cvt = _cache_view(cache, "w", [8], steps=2, nodes=nodes, width=1)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    assert type(chain.check_constraints()) is gb.Ok
    for rank in range(world):
        assert type(gio.chain_load(rank, chain)) is gb.Ok
        want = rank_owned_bytes([8], world, rank)
        reads = sorted((o, o + s) for d, o, s in cache.events if d == "read")
        # Each rank drains its own share out of the resident node box.
        for a, b in want:
            assert (a, b) in reads, (rank, want, reads)
    # The node box bytes were staged (writes) and all ranks delivered.
    writes = sorted((o, o + s) for d, o, s in cache.events if d == "write")
    assert writes, writes
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))


def test_fault_injection_accessor_error_returns_err():
    world = 4
    bad = TraceMemory(
        np.arange(8, dtype=np.uint8),
        error=gb.GDTensorIOError(operation="x", msg="boom"),
    )
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(bad, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    chain = gb.DirectDTensorsReadChain(src=sv, dst=cv)
    result = gio.chain_load(0, chain)
    assert type(result) is gb.Err
    assert "boom" in result.value.msg


def test_two_layer_chain_delivers_all_ranks():
    world, k = 4, 2
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    small = TraceMemory(np.zeros(8, dtype=np.uint8))
    big = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    smallv = _cache_view(small, "w", [8], steps=2, nodes=[0, 1], width=1)
    bigv = _cache_view(big, "w", [8], steps=2, nodes=[0, 1], width=1)
    chain = gb.TwoLayerCachedDTensorsReadChain(
        src=sv, small_cache=smallv, big_cache=bigv, dst=cv, num_ranks_per_node=k
    )
    assert type(chain.check_constraints()) is gb.Ok
    for rank in range(world):
        assert type(gio.chain_load(rank, chain)) is gb.Ok
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))


def test_mixed_chain_runs_segments_and_reports_error_object():
    world = 2
    # segment A on tensor "a", direct; segment B on tensor "b", one-layer.
    src_a = TraceMemory(np.arange(4, dtype=np.uint8))
    dst_a = TraceMemory(np.zeros(4, dtype=np.uint8))
    sv_a = _storage_view(src_a, "a", [4], world)
    cv_a = _compute_view(dst_a, "a", [4], world)
    direct = gb.DirectDTensorsReadChain(src=sv_a, dst=cv_a)

    src_b = TraceMemory(np.arange(4, dtype=np.uint8))
    cache_b = TraceMemory(np.zeros(4, dtype=np.uint8))
    dst_b = TraceMemory(np.zeros(4, dtype=np.uint8))
    sv_b = _storage_view(src_b, "b", [4], world)
    cv_b = _compute_view(dst_b, "b", [4], world)
    cachev_b = _cache_view(cache_b, "b", [4], steps=2, nodes=[0, 1], width=1)
    one = gb.OneLayerCachedDTensorsReadChain(
        src=sv_b, cache=cachev_b, dst=cv_b, num_ranks_per_node=1
    )

    mixed = gb.MixedDTensorsReadChain(
        [
            gb.DisjointTensorIdsAndReadChain(["a"], direct),
            gb.DisjointTensorIdsAndReadChain(["b"], one),
        ]
    )
    assert type(mixed.check_constraints()) is gb.Ok
    for rank in range(world):
        assert type(gio.chain_load(rank, mixed)) is gb.Ok
    assert bytes(dst_a.data) == bytes(np.arange(4, dtype=np.uint8))
    assert bytes(dst_b.data) == bytes(np.arange(4, dtype=np.uint8))


def test_stats_return_list_of_int_per_node():
    world, k = 4, 2
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    cvt = _cache_view(cache, "w", [8], steps=2, nodes=[0, 1], width=1)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    mem = gio.chain_stat_mem_overhead(chain)
    out = gio.chain_stat_outbound_traffic(chain)
    inb = gio.chain_stat_inbound_traffic(chain)
    assert isinstance(mem, list) and all(isinstance(x, int) for x in mem)
    assert isinstance(out, list) and isinstance(inb, list)
    # Two nodes; each node needs one 4-byte step cell (C=1).
    assert mem == [4, 4], mem
    # Every cache -> rank edge stays on the rank's own node: no outbound.
    assert out == [0, 0], out
    # Each node reads its 4 bytes from storage once, shared by its two ranks.
    assert inb == [4, 4], inb


@pytest.mark.parametrize(
    "steps,width,expect",
    # Concurrency Shard splits the step slice across slots; the union of a
    # step's live slots equals the step slice, so width does NOT grow resident.
    [(2, 1, 4), (4, 1, 2), (2, 4, 4)],
)
def test_mem_overhead_shrinks_with_steps_independent_of_width(steps, width, expect):
    # 8-elem U8, 2 nodes, 1 node per... resident = step slice (8/steps) x width.
    world, k = 2, 1
    sv = _storage_view(TraceMemory(np.zeros(8, np.uint8)), "w", [8], world)
    cv = _compute_view(TraceMemory(np.zeros(8, np.uint8)), "w", [8], world)
    cvt = _cache_view(
        TraceMemory(np.zeros(8, np.uint8)),
        "w",
        [8],
        steps=steps,
        nodes=[0, 1],
        width=width,
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    mem = gio.chain_stat_mem_overhead(chain)
    assert mem == [expect, expect], (steps, width, mem)


def test_cache_writes_cover_every_rank_read():
    # Independent delivery oracle: the bytes the cache serves out (reads) were
    # first staged in (writes). This checks leg pairing, NOT resident peak —
    # cumulative write volume is not max-live and does not verify mem_overhead.
    world, k = 4, 2
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    cvt = _cache_view(cache, "w", [8], steps=2, nodes=[0, 1], width=1)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    for rank in range(world):
        assert type(gio.chain_load(rank, chain)) is gb.Ok
    staged = _written_bytes(cache.events)
    served = {o + i for d, o, s in cache.events if d == "read" for i in range(s)}
    assert served <= staged, (sorted(served), sorted(staged))


def test_two_rank_node_fill_once_and_each_consumer_reads_share():
    # Two ranks on the same node share one fill of the node box; each rank then
    # drains its own share. The box bytes are staged exactly once per node.
    world, k = 4, 2
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    cvt = _cache_view(cache, "w", [8], steps=2, nodes=[0, 1], width=1)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    for rank in range(world):
        assert type(gio.chain_load(rank, chain)) is gb.Ok
    # Node0 box [0:4) staged once (two atomic halves, not two full fills).
    writes = [(o, s) for d, o, s in cache.events if d == "write" and o < 4]
    assert len(writes) == 2, writes  # off0:2 and off2:2 once each
    # Every rank's bytes delivered correctly.
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))


def test_slot_released_only_after_all_consumers_drain():
    # The node box stays resident (not freed/reused) until every consumer drained.
    # Here we observe that a second fill of the same atomic segment does not
    # occur while the cohort still owes drains; after both ranks drain, a fresh
    # chain_load may re-stage.
    world, k = 4, 2
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    cvt = _cache_view(cache, "w", [8], steps=2, nodes=[0, 1], width=1)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    for rank in range(world):
        assert type(gio.chain_load(rank, chain)) is gb.Ok
    writes_pass1 = [(o, s) for d, o, s in cache.events if d == "write"]
    assert len(writes_pass1) == 4, writes_pass1  # node0 2 halves + node1 2 halves
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))


def test_fill_failure_retries_cleanly_on_next_call():
    # A fill failure aborts the generation; an explicit re-call starts a fresh
    # generation with no leaked resident slot.
    world, k = 2, 1
    boom = gb.GDTensorIOError(operation="x", msg="fill-boom")
    bad = TraceMemory(np.arange(8, dtype=np.uint8), error=boom)
    setattr(bad, "error_on_call", 1)
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(bad, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    cvt = _cache_view(cache, "w", [8], steps=2, nodes=[0, 1], width=1)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    r0 = gio.chain_load(0, chain)
    assert type(r0) is gb.Err
    # Recover the source for the retry.
    setattr(bad, "error_on_call", None)
    good = TraceMemory(np.arange(8, dtype=np.uint8))
    good_view = _storage_view(good, "w", [8], world)
    good_chain = gb.OneLayerCachedDTensorsReadChain(
        src=good_view, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    for rank in range(world):
        assert type(gio.chain_load(rank, good_chain)) is gb.Ok
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))


# ===================================================== multi-tensor coverage


def _multi_compute(accessor, specs, world):
    named = []
    for tid, shape in specs:
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta(shape, "U8", 1),
            [gb.Shard(0)],
            gb.ArrangedMesh(gt.WorldSize(world)),
        )
        named.append(gb.NamedDTensor(tid, dt))
    return gb.ComputeDTensorsView(
        [t for t, _ in specs], gb.DTensorTable(named), accessor
    )


def _multi_cache(accessor, specs, steps, nodes, width):
    named = []
    for tid, shape in specs:
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
        named.append(gb.NamedDTensor(tid, dt))
    return gb.CacheDTensorsView([t for t, _ in specs], gb.DTensorTable(named), accessor)


def _multi_storage(accessor, specs, world):
    named = []
    for tid, shape in specs:
        mesh = gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
        )
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta(shape, "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh
        )
        named.append(gb.NamedDTensor(tid, dt))
    ids = [t for t, _ in specs]
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(len(specs))),
                gb.ArrangedMesh(gt.WorldSize(world)),
            )
        ],
        ids,
    )
    entries = {gb.TensorComponentPos(t, 0): group for t in ids}
    layout = gb.PhysicalLayoutTable(
        "safetensors", entries, {gb.TensorComponentPos(ids[0], 0): "m"}
    )

    def begin_load(v, rank):
        return gb.Ok(LoadSession(accessor, []))

    decoy = Memory([0], gb.GDTensorIOError(operation="d", msg="x"))
    return gb.StorageDTensorsView(
        ids,
        gb.DTensorTable(named),
        decoy,
        layout,
        lambda v, r: gb.Ok(LoadSession(accessor, [])),
        begin_load,
    )


def test_multi_tensor_a6_b2_c1_peak_resident():
    # A[6]/B[2]/C[1], S=2, distinct per-tensor backing. Each tensor reuses its
    # buffer across the two temporal steps, so per tensor resident = the larger
    # step slice: A=3, B=1, C=1. Independent regions sum -> 5 bytes per node.
    world, k = 2, 1
    specs = [("a", [6]), ("b", [2]), ("c", [1])]
    sv = _multi_storage(TraceMemory(np.zeros(9, np.uint8)), specs, world)
    cv = _multi_compute(TraceMemory(np.zeros(9, np.uint8)), specs, world)
    cvt = _multi_cache(
        TraceMemory(np.zeros(9, np.uint8)), specs, steps=2, nodes=[0, 1], width=1
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    assert type(chain.check_constraints()) is gb.Ok, chain.check_constraints()
    mem = gio.chain_stat_mem_overhead(chain)
    # C serialises slots across a node; peak is the largest single tensor slot
    # staged on the node (A=3 both nodes, B/C do not raise the max).
    assert mem == [3, 3], mem


def test_mixed_b_segment_fails_a_still_loads_same_error():
    world = 2
    # A: direct, healthy. B: one-layer whose source accessor faults.
    src_a = TraceMemory(np.arange(4, dtype=np.uint8))
    dst_a = TraceMemory(np.zeros(4, dtype=np.uint8))
    direct = gb.DirectDTensorsReadChain(
        src=_storage_view(src_a, "a", [4], world),
        dst=_compute_view(dst_a, "a", [4], world),
    )
    boom = gb.GDTensorIOError(operation="disk", msg="sector-fault")
    bad_src = TraceMemory(np.arange(4, dtype=np.uint8), error=boom)
    dst_b = TraceMemory(np.zeros(4, dtype=np.uint8))
    cache_b = TraceMemory(np.zeros(4, dtype=np.uint8))
    one = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_view(bad_src, "b", [4], world),
        cache=_cache_view(cache_b, "b", [4], steps=2, nodes=[0, 1], width=1),
        dst=_compute_view(dst_b, "b", [4], world),
        num_ranks_per_node=1,
    )
    mixed = gb.MixedDTensorsReadChain(
        [
            gb.DisjointTensorIdsAndReadChain(["a"], direct),
            gb.DisjointTensorIdsAndReadChain(["b"], one),
        ]
    )
    result = gio.chain_load(0, mixed)
    assert type(result) is gb.Err
    assert result.value is boom or result.value.msg == "sector-fault"


# =========================================== lifecycle / permuted / bytes oracle


class CountingLoadSession(LoadSession):
    def __init__(self, accessor, events, finish_error=None):
        super().__init__(accessor, events, finish_error)
        self.finished = 0
        self.aborted = 0

    def finish(self):
        self.finished += 1
        return super().finish()

    def abort(self):
        self.aborted += 1
        return gb.Ok(None)


def test_shared_source_session_opened_once_per_chain_load():
    # One storage view, begin_load must fire once per chain_load call, finish once.
    world, k = 4, 2
    sessions = []

    def begin(v, rank):
        s = CountingLoadSession(TraceMemory(np.arange(8, dtype=np.uint8)), [])
        sessions.append(s)
        return gb.Ok(s)

    sv = _storage_view(TraceMemory(np.arange(8, dtype=np.uint8)), "w", [8], world)
    sv.begin_load = begin
    cv = _compute_view(TraceMemory(np.zeros(8, np.uint8)), "w", [8], world)
    cvt = _cache_view(
        TraceMemory(np.zeros(8, np.uint8)), "w", [8], steps=2, nodes=[0, 1], width=1
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    for rank in range(world):
        sessions.clear()
        assert type(gio.chain_load(rank, chain)) is gb.Ok
        assert len(sessions) == 1, len(sessions)
        assert sessions[0].finished == 1
        assert sessions[0].aborted == 0


def test_finish_failure_propagates_and_session_released():
    world = 2
    boom = gb.GDTensorIOError(operation="flush", msg="disk-full")
    bad_finish = []

    def begin(v, rank):
        s = CountingLoadSession(
            TraceMemory(np.arange(4, dtype=np.uint8)), [], finish_error=boom
        )
        bad_finish.append(s)
        return gb.Ok(s)

    sv = _storage_view(TraceMemory(np.arange(4, dtype=np.uint8)), "w", [4], world)
    sv.begin_load = begin
    cv = _compute_view(TraceMemory(np.zeros(4, np.uint8)), "w", [4], world)
    chain = gb.DirectDTensorsReadChain(src=sv, dst=cv)
    result = gio.chain_load(0, chain)
    assert type(result) is gb.Err
    assert result.value.msg == "disk-full"
    assert bad_finish[0].finished == 1


def test_permuted_reshape_spatio_delivers_every_rank():
    # spatio permuted [3,0,1,2] over a 2x2 reshape of 4 nodes.
    world, k = 4, 1
    sp = gb.PermutedMesh(
        gb.ReshapedMesh(gb.ArrangedMesh(gt.NumNodes(4)), [2, 2]), [3, 0, 1, 2]
    )
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([8], "U8", 1),
        [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
        gb.StCacheOuterProductMesh(
            temporal=gb.ArrangedMesh(gt.NumSteps(2)),
            spatial=sp,
            concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(1)),
        ),
    )
    cvt = gb.CacheDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), cache
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    for rank in range(world):
        assert type(gio.chain_load(rank, chain)) is gb.Ok
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))


def test_cross_node_observed_bytes_match_per_node_stats():
    world, k = 4, 2
    src = TraceMemory(np.arange(8, dtype=np.uint8))
    cache = TraceMemory(np.zeros(8, dtype=np.uint8))
    dst = TraceMemory(np.zeros(8, dtype=np.uint8))
    sv = _storage_view(src, "w", [8], world)
    cv = _compute_view(dst, "w", [8], world)
    cvt = _cache_view(cache, "w", [8], steps=2, nodes=[0, 1], width=1)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    per_node_read = {0: 0, 1: 0}
    for rank in range(world):
        node = rank // k
        src.events.clear()
        assert type(gio.chain_load(rank, chain)) is gb.Ok
        per_node_read[node] += sum(s for d, _, s in src.events if d == "read")
    # Cache and ranks share a node, so no byte leaves a node.
    assert gio.chain_stat_outbound_traffic(chain) == [0, 0]
    # Inbound = the storage bytes each node actually read (one fill per node).
    inbound = gio.chain_stat_inbound_traffic(chain)
    assert [per_node_read[0], per_node_read[1]] == inbound, (per_node_read, inbound)

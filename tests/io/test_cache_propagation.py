"""Cache propagation regressions: fan-out topology, traffic, dedup and budget.

Every expected value comes from an independent oracle in this file (node
demand unions, cell boxes and edge rules), not from the stat functions. Only
public APIs are used.

Covered defects (all failed before the node-level propagation plan):
* a two-layer chain whose big cache covers more nodes than the small cache
  raised "1 is not in list" for ranks on the extra node;
* outbound counted cache -> rank bytes on the same node;
* inbound counted bytes per rank instead of bytes per node, so source reads
  were not deduplicated across ranks of one node;
* mem_overhead ignored the bytes a parent node forwards to child nodes;
* the source read buffer spanned a whole contiguous file run, not one segment;
* missing consumer ranks hung forever instead of failing with a clear error.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import threading
import time
import weakref

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.memory import Memory


class Probe(Memory):
    """Accessor that records (op, offset, size) and the live cache bytes."""

    def __init__(self, data, delay=0.0):
        super().__init__(data)
        self.lock = threading.Lock()
        self.events: list = []
        self.delay = delay

    @staticmethod
    def _off(rng):
        if hasattr(rng, "global_tensor_offset"):
            return rng.global_tensor_offset
        return rng.contiguous_payload_offset

    def read_to_buffer(self, global_range, read_to_buffer):
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.events.append(("read", self._off(global_range), global_range.size))
        return super().read_to_buffer(global_range, read_to_buffer)

    def write_from_buffer(self, global_range, write_from_buffer):
        with self.lock:
            self.events.append(("write", self._off(global_range), global_range.size))
        return super().write_from_buffer(global_range, write_from_buffer)

    def total(self, op):
        return sum(s for o, _off, s in self.events if o == op)


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
        Memory([0]),
        layout,
        lambda _v, _r: gb.Ok(_Load()),
        lambda _v, _r: gb.Ok(_Load()),
    )


def _compute(acc, size, world, placement=None):
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1),
        [placement or gb.Shard(0)],
        gb.ArrangedMesh(gt.WorldSize(world)),
    )
    return gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), acc
    )


def _cache(acc, size, steps, nodes, width):
    mesh = gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=gb.LiteralMesh(list(nodes), gt.NumNodes(len(nodes))),
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1),
        [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
        mesh,
    )
    return gb.CacheDTensorsView(["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), acc)


def _load_all(chain, world, threaded=False):
    results = {}

    def run(rank):
        results[rank] = gio.chain_load(rank, chain)

    if not threaded:
        for rank in range(world):
            run(rank)
        return results
    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "chain_load hung"
    return results


def _ok(results):
    bad = {r: v.value.msg for r, v in results.items() if type(v) is not gb.Ok}
    assert not bad, bad


# ---------------------------------------------------------------- fan-out


def _fanout(size=8, world=4, k=2, small=(0,), big=(0, 1), steps=1, width=1):
    src = Probe(np.arange(size, dtype=np.uint8))
    small_mem = Probe(np.zeros(size, np.uint8))
    big_mem = Probe(np.zeros(size, np.uint8))
    dst = Probe(np.zeros(size, np.uint8))
    chain = gb.TwoLayerCachedDTensorsReadChain(
        src=_storage(src, size, world),
        small_cache=_cache(small_mem, size, steps, small, width),
        big_cache=_cache(big_mem, size, steps, big, width),
        dst=_compute(dst, size, world),
        num_ranks_per_node=k,
    )
    return chain, src, small_mem, big_mem, dst


@pytest.mark.parametrize("threaded", [False, True])
def test_fanout_two_layer_chain_delivers_every_rank(threaded):
    chain, src, _small, _big, dst = _fanout()
    assert type(chain.check_constraints()) is gb.Ok
    _ok(_load_all(chain, 4, threaded))
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))
    # One upper-layer node reads storage once for both lower-layer nodes.
    assert src.total("read") == 8


def test_fanout_stats_follow_edges():
    # Oracle. Ranks 0,1 on node 0 own [0,4); ranks 2,3 on node 1 own [4,8).
    # big@0 holds [0,4), big@1 holds [4,8). small@0 is parent of both and holds
    # the union [0,8). Edges: storage->small@0 (8B, inbound node 0),
    # small@0->big@1 (4B, node 0 out, node 1 in). Other edges are local.
    chain, *_ = _fanout()
    assert gio.chain_stat_inbound_traffic(chain) == [8, 4]
    assert gio.chain_stat_outbound_traffic(chain) == [4, 0]
    # S=1, C=1: one cell per layer. Node 0 holds small (8) + big (4).
    assert gio.chain_stat_mem_overhead(chain) == [12, 4]


def test_fanout_spreads_extra_nodes_over_parents():
    # small on nodes {0,1}, big on {0,1,2,3}: nodes 2,3 read from 0,1 round-robin.
    chain, src, _s, big, dst = _fanout(
        size=16, world=8, k=2, small=(0, 1), big=(0, 1, 2, 3)
    )
    _ok(_load_all(chain, 8, threaded=True))
    assert bytes(dst.data) == bytes(np.arange(16, dtype=np.uint8))
    # Each node owns 4B. small@0 serves big@{0,2}, small@1 serves big@{1,3}.
    assert gio.chain_stat_inbound_traffic(chain) == [8, 8, 4, 4]
    assert gio.chain_stat_outbound_traffic(chain) == [4, 4, 0, 0]
    assert src.total("read") == 16


# ---------------------------------------------------------------- traffic


def _one_layer(size=8, world=4, k=2, nodes=(0, 1), steps=2, width=1, placement=None):
    src = Probe(np.arange(size, dtype=np.uint8))
    cache = Probe(np.zeros(size, np.uint8))
    dst = Probe(np.zeros(size, np.uint8))
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=_storage(src, size, world),
        cache=_cache(cache, size, steps, nodes, width),
        dst=_compute(dst, size, world, placement),
        num_ranks_per_node=k,
    )
    return chain, src, cache, dst


def test_same_node_drain_has_no_outbound_traffic():
    chain, *_ = _one_layer()
    assert gio.chain_stat_outbound_traffic(chain) == [0, 0]


def test_node_source_reads_are_deduplicated_and_match_inbound():
    chain, src, cache, dst = _one_layer()
    _ok(_load_all(chain, 4, threaded=True))
    assert bytes(dst.data) == bytes(np.arange(8, dtype=np.uint8))
    # Each node needs 4B; two ranks per node must not read them twice.
    assert src.total("read") == 8
    assert cache.total("write") == 8
    assert gio.chain_stat_inbound_traffic(chain) == [4, 4]


def test_replicate_dst_reads_storage_once_per_node():
    world, k = 4, 2
    chain, src, _c, _d = _one_layer(world=world, k=k, steps=1, placement=gb.Replicate())
    _ok(_load_all(chain, world, threaded=True))
    assert src.total("read") == 2 * 8
    assert gio.chain_stat_inbound_traffic(chain) == [8, 8]


# ---------------------------------------------------------------- memory


def _live_peak(cache: Probe) -> int:
    """Upper bound on resident bytes from accessor events: a byte is live from
    its cache write until the last read of that byte."""
    last_read = {
        b: i
        for i, (op, off, n) in enumerate(cache.events)
        if op == "read"
        for b in range(off, off + n)
    }
    live, peak = set(), 0
    for i, (op, off, n) in enumerate(cache.events):
        if op == "write":
            live.update(range(off, off + n))
        peak = max(peak, len(live))
        live = {b for b in live if last_read.get(b, -1) > i}
    return peak


@pytest.mark.parametrize(
    "steps,width", [(1, 1), (2, 1), (4, 1), (2, 2), (4, 2), (8, 2)]
)
def test_resident_cache_bytes_never_exceed_stat(steps, width):
    size, world = 16, 2
    chain, _src, cache, dst = _one_layer(size, world, 2, (0,), steps, width)
    stat = gio.chain_stat_mem_overhead(chain)
    # Oracle: one node needs the whole tensor; a cell is size/(S*C) bytes and
    # the budget holds C cells.
    assert stat == [width * (size // (steps * width))], stat
    _ok(_load_all(chain, world, threaded=True))
    assert bytes(dst.data) == bytes(np.arange(size, dtype=np.uint8))
    assert _live_peak(cache) <= stat[0]


def test_source_read_unit_is_one_segment_not_the_file_run():
    size, world = 64, 2
    chain, src, _cache_mem, _dst = _one_layer(size, world, 2, (0,), steps=8, width=1)
    _ok(_load_all(chain, world))
    sizes = {n for op, _o, n in src.events if op == "read"}
    assert max(sizes) <= gio.chain_stat_mem_overhead(chain)[0] == 8, sizes


# ---------------------------------------------------------------- liveness


def test_missing_consumer_fails_with_clear_error(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0.5")
    # Replicate dst: ranks 0 and 1 both need both cells. C=1 holds one cell.
    # Only rank 0 runs, so cell 0 stays owed to rank 1 and cell 1 never fits.
    chain, *_ = _one_layer(
        size=8, world=2, k=2, nodes=(0,), steps=2, width=1, placement=gb.Replicate()
    )
    result = gio.chain_load(0, chain)
    assert type(result) is gb.Err, result
    assert result.value.check_name == "cache_wait", result.value
    assert "rank', 1" in result.value.msg, result.value.msg


def test_wait_seconds_must_be_positive(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0")
    chain, *_ = _one_layer()
    result = gio.chain_load(0, chain)
    assert type(result) is gb.Err
    assert result.value.argument_name == "GDTENSOR_CACHE_WAIT_SECONDS"


# ---------------------------------------------------------------- adversarial regressions
#
# The six adversarial findings (budget unit, absent-cohort semantics, process
# coordination, in-flight progress, plan-cache liveness, up-front checks) each
# have a regression here. Every case uses only public APIs and failed against
# the pre-fix code:
# * budget was per (tid, s, c); two tensors sharing one cell double-counted;
# * a missing cohort rank hung forever instead of a clear cache_wait;
# * one process per rank could not coordinate (the plan state was in-memory);
# * a slow fill counted as no progress and timed out healthy peers;
# * the plan cache pinned every cache view for the lifetime of the process;
# * mixed node sets and out-of-range ranks passed check_constraints silently.


def _dt(tid, size, world, placement=None):
    return gb.NamedDTensor(
        tid,
        gb.NaiveDTensor(
            gb.GlobalTensorMeta([size], "U8", 1),
            [placement or gb.Shard(0)],
            gb.ArrangedMesh(gt.WorldSize(world)),
        ),
    )


def _storage_mt(acc, tids, size, world):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    named = [
        gb.NamedDTensor(
            t,
            gb.NaiveDTensor(
                gb.GlobalTensorMeta([size], "U8", 1),
                [gb.Replicate(), gb.Shard(0)],
                mesh,
            ),
        )
        for t in tids
    ]
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(len(tids))),
                gb.ArrangedMesh(gt.WorldSize(world)),
            )
        ],
        list(tids),
    )
    entries = {gb.TensorComponentPos(t, r): group for t in tids for r in range(world)}
    layout = gb.PhysicalLayoutTable(
        "safetensors", entries, {gb.TensorComponentPos(tids[0], 0): "mem"}
    )

    class _Load:
        def get_file_offset(self, pos):
            return gb.Ok(
                tids.index(pos.tensor_id) * size if hasattr(pos, "tensor_id") else 0
            )

        def init_data_accessor(self):
            return gb.Ok(acc)

        def finish(self):
            return gb.Ok(None)

    return gb.StorageDTensorsView(
        list(tids),
        gb.DTensorTable(named),
        Memory([0]),
        layout,
        lambda _v, _r: gb.Ok(_Load()),
        lambda _v, _r: gb.Ok(_Load()),
    )


def _compute_mt(acc, tids, size, world, placement=None):
    return gb.ComputeDTensorsView(
        list(tids),
        gb.DTensorTable([_dt(t, size, world, placement) for t in tids]),
        acc,
    )


def _cache_mesh(steps, nodes, width):
    return gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=gb.LiteralMesh(list(nodes), gt.NumNodes(len(nodes))),
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )


def _cache_mt(acc, tids, size, steps, nodes, width, per_tid_nodes=None):
    named = []
    for t in tids:
        nd = (per_tid_nodes or {}).get(t, nodes)
        named.append(
            gb.NamedDTensor(
                t,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([size], "U8", 1),
                    [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
                    _cache_mesh(steps, nd, width),
                ),
            )
        )
    return gb.CacheDTensorsView(list(tids), gb.DTensorTable(named), acc)


def _one_layer_mt(
    tids=("w",),
    size=8,
    world=2,
    k=2,
    nodes=(0,),
    steps=2,
    width=1,
    placement=None,
    per_tid_nodes=None,
    src_delay=0.0,
):
    placement = gb.Replicate() if placement == "rep" else placement
    tids = list(tids)
    n = len(tids)
    src = Probe(np.tile(np.arange(size, dtype=np.uint8), n), src_delay)
    cache_mem = Probe(np.zeros(size, np.uint8))
    dst = Probe(np.zeros(size, np.uint8))
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_mt(src, tids, size, world),
        cache=_cache_mt(cache_mem, tids, size, steps, nodes, width, per_tid_nodes),
        dst=_compute_mt(dst, tids, size, world, placement),
        num_ranks_per_node=k,
    )
    return chain, src, cache_mem, dst


def _mp_coord_worker(rank, cfg, q):
    """One process per rank: the real deployment shape of the file coordinator."""
    chain, _src, _cache_mem, _dst = _one_layer_mt(**cfg)
    r = gio.chain_load(rank, chain)
    q.put((rank, type(r) is gb.Ok))


def test_one_process_per_rank_coordinates_via_shared_state(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "3")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_mp_coord_worker,
            args=(r, dict(steps=2, width=1, placement="rep", world=2), q),
        )
        for r in range(2)
    ]
    for p in procs:
        p.start()
    out = [q.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(30)
    assert sorted(out) == [(0, True), (1, True)]


def test_one_process_per_rank_multi_tensor_coordinates(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "3")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_mp_coord_worker,
            args=(
                r,
                dict(tids=("a", "b"), steps=1, width=1, placement="rep", world=2),
                q,
            ),
        )
        for r in range(2)
    ]
    for p in procs:
        p.start()
    out = [q.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(30)
    assert sorted(out) == [(0, True), (1, True)]


@pytest.mark.parametrize("steps,width", [(1, 1), (2, 1), (2, 2)])
def test_multi_tensor_budget_aggregates_all_tensors_of_a_cell(
    tmp_path, monkeypatch, steps, width
):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    # Two 8B tensors share one (s, c) cell. The budget must be their sum, not
    # one tensor's cell bytes: 16 for S=1, 8 for S=2 at any concurrency width.
    chain, *_ = _one_layer_mt(
        tids=("a", "b"), size=8, world=2, k=2, nodes=(0,), steps=steps, width=width
    )
    assert gio.chain_stat_mem_overhead(chain) == [16 // steps]
    _ok(_load_all(chain, 2, threaded=True))


def test_rank_outside_read_chain_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    chain, *_ = _one_layer_mt(nodes=(0,), world=2)
    r = gio.chain_load(99, chain)
    assert type(r) is gb.Err, r
    assert r.value.argument_name == "rank", r.value
    assert "not in read-chain ranks" in r.value.msg, r.value.msg


def test_mixed_node_sets_rejected_up_front(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    size, world, k = 8, 4, 2
    src = Probe(np.tile(np.arange(size, dtype=np.uint8), 2))
    cv = _cache_mt(
        Probe(np.zeros(size, np.uint8)),
        ["a", "b"],
        size,
        1,
        (0, 1),
        1,
        per_tid_nodes={"a": (0, 1), "b": (0,)},
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_mt(src, ["a", "b"], size, world),
        cache=cv,
        dst=_compute_mt(Probe(np.zeros(size, np.uint8)), ["a", "b"], size, world),
        num_ranks_per_node=k,
    )
    check = chain.check_constraints()
    assert type(check) is gb.Err, check
    assert "all tensors of a cache view must cover one node set" in check.value.msg
    r = gio.chain_load(0, chain)
    assert type(r) is gb.Err, r
    assert "all tensors of a cache view must cover one node set" in r.value.msg


def test_serial_missing_cohort_rank_fails_every_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0.5")
    # Replicate dst with S=2, C=1: the budget holds one cell, both ranks need
    # both cells, so each rank must be present while the other runs.
    chain, *_ = _one_layer_mt(
        size=8, world=2, k=2, nodes=(0,), steps=2, width=1, placement="rep"
    )
    for rank in (0, 1):
        r = gio.chain_load(rank, chain)
        assert type(r) is gb.Err, r
        assert r.value.check_name == "cache_wait", r.value


def test_inflight_slow_fill_counts_as_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "1")
    # Rank 0 fills the only segment for 2.5s (> the 1s wait limit). Rank 1 must
    # treat the in-flight fill as progress instead of timing out.
    chain, src, _c, _d = _one_layer_mt(
        size=8,
        world=2,
        k=2,
        nodes=(0,),
        steps=1,
        width=1,
        placement="rep",
        src_delay=2.5,
    )
    _ok(_load_all(chain, 2, threaded=True))
    assert bytes(_d.data) == bytes(np.arange(8, dtype=np.uint8))
    assert src.total("read") == 8


def test_concurrent_slow_fill_full_budget_two_ranks_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "1")
    # S=2, C=1 with both ranks online: two slow fills back to back, every rank
    # drains both cells. A healthy peer must never hit the wait limit.
    chain, src, _c, _d = _one_layer_mt(
        size=8,
        world=2,
        k=2,
        nodes=(0,),
        steps=2,
        width=1,
        placement="rep",
        src_delay=2.5,
    )
    _ok(_load_all(chain, 2, threaded=True))
    assert bytes(_d.data) == bytes(np.arange(8, dtype=np.uint8))
    assert src.total("read") == 8


def test_plan_cache_does_not_pin_cache_views(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    chain, *_ = _one_layer_mt(size=8, world=2, k=2, nodes=(0,), steps=1, width=1)
    _ok(_load_all(chain, 2, threaded=True))
    view = chain.cache
    ref = weakref.ref(view)
    del chain, view
    gc.collect()
    assert ref() is None, "the plan cache must not pin cache views"

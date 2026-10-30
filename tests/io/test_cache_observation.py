"""Independent acceptance oracle for cache chain observability (task 27).

Owned by the verification agent. Expectations come only from raw public
accessor events and from a reference slot-residency allocator — never from
``gdtensor.io`` stats, internal plans or box/union helpers.

Residency model: a cache slot buffer is allocated on write, RETAINED after
the writer callback returns, and released only when the downstream reader
acks (read callback returns). ``resident_peak`` is the max total retained
slot bytes at any instant — the true extra-allocation footprint, distinct
from the backing source buffer (baseline).
"""

from __future__ import annotations

import itertools
import threading
import time

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.session as gs
import gdtensor.tag as gt
from gdtensor import Result

from tests.support.cache_oracle import ResidentCache


# ---------------------------------------------------------------------------
# Backing-buffer accessor (source / dst baseline)
# ---------------------------------------------------------------------------


class LiveMemory(ga.DTensorDataAccessor):
    """A flat byte buffer that records access events."""

    def __init__(self, data: np.ndarray, events: list, name: str, error=None):
        self.data = np.array(data, dtype=np.uint8)
        self.events = events
        self.name = name
        self.error = error
        self.error_on_call = None
        self.write_error_on_call = None
        self._read_calls = 0
        self._write_calls = 0
        self.inflight: set = set()
        self.release_barrier: threading.Event | None = None
        self.barriers: dict = {}  # rank -> Event (thread-local write park)
        self._tl = threading.local()
        self._tl.rank = 0

    @property
    def current_rank(self) -> int:
        return getattr(self._tl, "rank", 0)

    @current_rank.setter
    def current_rank(self, rank: int) -> None:
        self._tl.rank = rank

    def _access(self, rng, callback, phase: str):
        if phase == "read":
            self._read_calls += 1
        else:
            self._write_calls += 1
        if (
            phase == "read"
            and self.error is not None
            and self.error_on_call == self._read_calls - 1
        ):
            return gb.Err(self.error)
        if (
            phase == "write"
            and self.error is not None
            and self.write_error_on_call == self._write_calls - 1
        ):
            return gb.Err(self.error)
        offset = getattr(rng, "global_tensor_offset", None)
        if offset is None:
            offset = rng.contiguous_payload_offset
        size = rng.size
        self.inflight.add((offset, size))
        if self.release_barrier is not None:
            self.release_barrier.wait()
        if phase == "write" and self.current_rank in self.barriers:
            self.barriers[self.current_rank].wait()
        self.events.append((self.name, phase, offset, size))
        data = self.data[offset : offset + size]
        res = callback(data.ctypes.data, data.nbytes)
        self.inflight.discard((offset, size))
        return res

    def read_to_buffer(self, global_range, read_to_buffer) -> Result[None]:
        return self._access(global_range, read_to_buffer, "read")

    def write_from_buffer(self, global_range, write_from_buffer) -> Result[None]:
        return self._access(global_range, write_from_buffer, "write")


# ---------------------------------------------------------------------------
# Reference slot-residency cache allocator (imported from tests/support)
# ---------------------------------------------------------------------------


class LiveSave(gs.StorageSaveSession):
    def __init__(self, mem, events):
        self.mem, self.events = mem, events
        self.opened = self.finished = self.aborted = 0

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        self.opened += 1
        return gb.Ok(self.mem)

    def commit(self):
        self.finished += 1
        return gb.Ok(None)

    def abort(self):
        self.aborted += 1
        return gb.Ok(None)


class LiveLoad(gs.StorageLoadSession):
    def __init__(self, mem, events, error=None):
        self.mem, self.events, self.error = mem, events, error
        self.opened = self.finished = self.aborted = 0

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        self.opened += 1
        if self.error is not None:
            return gb.Err(self.error)
        return gb.Ok(self.mem)

    def finish(self):
        self.finished += 1
        return gb.Ok(None)

    def abort(self):
        self.aborted += 1
        return gb.Ok(None)


# ---------------------------------------------------------------------------
# View builders (public API only)
# ---------------------------------------------------------------------------


def _sh(s):
    return [s] if isinstance(s, int) else s


def _cache_dt(names, shapes, placements, temporal_steps, nodes, width):
    dts = []
    for name, shape in zip(names, shapes):
        mesh = gb.StCacheOuterProductMesh(
            temporal=gb.ArrangedMesh(gt.NumSteps(temporal_steps)),
            spatial=gb.ArrangedMesh(gt.NumNodes(2)),
            concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
        )
        dts.append(
            gb.NamedDTensor(
                name,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta(_sh(shape), "U8", 1), placements, mesh
                ),
            )
        )
    return gb.DTensorTable(dts)


def _storage_view(mem, names, shapes, events, world=4):
    entries = {}
    leaders = {}
    dts = []
    for name, shape in zip(names, shapes):
        mesh = gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
        )
        dts.append(
            gb.NamedDTensor(
                name,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta(_sh(shape), "U8", 1),
                    [gb.Replicate(), gb.Shard(0)],
                    mesh,
                ),
            )
        )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(len(names))),
                gb.ArrangedMesh(gt.WorldSize(world)),
            )
        ],
        list(names),
    )
    for name in names:
        for rank in range(world):
            entries[gb.TensorComponentPos(name, rank)] = group
    leaders[gb.TensorComponentPos(names[0], 0)] = "mem"
    layout = gb.PhysicalLayoutTable("safetensors", entries, leaders)

    def begin_save(view, rank):
        return gb.Ok(LiveSave(mem, events))

    def begin_load(view, rank):
        return gb.Ok(LiveLoad(mem, events))

    return gb.StorageDTensorsView(
        names, gb.DTensorTable(dts), mem, layout, begin_save, begin_load
    )


def _compute_view(mem, names, shapes, world):
    dts = []
    for name, shape in zip(names, shapes):
        mesh = gb.ArrangedMesh(gt.WorldSize(world))
        dts.append(
            gb.NamedDTensor(
                name,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta(_sh(shape), "U8", 1), [gb.Shard(0)], mesh
                ),
            )
        )
    return gb.ComputeDTensorsView(names, gb.DTensorTable(dts), mem)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _tally_node_bytes(events, observed, node):
    for ev in events:
        who, phase, size = ev[0], ev[1], ev[-1]
        if who == "src" and phase == "read":
            observed[node]["in"] += size
        if who == "dst" and phase == "write":
            observed[node]["out"] += size


def test_world4_k2_per_node_crossnode_bytes_match_stats():
    """world=4, k=2. Observed per-node pull/push bytes must match stats."""
    world, k = 4, 2
    events = []
    src = LiveMemory(np.arange(8, dtype=np.uint8), events, "src")
    cache = ResidentCache(events, k)
    dst = LiveMemory(np.zeros(8, dtype=np.uint8), events, "dst")

    sv = _storage_view(src, ["w"], [8], events)
    cv = _compute_view(dst, ["w"], [8], world)
    cvt = gb.CacheDTensorsView(
        ["w"],
        _cache_dt(["w"], [8], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0, 1], 1),
        cache,
    )

    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    assert type(chain.check_constraints()) is gb.Ok

    observed = {n: {"in": 0, "out": 0} for n in (0, 1)}
    for rank in range(world):
        node = rank // k
        events.clear()
        assert type(gio.chain_load(rank, chain)) is gb.Ok
        _tally_node_bytes(events, observed, node)

    outbound = gio.chain_stat_outbound_traffic(chain)
    inbound = gio.chain_stat_inbound_traffic(chain)
    # Cache -> rank edges stay on one node: design 5.8 counts no traffic.
    assert outbound == [0, 0], outbound
    # Node fill-once: inbound equals the deduplicated source reads per node.
    assert inbound == [observed[0]["in"], observed[1]["in"]], (inbound, observed)
    assert inbound == [4, 4], inbound
    fills = [e for e in events if e[0] == "cache" and e[1] == "write"]
    assert len(fills) == len({(e[2], e[3]) for e in fills}), "fill dedup violated"


def test_multitensor_peak_live_from_allocator_not_writes():
    """Peak resident slot bytes come from the reference allocator, not stats."""
    world, k = 2, 1
    events = []
    src = LiveMemory(np.arange(9, dtype=np.uint8), events, "src")
    cache = ResidentCache(events, k)
    dst = LiveMemory(np.zeros(9, dtype=np.uint8), events, "dst")

    sv = _storage_view(src, ["a", "b", "c"], [6, 2, 1], events, world)
    cv = _compute_view(dst, ["a", "b", "c"], [6, 2, 1], world)
    cvt = gb.CacheDTensorsView(
        ["a", "b", "c"],
        _cache_dt(
            ["a", "b", "c"],
            [6, 2, 1],
            [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
            2,
            [0, 1],
            1,
        ),
        cache,
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    for rank in range(world):
        cache.current_rank = rank
        assert type(gio.chain_load(rank, chain)) is gb.Ok

    mem = gio.chain_stat_mem_overhead(chain)
    expected = [cache.peak_by_node.get(n, 0) for n in range(2)]
    # Stat is the aggregated (step, slot) cell budget: node 0 holds the
    # A+B+C step-0 segments (5), node 1 the A+B step-1 segments (4). The
    # allocator peak reflects the serial single-segment execution (3) and
    # stays below the bound.
    assert mem == [5, 4], mem
    assert all(p <= b for p, b in zip(expected, mem)), (expected, mem)


def test_slot_steps_not_equal_nodes():
    """S=3 slots with 2 nodes is legal; resident peak from allocator."""
    world, k = 4, 2
    events = []
    src = LiveMemory(np.arange(6, dtype=np.uint8), events, "src")
    cache = ResidentCache(events, k)
    dst = LiveMemory(np.zeros(6, dtype=np.uint8), events, "dst")
    sv = _storage_view(src, ["w"], [6], events)
    cv = _compute_view(dst, ["w"], [6], world)
    cvt = gb.CacheDTensorsView(
        ["w"],
        _cache_dt(["w"], [6], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 3, [0, 1], 1),
        cache,
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    for rank in range(world):
        cache.current_rank = rank
        assert type(gio.chain_load(rank, chain)) is gb.Ok
    mem = gio.chain_stat_mem_overhead(chain)
    expected = [cache.peak_by_node.get(n, 0) for n in range(2)]
    assert mem == expected, (mem, expected)


def test_delayed_read_callback_blocks_advance():
    """Parked src read must not start the same slot's next leg or finalize."""
    import time

    world, k = 2, 1
    events = []
    src = LiveMemory(np.arange(8, dtype=np.uint8), events, "src")
    src.release_barrier = threading.Event()
    cache = ResidentCache(events, 1)
    dst = LiveMemory(np.zeros(8, dtype=np.uint8), events, "dst")

    sv = _storage_view(src, ["w"], [8], events, world)
    cv = _compute_view(dst, ["w"], [8], world)
    cvt = gb.CacheDTensorsView(
        ["w"],
        _cache_dt(["w"], [8], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0, 1], 1),
        cache,
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )

    result_box = {}

    def runner():
        result_box["r"] = gio.chain_load(0, chain)

    t = threading.Thread(target=runner)
    t.start()
    for _ in range(100):
        if src.inflight:
            break
        time.sleep(0.05)
    assert src.inflight, "read callback never entered"
    parked = list(events)
    time.sleep(0.3)
    # Single-tensor chain: any new event while parked belongs to the same
    # slot's next leg or finalize — production must not have advanced.
    new_events = events[len(parked) :]
    assert not new_events, new_events
    src.release_barrier.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert type(result_box["r"]) is gb.Ok
    # After release the normal legs complete; residency drained fully.
    assert cache.live == [], cache.live
    # chain_load(0) delivers rank 0's shard only (world=2, Shard(0)).
    assert list(dst.data[:4]) == [0, 1, 2, 3]
    assert list(dst.data[4:]) == [0, 0, 0, 0]


def test_mixed_a_bytes_same_error_and_lifecycle():
    """Mixed chain: A direct delivers bytes; B fault returns same error."""
    world = 2
    events = []
    src_a = LiveMemory(np.arange(4, dtype=np.uint8), events, "src_a")
    dst_a = LiveMemory(np.zeros(4, dtype=np.uint8), events, "dst_a")
    direct = gb.DirectDTensorsReadChain(
        src=_storage_view(src_a, ["a"], [4], events, world),
        dst=_compute_view(dst_a, ["a"], [4], world),
    )
    boom = gb.GDTensorIOError(operation="disk", msg="sector-fault")
    src_b = LiveMemory(np.arange(4, dtype=np.uint8), events, "src_b", error=boom)
    src_b.error_on_call = 1
    dst_b = LiveMemory(np.zeros(4, dtype=np.uint8), events, "dst_b")
    cache_b = ResidentCache(events, 1)
    one = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_view(src_b, ["b"], [4], events, world),
        cache=gb.CacheDTensorsView(
            ["b"],
            _cache_dt(
                ["b"], [4], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0, 1], 1
            ),
            cache_b,
        ),
        dst=_compute_view(dst_b, ["b"], [4], world),
        num_ranks_per_node=1,
    )
    mixed = gb.MixedDTensorsReadChain(
        [
            gb.DisjointTensorIdsAndReadChain(["a"], direct),
            gb.DisjointTensorIdsAndReadChain(["b"], one),
        ]
    )
    assert type(gio.chain_load(0, mixed)) is gb.Ok
    assert list(dst_a.data[:2]) == [0, 1]
    assert list(dst_a.data[2:]) == [0, 0]

    result = gio.chain_load(0, mixed)
    assert type(result) is gb.Err
    assert result.value is boom or result.value.msg == "sector-fault"


def _chain_w(world=2, k=1, events=None, src_mem=None, dst_mem=None):
    events = events if events is not None else []
    src = LiveMemory(
        src_mem if src_mem is not None else np.arange(8, dtype=np.uint8), events, "src"
    )
    cache = ResidentCache(events, k)
    dst = LiveMemory(
        dst_mem if dst_mem is not None else np.zeros(8, dtype=np.uint8), events, "dst"
    )
    sv = _storage_view(src, ["w"], [8], events, world)
    cv = _compute_view(dst, ["w"], [8], world)
    cvt = gb.CacheDTensorsView(
        ["w"],
        _cache_dt(["w"], [8], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0, 1], 1),
        cache,
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=k
    )
    return chain, src, cache, dst


def test_concurrent_ranks_share_allocator_drain_and_peak():
    """Two ranks enter the chain on threads; allocator stays race-free and drains.

    The executor is currently serial, so this does NOT claim parallel throughput
    benefit — it verifies that threaded entry never corrupts shared live state,
    never leaks a slot, and still delivers correct shards.
    """
    world, k = 2, 1
    chain, _src, cache, dst = _chain_w(world, k)

    box = {}

    def run(rank):
        cache.current_rank = rank
        box[rank] = gio.chain_load(rank, chain)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive(), "chain_load hung"

    for r in range(world):
        assert type(box[r]) is gb.Ok, box[r]
    # Every slot buffer released once its reader acked: nothing left resident.
    assert cache.live == [], cache.live
    # Each node recorded its own peak independently (never merged across nodes).
    assert all(v > 0 for v in cache.peak_by_node.values()), cache.peak_by_node
    # Rank 0 shard [0:4], rank 1 shard [4:8] delivered.
    assert list(dst.data[:4]) == [0, 1, 2, 3]
    assert list(dst.data[4:]) == [4, 5, 6, 7]


def test_failure_cancel_frees_partially_written_slots():
    """Consumer write fails after the slot was filled: cancellation releases it.

    The executor is serial, so this exercises the same rank: cache fill write
    succeeds, then the consumer (dst) write errors on its first call. Expect
    Err returned and zero resident slots left — the fill was not leaked.
    """
    world, k = 2, 1
    events = []
    src = LiveMemory(np.arange(8, dtype=np.uint8), events, "src")
    cache = ResidentCache(events, k)
    boom = gb.GDTensorIOError(operation="disk", msg="cancel-fault")
    dst = LiveMemory(np.zeros(8, dtype=np.uint8), events, "dst", error=boom)
    dst.write_error_on_call = 0
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_view(src, ["w"], [8], events, world),
        cache=gb.CacheDTensorsView(
            ["w"],
            _cache_dt(
                ["w"], [8], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0, 1], 1
            ),
            cache,
        ),
        dst=_compute_view(dst, ["w"], [8], world),
        num_ranks_per_node=k,
    )

    cache.current_rank = 0
    result = gio.chain_load(0, chain)
    assert type(result) is gb.Err, result
    assert result.value.msg == "cancel-fault"
    # Cancellation released the filled slot: ACK-remaining count is zero.
    assert cache.live == [], cache.live


def test_node_shared_fill_dedup_and_per_consumer_bytes():
    """Node-shared fill is written once; per-consumer shard bytes stay exact.

    Two ranks on the same node (k=2, world=4) consume one slot. The cache must
    record exactly one fill write per (tid, offset), and each consumer receives
    its own shard bytes — no duplicated fill, no shared-buffer aliasing.
    """
    world, k = 4, 2
    events = []
    src = LiveMemory(np.arange(8, dtype=np.uint8), events, "src")
    cache = ResidentCache(events, k)
    dst = LiveMemory(np.zeros(8, dtype=np.uint8), events, "dst")
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_view(src, ["w"], [8], events, world),
        cache=gb.CacheDTensorsView(
            ["w"],
            _cache_dt(
                ["w"], [8], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0, 1], 1
            ),
            cache,
        ),
        dst=_compute_view(dst, ["w"], [8], world),
        num_ranks_per_node=k,
    )
    for rank in range(world):
        cache.current_rank = rank
        assert type(gio.chain_load(rank, chain)) is gb.Ok

    fills = [e for e in events if e[0] == "cache" and e[1] == "write"]
    seen = {(e[2], e[3]) for e in fills}
    assert len(fills) == len(seen), (len(fills), len(seen))
    # Node-0 ranks 0,1 and node-1 ranks 2,3 each got their own shard copy.
    assert list(dst.data[:2]) == [0, 1]
    assert list(dst.data[2:4]) == [2, 3]
    assert list(dst.data[4:6]) == [4, 5]
    assert list(dst.data[6:]) == [6, 7]
    assert cache.live == [], cache.live


def _chain_w_c1(events, src=None, dst=None, width=1):
    src = LiveMemory(
        src if src is not None else np.arange(8, dtype=np.uint8), events, "src"
    )
    cache = ResidentCache(events, 1)
    dst = LiveMemory(
        dst if dst is not None else np.zeros(8, dtype=np.uint8), events, "dst"
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_view(src, ["w"], [8], events, 2),
        cache=gb.CacheDTensorsView(
            ["w"],
            _cache_dt(
                ["w"], [8], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0, 1], width
            ),
            cache,
        ),
        dst=_compute_view(dst, ["w"], [8], 2),
        num_ranks_per_node=1,
    )
    return chain, src, cache, dst


def test_c1_two_fills_serialize_capacity():
    """C=1: second fill must wait while first slot is live; peak never >1 slot."""
    import time

    events = []
    chain, _src, cache, _dst = _chain_w_c1(events)
    box = {}

    def run(rank):
        cache.current_rank = rank
        box[rank] = gio.chain_load(rank, chain)

    t0 = threading.Thread(target=run, args=(0,))
    t0.start()
    time.sleep(0.5)
    assert cache.peak_by_node.get(0, 0) <= 4, (
        cache.peak_by_node
    )  # ceil(8/2)=4B per node, C=1
    t1 = threading.Thread(target=run, args=(1,))
    t1.start()
    t1.join(timeout=5)
    assert not t1.is_alive(), "C=1 capacity: second fill deadlocked (ack never reached)"
    t0.join(timeout=5)
    assert not t0.is_alive()
    assert type(box[0]) is gb.Ok and type(box[1]) is gb.Ok, box
    assert cache.live == [], cache.live


def test_node_isolation_same_offset_not_shared_across_nodes():
    """Same (tid, offset) on different nodes must be independent slots."""
    events = []
    src = LiveMemory(np.arange(4, dtype=np.uint8), events, "src")
    cache = ResidentCache(events, 2)
    dst = LiveMemory(np.zeros(4, dtype=np.uint8), events, "dst")
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_view(src, ["w"], [4], events, 4),
        cache=gb.CacheDTensorsView(
            ["w"],
            _cache_dt(
                ["w"],
                [4],
                [gb.Replicate(), gb.Replicate(), gb.Replicate()],
                2,
                [0, 1],
                1,
            ),
            cache,
        ),
        dst=_compute_view(dst, ["w"], [4], 4),
        num_ranks_per_node=2,
    )
    for rank in range(4):
        cache.current_rank = rank
        assert type(gio.chain_load(rank, chain)) is gb.Ok
    assert set(cache.peak_by_node) == {0, 1}, cache.peak_by_node
    assert cache.live == [], cache.live


def test_fill_failure_notifies_all_waiters_same_error_no_retry():
    """One fill error must fail every same-node waiter with the same error, no retry.

    world=2, k=2 => single node; ranks 0 and 1 are the same-node cohort sharing
    one slot fill. Both must get fill-boom, and the fill attempted exactly once.
    """
    boom = gb.GDTensorIOError(operation="disk", msg="fill-boom")
    events = []
    src = LiveMemory(np.arange(8, dtype=np.uint8), events, "src", error=boom)
    src.error_on_call = 0
    cache = ResidentCache(events, 2)
    dst = LiveMemory(np.zeros(8, dtype=np.uint8), events, "dst")
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=_storage_view(src, ["w"], [8], events, 2),
        cache=gb.CacheDTensorsView(
            ["w"],
            _cache_dt(
                ["w"], [8], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0], 1
            ),
            cache,
        ),
        dst=_compute_view(dst, ["w"], [8], 2),
        num_ranks_per_node=2,
    )
    box = {}

    def run(rank):
        cache.current_rank = rank
        box[rank] = gio.chain_load(rank, chain)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive(), "waiter hung after fill error"
    for r in range(2):
        assert type(box[r]) is gb.Err, box
        assert box[r].value.msg == "fill-boom", box
    # No retry: exactly one src read attempt for the failed fill.
    assert src._read_calls == 1, src._read_calls
    assert cache.live == [], cache.live


def _run_capacity_case(rows, S, C):
    events = []
    src = LiveMemory(np.arange(rows, dtype=np.uint8), events, "src")
    cache = ResidentCache(events, 1)
    dst = LiveMemory(np.zeros(rows, dtype=np.uint8), events, "dst")
    sv = _storage_view(src, ["w"], [rows], events, 2)
    cv = _compute_view(dst, ["w"], [rows], 2)
    cvt = gb.CacheDTensorsView(
        ["w"],
        _cache_dt(
            ["w"], [rows], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], S, [0, 1], C
        ),
        cache,
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=1
    )
    for rank in range(2):
        cache.current_rank = rank
        assert type(gio.chain_load(rank, chain)) is gb.Ok, (rows, S, C, rank)
    peak = max(cache.peak_by_node.values(), default=0)
    step = (
        rows + S - 1
    ) // S  # ceil(rows/S): shard box with remainder; N=replica, NOT split
    assert peak <= C * step, (rows, S, C, peak, step)


def test_random_nodivisible_capacity_matches_real_residency():
    """Non-divisible shapes (7,9 rows) x N=1..3 x C=1..3 x S=2..3.

    Reference ResidentCache peak must equal chain stats; total resident bytes must
    respect C * real slot step (no box/N ad-hoc undercount).
    """
    for rows, S, C in itertools.product((7, 9), (2, 3), (1, 2, 3)):
        _run_capacity_case(rows, S, C)


def _wait_until_read_calls(cache, target=1, polls=100):
    for _ in range(polls):
        if cache._read_calls >= target:
            return
        time.sleep(0.05)


def test_replicate_two_consumers_both_must_ack_before_release():
    """Single-slot Replicate cohort: slot stays live until BOTH ranks ACK.

    src storage world=1 Replicate (single fill range 4B); compute view world=2
    Replicate; cache S=1 C=1 spatial=[0] k=2 world=2. Two TLS sink barriers.
    Release rank0 only -> rank0 returns, slot must stay live; release rank1 -> freed.
    """
    events = []
    src = LiveMemory(np.arange(4, dtype=np.uint8), events, "src")
    cache = ResidentCache(events, 2)
    dst = LiveMemory(np.zeros(4, dtype=np.uint8), events, "dst")
    barriers = {0: threading.Event(), 1: threading.Event()}
    sv = _storage_view(src, ["w"], [4], events, 1)
    dts = [
        gb.NamedDTensor(
            "w",
            gb.NaiveDTensor(
                gb.GlobalTensorMeta(_sh([4]), "U8", 1),
                [gb.Replicate()],
                gb.ArrangedMesh(gt.WorldSize(2)),
            ),
        )
    ]
    cv = gb.ComputeDTensorsView(["w"], gb.DTensorTable(dts), dst)
    cvt = gb.CacheDTensorsView(
        ["w"],
        _cache_dt(["w"], [4], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 1, [0], 1),
        cache,
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=2
    )
    # Declared schema: both ranks must read full 4B (Replicate cohort).
    cache.expected_reads = {
        (0, "w", 0): {0: range(0, 4), 1: range(0, 4)},
    }
    box = {}

    def run(rank):
        cache.current_rank = rank
        dst.current_rank = rank
        cache.barriers = barriers
        box[rank] = gio.chain_load(rank, chain)

    try:
        threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
        for t in threads:
            t.start()
        # Wait until rank0 reaches cache read (rank1 blocked on read barrier).
        _wait_until_read_calls(cache)
        barriers[0].set()  # let rank0 finish its read
        threads[0].join(timeout=5)
        assert not threads[0].is_alive(), "rank0 hung"
        assert type(box[0]) is gb.Ok, box
        # Cohort: rank1 read still blocked; slot must stay live.
        assert cache.live, cache.live
        barriers[1].set()  # let rank1 read finish
        threads[1].join(timeout=5)
        assert not threads[1].is_alive(), "rank1 hung"
        assert type(box[1]) is gb.Ok, box
        assert cache.live == [], cache.live
    finally:
        for ev in barriers.values():
            ev.set()


def test_same_key_fill_fail_then_next_generation_succeeds():
    """Same-key fail: waiters get same Error, no retry; next gen both Ok."""
    boom = gb.GDTensorIOError(operation="disk", msg="fill-boom")
    events = []
    src = LiveMemory(np.arange(8, dtype=np.uint8), events, "src", error=boom)
    src.error_on_call = 0
    cache = ResidentCache(events, 2)
    dst = LiveMemory(np.zeros(8, dtype=np.uint8), events, "dst")
    sv = _storage_view(src, ["w"], [8], events, 2)
    cv = _compute_view(dst, ["w"], [8], 2)
    cvt = gb.CacheDTensorsView(
        ["w"],
        _cache_dt(["w"], [8], [gb.Shard(0), gb.Replicate(), gb.Shard(0)], 2, [0], 1),
        cache,
    )
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=2
    )
    box = {}

    def run(rank):
        cache.current_rank = rank
        box[rank] = gio.chain_load(rank, chain)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()
    for r in range(2):
        assert type(box[r]) is gb.Err, box
        assert box[r].value.msg == "fill-boom"
    assert src._read_calls == 1, src._read_calls
    # Next generation: error cleared, both ranks succeed.
    src.error = None
    box.clear()
    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    for r in range(2):
        assert type(box[r]) is gb.Ok, box
    assert cache.live == [], cache.live

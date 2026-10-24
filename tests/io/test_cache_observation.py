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

import threading

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.session as gs
import gdtensor.tag as gt
from gdtensor import Result


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
        self._read_calls = 0
        self.inflight: set = set()
        self.release_barrier: threading.Event | None = None

    def _access(self, rng, callback, phase: str):
        if (
            phase == "read"
            and self.error is not None
            and self.error_on_call == self._read_calls
        ):
            return gb.Err(self.error)
        if phase == "read":
            self._read_calls += 1
        offset = getattr(rng, "global_tensor_offset", None)
        if offset is None:
            offset = rng.contiguous_payload_offset
        size = rng.size
        self.inflight.add((offset, size))
        if phase == "read" and self.release_barrier is not None:
            self.release_barrier.wait()
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
# Reference slot-residency cache allocator
# ---------------------------------------------------------------------------


class ResidentCache(ga.DTensorDataAccessor):
    """Per-node cache allocator: regions live from write-done to matching-tid ack.

    The test runner sets ``current_rank`` before each ``chain_load``; the
    owner node is derived as ``rank // num_ranks_per_node``. Live regions are
    keyed (node, tid, offset) — per-node, per-tensor, never shared.
    ``peak_by_node[node]`` is that node's max simultaneously-resident bytes.
    """

    def __init__(self, events: list, num_ranks_per_node: int, name: str = "cache"):
        self.data: dict[str, dict[int, np.ndarray]] = {}
        self.events = events
        self.name = name
        self.num_ranks_per_node = num_ranks_per_node
        self.current_rank = 0
        self.live: list[tuple[int, str, int, int]] = []  # (node, tid, off, size)
        self.peak_by_node: dict[int, int] = {}

    @property
    def node(self) -> int:
        return self.current_rank // self.num_ranks_per_node

    def _snapshot(self):
        for node in {n for n, _t, _o, _s in self.live}:
            total = sum(sz for n, _t, _o, sz in self.live if n == node)
            self.peak_by_node[node] = max(self.peak_by_node.get(node, 0), total)

    def write_from_buffer(self, global_range, write_from_buffer) -> Result[None]:
        rng = global_range
        callback = write_from_buffer
        tid = getattr(rng, "global_tensor_id", "") or "?"
        offset = getattr(rng, "global_tensor_offset", None)
        if offset is None:
            offset = rng.contiguous_payload_offset
        size = rng.size
        buf = np.zeros(size, dtype=np.uint8)
        self.events.append((self.name, "write", tid, offset, size))
        res = callback(buf.ctypes.data, buf.nbytes)
        self.data.setdefault(tid, {})[offset] = buf
        self.live.append((self.node, tid, offset, size))
        self._snapshot()
        return res

    def read_to_buffer(self, global_range, read_to_buffer) -> Result[None]:
        rng = global_range
        callback = read_to_buffer
        tid = getattr(rng, "global_tensor_id", "") or "?"
        offset = getattr(rng, "global_tensor_offset", None)
        if offset is None:
            offset = rng.contiguous_payload_offset
        size = rng.size
        buf = self.data[tid][offset]
        self.events.append((self.name, "read", tid, offset, size))
        res = callback(buf.ctypes.data, buf.nbytes)
        self.live.remove((self.node, tid, offset, size))
        return res


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
    assert outbound == [observed[0]["out"], observed[1]["out"]], (outbound, observed)
    assert inbound == [observed[0]["in"], observed[1]["in"]], (inbound, observed)


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
    assert mem == expected, (mem, expected)


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

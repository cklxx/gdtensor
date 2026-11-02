"""Public two-layer IO with reclaimed file slots and uneven F32 columns.

Ranks run in threads. Cache files are shared, but this does not test independent
rank processes. Resident-byte counts exclude temporary IO callback buffers.
"""

import threading
import time
import zlib
from collections import Counter

import numpy as np
import pytest
import safetensors.numpy

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_sim.formats import make_storage_view
from tests.support.cache_oracle import storage_schema
from tests.support.cluster_sim import Cluster, layer_targets
from tests.support.oracle import ArrayFabric


@pytest.fixture(autouse=True, params=["file", "thread"])
def isolated_coordinator(request, tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", request.param)
    coord = tmp_path / "coord"
    coord.mkdir()
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(coord))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "3")


def _bytes(rng):
    return set(range(rng.global_tensor_offset, rng.global_tensor_offset + rng.size))


class ReusedFiles(ga.DTensorDataAccessor):
    """Delete after all independently declared consumers read a segment.

    A freed path is reused by a later segment. No full-tensor shadow copy exists.
    Upper consumers are child nodes; lower consumers are individual ranks.
    """

    def __init__(self, root, cluster, target, consumers, budgets, upper):
        root.mkdir()
        self.root, self.cluster, self.target = root, cluster, target
        self.consumers, self.budgets, self.upper = consumers, budgets, upper
        self.lock = threading.Lock()
        self.slots, self.coverage = {}, {}
        self.free = {node: [] for node in budgets}
        self.live = dict.fromkeys(budgets, 0)
        self.peak = dict.fromkeys(budgets, 0)
        self.writes, self.reads, self.deleted, self.reused = [], [], [], []
        self.block = None
        self.entered, self.release, self.fast = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

    def _address(self, rng):
        rank = self.cluster.rank
        node = getattr(rng, "cache_node", self.target(rank))
        consumer = rank // self.cluster.k if self.upper else rank
        return node, consumer

    def write_from_buffer(self, global_range, write_from_buffer):
        rng = global_range
        data = np.empty(rng.size, dtype=np.uint8)
        result = write_from_buffer(data.ctypes.data, data.nbytes)
        if type(result) is not gb.Ok:
            return result
        node, _ = self._address(rng)
        key = (node, rng.global_tensor_id, rng.global_tensor_offset, rng.size)
        with self.lock:
            self._persist(key, data)
        return result

    def _persist(self, key, data):
        node, _, _, size = key
        assert key not in self.slots, "duplicate fill of a resident segment"
        assert self.live[node] + size <= self.budgets[node], "resident budget exceeded"
        path = self._allocate(node, key)
        assert not path.exists(), "reused path still contains an old generation"
        assert path.write_bytes(data.tobytes()) == size
        self.slots[key] = path
        self.coverage[key] = {}
        self.live[node] += size
        self.peak[node] = max(self.peak[node], self.live[node])
        self.writes.append((key, path))

    def _allocate(self, node, key):
        if self.free[node]:
            path = self.free[node].pop()
            self.reused.append((key, path))
            return path
        return self.root / f"node-{node}-slot-{len(self.writes)}.bin"

    def read_to_buffer(self, global_range, read_to_buffer):
        rng = global_range
        node, consumer = self._address(rng)
        if self.block == (node, consumer) and rng.global_tensor_offset == 0:
            self.entered.set()
            assert self.release.wait(10), "slow cache consumer was not released"
        with self.lock:
            key, data = self._read(node, consumer, rng)
        result = read_to_buffer(data.ctypes.data, data.nbytes)
        if type(result) is not gb.Ok:
            return result
        with self.lock:
            self._ack(key, consumer, rng)
        return result

    def _read(self, node, consumer, rng):
        needed = _bytes(rng)
        assert needed <= self.consumers[node][consumer], "unexpected consumer bytes"
        hits = [
            (key, path)
            for key, path in self.slots.items()
            if key[0] == node
            and key[1] == rng.global_tensor_id
            and key[2] <= rng.global_tensor_offset
            and rng.global_tensor_offset + rng.size <= key[2] + key[3]
        ]
        assert len(hits) == 1, "required cache bytes were deleted or never filled"
        key, path = hits[0]
        raw = path.read_bytes()
        assert len(raw) == key[3], "physical slot was truncated"
        start = rng.global_tensor_offset - key[2]
        return key, np.frombuffer(raw[start : start + rng.size], dtype=np.uint8).copy()

    def _ack(self, key, consumer, rng):
        covered = self.coverage[key]
        covered.setdefault(consumer, set()).update(_bytes(rng))
        self.reads.append((key[0], consumer, rng.global_tensor_offset, rng.size))
        if (
            self.block
            and key[0] == self.block[0]
            and consumer != self.block[1]
            and rng.global_tensor_offset == 0
        ):
            self.fast.set()
        segment = set(range(key[2], key[2] + key[3]))
        required = self.consumers[key[0]]
        if all(
            (wanted & segment) <= covered.get(c, set())
            for c, wanted in required.items()
        ):
            path = self.slots.pop(key)
            path.unlink()
            assert not path.exists(), "released bytes remain on disk"
            self.live[key[0]] -= key[3]
            self.free[key[0]].append(path)
            self.deleted.append((key, path))

    def assert_slow_bytes_live(self):
        block = self.block
        assert block is not None, "blocked consumer never entered"
        node, slow = block
        with self.lock:
            held = [k for k in self.slots if k[0] == node and k[2] == 0]
            assert held, "first segment disappeared while its consumer was blocked"
            key = held[0]
            assert not self.coverage[key].get(slow)
            assert self.slots[key].stat().st_size == key[3]

    def assert_finished(self):
        assert not self.slots
        assert not any(self.live.values())
        assert not list(self.root.glob("*.bin"))
        assert len(self.deleted) == len(self.writes)
        assert self.reused, "test did not reuse a released physical slot"
        for node, consumers in self.consumers.items():
            self._assert_node(node, consumers)

    def _assert_node(self, node, consumers):
        wanted = set().union(*consumers.values())
        writes = Counter(
            b
            for (n, _, off, size), _ in self.writes
            if n == node
            for b in range(off, off + size)
        )
        assert writes == Counter({b: 1 for b in wanted}), "missing or repeated fill"
        for consumer, expected in consumers.items():
            reads = Counter(
                b
                for n, c, off, size in self.reads
                if (n, c) == (node, consumer)
                for b in range(off, off + size)
            )
            assert reads == Counter({b: 1 for b in expected})


def _table(meta, placements, mesh):
    return gb.DTensorTable(
        [gb.NamedDTensor("w", gb.NaiveDTensor(meta, placements, mesh))]
    )


def _cache_view(meta, accessor, nodes, steps):
    mesh = gb.StCacheOuterProductMesh(
        gb.ArrangedMesh(gt.NumSteps(steps)),
        gb.LiteralMesh(nodes, gt.NumNodes(max(nodes) + 1)),
        gb.ArrangedMesh(gt.ConcurrencyWidth(1)),
    )
    return gb.CacheDTensorsView(
        ["w"], _table(meta, [gb.Shard(0), gb.Replicate(), gb.Shard(0)], mesh), accessor
    )


def _save(root, values, meta):
    table, layout = storage_schema(["w"], size=values.size)
    table.named_d_tensors[0].d_tensor.set_global_tensor_meta(meta)
    dense = _table(meta, [gb.Replicate()], gb.ArrangedMesh(gt.WorldSize(1)))
    source = ArrayFabric(dense, {0: {"w": np.arange(values.size)}}, {"w": values})
    saved = make_storage_view(["w"], table, layout, str(root))
    result = gio.rank_save(0, source.view(0), saved)
    assert type(result) is gb.Ok, result
    decoded = safetensors.numpy.load_file(root / "model.safetensors")
    np.testing.assert_array_equal(decoded["w"], values)
    return make_storage_view(["w"], table, layout, str(root))


def _build(root, values, points, placement, steps):
    itemsize = values.dtype.itemsize
    meta = gb.GlobalTensorMeta(
        list(values.shape), "F32" if itemsize == 4 else "U8", itemsize
    )
    cluster = Cluster(2, 2)
    targets = layer_targets(cluster, [[0], [0, 1]])
    rank_bytes = {
        r: set((p[:, None] * itemsize + np.arange(itemsize)).ravel())
        for r, p in points.items()
    }
    lower = {
        node: {r: rank_bytes[r] for r in range(node * 2, node * 2 + 2)}
        for node in range(2)
    }
    upper = {0: {node: set().union(*ranks.values()) for node, ranks in lower.items()}}
    first_rows = (values.shape[0] + steps - 1) // steps
    cell_bytes = first_rows * (values.size // values.shape[0]) * itemsize
    first_cell = set(range(cell_bytes))
    budgets = [
        {
            node: len(first_cell & set().union(*consumers.values()))
            for node, consumers in demands.items()
        }
        for demands in (upper, lower)
    ]
    caches = [
        ReusedFiles(
            root / f"layer-{i}", cluster, targets[i], demand, budgets[i], i == 0
        )
        for i, demand in enumerate((upper, lower))
    ]
    table = _table(meta, [placement], gb.ArrangedMesh(gt.WorldSize(4)))
    dst = ArrayFabric(table, {r: {"w": p} for r, p in points.items()})
    src = _save(root / "checkpoint", values, meta)
    views = [
        _cache_view(meta, c, nodes, steps) for c, nodes in zip(caches, ([0], [0, 1]))
    ]
    chains = [
        gb.TwoLayerCachedDTensorsReadChain(
            src=src,
            small_cache=views[0],
            big_cache=views[1],
            dst=dst.view(r),
            num_ranks_per_node=2,
        )
        for r in range(4)
    ]
    assert type(chains[0].check_constraints()) is gb.Ok
    expected_peak = [budgets[0].get(n, 0) + budgets[1][n] for n in range(2)]
    assert gio.chain_stat_mem_overhead(chains[0]) == expected_peak
    return chains, cluster, caches, dst


def _worker(rank, chains, cluster, results):
    cluster.rank = rank
    try:
        results[rank] = gio.chain_load(rank, chains[rank])
    except BaseException as error:
        results[rank] = error


def _join(threads, timeout):
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0, deadline - time.monotonic()))


def _run(chains, cluster, caches, blocked=None):
    results = {}
    threads = [
        threading.Thread(
            target=_worker, args=(r, chains, cluster, results), daemon=True
        )
        for r in range(4)
    ]
    for thread in threads:
        thread.start()
    try:
        if blocked is not None:
            assert blocked.entered.wait(5), "slow consumer did not enter its read"
            assert blocked.fast.wait(5), "other consumer did not finish its read"
            blocked.assert_slow_bytes_live()
            blocked.release.set()
        _join(threads, 20)
        assert not any(t.is_alive() for t in threads), "chain_load hung"
        assert set(results) == set(range(4)), results
        assert all(type(r) is gb.Ok for r in results.values()), results
    finally:
        for cache in caches:
            cache.release.set()
        _join(threads, 10)


def _check(caches, dst, values):
    dst.assert_values({"w": values})
    for rank, tensors in dst.buffers.items():
        expected = values.reshape(-1)[dst.points[rank]["w"]].tobytes()
        assert zlib.crc32(tensors["w"].tobytes()) == zlib.crc32(expected)
    for cache in caches:
        cache.assert_finished()
        assert all(0 < cache.peak[n] <= budget for n, budget in cache.budgets.items())


@pytest.mark.parametrize("blocked_layer", [0, 1], ids=["slow-child-node", "slow-rank"])
def test_two_layer_physical_reuse_waits_for_slow_consumer(tmp_path, blocked_layer):
    values = np.random.default_rng(631).integers(0, 256, 32, dtype=np.uint8)
    points = {rank: np.arange(32) for rank in range(4)}
    chains, cluster, caches, dst = _build(
        tmp_path, values, points, gb.Replicate(), steps=4
    )
    blocked = caches[blocked_layer]
    blocked.block = (0, 1) if blocked_layer == 0 else (1, 3)
    _run(chains, cluster, caches, blocked)
    _check(caches, dst, values)


@pytest.mark.parametrize("columns", [11, 5], ids=["uneven-columns", "empty-last-rank"])
def test_two_layer_f32_uneven_column_shards(tmp_path, columns):
    values = np.random.default_rng(842).normal(size=(3, columns)).astype(np.float32)
    chunk = (columns + 3) // 4
    indices = np.arange(values.size).reshape(values.shape)
    points = {
        rank: indices[:, rank * chunk : (rank + 1) * chunk].ravel() for rank in range(4)
    }
    chains, cluster, caches, dst = _build(
        tmp_path, values, points, gb.Shard(1), steps=2
    )
    _run(chains, cluster, caches)
    _check(caches, dst, values)

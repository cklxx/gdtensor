"""Randomized stress for the co-located cache read chain.

Each seed draws one legal deployment: ranks per node, node count, one or two
cache layers, cache node sets, steps, concurrency width, tensor count and
size, compute placement, coordinator, start order, start delay, NIC bandwidth
and repeated loads. Payloads are random bytes, so outputs use exact-byte and
CRC checks.

An independent oracle derives the wanted bytes of every rank and the demand
of every cache node. It does not call the gdtensor planner. The test checks:

* every rank receives exact bytes with a matching CRC; guard bytes stay intact;
* every cache read finds valid bytes on the addressed node;
* every cache write is a local fill: filler node equals the DN node;
* every drain reads the serve node the oracle derives for the rank;
* every rank drains exactly its wanted bytes from the last cache layer, so
  each output byte passed through the cache;
* every demanded cache byte is filled once per generation, no more, no less;
* storage bytes are read once per first-layer node, by a rank on that node;
* the stat inbound and outbound traffic equal the simulated NIC bytes.

Each generation starts from empty cache bytes and zero counters and is checked
on its own, so a 2+0 fill split across generations fails.

A second sampler keeps the common shape of a real hybrid deployment: 2 or 4
nodes, 4 or 8 ranks per node, 2 or 4 KiB-class tensors, 4 or 8 steps, one or
two layers, file coordinator. It guards the sizes the uniform sampler never
reaches. GDTENSOR_STRESS_COMMON_SEEDS sets its seed count (default 8).

GDTENSOR_STRESS_SEEDS sets the seed count; GDTENSOR_STRESS_BASE sets the
first seed. A failure message prints the seed and the drawn case.
GDTENSOR_STRESS_REPORT names a file; each passed generation appends one JSON
line with the observed values next to the oracle values.
"""

import json
import os
import threading
import time

import numpy as np
import pytest

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.cluster_sim import Cluster, NodeCache, layer_targets, parent_of
from tests.support.crc import crc_of

SEEDS = int(os.environ.get("GDTENSOR_STRESS_SEEDS", "128"))
SEED_BASE = int(os.environ.get("GDTENSOR_STRESS_BASE", "0"))
COMMON_SEEDS = int(os.environ.get("GDTENSOR_STRESS_COMMON_SEEDS", "8"))
GUARD = 0xA5


# ---------------------------------------------------------------- draw


def _subset(rng, items) -> list:
    items = list(items)
    count = int(rng.integers(1, len(items) + 1))
    return sorted(int(x) for x in rng.choice(items, size=count, replace=False))


def _draw(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nodes = int(rng.integers(1, 5))
    big = _subset(rng, range(nodes))
    layers = [big] if rng.random() < 0.5 else [_subset(rng, big), big]
    sizes = [int(rng.integers(1, 33)) for _ in range(int(rng.integers(1, 4)))]
    return dict(
        seed=seed,
        k=int(rng.integers(1, 4)),
        nodes=nodes,
        layers=layers,
        steps=[int(rng.choice([1, 2, 3, 4, 8])) for _ in layers],
        width=[int(rng.integers(1, 4)) for _ in layers],
        sizes=sizes,
        replicate=[rng.random() < 0.3 for _ in sizes],
        coordinator=str(rng.choice(["file", "thread"])),
        bandwidth=float(rng.choice([0.0, 0.0, 0.0, 16384.0])),
        repeat=int(rng.integers(1, 3)),
    )


def _draw_common(seed: int) -> dict:
    """Draw a common deployment shape, above the uniform sampler's ceiling.

    Ranks and DNs share a node, so k stays above one. Tensors reach 256 to
    1024 bytes and steps reach 8, so one cache layer costs real bytes and
    the segment count passes the 32-byte, 8-step limit of `_draw`.
    """
    rng = np.random.default_rng(seed + 100_000)
    nodes = int(rng.choice([2, 4]))
    big = _subset(rng, range(nodes))
    layers = [big] if rng.random() < 0.15 else [_subset(rng, big), big]
    count = int(rng.integers(2, 5))
    sizes = [int(rng.choice([256, 512, 1024])) for _ in range(count)]
    return dict(
        seed=seed,
        k=int(rng.choice([4, 8])),
        nodes=nodes,
        layers=layers,
        steps=[int(rng.choice([4, 8])) for _ in layers],
        width=[int(rng.choice([1, 2, 3])) for _ in layers],
        sizes=sizes,
        replicate=[rng.random() < 0.2 for _ in sizes],
        coordinator=str(rng.choice(["file", "file", "file", "thread"])),
        bandwidth=float(rng.choice([0.0, 0.0, 0.0, 65536.0])),
        repeat=2,
    )


# ---------------------------------------------------------------- accessors


class FileStorage(ga.DTensorDataAccessor):
    """One packed storage file; a read lands on the calling rank node.

    The payload holds the tensors back to back in table order. A read maps
    its payload range back to (tensor id, tensor offset) independently of
    the planner, so a planner offset error reads wrong bytes or fails.
    """

    def __init__(self, cluster: Cluster, data: dict) -> None:
        self.cluster = cluster
        self.payload = np.concatenate(list(data.values()))
        self.spans, start = [], 0
        for tid, value in data.items():
            self.spans.append((start, start + value.size, tid))
            start += value.size
        self.reads = []  # (actor node, tid, offset, size)

    def _tensor_of(self, off: int, size: int):
        hits = [(lo, hi, tid) for lo, hi, tid in self.spans if lo <= off < hi]
        assert len(hits) == 1, f"storage read {off}+{size} outside payload"
        lo, hi, tid = hits[0]
        assert off + size <= hi, f"storage read {off}+{size} crosses tensors"
        return tid, off - lo

    def read_to_buffer(self, global_range, read_to_buffer):
        node = self.cluster.node_of_rank(self.cluster.rank)
        off, size = global_range.contiguous_payload_offset, global_range.size
        tid, toff = self._tensor_of(off, size)
        with self.cluster.lock:
            self.reads.append((node, tid, toff, size))
            self.cluster.storage_in[node] += size
        view = self.payload[off : off + size].copy()
        return read_to_buffer(view.ctypes.data, size)

    def write_from_buffer(self, global_range, write_from_buffer):
        return gb.Err(gb.GDTensorIOError(operation="write", msg="read-only storage"))


class RankBuffers(ga.DTensorDataAccessor):
    """Private compute memory per rank and tensor, preset to guard bytes."""

    def __init__(self, cluster: Cluster, sizes: dict, world: int) -> None:
        self.cluster = cluster
        self.buffers = {
            rank: {tid: np.full(n, GUARD, np.uint8) for tid, n in sizes.items()}
            for rank in range(world)
        }

    def _view(self, rng):
        off, size = rng.global_tensor_offset, rng.size
        return self.buffers[self.cluster.rank][rng.global_tensor_id][off : off + size]

    def write_from_buffer(self, global_range, write_from_buffer):
        return write_from_buffer(
            self._view(global_range).ctypes.data, global_range.size
        )

    def read_to_buffer(self, global_range, read_to_buffer):
        view = self._view(global_range).copy()
        return read_to_buffer(view.ctypes.data, global_range.size)


class _Load:
    def __init__(self, acc) -> None:
        self.acc = acc

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        return gb.Ok(self.acc)

    def finish(self):
        return gb.Ok(None)


# ---------------------------------------------------------------- views


def _storage_view(acc, tids, sizes, world):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([sizes[tid]], "U8", 1),
                    [gb.Replicate(), gb.Shard(0)],
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
                gb.ArrangedMesh(gt.WorldSize(world)),
            )
        ],
        list(tids),
    )
    entries = {gb.TensorComponentPos(t, r): group for t in tids for r in range(world)}
    layout = gb.PhysicalLayoutTable(
        "safetensors", entries, {gb.TensorComponentPos(tids[0], 0): "model.bin"}
    )

    def load(_view, _rank):
        return gb.Ok(_Load(acc))

    return gb.StorageDTensorsView(list(tids), table, acc, layout, load, load)


def _cache_view(acc, tids, sizes, steps, nodes, width):
    mesh = gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=gb.LiteralMesh(list(nodes), gt.NumNodes(max(nodes) + 1)),
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    named = [
        gb.NamedDTensor(
            tid,
            gb.NaiveDTensor(
                gb.GlobalTensorMeta([sizes[tid]], "U8", 1),
                [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
                mesh,
            ),
        )
        for tid in tids
    ]
    return gb.CacheDTensorsView(list(tids), gb.DTensorTable(named), acc)


def _compute_view(acc, tids, sizes, world, replicate):
    named = [
        gb.NamedDTensor(
            tid,
            gb.NaiveDTensor(
                gb.GlobalTensorMeta([sizes[tid]], "U8", 1),
                [gb.Replicate() if replicate[tid] else gb.Shard(0)],
                gb.ArrangedMesh(gt.WorldSize(world)),
            ),
        )
        for tid in tids
    ]
    return gb.ComputeDTensorsView(list(tids), gb.DTensorTable(named), acc)


def _build(case):
    world = case["k"] * case["nodes"]
    tids = [f"t{i}" for i in range(len(case["sizes"]))]
    sizes = dict(zip(tids, case["sizes"]))
    replicate = dict(zip(tids, case["replicate"]))
    cluster = Cluster(case["nodes"], case["k"], case["bandwidth"])
    rng = np.random.default_rng(case["seed"] + 10_000)
    data = {tid: rng.integers(0, 256, sizes[tid], dtype=np.uint8) for tid in tids}
    targets = layer_targets(cluster, case["layers"])
    caches = [NodeCache(cluster, sizes, target) for target in targets]
    src = FileStorage(cluster, data)
    dst = RankBuffers(cluster, sizes, world)
    views = [
        _cache_view(cache, tids, sizes, s, nodes, w)
        for cache, nodes, s, w in zip(
            caches, case["layers"], case["steps"], case["width"]
        )
    ]
    common = dict(
        src=_storage_view(src, tids, sizes, world),
        dst=_compute_view(dst, tids, sizes, world, replicate),
        num_ranks_per_node=case["k"],
    )
    if len(views) == 1:
        chain = gb.OneLayerCachedDTensorsReadChain(cache=views[0], **common)
    else:
        chain = gb.TwoLayerCachedDTensorsReadChain(
            small_cache=views[0], big_cache=views[1], **common
        )
    return dict(
        chain=chain,
        cluster=cluster,
        caches=caches,
        src=src,
        dst=dst,
        data=data,
        sizes=sizes,
        replicate=replicate,
        world=world,
        targets=targets,
    )


# ---------------------------------------------------------------- oracle


def _wants(size: int, world: int, replicate: bool, rank: int) -> tuple:
    if replicate:
        return 0, size
    chunk = -(-size // world)
    return min(size, rank * chunk), min(size, (rank + 1) * chunk)


def _empty(nodes, sizes) -> dict:
    return {
        n: {tid: np.zeros(size, bool) for tid, size in sizes.items()} for n in nodes
    }


def _merge(dst: dict, src: dict) -> None:
    for tid, mask in src.items():
        dst[tid] |= mask


def _demand(case, built) -> list:
    """Per layer, node -> tid -> demanded-byte mask (independent of gdtensor)."""
    layers, sizes = case["layers"], built["sizes"]
    serve = parent_of(list(range(case["nodes"])), layers[-1])
    last = _empty(layers[-1], sizes)
    for rank in range(built["world"]):
        _merge(last[serve[rank // case["k"]]], _rank_mask(case, built, rank))
    if len(layers) == 1:
        return [last]
    up = parent_of(layers[1], layers[0])
    first = _empty(layers[0], sizes)
    for child, want in last.items():
        _merge(first[up[child]], want)
    return [first, last]


def _rank_mask(case, built, rank) -> dict:
    out = {}
    for tid, size in built["sizes"].items():
        lo, hi = _wants(size, built["world"], built["replicate"][tid], rank)
        out[tid] = np.zeros(size, bool)
        out[tid][lo:hi] = True
    return out


def _counts(records, nodes, sizes) -> dict:
    counts = {
        n: {tid: np.zeros(size, np.int64) for tid, size in sizes.items()} for n in nodes
    }
    for node, tid, off, size in records:
        counts[node][tid][off : off + size] += 1
    return counts


# ---------------------------------------------------------------- run + check


def _run(case, built) -> None:
    world, cluster = built["world"], built["cluster"]
    rng = np.random.default_rng(case["seed"] + 20_000)
    order = rng.permutation(world).tolist()
    delay = rng.uniform(0.0, 0.004, world)
    results = {}

    def one(rank):
        time.sleep(float(delay[rank]))
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    threads = [threading.Thread(target=one, args=(r,), daemon=True) for r in order]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 60
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), ("chain_load hung", case)
    bad = {r: v.value for r, v in results.items() if type(v) is not gb.Ok}
    assert len(results) == world and not bad, (bad, case)


def _check_outputs(case, built) -> None:
    for rank, tensors in built["dst"].buffers.items():
        for tid, got in tensors.items():
            lo, hi = _wants(got.size, built["world"], built["replicate"][tid], rank)
            want = np.full(got.size, GUARD, np.uint8)
            want[lo:hi] = built["data"][tid][lo:hi]
            assert crc_of(got.tobytes()) == crc_of(want.tobytes()), (rank, tid, case)
            np.testing.assert_array_equal(got, want, err_msg=f"{rank} {tid} {case}")


def _check_cache_fills(case, built, demand) -> None:
    serve = built["targets"][-1]
    drains = [(r, t) for r, t, _s in built["caches"][-1].reads if t != serve(r)]
    assert not drains, ("drain off serve node", drains[:4], case)
    got = _drain_bytes(built)
    want = [_bytes(_rank_mask(case, built, r)) for r in range(built["world"])]
    assert got == want, ("drain bytes", got, want, case)
    for layer, cache in enumerate(built["caches"]):
        assert cache.wrong_node == 0, (layer, case)
        remote = [f for f in cache.fills if f[0] != f[1]]
        assert not remote, ("remote fill", layer, remote[:4], case)
        records = [(t, tid, o, s) for _a, t, tid, o, s in cache.fills]
        counts = _counts(records, case["layers"][layer], built["sizes"])
        _check_once(counts, demand[layer], ("cache", layer, case))


def _check_once(counts, demand, label) -> None:
    for node, tensors in demand.items():
        for tid, mask in tensors.items():
            want = mask.astype(np.int64)
            np.testing.assert_array_equal(
                counts[node][tid], want, err_msg=f"{label} node={node} tid={tid}"
            )


def _check_storage(case, built, demand) -> None:
    reads = built["src"].reads
    first = case["layers"][0]
    assert all(node in first for node, *_ in reads), ("storage read off DN", case)
    counts = _counts(reads, first, built["sizes"])
    _check_once(counts, demand[0], ("storage", case))


def _check_traffic(case, built) -> None:
    cluster = built["cluster"]
    rx, tx = cluster.nic_bytes()
    nodes = sorted(cluster.nics)
    observed_in = [rx[n] + cluster.storage_in[n] for n in nodes]
    stat_in = list(gio.chain_stat_inbound_traffic(built["chain"]))
    stat_out = list(gio.chain_stat_outbound_traffic(built["chain"]))
    assert stat_out == tx, (stat_out, tx, case)
    assert stat_in == observed_in, (stat_in, observed_in, case)


def _bytes(masks: dict) -> int:
    return int(sum(mask.sum() for mask in masks.values()))


def _drain_bytes(built) -> list:
    out = [0] * built["world"]
    for rank, _target, size in built["caches"][-1].reads:
        out[rank] += size
    return out


def _report(case, built, demand, gen) -> None:
    """Append one evidence line: observed values next to oracle values."""
    path = os.environ.get("GDTENSOR_STRESS_REPORT")
    if not path:
        return
    caches, cluster = built["caches"], built["cluster"]
    rx, tx = cluster.nic_bytes()
    row = dict(
        seed=case["seed"],
        gen=gen,
        ranks=built["world"],
        layers=len(case["layers"]),
        coordinator=case["coordinator"],
        fill_bytes=[sum(f[4] for f in c.fills) for c in caches],
        oracle_fill_bytes=[sum(map(_bytes, d.values())) for d in demand],
        drain_bytes=sum(_drain_bytes(built)),
        output_bytes=sum(
            _bytes(_rank_mask(case, built, r)) for r in range(built["world"])
        ),
        storage_bytes=sum(r[3] for r in built["src"].reads),
        oracle_storage_bytes=sum(map(_bytes, demand[0].values())),
        stat_out=list(gio.chain_stat_outbound_traffic(built["chain"])),
        nic_out=tx,
    )
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _reset_generation(built) -> None:
    """Clear outputs, cache bytes and counters, so each generation must refill."""
    for tensors in built["dst"].buffers.values():
        for buf in tensors.values():
            buf[:] = GUARD
    for cache in built["caches"]:
        cache.mem.clear()
        cache.fills.clear()
        cache.reads.clear()
    built["src"].reads.clear()
    cluster = built["cluster"]
    for nic in cluster.nics.values():
        nic.rx = nic.tx = 0
    for node in cluster.storage_in:
        cluster.storage_in[node] = 0


@pytest.mark.parametrize("seed", range(SEED_BASE, SEED_BASE + SEEDS))
def test_random_cache_strategy_is_correct(seed, tmp_path, monkeypatch):
    _verify_case(_draw(seed), tmp_path, monkeypatch)


@pytest.mark.parametrize("seed", range(COMMON_SEEDS))
def test_common_shape_stays_correct(seed, tmp_path, monkeypatch):
    _verify_case(_draw_common(seed), tmp_path, monkeypatch)


def _verify_case(case, tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", case["coordinator"])
    built = _build(case)
    assert type(built["chain"].check_constraints()) is gb.Ok, case
    demand = _demand(case, built)
    for gen in range(case["repeat"]):
        _reset_generation(built)
        _run(case, built)
        _check_outputs(case, built)
        _check_cache_fills(case, built, demand)
        _check_storage(case, built, demand)
        _check_traffic(case, built)
        _report(case, built, demand, gen)


@pytest.fixture(
    params=[
        pytest.param(("file", [[0, 2]]), id="file-one-layer"),
        pytest.param(("thread", [[0, 2]]), id="thread-one-layer"),
        pytest.param(("file", [[0], [0, 2]]), id="file-two-layers"),
        pytest.param(("thread", [[0], [0, 2]]), id="thread-two-layers"),
    ]
)
def changed_payload_generation(request, tmp_path, monkeypatch):
    coordinator, layers = request.param
    case = dict(
        seed=0,
        k=2,
        nodes=3,
        layers=layers,
        steps=[3] * len(layers),
        width=[2] * len(layers),
        sizes=[17, 9],
        replicate=[False, True],
        coordinator=coordinator,
        bandwidth=0.0,
        repeat=2,
    )
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", coordinator)
    built = _build(case)
    assert type(built["chain"].check_constraints()) is gb.Ok, case
    demand = _demand(case, built)
    _reset_generation(built)
    _run(case, built)
    _check_outputs(case, built)
    _check_cache_fills(case, built, demand)
    _check_storage(case, built, demand)
    _check_traffic(case, built)

    # Change every byte in storage and in the separate oracle, on the same chain.
    old_data = built["data"]
    built["src"].payload ^= np.uint8(0xFF)
    built["data"] = {tid: value ^ np.uint8(0xFF) for tid, value in old_data.items()}
    _reset_generation(built)
    return case, built, demand, old_data


def test_second_generation_reads_changed_payload(changed_payload_generation):
    case, built, demand, _old_data = changed_payload_generation
    _run(case, built)
    _check_outputs(case, built)
    _check_cache_fills(case, built, demand)
    _check_storage(case, built, demand)
    _check_traffic(case, built)


def test_second_generation_rejects_stale_last_layer(
    changed_payload_generation, monkeypatch
):
    case, built, demand, old_data = changed_payload_generation
    cache = built["caches"][-1]
    write = cache.write_from_buffer

    def stale_write(global_range, write_from_buffer):
        result = write(global_range, write_from_buffer)
        if type(result) is not gb.Ok:
            return result
        _actor, target = cache._ends(global_range)
        tid = global_range.global_tensor_id
        off, size = global_range.global_tensor_offset, global_range.size
        with cache.lock:
            data, _valid = cache._buf(target, tid)
            data[off : off + size] = old_data[tid][off : off + size]
        return result

    monkeypatch.setattr(cache, "write_from_buffer", stale_write)
    _run(case, built)
    _check_cache_fills(case, built, demand)
    _check_storage(case, built, demand)
    _check_traffic(case, built)
    with pytest.raises(AssertionError):
        _check_outputs(case, built)


def test_draw_covers_every_strategy_axis():
    cases = [_draw(s) for s in range(SEED_BASE, SEED_BASE + SEEDS)]
    assert {len(c["layers"]) for c in cases} == {1, 2}
    assert {c["coordinator"] for c in cases} == {"file", "thread"}
    assert {c["repeat"] for c in cases} == {1, 2}
    assert {c["k"] for c in cases} == {1, 2, 3}
    assert any(any(c["replicate"]) for c in cases)
    assert any(c["bandwidth"] > 0 for c in cases)
    assert any(len(c["layers"][-1]) < c["nodes"] for c in cases)
    assert any(max(c["steps"]) * max(c["width"]) > min(c["sizes"]) for c in cases)

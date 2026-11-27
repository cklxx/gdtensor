"""Randomized stress for the co-located cache write chain (`gio.chain_save`).

Each seed draws a legal compute -> cache -> storage deployment: ranks per node,
node count, 1 or 2 cache layers (`first_cache`, then optional `second_cache`), cache
node subsets, temporal steps, concurrency width, 1..3 tensors with non-divisible
sizes, mixed dtypes (`U8`, `I32`), 1D and 2D compute placements (`Shard`,
`Replicate`), `file`/`thread` coordinator, start order/delay, NIC bandwidth, and
repeated generations. Payloads are random bytes checked by exact bytes and CRC.

An independent oracle derives designated writers, owned byte masks, per-layer
supply masks, peak budgets, and expected network/storage traffic without calling
the gdtensor planner. It verifies:
* written storage bytes match ground truth with matching CRC and intact guards;
* every storage byte is written once by a node in the terminal cache layer;
* every cache fill/read addresses valid node-local bytes (`wrong_node == 0`);
* every cache layer receives each supplied byte once per generation;
* resident bytes never exceed segment budgets and sum to `<= mem_overhead`;
* zero resident bytes and empty slot tables remain after each generation;
* simulated NIC rx/tx and storage writes match inbound/outbound stats.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

import numpy as np
import pytest

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.cluster_sim import Cluster, NodeCache, parent_of
from tests.support.crc import crc_of


GUARD = 0xA5
SENTINEL_BYTES = 16
SEEDS = int(os.environ.get("GDTENSOR_WRITE_STRESS_SEEDS", "48"))
SEED_BASE = int(os.environ.get("GDTENSOR_WRITE_STRESS_BASE", "0"))
COMMON_SEEDS = int(os.environ.get("GDTENSOR_WRITE_STRESS_COMMON_SEEDS", "4"))
FANIN_SEEDS = int(os.environ.get("GDTENSOR_WRITE_STRESS_FANIN_SEEDS", "12"))


# ---------------------------------------------------------------- accessors


class WriteFileStorage(ga.DTensorDataAccessor):
    """Packed simulated storage file with trailing guard sentinel bytes."""

    def __init__(self, cluster: Cluster, sizes: dict[str, int]) -> None:
        self.cluster = cluster
        self.total = sum(sizes.values())
        self.payload = np.full(self.total + SENTINEL_BYTES, GUARD, dtype=np.uint8)
        self.spans: list[tuple[int, int, str]] = []
        start = 0
        for tid, nbytes in sizes.items():
            self.spans.append((start, start + nbytes, tid))
            start += nbytes
        self.writes: list[tuple[int, str, int, int]] = []
        self.storage_out = {n: 0 for n in cluster.nics}

    def reset_payload(self) -> None:
        self.payload.fill(GUARD)

    def _tensor_of(self, off: int, size: int) -> tuple[str, int]:
        hits = [(lo, hi, tid) for lo, hi, tid in self.spans if lo <= off < hi]
        assert len(hits) == 1, f"storage write {off}+{size} outside payload"
        lo, hi, tid = hits[0]
        assert off + size <= hi, f"storage write {off}+{size} crosses tensors"
        return tid, off - lo

    def tensor_bytes(self, tid: str) -> np.ndarray:
        lo, hi = next((lo, hi) for lo, hi, name in self.spans if name == tid)
        return self.payload[lo:hi]

    def guards_intact(self) -> bool:
        return bool(np.all(self.payload[self.total :] == GUARD))

    def read_to_buffer(self, global_range, read_to_buffer):
        return gb.Err(gb.GDTensorIOError(operation="read", msg="write-only storage"))

    def write_from_buffer(self, global_range, write_from_buffer):
        off, size = global_range.contiguous_payload_offset, global_range.size
        buf = np.empty(size, dtype=np.uint8)
        res = write_from_buffer(buf.ctypes.data, size)
        if type(res) is not gb.Ok:
            return res
        tid, toff = self._tensor_of(off, size)
        node = self.cluster.node_of_rank(self.cluster.rank)
        with self.cluster.lock:
            self.payload[off : off + size] = buf
            self.writes.append((node, tid, toff, size))
            self.storage_out[node] += size
        return res


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


class WriteRankBuffers(ga.DTensorDataAccessor):
    """Per-rank compute buffers: owned bytes from ground truth, rest GUARD."""

    def __init__(
        self,
        cluster: Cluster,
        data: dict[str, np.ndarray],
        owned: dict[int, dict[str, np.ndarray]],
        world: int,
    ) -> None:
        self.cluster = cluster
        self.owned = owned
        self.buffers: dict[int, dict[str, np.ndarray]] = {}
        for rank in range(world):
            self.buffers[rank] = _init_rank_tensors(data, owned[rank])

    def read_to_buffer(self, global_range, read_to_buffer):
        rank = self.cluster.rank
        tid, off, size = (
            global_range.global_tensor_id,
            global_range.global_tensor_offset,
            global_range.size,
        )
        assert self.owned[rank][tid][off : off + size].all(), (
            f"rank {rank} read unowned or deduped bytes {tid}@{off}+{size}"
        )
        view = self.buffers[rank][tid][off : off + size].copy()
        return read_to_buffer(view.ctypes.data, size)

    def write_from_buffer(self, global_range, write_from_buffer):
        return gb.Err(gb.GDTensorIOError(operation="write", msg="read-only compute"))


def _init_rank_tensors(
    data: dict[str, np.ndarray], masks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tid, raw in data.items():
        arr = np.full(raw.size, GUARD, dtype=np.uint8)
        arr[masks[tid]] = raw[masks[tid]]
        out[tid] = arr
    return out


# ---------------------------------------------------------------- samplers


def _subset(rng, items) -> list[int]:
    pool = list(items)
    count = int(rng.integers(1, len(pool) + 1))
    return sorted(int(x) for x in rng.choice(pool, size=count, replace=False))


def _subset_size(rng, items, count: int) -> list[int]:
    return sorted(int(x) for x in rng.choice(list(items), size=count, replace=False))


def _factor_2d(world: int, rng) -> tuple[int, int]:
    pairs = [(a, world // a) for a in range(1, world + 1) if world % a == 0]
    return pairs[int(rng.integers(0, len(pairs)))]


def _draw_tensor_spec(rng, world: int, force_1d: bool = False) -> dict[str, Any]:
    dtype, itemsize = ("I32", 4) if rng.random() < 0.4 else ("U8", 1)
    if not force_1d and world >= 2 and rng.random() < 0.4:
        shape = [int(rng.integers(1, 9)), int(rng.integers(1, 9))]
        kind = str(rng.choice(["shard2d", "rep0_shard1", "shard0_rep1"]))
        return dict(shape=shape, dtype=dtype, itemsize=itemsize, placement=kind)
    shape = [int(rng.integers(1, 34))]
    kind = "replicate" if rng.random() < 0.3 else "shard0"
    return dict(shape=shape, dtype=dtype, itemsize=itemsize, placement=kind)


def _draw(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nodes = int(rng.integers(1, 5))
    k = int(rng.integers(1, 4))
    world = nodes * k
    big = _subset(rng, range(nodes))
    layers = [big] if rng.random() < 0.5 else [big, _subset(rng, big)]
    specs = [_draw_tensor_spec(rng, world) for _ in range(int(rng.integers(1, 4)))]
    return dict(
        seed=seed,
        k=k,
        nodes=nodes,
        layers=layers,
        steps=[int(rng.choice([1, 2, 3, 4, 8])) for _ in layers],
        width=[int(rng.choice([1, 2, 3])) for _ in layers],
        mesh2d=_factor_2d(world, rng),
        tensors=specs,
        coordinator=str(rng.choice(["file", "thread"])),
        bandwidth=float(rng.choice([0.0, 0.0, 0.0, 16384.0])),
        repeat=int(rng.integers(1, 3)),
    )


def _draw_common(seed: int) -> dict[str, Any]:
    """Larger hybrid cluster shape (up to 32 ranks, 256..1024 byte tensors)."""
    rng = np.random.default_rng(seed + 100_000)
    nodes = int(rng.choice([2, 4]))
    k = int(rng.choice([4, 8]))
    big = _subset(rng, range(nodes))
    layers = [big] if rng.random() < 0.15 else [big, _subset(rng, big)]
    count = int(rng.integers(2, 5))
    specs = [
        dict(
            shape=[int(rng.choice([256, 512, 1024]))],
            dtype="U8",
            itemsize=1,
            placement="replicate" if rng.random() < 0.2 else "shard0",
        )
        for _ in range(count)
    ]
    return dict(
        seed=seed,
        k=k,
        nodes=nodes,
        layers=layers,
        steps=[int(rng.choice([4, 8])) for _ in layers],
        width=[int(rng.choice([1, 2, 3])) for _ in layers],
        mesh2d=_factor_2d(nodes * k, rng),
        tensors=specs,
        coordinator=str(rng.choice(["file", "file", "file", "thread"])),
        bandwidth=float(rng.choice([0.0, 0.0, 0.0, 65536.0])),
        repeat=2,
    )


def _draw_fanin(seed: int) -> dict[str, Any]:
    """Two-layer fan-in shape: all nodes in L0 converging onto subset L1."""
    rng = np.random.default_rng(seed + 200_000)
    nodes = int(rng.choice([2, 3, 4]))
    small = _subset_size(rng, range(nodes), int(rng.integers(1, nodes)))
    count = int(rng.integers(2, 5))
    specs = [
        dict(
            shape=[int(rng.choice([64, 128, 256, 512]))],
            dtype="U8",
            itemsize=1,
            placement="shard0",
        )
        for _ in range(count)
    ]
    return dict(
        seed=seed,
        k=1,
        nodes=nodes,
        layers=[list(range(nodes)), small],
        steps=[int(rng.choice([1, 2, 4])), int(rng.choice([1, 2, 4]))],
        width=[int(rng.choice([1, 2, 3])) for _ in range(2)],
        mesh2d=(nodes, 1),
        tensors=specs,
        coordinator=str(rng.choice(["file", "thread"])),
        bandwidth=0.0,
        repeat=2,
    )


# ---------------------------------------------------------------- oracle + build


def _chunk(extent: int, parts: int, coord: int) -> tuple[int, int]:
    step = -(-extent // parts)
    return min(extent, coord * step), min(extent, (coord + 1) * step)


def _placement_info(spec: dict[str, Any], world: int, mesh2d: tuple[int, int]):
    kind = spec["placement"]
    if kind == "replicate":
        return [gb.Replicate()], (world,)
    if kind == "shard0":
        return [gb.Shard(0)], (world,)
    if kind == "shard2d":
        return [gb.Shard(0), gb.Shard(1)], mesh2d
    if kind == "rep0_shard1":
        return [gb.Replicate(), gb.Shard(1)], mesh2d
    return [gb.Shard(0), gb.Replicate()], mesh2d


def _compute_mesh_for(shape: tuple[int, ...], world: int):
    if len(shape) == 1:
        return gb.ArrangedMesh(gt.WorldSize(world))
    return gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(world)), list(shape))


def _rank_owned_mask(
    spec: dict[str, Any], order: int, rank: int, world: int, mesh2d: tuple[int, int]
) -> np.ndarray:
    shape = spec["shape"]
    itemsize = spec["itemsize"]
    nbytes = int(np.prod(shape)) * itemsize
    mask = np.zeros(nbytes, dtype=bool)
    placements, mshape = _placement_info(spec, world, mesh2d)
    coords = (rank,) if len(mshape) == 1 else (rank // mshape[1], rank % mshape[1])
    for dim, p in enumerate(placements):
        if (
            type(p) is gb.Replicate
            and mshape[dim] > 1
            and coords[dim] != order % mshape[dim]
        ):
            return mask
    lo, hi = [0] * len(shape), list(shape)
    for dim, p in enumerate(placements):
        if type(p) is gb.Shard:
            a, b = _chunk(hi[p.axis] - lo[p.axis], mshape[dim], coords[dim])
            lo[p.axis], hi[p.axis] = lo[p.axis] + a, lo[p.axis] + b
    box = tuple(slice(lo[i], hi[i]) for i in range(len(shape))) + (slice(None),)
    mask.reshape(*shape, itemsize)[box] = True
    return mask


def _write_layer_targets(cluster: Cluster, layers: list[list[int]]) -> list:
    """Oracle mapping rank -> target DN node for each write cache layer."""
    serve = parent_of(sorted(cluster.nics), layers[0])
    if len(layers) == 1:
        return [lambda r, s=serve, k=cluster.k: s[r // k]]
    up = parent_of(layers[0], layers[1])
    return [
        lambda r, s=serve, k=cluster.k: s[r // k],
        lambda r, s=serve, u=up, k=cluster.k: u[s[r // k]],
    ]


def _storage_view(acc, tids, specs, world):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta(spec["shape"], spec["dtype"], spec["itemsize"]),
                    [gb.Replicate(), gb.Shard(0)],
                    mesh,
                ),
            )
            for tid, spec in zip(tids, specs)
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
    session_cb = lambda _v, _r: gb.Ok(_WriteSession(acc))
    return gb.StorageDTensorsView(
        list(tids),
        table,
        acc,
        layout,
        begin_save=session_cb,
        begin_load=session_cb,
    )


def _cache_view(acc, tids, specs, steps, nodes, width):
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
                    gb.GlobalTensorMeta(spec["shape"], spec["dtype"], spec["itemsize"]),
                    [gb.Shard(0), gb.Shard(0), gb.Shard(0)],
                    mesh,
                ),
            )
            for tid, spec in zip(tids, specs)
        ]
    )
    return gb.CacheDTensorsView(list(tids), table, acc)


def _compute_view(acc, tids, specs, world, mesh2d):
    named = []
    for tid, spec in zip(tids, specs):
        placements, mshape = _placement_info(spec, world, mesh2d)
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta(spec["shape"], spec["dtype"], spec["itemsize"]),
            placements,
            _compute_mesh_for(mshape, world),
        )
        named.append(gb.NamedDTensor(tid, dt))
    return gb.ComputeDTensorsView(list(tids), gb.DTensorTable(named), acc)


def _build(case: dict[str, Any]) -> dict[str, Any]:
    world = case["k"] * case["nodes"]
    specs = case["tensors"]
    tids = [f"t{i}" for i in range(len(specs))]
    sizes = {
        tid: int(np.prod(s["shape"])) * s["itemsize"] for tid, s in zip(tids, specs)
    }
    rng = np.random.default_rng(case["seed"] + 10_000)
    data = {tid: rng.integers(0, 256, sizes[tid], dtype=np.uint8) for tid in tids}
    owned = {
        rank: {
            tid: _rank_owned_mask(spec, order, rank, world, case["mesh2d"])
            for order, (tid, spec) in enumerate(zip(tids, specs))
        }
        for rank in range(world)
    }
    cluster = Cluster(case["nodes"], case["k"], case["bandwidth"])
    targets = _write_layer_targets(cluster, case["layers"])
    caches = [NodeCache(cluster, sizes, target) for target in targets]
    dst = WriteFileStorage(cluster, sizes)
    src = WriteRankBuffers(cluster, data, owned, world)
    views = [
        _cache_view(c, tids, specs, s, n, w)
        for c, n, s, w in zip(caches, case["layers"], case["steps"], case["width"])
    ]
    common = dict(
        src=_compute_view(src, tids, specs, world, case["mesh2d"]),
        dst=_storage_view(dst, tids, specs, world),
        num_ranks_per_node=case["k"],
    )
    chain = (
        gb.OneLayerCachedDTensorsRWChain(cache=views[0], **common)
        if len(views) == 1
        else gb.TwoLayerCachedDTensorsRWChain(
            first_cache=views[0], second_cache=views[1], **common
        )
    )
    return dict(
        chain=chain,
        cluster=cluster,
        caches=caches,
        src=src,
        dst=dst,
        data=data,
        owned=owned,
        sizes=sizes,
        world=world,
        targets=targets,
    )


def _node_shard_mask(
    spec: dict[str, Any], steps: int, nodes: list[int], idx: int
) -> np.ndarray:
    shape, itemsize = spec["shape"], spec["itemsize"]
    nbytes = int(np.prod(shape)) * itemsize
    mask = np.zeros(nbytes, dtype=bool)
    view = mask.reshape(*shape, itemsize)
    rest = (slice(None),) * len(shape)
    for step in range(steps):
        t_lo, t_hi = _chunk(shape[0], steps, step)
        s_lo, s_hi = _chunk(t_hi - t_lo, len(nodes), idx)
        view[(slice(t_lo + s_lo, t_lo + s_hi),) + rest] = True
    return mask


def _supply_oracle(case: dict[str, Any], built: dict[str, Any]) -> list[dict]:
    tids = [f"t{i}" for i in range(len(case["tensors"]))]
    total = {tid: np.zeros(sz, bool) for tid, sz in built["sizes"].items()}
    for mask_map in built["owned"].values():
        for tid, mask in mask_map.items():
            total[tid] |= mask
    out = []
    for nodes, steps in zip(case["layers"], case["steps"]):
        layer_supply = {
            node: {
                tid: total[tid] & _node_shard_mask(spec, steps, nodes, idx)
                for tid, spec in zip(tids, case["tensors"])
            }
            for idx, node in enumerate(nodes)
        }
        out.append(layer_supply)
    return out


# ---------------------------------------------------------------- run + check


def _counts(records, nodes, sizes) -> dict:
    out = {n: {tid: np.zeros(sz, np.int64) for tid, sz in sizes.items()} for n in nodes}
    for node, tid, off, size in records:
        out[node][tid][off : off + size] += 1
    return out


def _check_once(counts, supply, label) -> None:
    for node, tensors in supply.items():
        for tid, mask in tensors.items():
            np.testing.assert_array_equal(
                counts[node][tid],
                mask.astype(np.int64),
                err_msg=f"{label} n={node} {tid}",
            )


def _run_round(case, built, monkeypatch) -> dict[tuple[int, int], int]:
    internal_wc = sys.modules[gio.chain_save.__module__]
    orig_admit = internal_wc._WriteNodeState.admit
    peaks: dict[tuple[int, int], int] = {}

    def checked_admit(state, seg):
        ok = orig_admit(state, seg)
        assert state.resident <= state.budget, (
            seg.layer,
            seg.node,
            state.resident,
            state.budget,
        )
        if ok:
            key = (seg.layer, seg.node)
            peaks[key] = max(peaks.get(key, 0), state.resident)
        return ok

    monkeypatch.setattr(internal_wc._WriteNodeState, "admit", checked_admit)
    world, cluster = built["world"], built["cluster"]
    rng = np.random.default_rng(case["seed"] + 20_000)
    results: dict[int, Any] = {}
    delay = rng.uniform(0.0, 0.004, world)

    def one(rank: int) -> None:
        time.sleep(float(delay[rank]))
        cluster.rank = rank
        results[rank] = gio.chain_save(rank, built["chain"])

    threads = [
        threading.Thread(target=one, args=(r,), daemon=True)
        for r in rng.permutation(world)
    ]
    for th in threads:
        th.start()
    deadline = time.monotonic() + 60.0
    for th in threads:
        th.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(th.is_alive() for th in threads), ("chain_save hung", case)
    bad = {r: v.value for r, v in results.items() if type(v) is not gb.Ok}
    assert len(results) == world and not bad, (bad, case)
    return peaks


def _check_outputs(case, built) -> None:
    dst = built["dst"]
    assert dst.guards_intact(), ("storage guard corrupted", case)
    for tid, want in built["data"].items():
        got = dst.tensor_bytes(tid)
        assert crc_of(got.tobytes()) == crc_of(want.tobytes()), (tid, case)
        np.testing.assert_array_equal(got, want, err_msg=f"{tid} {case}")


def _check_storage(case, built, supply) -> None:
    last = case["layers"][-1]
    writes = built["dst"].writes
    assert all(node in last for node, *_ in writes), (
        "storage write off terminal DN",
        case,
    )
    counts = _counts(writes, last, built["sizes"])
    _check_once(counts, supply[-1], ("storage", case))


def _check_cache_writes(case, built, supply) -> None:
    for layer, cache in enumerate(built["caches"]):
        assert cache.wrong_node == 0, (layer, case)
        records = [(t, tid, o, s) for _a, t, tid, o, s in cache.fills]
        counts = _counts(records, case["layers"][layer], built["sizes"])
        _check_once(counts, supply[layer], ("cache", layer, case))


def _check_traffic(case, built) -> None:
    cluster = built["cluster"]
    rx, tx = cluster.nic_bytes()
    nodes = sorted(cluster.nics)
    obs_out = [tx[n] + built["dst"].storage_out[n] for n in nodes]
    stat_in = list(gio.chain_stat_inbound_traffic(built["chain"]))
    stat_out = list(gio.chain_stat_outbound_traffic(built["chain"]))
    assert stat_in == rx, (stat_in, rx, case)
    assert stat_out == obs_out, (stat_out, obs_out, case)


def _check_budget_and_released(case, built, peaks, generation) -> None:
    stat_mem = list(gio.chain_stat_mem_overhead(built["chain"]))
    for node in range(case["nodes"]):
        cap = stat_mem[node] if node < len(stat_mem) else 0
        obs = sum(peaks.get((layer, node), 0) for layer in range(len(case["layers"])))
        assert obs <= cap, (node, obs, cap, case)
    internal_wc = sys.modules[gio.chain_save.__module__]
    active = internal_wc._fp_entry(built["chain"])
    try:
        table = internal_wc._write_slots(built["chain"], active)
    finally:
        internal_wc._fp_release(id(built["chain"]), active[0])
    with table.cond:
        left = {
            ln: (st.resident, dict(st.slots))
            for ln, st in table.nodes.items()
            if st.slots or st.resident
        }
    assert not left, (
        "resident or slots leaked after write round",
        generation,
        left,
        case,
    )


def _check_fanin(case, built, supply) -> None:
    """Verify 2-layer L0 -> L1 spatial-Shard fan-in structure from observed transfers."""
    l0, l1 = case["layers"]
    assert l0 == list(range(case["nodes"])) and len(l1) < len(l0)
    want_edges = {
        (n0, n1)
        for n0 in l0
        for n1 in l1
        if any(
            (supply[0][n0][tid] & supply[1][n1][tid]).any() for tid in built["sizes"]
        )
    }
    edges = {(actor, target) for actor, target, *_ in built["caches"][1].fills}
    assert all(actor in l0 and target in l1 for actor, target in edges), (edges, case)
    assert edges == want_edges, (edges, want_edges, case)
    by_parent: dict[int, set[int]] = {}
    for child, target in edges:
        by_parent.setdefault(target, set()).add(child)
    assert set(by_parent) == set(l1) and max(len(v) for v in by_parent.values()) >= 2


def _reset_generation(built) -> None:
    built["dst"].reset_payload()
    built["dst"].writes.clear()
    for node in built["dst"].storage_out:
        built["dst"].storage_out[node] = 0
    for cache in built["caches"]:
        cache.mem.clear()
        cache.fills.clear()
        cache.reads.clear()
    for nic in built["cluster"].nics.values():
        nic.rx = nic.tx = 0


def _verify_case(case, tmp_path, monkeypatch, check_fanin: bool = False) -> None:
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "15")
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", case["coordinator"])
    built = _build(case)
    assert type(built["chain"].check_constraints()) is gb.Ok, case
    supply = _supply_oracle(case, built)
    for gen in range(case["repeat"]):
        _reset_generation(built)
        peaks = _run_round(case, built, monkeypatch)
        _check_outputs(case, built)
        _check_cache_writes(case, built, supply)
        _check_storage(case, built, supply)
        _check_traffic(case, built)
        _check_budget_and_released(case, built, peaks, gen)
        if check_fanin:
            _check_fanin(case, built, supply)


# ---------------------------------------------------------------- tests


@pytest.mark.parametrize("seed", range(SEED_BASE, SEED_BASE + SEEDS))
def test_random_write_chain_strategy_is_correct(seed, tmp_path, monkeypatch):
    _verify_case(_draw(seed), tmp_path, monkeypatch)


@pytest.mark.parametrize("seed", range(COMMON_SEEDS))
def test_common_write_shape_stays_correct(seed, tmp_path, monkeypatch):
    _verify_case(_draw_common(seed), tmp_path, monkeypatch)


@pytest.mark.parametrize("seed", range(FANIN_SEEDS))
def test_two_layer_write_fan_in_is_correct(seed, tmp_path, monkeypatch):
    _verify_case(_draw_fanin(seed), tmp_path, monkeypatch, check_fanin=True)


def test_draw_covers_every_write_strategy_axis():
    cases = [_draw(s) for s in range(SEED_BASE, SEED_BASE + SEEDS)]
    assert {len(c["layers"]) for c in cases} == {1, 2}
    assert {c["coordinator"] for c in cases} == {"file", "thread"}
    assert {c["repeat"] for c in cases} == {1, 2}
    assert {c["k"] for c in cases} == {1, 2, 3}
    assert any(any(t["placement"] == "replicate" for t in c["tensors"]) for c in cases)
    assert any(any(t["dtype"] == "I32" for t in c["tensors"]) for c in cases)
    assert any(any(len(t["shape"]) == 2 for t in c["tensors"]) for c in cases)
    assert any(c["bandwidth"] > 0 for c in cases)
    assert any(len(c["layers"][0]) < c["nodes"] for c in cases)
    assert any(
        len(c["layers"]) == 2 and len(c["layers"][1]) < len(c["layers"][0])
        for c in cases
    )
    assert any(
        max(c["steps"]) * max(c["width"]) > min(t["shape"][0] for t in c["tensors"])
        for c in cases
    )

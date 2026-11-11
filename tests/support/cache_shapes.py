"""Shared cluster shapes for cache benchmark tests.

Storage payload, compute buffers, the three views and the independent
demand oracle live here so more than one test module can build them.
"""

import hashlib
from typing import Any

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.tag as gt
from tests.support.cluster_sim import Cluster, NodeCache, layer_targets, parent_of

GUARD = 0xA5


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

    def cache_identity(self) -> object | None:
        # Memory fixture identity changes with its payload, not with a test epoch.
        return hashlib.sha256(self.payload.tobytes()).hexdigest()

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


def _build(case) -> dict[str, Any]:
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


def _bytes(masks: dict) -> int:
    return int(sum(mask.sum() for mask in masks.values()))

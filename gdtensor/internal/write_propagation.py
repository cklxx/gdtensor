"""Node-level propagation plan for cached RW write chains (spatial: Shard).

For ``Compute -> first_cache -> second_cache -> Storage`` writes, every cache
layer partitions tensors across its nodes via ``spatial: Shard``. Compute
supplies from designated writers route directly to the cache nodes owning
each byte interval, coalesce along contiguous (step, concurrency) cells, and
converge through downstream Shard layers before flushing to storage.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .dtensor import (
    box_of,
    box_to_intervals,
    global_tensor_meta_of,
    is_designated_writer,
    iv_intersect,
    naive_of,
    named_of,
    order_of,
    owned_intervals,
)
from .mesh import (
    cache_concurrency_width,
    cache_mesh_permutation,
    cache_num_nodes,
    cache_num_steps,
)
from .plan import (
    CHECK,
    _Side,
    _assert_disjoint,
    _check_bounds,
    _save_pieces,
    is_storage,
)
from .propagation import (
    _crange,
    _dst_ranks,
    _hop,
    _merge_pending,
    _split,
    _view_nodes,
    _view_width,
)
from .result import check_failed, invalid

WriteSegKey = Tuple[str, int, int]  # (tensor id, byte offset, byte size)


@dataclass
class WriteSegment:
    layer: int
    node: int
    tid: str
    off: int
    size: int
    cell: tuple
    # producer -> [(global_off, size)]; producer is ("rank", r) on layer 0
    # or ("seg", layer - 1, node, key) on layer > 0
    producers: dict = field(default_factory=dict)
    # layer < last: [(tnode, dst_key, global_off, size)] on the target layer
    children: list = field(default_factory=list)
    # layer == last: [(storage range, byte offset inside this segment)]
    storage: list = field(default_factory=list)

    @property
    def key(self) -> WriteSegKey:
        return (self.tid, self.off, self.size)

    def expected(self) -> set:
        """Every (producer, offset, size) piece that must stage before flush."""
        return {
            (producer, off, size)
            for producer, pieces in self.producers.items()
            for off, size in pieces
        }


@dataclass
class WriteChainPlan:
    tids: List[List[str]]
    k: int
    layer_nodes: List[List[int]]
    targets: List[Dict[int, Tuple[int, ...]]]
    ranks: List[int]
    segments: List[Dict[int, Dict[WriteSegKey, WriteSegment]]] = field(
        default_factory=list
    )
    budget: List[Dict[int, int]] = field(default_factory=list)
    supply: List[Dict[int, int]] = field(default_factory=list)
    stages: Dict[int, list] = field(default_factory=dict)
    cohort: List[Dict[int, set]] = field(default_factory=list)
    flush_ranks: Dict[tuple, tuple] = field(default_factory=dict)
    services: Dict[int, list] = field(default_factory=dict)


def write_cache_layers(chain) -> list:
    from .chain import OneLayerCachedDTensorsRWChain

    if type(chain) is OneLayerCachedDTensorsRWChain:
        return [chain.cache]
    return [chain.first_cache, chain.second_cache]


def build_write_chain_plan(chain, with_storage: bool = True) -> WriteChainPlan:
    """Node-level write aggregation plan with spatial Shard ownership routing."""
    k = chain.num_ranks_per_node
    if k <= 0:
        raise invalid("num_ranks_per_node", "must be positive")
    caches = write_cache_layers(chain)
    layer_nodes = [_view_nodes(view) for view in caches]
    ranks = _dst_ranks(chain.src)
    targets: List[Dict[int, Tuple[int, ...]]] = [
        {n: () for n in nodes} for nodes in layer_nodes
    ]
    plan = WriteChainPlan(
        [list(view.tensor_ids) for view in caches],
        k,
        layer_nodes,
        targets,
        ranks,
    )
    supplies = {rank: _rank_supplies(chain.src, rank) for rank in plan.ranks}
    total_supply = _total_supply(plan, supplies)
    side = _Side(chain.dst) if with_storage else None
    _build_write_segments(plan, caches, total_supply, side)
    plan.budget = [
        {n: _write_budget(view, segs) for n, segs in plan.segments[layer].items()}
        for layer, view in enumerate(caches)
    ]
    plan.supply = [
        {
            n: sum(seg.size for seg in segs.values())
            for n, segs in plan.segments[layer].items()
        }
        for layer in range(len(caches))
    ]
    _link_write_plan(plan, chain, supplies, side)
    return plan


def _build_write_segments(
    plan: WriteChainPlan, caches: list, total_supply: Dict[str, np.ndarray], side
) -> None:
    last = len(caches) - 1
    plan.segments = [{} for _ in caches]
    for layer in range(last, -1, -1):
        view = caches[layer]
        cuts = _write_cuts(plan, side, layer, last)
        plan.segments[layer] = {
            n: _layer_write_segments(view, layer, n, total_supply, cuts)
            for n in plan.layer_nodes[layer]
        }


def _link_write_plan(plan: WriteChainPlan, chain, supplies, side) -> None:
    last = len(plan.tids) - 1
    if side is not None:
        _link_write_storage(plan, chain.dst, side)
    for layer in range(last):
        _link_write_children(plan, layer)
    _link_stages(plan, chain.src, supplies)
    closures = _link_flush_ranks(plan)
    _link_write_cohorts(plan, closures)


def _rank_supplies(src, rank: int) -> Dict[str, np.ndarray]:
    supplies = {}
    table = src.d_tensor_table
    for tid in src.tensor_ids:
        nd = named_of(table, tid)
        if not is_designated_writer(nd, rank, order_of(table, tid)):
            continue
        iv = owned_intervals(nd, "compute", rank)
        if iv.size:
            supplies[tid] = iv
    return supplies


def _total_supply(
    plan: WriteChainPlan, supplies: Dict[int, Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    pending: Dict[str, list] = {}
    for rank in plan.ranks:
        _ = [
            pending.setdefault(tid, []).append(iv) for tid, iv in supplies[rank].items()
        ]
    return _merge_pending(pending)


def _write_cuts(plan: WriteChainPlan, side, layer: int, last: int) -> Dict[str, set]:
    cuts: Dict[str, set] = {}
    if layer < last:
        for segs in plan.segments[layer + 1].values():
            _add_segment_cuts(segs.values(), cuts)
        return cuts
    if side is None:
        return cuts
    for tid in plan.tids[last]:
        _add_storage_cuts(side, tid, cuts)
    return cuts


def _add_segment_cuts(segs, cuts: Dict[str, set]) -> None:
    for seg in segs:
        cuts.setdefault(seg.tid, set()).update((seg.off, seg.off + seg.size))


def _add_storage_cuts(side, tid: str, cuts: Dict[str, set]) -> None:
    points = cuts.setdefault(tid, set())
    for iv, _path, _cpo, isz in side.storage_pieces(tid):
        points.update(int(x) * isz for x in iv.ravel())


def _cell_segments(
    naive,
    tid: str,
    p_shape: tuple,
    perm: tuple,
    s: int,
    n_index: int,
    c: int,
    supply_iv,
) -> List[Tuple[int, int, tuple]]:
    meta = naive.global_tensor_meta
    p_coord = tuple((s, n_index, c)[i] for i in perm)
    lo, hi = box_of(naive, p_shape, p_coord)
    box = box_to_intervals(meta.global_shape, lo, hi)
    return [
        (int(a) * meta.itemsize, int(b - a) * meta.itemsize, (tid, s, c))
        for a, b in iv_intersect(box, supply_iv)
    ]


def _write_tensor_segments(
    view, n_index: int, tid: str, supply_iv
) -> Iterable[Tuple[int, int, tuple]]:
    naive = naive_of(named_of(view.d_tensor_table, tid))
    perm = cache_mesh_permutation(naive.mesh)
    base_shape = (
        cache_num_steps(naive.mesh),
        cache_num_nodes(naive.mesh),
        cache_concurrency_width(naive.mesh),
    )
    p_shape = tuple(base_shape[i] for i in perm)
    for s in range(base_shape[0]):
        for c in range(base_shape[2]):
            yield from _cell_segments(
                naive, tid, p_shape, perm, s, n_index, c, supply_iv
            )


def _layer_write_segments(
    view, layer: int, node: int, have, cuts
) -> Dict[WriteSegKey, WriteSegment]:
    segs: Dict[WriteSegKey, WriteSegment] = {}
    n_index = _view_nodes(view).index(node)
    for tid in view.tensor_ids:
        supply_iv = have.get(tid)
        if supply_iv is None or not supply_iv.size:
            continue
        points = cuts.get(tid, ())
        segs.update(
            (seg.key, seg)
            for off, size, cell in _write_tensor_segments(view, n_index, tid, supply_iv)
            for a, b in _split(off, off + size, points)
            for seg in (WriteSegment(layer, node, tid, a, b - a, cell),)
        )
    return segs


def _link_write_storage(plan: WriteChainPlan, dst, side) -> None:
    by_raw: Dict[str, list] = {}
    for segs in plan.segments[-1].values():
        for seg in segs.values():
            _assign_seg_storage(side, seg, by_raw)
    _validate_write_storage(side, dst, by_raw)


def _assign_seg_storage(side, seg: WriteSegment, by_raw: Dict[str, list]) -> None:
    isz = global_tensor_meta_of(named_of(side.table, seg.tid)).itemsize
    iv = np.array([[seg.off // isz, (seg.off + seg.size) // isz]], dtype=np.int64)
    pairs = _save_pieces(seg.tid, iv, isz, side)
    seg.storage = sorted(
        ((dst_r, src_r.global_tensor_offset - seg.off) for src_r, dst_r in pairs),
        key=lambda p: p[1],
    )
    if sum(dst_r.size for dst_r, _ in seg.storage) != seg.size:
        raise check_failed(CHECK, f"write_chain {seg.tid}: incomplete storage mapping")
    for dst_r, _ in seg.storage:
        if is_storage(dst_r):
            by_raw.setdefault(dst_r.file_path, []).append(
                (dst_r.contiguous_payload_offset, dst_r.size)
            )


def _validate_write_storage(side, _dst, by_raw: Dict[str, list]) -> None:
    index = side.index
    for path, pairs in by_raw.items():
        if path not in index.files:
            raise check_failed(CHECK, f"write_chain: unknown storage file {path}")
        iv = _check_bounds(
            pairs, index.files[path].payload_size, f"storage {path}", CHECK
        )
        _assert_disjoint(iv, f"storage {path}", CHECK)


def _write_by_tid_all(
    nodes_map: Dict[int, Dict[WriteSegKey, WriteSegment]],
) -> Dict[str, tuple]:
    grouped: Dict[str, List[Tuple[int, WriteSegment]]] = {}
    for node, segs in nodes_map.items():
        for seg in segs.values():
            grouped.setdefault(seg.tid, []).append((node, seg))
    for items in grouped.values():
        items.sort(key=lambda item: (item[1].off, item[0]))
    return {
        tid: (items, [seg.off for _node, seg in items])
        for tid, items in grouped.items()
    }


def _write_cover_all(index, tid: str, off: int, size: int, producer) -> list:
    end, pieces = off + size, []
    items, starts = index.get(tid, ((), ()))
    start = max(0, bisect_right(starts, off) - 1)
    for node, seg in items[start:]:
        if seg.off >= end:
            break
        a, b = max(off, seg.off), min(end, seg.off + seg.size)
        if b > a:
            pieces.append((node, seg.key, a, b - a))
            seg.producers.setdefault(producer, []).append((a, b - a))
    if sum(p[3] for p in pieces) != size:
        raise check_failed(CHECK, f"write_chain {tid}@{off}+{size}: downstream gap")
    return pieces


def _write_budget(view, segs: Dict[WriteSegKey, WriteSegment]) -> int:
    cells: Dict[tuple, int] = {}
    for seg in segs.values():
        key = (seg.cell[1], seg.cell[2])
        cells[key] = cells.get(key, 0) + seg.size
    return sum(sorted(cells.values(), reverse=True)[: _view_width(view)])


def _link_write_children(plan: WriteChainPlan, layer: int) -> None:
    index = _write_by_tid_all(plan.segments[layer + 1])
    for node, segs in plan.segments[layer].items():
        tnodes: set = set()
        for seg in segs.values():
            producer = ("seg", layer, node, seg.key)
            seg.children = _write_cover_all(index, seg.tid, seg.off, seg.size, producer)
            tnodes.update(tn for tn, _tk, _o, _sz in seg.children)
        plan.targets[layer][node] = tuple(sorted(tnodes))


def _link_stages(plan: WriteChainPlan, src, supplies) -> None:
    index = _write_by_tid_all(plan.segments[0])
    for rank in plan.ranks:
        plan.stages[rank] = [
            piece
            for tid, iv in supplies[rank].items()
            for piece in _rank_stage_pieces(index, src, tid, iv, rank)
        ]


def _rank_stage_pieces(index, src, tid: str, iv, rank: int) -> list:
    isz = global_tensor_meta_of(named_of(src.d_tensor_table, tid)).itemsize
    pieces = []
    for a, b in iv:
        pieces.extend(
            _write_cover_all(index, tid, int(a) * isz, int(b - a) * isz, ("rank", rank))
        )
    return pieces


def _producer_ranks(plan: WriteChainPlan) -> Dict[tuple, set]:
    out: Dict[tuple, set] = {}
    for layer, nodes in enumerate(plan.segments):
        for node, segs in nodes.items():
            _collect_layer_producers(out, layer, node, segs)
    return out


def _producer_rank_set(producer: tuple, out: dict, layer: int) -> set:
    if producer[0] == "rank":
        return {producer[1]}
    return out[(layer - 1, producer[2], producer[3])]


def _collect_layer_producers(out: dict, layer: int, node: int, segs: dict) -> None:
    for key, seg in segs.items():
        ranks: set = set()
        for producer in seg.producers:
            ranks |= _producer_rank_set(producer, out, layer)
        out[(layer, node, key)] = ranks


def _downstream_closure(plan: WriteChainPlan, todo: list) -> set:
    seen: set = set()
    last = len(plan.tids) - 1
    while todo:
        item = todo.pop()
        if item in seen:
            continue
        seen.add(item)
        layer, node, key = item
        if layer >= last:
            continue
        seg = plan.segments[layer][node][key]
        todo.extend((layer + 1, tn, tk) for tn, tk, _o, _s in seg.children)
    return seen


def write_stage_closure(plan: WriteChainPlan, rank: int) -> set:
    """Every (layer, node, key) downstream of stages pushed by this rank."""
    todo = [(0, n, key) for n, key, _o, _s in plan.stages.get(rank, ())]
    return _downstream_closure(plan, todo)


def write_execution_closure(plan: WriteChainPlan, rank: int) -> set:
    """Staged dependency closure plus flush service work assigned to this rank."""
    todo = [(0, n, key) for n, key, _o, _s in plan.stages.get(rank, ())]
    todo.extend(plan.services.get(rank, ()))
    return _downstream_closure(plan, todo)


def _link_flush_ranks(plan: WriteChainPlan) -> Dict[int, set]:
    producers = _producer_ranks(plan)
    plan.services = {r: [] for r in plan.ranks}
    local = {
        node: [r for r in plan.ranks if r // plan.k == node]
        for nodes in plan.layer_nodes
        for node in nodes
    }
    for layer, nodes in enumerate(plan.segments):
        for node, segments in nodes.items():
            _assign_node_flushes(plan, producers, local[node], layer, node, segments)
    return {r: write_execution_closure(plan, r) for r in plan.ranks}


def _assign_node_flushes(plan, producers, local, layer, node, segments) -> None:
    for key in sorted(segments):
        item = (layer, node, key)
        ranks = producers.get(item, set())
        eligible = [r for r in local if r in ranks]
        helper = (local or plan.ranks)[0]
        flushers = tuple(eligible or [helper])
        plan.flush_ranks[item] = flushers
        if not (ranks <= set(flushers)):
            plan.services[flushers[0]].append(item)


def _link_write_cohorts(plan: WriteChainPlan, closures: Dict[int, set]) -> None:
    plan.cohort = [{n: set() for n in nodes} for nodes in plan.layer_nodes]
    for rank in plan.ranks:
        for layer, node, _key in closures[rank]:
            plan.cohort[layer][node].add(rank)


# ------------------------------------------------------------------ derived


def _flush_pairs(plan: WriteChainPlan, layer: int, seg: WriteSegment) -> list:
    if layer == len(plan.tids) - 1:
        return [
            (_crange(seg.tid, seg.off + rel, rng.size), rng) for rng, rel in seg.storage
        ]
    return [
        (_crange(seg.tid, off, size), _crange(seg.tid, off, size))
        for _tnode, _tkey, off, size in seg.children
    ]


def rank_write_copy_plans(plan: WriteChainPlan, rank: int) -> list:
    """One CopyPlan per leg in data-flow order: compute -> cache -> ... -> storage."""
    if rank not in plan.ranks:
        raise invalid("rank", f"rank {rank} not in write-chain ranks {plan.ranks}")
    legs: list = [[] for _ in range(len(plan.tids) + 1)]
    legs[0] = [
        (_crange(key[0], off, size), _crange(key[0], off, size))
        for _node, key, off, size in plan.stages[rank]
    ]
    for layer, node, key in sorted(write_execution_closure(plan, rank)):
        if rank in plan.flush_ranks[(layer, node, key)]:
            legs[layer + 1].extend(
                _flush_pairs(plan, layer, plan.segments[layer][node][key])
            )
    return legs


def plan_write_mem(plan: WriteChainPlan) -> Dict[int, int]:
    """Per-node cache bytes: the layer budgets, summed over layers."""
    acc: Dict[int, int] = {}
    for budgets in plan.budget:
        for node, nbytes in budgets.items():
            acc[node] = acc.get(node, 0) + nbytes
    return acc


def plan_write_traffic(plan: WriteChainPlan) -> Tuple[Dict[int, int], Dict[int, int]]:
    """(inbound, outbound) bytes per node across compute->cache, cache->cache, cache->storage."""
    nodes = {n for layer in plan.layer_nodes for n in layer}
    nodes.update(r // plan.k for r in plan.ranks)
    inbound = {n: 0 for n in nodes}
    outbound = {n: 0 for n in nodes}
    for rank, pieces in plan.stages.items():
        _stage_traffic(rank // plan.k, pieces, inbound, outbound)
    last = len(plan.tids) - 1
    for layer in range(last):
        _write_edge_traffic(plan, layer, inbound, outbound)
    _terminal_write_traffic(plan, last, inbound, outbound)
    return inbound, outbound


def _stage_traffic(rank_node: int, pieces, inbound, outbound) -> None:
    for node, _key, _off, size in pieces:
        _hop(rank_node, node, size, inbound, outbound)


def _write_edge_traffic(plan: WriteChainPlan, layer: int, inbound, outbound) -> None:
    for node, segs in plan.segments[layer].items():
        _node_edge_traffic(plan, layer, node, segs, inbound, outbound)


def _node_edge_traffic(
    plan: WriteChainPlan, layer: int, node: int, segs, inbound, outbound
) -> None:
    for seg in segs.values():
        actor = plan.flush_ranks[(layer, node, seg.key)][0] // plan.k
        for tnode, _tkey, _off, size in seg.children:
            _hop(node, actor, size, inbound, outbound)
            _hop(actor, tnode, size, inbound, outbound)


def _terminal_write_traffic(plan: WriteChainPlan, last: int, inbound, outbound) -> None:
    for node, segs in plan.segments[last].items():
        for seg in segs.values():
            actor = plan.flush_ranks[(last, node, seg.key)][0] // plan.k
            _hop(node, actor, seg.size, inbound, outbound)
            outbound[actor] += seg.size

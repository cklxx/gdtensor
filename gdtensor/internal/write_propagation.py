"""Node-level propagation plan for cached write chains.

Mirrors propagation.py for compute -> cache -> storage writes: per-rank
compute supplies coalesce along contiguous byte intervals inside node-local
cache segments, and flush to storage (or a parent convergence cache layer) once
every contributing producer piece has staged. Budget, traffic, and admission
share one plan definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .dtensor import (
    global_tensor_meta_of,
    is_designated_writer,
    named_of,
    order_of,
    owned_intervals,
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
    _tensor_segments,
    _view_nodes,
    _view_width,
    parent_map,
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
    # layer < last: [(dst_key, global_off, size)] on the target layer node
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
    serve: Dict[int, int]
    target: List[Dict[int, int]]
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
    from .chain import OneLayerCachedDTensorsWriteChain

    if type(chain) is OneLayerCachedDTensorsWriteChain:
        return [chain.cache]
    return [chain.big_cache, chain.small_cache]


def _target_maps(layer_nodes: List[List[int]]) -> List[Dict[int, int]]:
    out = [
        parent_map(layer_nodes[layer], layer_nodes[layer + 1])
        for layer in range(len(layer_nodes) - 1)
    ]
    out.append({})
    return out


def build_write_chain_plan(chain, with_storage: bool = True) -> WriteChainPlan:
    """Node-level write aggregation plan."""
    k = chain.num_ranks_per_node
    if k <= 0:
        raise invalid("num_ranks_per_node", "must be positive")
    caches = write_cache_layers(chain)
    layer_nodes = [_view_nodes(view) for view in caches]
    ranks = _dst_ranks(chain.src)
    serve = parent_map(sorted({r // k for r in ranks}), layer_nodes[0])
    target = _target_maps(layer_nodes)
    plan = WriteChainPlan(
        [list(view.tensor_ids) for view in caches],
        k,
        layer_nodes,
        serve,
        target,
        ranks,
    )
    supplies = {rank: _rank_supplies(chain.src, rank) for rank in plan.ranks}
    supply = _coalesce_supplies(plan, supplies)
    side = _Side(chain.dst) if with_storage else None
    _build_write_segments(plan, caches, supply, side)
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


def _build_write_segments(plan: WriteChainPlan, caches: list, supply, side) -> None:
    last = len(caches) - 1
    plan.segments = [{} for _ in caches]
    for layer in range(last, -1, -1):
        view = caches[layer]
        plan.segments[layer] = {
            n: _layer_write_segments(
                view,
                layer,
                n,
                supply[layer][n],
                _write_cuts(plan, side, layer, n, last),
            )
            for n in supply[layer]
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


def _coalesce_supplies(
    plan: WriteChainPlan, supplies: Dict[int, Dict[str, np.ndarray]]
):
    pending = [{n: {} for n in nodes} for nodes in plan.layer_nodes]
    for rank in plan.ranks:
        node_pending = pending[0][plan.serve[rank // plan.k]]
        _ = [
            node_pending.setdefault(tid, []).append(iv)
            for tid, iv in supplies[rank].items()
        ]
    supply = [{} for _ in plan.tids]
    last = len(plan.tids) - 1
    for layer in range(len(plan.tids)):
        supply[layer] = {
            node: _merge_pending(node_pending)
            for node, node_pending in pending[layer].items()
        }
        if layer < last:
            _propagate_pending(plan, supply, pending, layer)
    return supply


def _propagate_pending(plan: WriteChainPlan, supply, pending, layer: int) -> None:
    for node, have in supply[layer].items():
        t_pending = pending[layer + 1][plan.target[layer][node]]
        _ = [t_pending.setdefault(tid, []).append(iv) for tid, iv in have.items()]


def _write_cuts(
    plan: WriteChainPlan, side, layer: int, node: int, last: int
) -> Dict[str, set]:
    cuts: Dict[str, set] = {}
    if layer < last:
        for seg in plan.segments[layer + 1][plan.target[layer][node]].values():
            cuts.setdefault(seg.tid, set()).update((seg.off, seg.off + seg.size))
        return cuts
    if side is None:
        return cuts
    for tid in plan.tids[last]:
        _add_storage_cuts(side, tid, cuts)
    return cuts


def _add_storage_cuts(side, tid: str, cuts: Dict[str, set]) -> None:
    points = cuts.setdefault(tid, set())
    for iv, _path, _cpo, isz in side.storage_pieces(tid):
        points.update(int(x) * isz for x in iv.ravel())


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
            for off, size, cell in _tensor_segments(view, n_index, tid, supply_iv)
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


def _validate_write_storage(side, dst, by_raw: Dict[str, list]) -> None:
    index = side.index
    for path, pairs in by_raw.items():
        if path not in index.files:
            raise check_failed(CHECK, f"write_chain: unknown storage file {path}")
        iv = _check_bounds(
            pairs, index.files[path].payload_size, f"storage {path}", CHECK
        )
        _assert_disjoint(iv, f"storage {path}", CHECK)


def _write_cover(index, tid: str, off: int, size: int, producer) -> list:
    from bisect import bisect_right

    end, pieces = off + size, []
    segments, starts = index.get(tid, ((), ()))
    start = max(0, bisect_right(starts, off) - 1)
    for seg in segments[start:]:
        if seg.off >= end:
            break
        a, b = max(off, seg.off), min(end, seg.off + seg.size)
        if b > a:
            pieces.append((seg.key, a, b - a))
            seg.producers.setdefault(producer, []).append((a, b - a))
    if sum(p[2] for p in pieces) != size:
        raise check_failed(CHECK, f"write_chain {tid}@{off}+{size}: downstream gap")
    return pieces


def _write_budget(view, segs: Dict[WriteSegKey, WriteSegment]) -> int:
    cells: Dict[tuple, int] = {}
    for seg in segs.values():
        key = (seg.cell[1], seg.cell[2])
        cells[key] = cells.get(key, 0) + seg.size
    return sum(sorted(cells.values(), reverse=True)[: _view_width(view)])


def _write_by_tid(segs: Dict[WriteSegKey, WriteSegment]) -> Dict[str, tuple]:
    grouped: Dict[str, List[WriteSegment]] = {}
    for seg in sorted(segs.values(), key=lambda s: s.off):
        grouped.setdefault(seg.tid, []).append(seg)
    return {
        tid: (segments, [seg.off for seg in segments])
        for tid, segments in grouped.items()
    }


def _link_write_children(plan: WriteChainPlan, layer: int) -> None:
    for node, segs in plan.segments[layer].items():
        index = _write_by_tid(plan.segments[layer + 1][plan.target[layer][node]])
        for seg in segs.values():
            producer = ("seg", layer, node, seg.key)
            seg.children = _write_cover(index, seg.tid, seg.off, seg.size, producer)


def _link_stages(plan: WriteChainPlan, src, supplies) -> None:
    indices = {
        node: _write_by_tid(segments) for node, segments in plan.segments[0].items()
    }
    for rank in plan.ranks:
        node = plan.serve[rank // plan.k]
        plan.stages[rank] = [
            (node, key, off, size)
            for tid, iv in supplies[rank].items()
            for key, off, size in _rank_stage_pieces(indices[node], src, tid, iv, rank)
        ]


def _rank_stage_pieces(index, src, tid: str, iv, rank: int) -> list:
    isz = global_tensor_meta_of(named_of(src.d_tensor_table, tid)).itemsize
    pieces = []
    for a, b in iv:
        pieces.extend(
            _write_cover(index, tid, int(a) * isz, int(b - a) * isz, ("rank", rank))
        )
    return pieces


def _producer_ranks(plan: WriteChainPlan) -> Dict[tuple, set]:
    out: Dict[tuple, set] = {}
    for layer, nodes in enumerate(plan.segments):
        for node, segs in nodes.items():
            _collect_layer_producers(plan, out, layer, node, segs)
    return out


def _producer_rank_set(producer: tuple, out: dict, layer: int) -> set:
    if producer[0] == "rank":
        return {producer[1]}
    return out[(layer - 1, producer[2], producer[3])]


def _collect_layer_producers(
    plan: WriteChainPlan, out: dict, layer: int, node: int, segs: dict
) -> None:
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
        tnode = plan.target[layer][node]
        seg = plan.segments[layer][node][key]
        todo.extend((layer + 1, tnode, tkey) for tkey, _o, _s in seg.children)
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
        for _tkey, off, size in seg.children
    ]


def rank_write_copy_plans(plan: WriteChainPlan, rank: int) -> list:
    """One CopyPlan per leg in data-flow order: compute -> cache -> ... -> storage."""
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
    nodes.update(plan.serve)
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
        tnode = plan.target[layer][node]
        for seg in segs.values():
            actor = plan.flush_ranks[(layer, node, seg.key)][0] // plan.k
            _hop(node, actor, seg.size, inbound, outbound)
            _hop(actor, tnode, seg.size, inbound, outbound)


def _terminal_write_traffic(plan: WriteChainPlan, last: int, inbound, outbound) -> None:
    for node, segs in plan.segments[last].items():
        for seg in segs.values():
            actor = plan.flush_ranks[(last, node, seg.key)][0] // plan.k
            _hop(node, actor, seg.size, inbound, outbound)
            outbound[actor] += seg.size

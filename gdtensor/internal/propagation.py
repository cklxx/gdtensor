"""Node-level propagation plan for cached read chains.

One plan describes a concrete cached chain for every node at once. Stats, the
per-rank copy plans and the executor all derive from it, so capacity, traffic
and runtime admission use one definition.

Terms:
* layer: one cache view of the chain, in data-flow order (layer 0 reads storage).
* parent: every node of layer l reads from exactly one node of layer l - 1. A
  node that is present in both layers reads locally. The other nodes are spread
  round-robin over the sorted parent nodes. The last layer serves the rank
  nodes with the same rule (rank r is on node r // num_ranks_per_node).
* demand: elements a node must hold. The last layer holds the union of what its
  served ranks own on the compute view. A parent holds the union of what its
  children hold.
* cell: one (temporal step, concurrency slot) box of a cache tensor.
* segment: one contiguous byte run of a cell intersected with the demand. It is
  the unit of fill, residency and release.
* segment boundaries: layer 0 splits at storage slot boundaries (one segment
  reads one storage range); a lower layer splits at its parent's segments (one
  segment has one parent segment).
* budget: the C largest cells of a (layer, node), summed over all tensors of
  the layer. A cell is one (temporal step, concurrency slot) box of every
  tensor of the layer: one step-slot of a multi-tensor layer counts once, so
  S=1 holds the whole step. C is the concurrency width. The executor admits a
  segment only while the resident bytes stay inside this budget, so the stat
  is a hard bound. A segment never exceeds its cell, so one segment always
  fits.
* cohort: the ranks that read a (layer, node). When a node's budget cannot
  hold the node demand, every cohort rank must be present at the same time;
  the executor reports a missing consumer instead of hanging (api.py
  _wait_cohort).
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
    iv_intersect,
    iv_normalize,
    iv_prefix,
    named_of,
    naive_of,
    owned_intervals,
    participants_of,
)
from .mesh import (
    cache_concurrency_width,
    cache_node_set,
    cache_num_nodes,
    cache_num_steps,
)
from .plan import CHECK, _Side, _pair
from .ranges import RangeFromComputeView, RangeFromStorageView
from .result import check_failed, invalid

SegKey = Tuple[str, int, int]  # (tensor id, byte offset, byte size)


@dataclass
class Segment:
    layer: int
    node: int
    tid: str
    off: int
    size: int
    cell: tuple
    # layer 0: [(storage range, byte offset inside the segment)]
    storage: list = field(default_factory=list)
    # layer > 0: [(parent key, byte offset, byte size)] on the parent node
    parents: list = field(default_factory=list)
    # consumer -> [(byte offset, byte size)]; a consumer is ("rank", r) or
    # ("seg", layer, node, key) for a child segment
    consumers: dict = field(default_factory=dict)

    @property
    def key(self) -> SegKey:
        return (self.tid, self.off, self.size)

    def owed(self) -> set:
        """Every (consumer, offset, size) piece that must ACK before release."""
        return {
            (consumer, off, size)
            for consumer, pieces in self.consumers.items()
            for off, size in pieces
        }


@dataclass
class ChainPlan:
    tids: List[List[str]]
    k: int
    layer_nodes: List[List[int]]
    parent: List[Dict[int, int]]
    serve: Dict[int, int]
    ranks: List[int]
    segments: List[Dict[int, Dict[SegKey, Segment]]] = field(default_factory=list)
    budget: List[Dict[int, int]] = field(default_factory=list)
    demand: List[Dict[int, int]] = field(default_factory=list)
    drains: Dict[int, list] = field(default_factory=dict)
    cohort: List[Dict[int, set]] = field(default_factory=list)
    fill_ranks: Dict[tuple, tuple] = field(default_factory=dict)
    services: Dict[int, list] = field(default_factory=dict)


# ------------------------------------------------------------------ build


def cache_layers(chain) -> list:
    from .chain import OneLayerCachedDTensorsReadChain

    if type(chain) is OneLayerCachedDTensorsReadChain:
        return [chain.cache]
    return [chain.small_cache, chain.big_cache]


def build_chain_plan(chain, with_storage: bool = True) -> ChainPlan:
    """Node-level plan. Stats pass with_storage=False: they need only meshes
    and global meta, so they never build the storage payload index."""
    k = chain.num_ranks_per_node
    if k <= 0:
        raise invalid("num_ranks_per_node", "must be positive")
    caches = cache_layers(chain)
    layer_nodes = [_view_nodes(view) for view in caches]
    ranks = _dst_ranks(chain.dst)
    serve = parent_map(sorted({r // k for r in ranks}), layer_nodes[-1])
    parent: List[Dict[int, int]] = [{}]
    for layer in range(1, len(caches)):
        parent.append(parent_map(layer_nodes[layer], layer_nodes[layer - 1]))
    plan = ChainPlan(
        [list(view.tensor_ids) for view in caches], k, layer_nodes, parent, serve, ranks
    )
    wants = {rank: _rank_wants(chain.dst, rank) for rank in plan.ranks}
    demand = _demands(plan, wants)
    side = _Side(chain.src) if with_storage else None
    for layer, view in enumerate(caches):
        plan.segments.append(
            {
                n: _layer_segments(
                    view, layer, n, demand[layer][n], _cuts(plan, side, layer, n)
                )
                for n in demand[layer]
            }
        )
    plan.budget = [
        {n: _budget(view, segs) for n, segs in plan.segments[layer].items()}
        for layer, view in enumerate(caches)
    ]
    plan.demand = [
        {
            n: sum(seg.size for seg in segs.values())
            for n, segs in plan.segments[layer].items()
        }
        for layer in range(len(caches))
    ]
    if side is not None:
        _link_storage(plan, side)
    for layer in range(1, len(caches)):
        _link_parents(plan, layer)
    _link_drains(plan, chain.dst, wants)
    closures = _link_fill_ranks(plan)
    _link_cohorts(plan, closures)
    return plan


def parent_map(children, parents) -> Dict[int, int]:
    """Child node -> serving parent node; a local parent wins."""
    if not parents:
        raise invalid("read_chain", "a cache layer covers no nodes")
    local = set(parents)
    remote = [c for c in sorted(children) if c not in local]
    out = {c: c for c in children if c in local}
    out.update({c: parents[j % len(parents)] for j, c in enumerate(remote)})
    return out


def _view_nodes(view) -> List[int]:
    sets = {
        tuple(cache_node_set(naive_of(nd).mesh))
        for nd in view.d_tensor_table.named_d_tensors
    }
    if len(sets) != 1:
        raise invalid(
            "cache_view", "all tensors of a cache view must cover one node set"
        )
    return list(sets.pop())


def _view_width(view) -> int:
    widths = {
        cache_concurrency_width(naive_of(nd).mesh)
        for nd in view.d_tensor_table.named_d_tensors
    }
    if len(widths) != 1:
        raise invalid("cache_view", "all tensors of a cache view must share one width")
    return widths.pop()


def _dst_ranks(dst) -> List[int]:
    ranks: set = set()
    for nd in dst.d_tensor_table.named_d_tensors:
        ranks.update(participants_of(nd, "compute"))
    return sorted(ranks)


def _rank_wants(dst, rank: int) -> Dict[str, np.ndarray]:
    wants = {}
    for tid in dst.tensor_ids:
        iv = owned_intervals(named_of(dst.d_tensor_table, tid), "compute", rank)
        if iv.size:
            wants[tid] = iv
    return wants


def _merge_pending(pending: Dict[str, list]) -> Dict[str, np.ndarray]:
    return {
        tid: intervals[0]
        if len(intervals) == 1
        else iv_normalize(np.concatenate(intervals))
        for tid, intervals in pending.items()
    }


def _demands(plan: ChainPlan, wants: Dict[int, Dict[str, np.ndarray]]):
    """Merge each node once after collecting its rank or child intervals."""
    pending = [{n: {} for n in nodes} for nodes in plan.layer_nodes]
    for rank in plan.ranks:
        for tid, iv in wants[rank].items():
            node_pending = pending[-1][plan.serve[rank // plan.k]]
            node_pending.setdefault(tid, []).append(iv)
    demand = [{} for _ in plan.tids]
    for layer in range(len(plan.tids) - 1, -1, -1):
        demand[layer] = {
            node: _merge_pending(node_pending)
            for node, node_pending in pending[layer].items()
        }
        if layer == 0:
            continue
        for child, want in demand[layer].items():
            parent_pending = pending[layer - 1][plan.parent[layer][child]]
            _ = [
                parent_pending.setdefault(tid, []).append(iv)
                for tid, iv in want.items()
            ]
    return demand


def _cuts(plan: ChainPlan, side, layer: int, node: int) -> Dict[str, set]:
    """Byte boundaries a segment must not cross.

    Layer 0 splits at storage slot boundaries, so one segment reads exactly
    one storage range. A lower layer splits at the segments of its parent
    node, so one segment has exactly one parent segment.
    """
    cuts: Dict[str, set] = {}
    if layer > 0:
        for seg in plan.segments[layer - 1][plan.parent[layer][node]].values():
            cuts.setdefault(seg.tid, set()).update((seg.off, seg.off + seg.size))
        return cuts
    if side is None:
        return cuts
    for tid in plan.tids[0]:
        for iv, _path, _cpo, isz in side.storage_pieces(tid):
            points = cuts.setdefault(tid, set())
            points.update(int(x) * isz for x in iv.ravel())
    return cuts


def _split(a: int, b: int, points) -> List[Tuple[int, int]]:
    edges = [a] + sorted(p for p in points if a < p < b) + [b]
    return list(zip(edges[:-1], edges[1:]))


def _layer_segments(view, layer: int, node: int, demand, cuts) -> Dict[SegKey, Segment]:
    """Demanded cell bytes, split at the boundaries from _cuts."""
    segs: Dict[SegKey, Segment] = {}
    n_index = _view_nodes(view).index(node)
    for tid in view.tensor_ids:
        want = demand.get(tid)
        if want is None or not want.size:
            continue
        points = cuts.get(tid, ())
        segs.update(
            (seg.key, seg)
            for off, size, cell in _tensor_segments(view, n_index, tid, want)
            for a, b in _split(off, off + size, points)
            for seg in (Segment(layer, node, tid, a, b - a, cell),)
        )
    return segs


def _tensor_segments(
    view, n_index: int, tid: str, want
) -> Iterable[Tuple[int, int, tuple]]:
    naive = naive_of(named_of(view.d_tensor_table, tid))
    shape = (
        cache_num_steps(naive.mesh),
        cache_num_nodes(naive.mesh),
        cache_concurrency_width(naive.mesh),
    )
    meta = naive.global_tensor_meta
    for s in range(shape[0]):
        for c in range(shape[2]):
            lo, hi = box_of(naive, shape, (s, n_index, c))
            box = box_to_intervals(meta.global_shape, lo, hi)
            yield from (
                (
                    int(a) * meta.itemsize,
                    int(b - a) * meta.itemsize,
                    (tid, s, c),
                )
                for a, b in iv_intersect(box, want)
            )


def _budget(view, segs: Dict[SegKey, Segment]) -> int:
    """The C largest (step, slot) cells, each summed over every tensor."""
    cells: Dict[tuple, int] = {}
    for seg in segs.values():
        key = (seg.cell[1], seg.cell[2])
        cells[key] = cells.get(key, 0) + seg.size
    return sum(sorted(cells.values(), reverse=True)[: _view_width(view)])


# ------------------------------------------------------------------ links


def _link_storage(plan: ChainPlan, side) -> None:
    grouped: Dict[str, List[Segment]] = {}
    for segs in plan.segments[0].values():
        for seg in segs.values():
            grouped.setdefault(seg.tid, []).append(seg)
    for tid, segments in grouped.items():
        for seg in segments:
            seg.storage = _storage_pieces(side, seg)
        side._propagation_storage.pop(tid, None)


def _storage_pieces_generic(side, seg: Segment) -> list:
    isz = global_tensor_meta_of(named_of(side.table, seg.tid)).itemsize
    iv = np.array([[seg.off // isz, (seg.off + seg.size) // isz]], dtype=np.int64)
    pairs = _pair([(iv, ("compute", seg.tid, isz))], side.src_candidates(seg.tid, 0))
    return sorted(
        ((src, dst.global_tensor_offset - seg.off) for src, dst in pairs),
        key=lambda p: p[1],
    )


def _storage_index(side, tid: str) -> tuple | None:
    """Flatten disjoint storage slots once; retain source order for overlaps."""
    cache = getattr(side, "_propagation_storage", None)
    if cache is None:
        cache = side._propagation_storage = {}
    if tid not in cache:
        entries = _storage_entries(side, tid)
        overlap = any(
            previous[1] > entry[0] for previous, entry in zip(entries, entries[1:])
        )
        isz = global_tensor_meta_of(named_of(side.table, tid)).itemsize
        cache[tid] = (
            None if overlap else (entries, [entry[0] for entry in entries], isz)
        )
    return cache[tid]


def _storage_entries(side, tid: str) -> list:
    entries = [
        (int(a) * isz, int(b) * isz, path, cpo + int(prefix) * isz, isz)
        for iv, path, cpo, isz in side.storage_pieces(tid)
        for (a, b), prefix in zip(iv, iv_prefix(iv, iv[:, 0]))
    ]
    entries.sort(key=lambda entry: entry[0])
    return entries


def _storage_pieces(side, seg: Segment) -> list:
    indexed = _storage_index(side, seg.tid)
    if indexed is None:
        return _storage_pieces_generic(side, seg)
    entries, starts, isz = indexed
    cursor, end, pieces, missing = seg.off, seg.off + seg.size, [], 0
    position = max(0, bisect_right(starts, cursor) - 1)
    for start, stop, path, source_off, _isz in entries[position:]:
        if start >= end or cursor == end:
            break
        a, b = max(cursor, start), min(end, stop)
        if b <= a:
            continue
        if a > cursor:
            missing += a - cursor
        src = RangeFromStorageView(path, source_off + a - start, b - a)
        pieces.append((src, a - seg.off))
        cursor = b
    missing += end - cursor
    if missing:
        raise invalid(
            "src_view", f"no source covers {missing // isz} destination elements"
        )
    return pieces


def _by_tid(segs: Dict[SegKey, Segment]) -> Dict[str, tuple]:
    grouped: Dict[str, List[Segment]] = {}
    for seg in sorted(segs.values(), key=lambda s: s.off):
        grouped.setdefault(seg.tid, []).append(seg)
    return {
        tid: (segments, [seg.off for seg in segments])
        for tid, segments in grouped.items()
    }


def _cover(index, tid: str, off: int, size: int, consumer) -> list:
    """Upstream pieces covering [off, off + size); registers the consumer."""
    end, pieces = off + size, []
    segments, starts = index.get(tid, ((), ()))
    start = max(0, bisect_right(starts, off) - 1)
    for seg in segments[start:]:
        if seg.off >= end:
            break
        a, b = max(off, seg.off), min(end, seg.off + seg.size)
        if b > a:
            pieces.append((seg.key, a, b - a))
            seg.consumers.setdefault(consumer, []).append((a, b - a))
    if sum(p[2] for p in pieces) != size:
        raise check_failed(CHECK, f"cache {tid}@{off}+{size}: upstream gap")
    return pieces


def _link_parents(plan: ChainPlan, layer: int) -> None:
    for node, segs in plan.segments[layer].items():
        index = _by_tid(plan.segments[layer - 1][plan.parent[layer][node]])
        for seg in segs.values():
            consumer = ("seg", layer, node, seg.key)
            seg.parents = _cover(index, seg.tid, seg.off, seg.size, consumer)


def _link_drains(plan: ChainPlan, dst, wants) -> None:
    indices = {node: _by_tid(segments) for node, segments in plan.segments[-1].items()}
    for rank in plan.ranks:
        node = plan.serve[rank // plan.k]
        plan.drains[rank] = [
            (node, key, off, size)
            for tid, iv in wants[rank].items()
            for key, off, size in _rank_pieces(indices[node], dst, tid, iv, rank)
        ]


def _rank_pieces(index, dst, tid: str, iv, rank: int) -> list:
    isz = global_tensor_meta_of(named_of(dst.d_tensor_table, tid)).itemsize
    pieces = []
    for a, b in iv:
        pieces.extend(
            _cover(index, tid, int(a) * isz, int(b - a) * isz, ("rank", rank))
        )
    return pieces


def rank_closure(plan: ChainPlan, rank: int) -> set:
    """Every (layer, node, key) segment the rank reads, directly or upstream."""
    todo = [(len(plan.tids) - 1, n, key) for n, key, _o, _s in plan.drains[rank]]
    seen: set = set()
    while todo:
        item = todo.pop()
        if item not in seen:
            seen.add(item)
            todo.extend(_parent_items(plan, *item))
    return seen


def _parent_items(plan: ChainPlan, layer: int, node: int, key) -> list:
    if layer == 0:
        return []
    pnode = plan.parent[layer][node]
    seg = plan.segments[layer][node][key]
    return [(layer - 1, pnode, pkey) for pkey, _o, _s in seg.parents]


def _execution_closure(plan: ChainPlan, rank: int, closure: set) -> set:
    todo = list(plan.services.get(rank, ()))
    seen = set(closure)
    while todo:
        item = todo.pop()
        if item not in seen:
            seen.add(item)
            todo.extend(_parent_items(plan, *item))
    return seen


def execution_closure(plan: ChainPlan, rank: int) -> set:
    """Drain dependencies plus DN service work assigned to this rank."""
    return _execution_closure(plan, rank, rank_closure(plan, rank))


def _link_fill_ranks(plan: ChainPlan) -> Dict[int, set]:
    closures = {r: rank_closure(plan, r) for r in plan.ranks}
    consumers: Dict[tuple, list] = {}
    for rank, closure in closures.items():
        for item in closure:
            consumers.setdefault(item, []).append(rank)
    plan.services = {r: [] for r in plan.ranks}
    local = {
        node: [r for r in plan.ranks if r // plan.k == node]
        for nodes in plan.layer_nodes
        for node in nodes
    }
    for layer, nodes in enumerate(plan.segments):
        for node, segments in nodes.items():
            _assign_node_fills(plan, consumers, local[node], layer, node, segments)
    del consumers, local
    for rank in plan.ranks:
        closures[rank] = _execution_closure(plan, rank, closures[rank])
    return closures


def _assign_node_fills(plan, consumers, local, layer, node, segments) -> None:
    for key in sorted(segments):
        item = (layer, node, key)
        ranks = consumers.get(item, ())
        eligible = [r for r in local if r in ranks]
        helper = (local or plan.ranks)[0]
        plan.fill_ranks[item] = tuple(eligible or [helper])
        if not eligible:
            plan.services[helper].append(item)


def _link_cohorts(plan: ChainPlan, closures) -> None:
    plan.cohort = [{n: set() for n in nodes} for nodes in plan.layer_nodes]
    for rank in plan.ranks:
        for layer, node, _key in closures[rank]:
            plan.cohort[layer][node].add(rank)


# ------------------------------------------------------------------ derived


def _crange(tid: str, off: int, size: int) -> RangeFromComputeView:
    return RangeFromComputeView(tid, off, size)


def _fill_pairs(seg: Segment) -> list:
    if seg.layer == 0:
        return [
            (rng, _crange(seg.tid, seg.off + o, rng.size)) for rng, o in seg.storage
        ]
    return [
        (_crange(seg.tid, o, n), _crange(seg.tid, o, n)) for _k, o, n in seg.parents
    ]


def rank_copy_plans(plan: ChainPlan, rank: int) -> list:
    """One CopyPlan per leg: the rank's fill closure, then its own drains."""
    legs: list = [[] for _ in range(len(plan.tids) + 1)]
    for layer, node, key in sorted(execution_closure(plan, rank)):
        if rank in plan.fill_ranks[(layer, node, key)]:
            legs[layer].extend(_fill_pairs(plan.segments[layer][node][key]))
    legs[-1] = [
        (_crange(key[0], off, size), _crange(key[0], off, size))
        for _node, key, off, size in plan.drains[rank]
    ]
    return legs


def plan_mem(plan: ChainPlan) -> Dict[int, int]:
    """Per-node cache bytes: the layer budgets, summed over layers."""
    acc: Dict[int, int] = {}
    for budgets in plan.budget:
        for node, nbytes in budgets.items():
            acc[node] = acc.get(node, 0) + nbytes
    return acc


def plan_traffic(plan: ChainPlan) -> Tuple[Dict[int, int], Dict[int, int]]:
    """(inbound, outbound) bytes per node.

    Storage reads count on the assigned filler node. Cache fills count the
    parent-to-filler and filler-to-DN hops; a local filler reduces these to
    the parent-to-DN edge. Only cross-node hops count. Drains count DN-to-rank.
    """
    nodes = {n for layer in plan.layer_nodes for n in layer}
    nodes.update(plan.serve)
    inbound = {n: 0 for n in nodes}
    outbound = {n: 0 for n in nodes}
    for node, segs in plan.segments[0].items():
        for seg in segs.values():
            actor = plan.fill_ranks[(0, node, seg.key)][0] // plan.k
            inbound[actor] += seg.size
            _hop(actor, node, seg.size, inbound, outbound)
    for layer in range(1, len(plan.tids)):
        _edge_traffic(plan, layer, inbound, outbound)
    for rank, pieces in plan.drains.items():
        _drain_traffic(rank // plan.k, pieces, inbound, outbound)
    return inbound, outbound


def _edge_traffic(plan: ChainPlan, layer: int, inbound, outbound) -> None:
    for node, segs in plan.segments[layer].items():
        pnode = plan.parent[layer][node]
        for seg in segs.values():
            actor = plan.fill_ranks[(layer, node, seg.key)][0] // plan.k
            _hop(pnode, actor, seg.size, inbound, outbound)
            _hop(actor, node, seg.size, inbound, outbound)


def _hop(src: int, dst: int, size: int, inbound, outbound) -> None:
    if src != dst:
        outbound[src] += size
        inbound[dst] += size


def _drain_traffic(rank_node: int, pieces, inbound, outbound) -> None:
    for node, _key, _off, size in pieces:
        if node != rank_node:
            outbound[node] += size
            inbound[rank_node] += size

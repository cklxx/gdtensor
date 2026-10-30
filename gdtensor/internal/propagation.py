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

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .dtensor import (
    EMPTY,
    box_of,
    box_to_intervals,
    global_tensor_meta_of,
    iv_intersect,
    iv_normalize,
    iv_total,
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
from .ranges import RangeFromComputeView
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
    demand = _demands(plan, chain.dst)
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
    _link_drains(plan, chain.dst)
    _link_cohorts(plan)
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
        if iv_total(iv):
            wants[tid] = iv_normalize(iv)
    return wants


def _merge(acc: Dict[str, np.ndarray], wants: Dict[str, np.ndarray]) -> None:
    for tid, iv in wants.items():
        acc[tid] = iv_normalize(np.concatenate((acc.get(tid, EMPTY), iv)))


def _demands(plan: ChainPlan, dst) -> List[Dict[int, Dict[str, np.ndarray]]]:
    demand: List[Dict[int, Dict[str, np.ndarray]]] = [
        {n: {} for n in nodes} for nodes in plan.layer_nodes
    ]
    for rank in plan.ranks:
        _merge(demand[-1][plan.serve[rank // plan.k]], _rank_wants(dst, rank))
    for layer in range(len(plan.tids) - 1, 0, -1):
        for child, want in demand[layer].items():
            _merge(demand[layer - 1][plan.parent[layer][child]], want)
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
    for tid in view.tensor_ids:
        want = demand.get(tid)
        if want is None or not iv_total(want):
            continue
        points = cuts.get(tid, ())
        for seg in _tensor_segments(view, layer, node, tid, want):
            segs.update(
                (p.key, p)
                for p in (
                    Segment(layer, node, tid, a, b - a, seg.cell)
                    for a, b in _split(seg.off, seg.off + seg.size, points)
                )
            )
    return segs


def _tensor_segments(view, layer: int, node: int, tid: str, want) -> List[Segment]:
    naive = naive_of(named_of(view.d_tensor_table, tid))
    shape = (
        cache_num_steps(naive.mesh),
        cache_num_nodes(naive.mesh),
        cache_concurrency_width(naive.mesh),
    )
    meta = naive.global_tensor_meta
    n_index = cache_node_set(naive.mesh).index(node)
    out: List[Segment] = []
    for s in range(shape[0]):
        for c in range(shape[2]):
            lo, hi = box_of(naive, shape, (s, n_index, c))
            box = iv_normalize(box_to_intervals(meta.global_shape, lo, hi))
            out.extend(
                Segment(
                    layer,
                    node,
                    tid,
                    int(a) * meta.itemsize,
                    int(b - a) * meta.itemsize,
                    (tid, s, c),
                )
                for a, b in iv_intersect(box, want)
            )
    return out


def _budget(view, segs: Dict[SegKey, Segment]) -> int:
    """The C largest (step, slot) cells, each summed over every tensor."""
    cells: Dict[tuple, int] = {}
    for seg in segs.values():
        key = (seg.cell[1], seg.cell[2])
        cells[key] = cells.get(key, 0) + seg.size
    return sum(sorted(cells.values(), reverse=True)[: _view_width(view)])


# ------------------------------------------------------------------ links


def _link_storage(plan: ChainPlan, side) -> None:
    for segs in plan.segments[0].values():
        for seg in segs.values():
            seg.storage = _storage_pieces(side, seg)


def _storage_pieces(side, seg: Segment) -> list:
    isz = global_tensor_meta_of(named_of(side.table, seg.tid)).itemsize
    iv = np.array([[seg.off // isz, (seg.off + seg.size) // isz]], dtype=np.int64)
    pairs = _pair([(iv, ("compute", seg.tid, isz))], side.src_candidates(seg.tid, 0))
    pieces = sorted(
        ((src, dst.global_tensor_offset - seg.off) for src, dst in pairs),
        key=lambda p: p[1],
    )
    if sum(src.size for src, _ in pieces) != seg.size:
        raise check_failed(CHECK, f"storage does not cover cache {seg.key}")
    return pieces


def _by_tid(segs: Dict[SegKey, Segment]) -> Dict[str, List[Segment]]:
    index: Dict[str, List[Segment]] = {}
    for seg in sorted(segs.values(), key=lambda s: s.off):
        index.setdefault(seg.tid, []).append(seg)
    return index


def _cover(index, tid: str, off: int, size: int, consumer) -> list:
    """Upstream pieces covering [off, off + size); registers the consumer."""
    end, pieces = off + size, []
    for seg in index.get(tid, ()):
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


def _link_drains(plan: ChainPlan, dst) -> None:
    for rank in plan.ranks:
        node = plan.serve[rank // plan.k]
        index = _by_tid(plan.segments[-1][node])
        plan.drains[rank] = [
            (node, key, off, size)
            for tid, iv in _rank_wants(dst, rank).items()
            for key, off, size in _rank_pieces(index, dst, tid, iv, rank)
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


def _link_cohorts(plan: ChainPlan) -> None:
    plan.cohort = [{n: set() for n in nodes} for nodes in plan.layer_nodes]
    for rank in plan.ranks:
        for layer, node, _key in rank_closure(plan, rank):
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
    for layer, node, key in sorted(rank_closure(plan, rank)):
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

    Storage -> layer 0 counts as inbound of the filling node (storage has no
    node id). A cache -> cache edge or cache -> rank edge counts only when the
    two ends are on different nodes. Every segment moves once per edge.
    """
    nodes = {n for layer in plan.layer_nodes for n in layer}
    nodes.update(plan.serve)
    inbound = {n: 0 for n in nodes}
    outbound = {n: 0 for n in nodes}
    for node, segs in plan.segments[0].items():
        inbound[node] += sum(seg.size for seg in segs.values())
    for layer in range(1, len(plan.tids)):
        _edge_traffic(plan, layer, inbound, outbound)
    for rank, pieces in plan.drains.items():
        _drain_traffic(rank // plan.k, pieces, inbound, outbound)
    return inbound, outbound


def _edge_traffic(plan: ChainPlan, layer: int, inbound, outbound) -> None:
    for node, segs in plan.segments[layer].items():
        pnode = plan.parent[layer][node]
        if pnode != node:
            nbytes = sum(seg.size for seg in segs.values())
            outbound[pnode] += nbytes
            inbound[node] += nbytes


def _drain_traffic(rank_node: int, pieces, inbound, outbound) -> None:
    for node, _key, _off, size in pieces:
        if node != rank_node:
            outbound[node] += size
            inbound[rank_node] += size

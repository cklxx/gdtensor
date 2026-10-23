"""Read chains (design B.1.1 chain_load, B.3.2), plans, and stats.

A read chain describes a storage->compute load through zero to two
node-local caches. Chain types, plans and stats are internal; the
gdtensor.builder namespace exposes user-visible chain products and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, cast

import numpy as np

from .builder_mixin import field_hash, make_builder_mixin
from .dtensor import (
    EMPTY,
    box_of,
    box_to_intervals,
    global_tensor_meta_of,
    iv_normalize,
    iv_total,
    named_of,
    naive_of,
    owned_intervals,
)
from .mesh import (
    cache_concurrency_width,
    cache_node_set,
    cache_num_nodes,
    cache_num_steps,
)
from .plan import (
    CHECK,
    _Side,
    _assert_disjoint,
    _byte_iv,
    _pair,
    _storage_side_within,
    is_compute,
)
from .ranges import CopyPlan
from .result import catch, check_failed, invalid
from .views import (
    CacheDTensorsView,
    ComputeDTensorsView,
    StorageDTensorsView,
)
from .products import Ok
from ..chain_types import Result
# ------------------------------------------------------------------ types


DirectDTensorsReadChainBuilderMixin = make_builder_mixin(
    "DirectDTensorsReadChain", [("src", "object"), ("dst", "object")]
)
OneLayerCachedDTensorsReadChainBuilderMixin = make_builder_mixin(
    "OneLayerCachedDTensorsReadChain",
    [
        ("src", "object"),
        ("cache", "object"),
        ("dst", "object"),
        ("num_ranks_per_node", "scalar"),
    ],
)
TwoLayerCachedDTensorsReadChainBuilderMixin = make_builder_mixin(
    "TwoLayerCachedDTensorsReadChain",
    [
        ("src", "object"),
        ("small_cache", "object"),
        ("big_cache", "object"),
        ("dst", "object"),
        ("num_ranks_per_node", "scalar"),
    ],
)
DisjointTensorIdsAndReadChainBuilderMixin = make_builder_mixin(
    "DisjointTensorIdsAndReadChain", [("tensor_ids", "list"), ("read_chain", "object")]
)
MixedDTensorsReadChainBuilderMixin = make_builder_mixin(
    "MixedDTensorsReadChain",
    [("disjoint_tensor_ids_and_read_chain_list", "list")],
)


@dataclass(eq=True)
class DirectDTensorsReadChain(DirectDTensorsReadChainBuilderMixin):
    src: object = None
    dst: object = None
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class OneLayerCachedDTensorsReadChain(OneLayerCachedDTensorsReadChainBuilderMixin):
    src: object = None
    cache: object = None
    dst: object = None
    num_ranks_per_node: int = 1
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class TwoLayerCachedDTensorsReadChain(TwoLayerCachedDTensorsReadChainBuilderMixin):
    src: object = None
    small_cache: object = None
    big_cache: object = None
    dst: object = None
    num_ranks_per_node: int = 1
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


ConcreteDTensorsReadChain = (
    DirectDTensorsReadChain
    | OneLayerCachedDTensorsReadChain
    | TwoLayerCachedDTensorsReadChain
)


@dataclass(eq=True)
class DisjointTensorIdsAndReadChain(DisjointTensorIdsAndReadChainBuilderMixin):
    tensor_ids: List[str]
    read_chain: object = None
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class MixedDTensorsReadChain(MixedDTensorsReadChainBuilderMixin):
    disjoint_tensor_ids_and_read_chain_list: List[DisjointTensorIdsAndReadChain]
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


DTensorsReadChain = (
    DirectDTensorsReadChain
    | OneLayerCachedDTensorsReadChain
    | TwoLayerCachedDTensorsReadChain
    | MixedDTensorsReadChain
)


# --------------------------------------------------------------- plan chain


@dataclass(eq=True)
class DirectCopyPlanChain:
    src_to_dst: CopyPlan
    __hash__ = field_hash


@dataclass(eq=True)
class OneLayerCachedCopyPlanChain:
    src_to_cache: CopyPlan
    cache_to_dst: CopyPlan
    __hash__ = field_hash


@dataclass(eq=True)
class TwoLayerCachedCopyPlanChain:
    src_to_small_cache: CopyPlan
    small_cache_to_big_cache: CopyPlan
    big_cache_to_dst: CopyPlan
    __hash__ = field_hash


ConcreteCopyPlanChain = (
    DirectCopyPlanChain | OneLayerCachedCopyPlanChain | TwoLayerCachedCopyPlanChain
)


@dataclass(eq=True)
class DisjointTensorIdsAndCopyPlanChain:
    tensor_ids: List[str]
    copy_plan_chain: ConcreteCopyPlanChain
    __hash__ = field_hash


@dataclass(eq=True)
class MixedCopyPlanChain:
    disjoint_tensor_ids_and_copy_plan_chain_list: List[
        DisjointTensorIdsAndCopyPlanChain
    ]
    __hash__ = field_hash


CopyPlanChain = (
    DirectCopyPlanChain
    | OneLayerCachedCopyPlanChain
    | TwoLayerCachedCopyPlanChain
    | MixedCopyPlanChain
)


# ----------------------------------------------------------------- helpers


def get_tensor_ids_of_read_chain(read_chain: DTensorsReadChain) -> List[str]:
    t = type(read_chain)
    if t in (
        DirectDTensorsReadChain,
        OneLayerCachedDTensorsReadChain,
        TwoLayerCachedDTensorsReadChain,
    ):
        c = cast(DirectDTensorsReadChain, read_chain)
        return list(cast("ComputeDTensorsView", c.dst).tensor_ids)
    if t is MixedDTensorsReadChain:
        ids: List[str] = []
        for seg in read_chain.disjoint_tensor_ids_and_read_chain_list:
            ids.extend(
                get_tensor_ids_of_read_chain(cast(DTensorsReadChain, seg.read_chain))
            )
        return ids
    raise invalid("read_chain", f"unknown chain type {t.__name__}")


def legs_of(read_chain: DTensorsReadChain):
    """(src, dst) view pairs per leg, data-flow order; Mixed flattened."""
    t = type(read_chain)
    if t is DirectDTensorsReadChain:
        c = cast(DirectDTensorsReadChain, read_chain)
        return [(c.src, c.dst)]
    if t is OneLayerCachedDTensorsReadChain:
        c = cast(OneLayerCachedDTensorsReadChain, read_chain)
        return [(c.src, c.cache), (c.cache, c.dst)]
    if t is TwoLayerCachedDTensorsReadChain:
        c = cast(TwoLayerCachedDTensorsReadChain, read_chain)
        return [
            (c.src, c.small_cache),
            (c.small_cache, c.big_cache),
            (c.big_cache, c.dst),
        ]
    if t is MixedDTensorsReadChain:
        c = cast(MixedDTensorsReadChain, read_chain)
        legs = []
        for seg in c.disjoint_tensor_ids_and_read_chain_list:
            legs.extend(legs_of(cast(DTensorsReadChain, seg.read_chain)))
        return legs
    raise invalid("read_chain", f"unknown chain type {t.__name__}")


def _concrete_legs(chain: ConcreteDTensorsReadChain):
    t = type(chain)
    if t is DirectDTensorsReadChain:
        return [
            {
                "views": (chain.src, chain.dst),
                "attr": "src_to_dst",
            }
        ]
    if t is OneLayerCachedDTensorsReadChain:
        return [
            {"views": (chain.src, chain.cache), "attr": "src_to_cache"},
            {"views": (chain.cache, chain.dst), "attr": "cache_to_dst"},
        ]
    return [
        {
            "views": (chain.src, chain.small_cache),
            "attr": "src_to_small_cache",
        },
        {
            "views": (chain.small_cache, chain.big_cache),
            "attr": "small_cache_to_big_cache",
        },
        {
            "views": (chain.big_cache, chain.dst),
            "attr": "big_cache_to_dst",
        },
    ]


def _wanted_intervals(compute_view: ComputeDTensorsView, tid: str, rank: int):
    """Global element intervals the rank must finally hold on the compute side.
    Every leg of the rank's chain moves exactly these elements."""
    nd = named_of(compute_view.d_tensor_table, tid)
    return owned_intervals(nd, "compute", rank)


def _cache_owned_blocks(nd, tid: str, rank: int, num_ranks_per_node: int):
    """Element intervals the cache on this rank's node offers for this rank.

    The executor serially walks the full S x C slot grid for the node, so the
    node's cache offers the union of every non-empty slot box on its spatial
    node. We do NOT pin step n = node n (no design basis); the union over all
    (step, slot) coordinates on this node is exactly the region the node can
    serve. ``_pair`` then intersects that with the wanted compute share, which
    is why the cache need not know the rank's step.
    """
    naive = naive_of(nd)
    mesh = naive.mesh
    S = cache_num_steps(mesh)
    N = cache_num_nodes(mesh)
    C = cache_concurrency_width(mesh)
    nodes = sorted(cache_node_set(mesh))
    n_index = nodes.index(rank // num_ranks_per_node)
    shape = global_tensor_meta_of(nd).global_shape
    isz = global_tensor_meta_of(nd).itemsize
    union = []
    for s in range(S):
        for c in range(C):
            lo, hi = box_of(naive, (S, N, C), (s, n_index, c))
            union.append(box_to_intervals(shape, lo, hi))
    intervals = iv_normalize(np.concatenate(union)) if union else EMPTY
    return [(intervals, ("compute", tid, isz))] if intervals.size else []


def _element_blocks(view, tid: str, rank: int, want, num_ranks_per_node: int):
    """Full owner blocks (interval, address) this side offers.

    ``_pair`` intersects these blocks with the wanted elements, so each
    interval here must be the full owner slot. Pre-intersecting would lose
    the owner-relative offset used to map to a physical file offset.

    Storage maps global elements through physical slots; compute addresses by
    global tensor offset; a cache view addresses by global offset too but its
    offered box is its own (step, node, slot) ownership, not the compute share.
    """
    if type(view) is StorageDTensorsView:
        return list(_Side(view).src_candidates(tid, rank))
    nd = named_of(view.d_tensor_table, tid)
    if type(view) is CacheDTensorsView:
        return _cache_owned_blocks(nd, tid, rank, num_ranks_per_node)
    isz = global_tensor_meta_of(nd).itemsize
    owned = owned_intervals(nd, "compute", rank)
    return [(owned, ("compute", tid, isz))] if owned.size else []


def _plan_leg_tid(
    rank: int, src, dst, tid: str, want, num_ranks_per_node: int
) -> CopyPlan:
    src_blocks = _element_blocks(src, tid, rank, want, num_ranks_per_node)
    dst_blocks = _element_blocks(dst, tid, rank, want, num_ranks_per_node)
    return _pair(dst_blocks, src_blocks)


# ------------------------------------------------------------------ B.1.2


@catch
def d_tensors_view_rank_plan_chain_load(
    rank: int, read_chain: DTensorsReadChain
) -> Result[CopyPlanChain]:
    from .plan import _check_same_tensors

    # B.1.2 Assert: every leg moves the same tensors with identical global
    # meta. Checked before any session is opened.
    for src, dst in legs_of(read_chain):
        _check_same_tensors(src, dst)
    t = type(read_chain)
    if t in (
        DirectDTensorsReadChain,
        OneLayerCachedDTensorsReadChain,
        TwoLayerCachedDTensorsReadChain,
    ):
        return Ok(_plan_concrete_chain(rank, read_chain))
    if t is MixedDTensorsReadChain:
        pieces = []
        for seg in read_chain.disjoint_tensor_ids_and_read_chain_list:
            pieces.append(
                DisjointTensorIdsAndCopyPlanChain(
                    list(seg.tensor_ids),
                    _plan_concrete_chain(
                        rank, cast(ConcreteDTensorsReadChain, seg.read_chain)
                    ),
                )
            )
        return Ok(MixedCopyPlanChain(pieces))
    raise invalid("read_chain", f"unknown chain type {t.__name__}")


def _chain_leg_plans(rank: int, chain: ConcreteDTensorsReadChain):
    """CopyPlan per leg, in data-flow order."""
    k = getattr(chain, "num_ranks_per_node", 1)
    return [
        _chain_one_leg(rank, src, dst, chain.dst, k)
        for src, dst in (spec["views"] for spec in _concrete_legs(chain))
    ]


def _chain_one_leg(rank: int, src, dst, compute_view, k: int) -> CopyPlan:
    plan: CopyPlan = []
    for tid in compute_view.tensor_ids:
        want = _wanted_intervals(compute_view, tid, rank)
        if iv_total(want):
            plan.extend(_plan_leg_tid(rank, src, dst, tid, want, k))
    return plan


def _plan_concrete_chain(rank: int, chain: ConcreteDTensorsReadChain):
    # Every leg moves the same elements: the rank's owned share on the final
    # compute view. The wanted set is computed once per tensor and reused
    # across the legs so src->cache and cache->dst carry identical bytes.
    plans = _chain_leg_plans(rank, chain)
    t = type(chain)
    if t is DirectDTensorsReadChain:
        return DirectCopyPlanChain(plans[0])
    if t is OneLayerCachedDTensorsReadChain:
        return OneLayerCachedCopyPlanChain(plans[0], plans[1])
    return TwoLayerCachedCopyPlanChain(plans[0], plans[1], plans[2])


def _storage_leg_out_of_bounds(view, plan: CopyPlan, pairs_side: str):
    index, by_f = _storage_side_bounds(view, plan, pairs_side)
    for path, iv in by_f.items():
        total = index.files[path].payload_size
        if iv.size and int(iv[:, 1].max()) > total:
            raise check_failed(CHECK, f"storage {path}: range outside [0, {total})")


def _compute_dst_intervals(view, plan: CopyPlan, tid: str):
    return _byte_iv(
        [
            (r.global_tensor_offset, r.size)
            for r in _chain_ranges(plan, "dst")
            if is_compute(r) and r.global_tensor_id == tid
        ]
    )


def validate_chain_leg(plan: CopyPlan, src, dst, tid: str, want) -> None:
    """One leg must move exactly `want` elements, disjointly and in bounds.

    Storage ranges stay inside their files; compute/cache ranges stay inside
    the global tensor. The check is element aligned and per leg, so a leg that
    drops half the shard cannot be masked by another leg.
    """
    for view, pairs_side in ((src, "src"), (dst, "dst")):
        if type(view) is StorageDTensorsView:
            _storage_leg_out_of_bounds(view, plan, pairs_side)
        elif pairs_side == "dst":
            nd = named_of(view.d_tensor_table, tid)
            itemsize = global_tensor_meta_of(nd).itemsize
            iv = _compute_dst_intervals(view, plan, tid)
            _assert_chain_covering(iv, want, itemsize, tid)
    _assert_src_disjoint(plan, tid)


def _assert_src_disjoint(plan: CopyPlan, tid: str) -> None:
    for r in _chain_ranges(plan, "src"):
        if is_compute(r) and r.global_tensor_id == tid:
            iv = _byte_iv([(r.global_tensor_offset, r.size)])
            _assert_disjoint(iv, f"compute {tid}", CHECK)


def _chain_ranges(plan: CopyPlan, which: str):
    return [src if which == "src" else dst for src, dst in plan]


def _storage_side_bounds(view, plan: CopyPlan, which: str):

    return _storage_side_within(view, plan, which)


def _assert_chain_covering(byte_iv, want, itemsize: int, tid: str) -> None:
    _assert_disjoint(byte_iv, f"compute {tid}", CHECK)
    got = (
        iv_normalize((byte_iv // itemsize).astype(np.int64)) if byte_iv.size else EMPTY
    )
    expected = iv_normalize(want)
    if not (got.shape == expected.shape and np.array_equal(got, expected)):
        raise check_failed(
            CHECK,
            f"chain leg compute {tid}: covers {iv_total(got)} of "
            f"{iv_total(expected)} elements",
        )


# ------------------------------------------------------------------ B.1.6
#
# Stats return list[int] indexed by spatial node id, length = number of nodes
# the chain's caches cover. Accounting is derived from the real meshes:
#   * mem_overhead: resident cache bytes on each cache node = concurrency_width
#     copies of the largest single step's slice (temporal reuse shrinks it as
#     num_steps grows; each concurrency slot adds one copy).
#   * inbound/outbound: bytes that actually cross a node boundary on the legs,
#     attributed per node from what that node's rank block owns on the final
#     compute mesh. Legs whose two ends live on the same node are local memory
#     motion and carry no traffic. Storage/compute endpoints have no node id.


def _view_node_ids(view):
    """Sorted spatial node ids a cache view covers; None for storage/compute."""
    if type(view) is CacheDTensorsView:
        mesh = view.d_tensor_table.named_d_tensors[0].d_tensor.mesh
        return cache_node_set(mesh)
    return None


def _cache_tensor_resident_bytes(nd, node: int) -> int:
    """Peak resident cache bytes on ``node`` for one tensor.

    The executor serially walks the full S x C slot grid: for each step it
    allocates the slots the concurrency Shard split that step into, consumes
    them, then frees them before the next step (temporal reuse). Peak live
    bytes = the maximum, over steps, of the union of that step's slot boxes.
    Each slot box is taken from the real mesh via box_of; no whole-step-copy
    factor is assumed.
    """
    mesh = nd.mesh
    S = cache_num_steps(mesh)
    N = cache_num_nodes(mesh)
    C = cache_concurrency_width(mesh)
    shape = nd.global_tensor_meta.global_shape
    isz = nd.global_tensor_meta.itemsize
    # node index into the spatial LiteralMesh order.
    n_index = sorted(cache_node_set(mesh)).index(node)

    def slot_boxes(s: int):
        for c in range(C):
            lo, hi = box_of(nd, (S, N, C), (s, n_index, c))
            yield box_to_intervals(shape, lo, hi)

    def step_live(s: int) -> int:
        union = {i for iv in slot_boxes(s) for a, b in iv for i in range(a, b)}
        return len(union) * isz

    return max((step_live(s) for s in range(S)), default=0)


def _node_compute_bytes(dst_view, node: int, k: int) -> int:
    """Bytes the ranks on ``node`` actually own on the final compute mesh."""
    total = 0
    for named in dst_view.d_tensor_table.named_d_tensors:
        isz = named.d_tensor.global_tensor_meta.itemsize
        for r in range(node * k, (node + 1) * k):
            total += iv_total(owned_intervals(named, "compute", r)) * isz
    return total


def get_stat_mem_overhead_of_chain_load(read_chain: DTensorsReadChain) -> List[int]:
    """Resident cache bytes per node (bytes). No session, no payload read."""
    return _stat_overhead(read_chain)


def get_stat_outbound_traffic_of_chain_load(read_chain: DTensorsReadChain) -> List[int]:
    return _stat_traffic(read_chain, outbound=True)


def get_stat_inbound_traffic_of_chain_load(read_chain: DTensorsReadChain) -> List[int]:
    return _stat_traffic(read_chain, outbound=False)


def _stat_overhead(chain) -> List[int]:
    acc: dict = {}
    for src, dst in legs_of(chain):
        nodes = _view_node_ids(dst)
        if nodes is None:
            continue
        for node in nodes:
            resident = sum(
                _cache_tensor_resident_bytes(nd.d_tensor, node)
                for nd in cast(CacheDTensorsView, dst).d_tensor_table.named_d_tensors
            )
            acc[node] = acc.get(node, 0) + resident
    n = (max(acc) + 1) if acc else 0
    return [acc.get(i, 0) for i in range(n)]


def _add_node_bytes(acc, nodes, dst_view, k: int) -> None:
    for node in nodes:
        acc[node] = acc.get(node, 0) + _node_compute_bytes(dst_view, node, k)


def _stat_traffic(chain, outbound: bool) -> List[int]:
    k = getattr(chain, "num_ranks_per_node", 1)
    dst_view = chain.dst
    out_acc: dict = {}
    in_acc: dict = {}
    for src, dst in legs_of(chain):
        src_nodes, dst_nodes = _view_node_ids(src), _view_node_ids(dst)
        if src_nodes is None and dst_nodes is None:
            continue
        if (
            src_nodes is not None
            and dst_nodes is not None
            and set(src_nodes) == set(dst_nodes)
        ):
            # Both ends on the same node set: local cache motion, no traffic.
            continue
        if dst_nodes is not None:
            _add_node_bytes(in_acc, dst_nodes, dst_view, k)
        if src_nodes is not None:
            _add_node_bytes(out_acc, src_nodes, dst_view, k)
    acc = out_acc if outbound else in_acc
    n = (max(acc) + 1) if acc else 0
    return [acc.get(i, 0) for i in range(n)]

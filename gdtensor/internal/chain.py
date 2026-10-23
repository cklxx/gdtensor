"""Read chains (design B.1.1 chain_load, B.3.2), plans, and stats.

A read chain describes a storage->compute load through zero to two
node-local caches. Chain types, plans and stats are internal; the
gdtensor.builder namespace exposes user-visible chain products and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .builder_mixin import field_hash, make_builder_mixin
from .dtensor import (
    EMPTY,
    global_tensor_meta_of,
    iv_normalize,
    iv_total,
    named_of,
    owned_intervals,
)
from .mesh import (
    resolve_cache_mesh,
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
    if t is DirectDTensorsReadChain:
        return list(read_chain.dst.tensor_ids)
    if t is OneLayerCachedDTensorsReadChain:
        return list(read_chain.dst.tensor_ids)
    if t is TwoLayerCachedDTensorsReadChain:
        return list(read_chain.dst.tensor_ids)
    if t is MixedDTensorsReadChain:
        ids: List[str] = []
        for seg in read_chain.disjoint_tensor_ids_and_read_chain_list:
            ids.extend(get_tensor_ids_of_read_chain(seg.read_chain))
        return ids
    raise invalid("read_chain", f"unknown chain type {t.__name__}")


def legs_of(read_chain: DTensorsReadChain):
    """(src, dst) view pairs per leg, data-flow order; Mixed flattened."""
    t = type(read_chain)
    if t is DirectDTensorsReadChain:
        return [(read_chain.src, read_chain.dst)]
    if t is OneLayerCachedDTensorsReadChain:
        return [
            (read_chain.src, read_chain.cache),
            (read_chain.cache, read_chain.dst),
        ]
    if t is TwoLayerCachedDTensorsReadChain:
        return [
            (read_chain.src, read_chain.small_cache),
            (read_chain.small_cache, read_chain.big_cache),
            (read_chain.big_cache, read_chain.dst),
        ]
    if t is MixedDTensorsReadChain:
        legs = []
        for seg in read_chain.disjoint_tensor_ids_and_read_chain_list:
            legs.extend(legs_of(seg.read_chain))
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


def _element_blocks(view, tid: str, rank: int, want):
    """Full owner blocks (interval, address) this side offers.

    ``_pair`` intersects these blocks with the wanted elements, so each
    interval here must be the full owner slot. Pre-intersecting would lose
    the owner-relative offset used to map to a physical file offset.

    Storage maps global elements through physical slots; compute and cache
    address elements by global tensor offset with the same itemsize.
    """
    if type(view) is StorageDTensorsView:
        return list(_Side(view).src_candidates(tid, rank))
    nd = named_of(view.d_tensor_table, tid)
    isz = global_tensor_meta_of(nd).itemsize
    owned = owned_intervals(nd, "compute", rank)
    return [(owned, ("compute", tid, isz))] if owned.size else []


def _plan_leg_tid(rank: int, src, dst, tid: str, want) -> CopyPlan:
    src_blocks = _element_blocks(src, tid, rank, want)
    dst_blocks = _element_blocks(dst, tid, rank, want)
    return _pair(dst_blocks, src_blocks)


# ------------------------------------------------------------------ B.1.2


@catch
def d_tensors_view_rank_plan_chain_load(
    rank: int, read_chain: DTensorsReadChain
) -> Result[CopyPlanChain]:
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
                    _plan_concrete_chain(rank, seg.read_chain),
                )
            )
        return Ok(MixedCopyPlanChain(pieces))
    raise invalid("read_chain", f"unknown chain type {t.__name__}")


def _chain_leg_plans(rank: int, chain: ConcreteDTensorsReadChain):
    """CopyPlan per leg, in data-flow order."""
    return [
        _chain_one_leg(rank, src, dst, chain.dst)
        for src, dst in (spec["views"] for spec in _concrete_legs(chain))
    ]


def _chain_one_leg(rank: int, src, dst, compute_view) -> CopyPlan:
    plan: CopyPlan = []
    for tid in compute_view.tensor_ids:
        want = _wanted_intervals(compute_view, tid, rank)
        if iv_total(want):
            plan.extend(_plan_leg_tid(rank, src, dst, tid, want))
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
    if not (got.shape == expected.shape and bool(np.array_equal(got, expected))):
        raise check_failed(
            CHECK,
            f"chain leg compute {tid}: covers {iv_total(got)} of "
            f"{iv_total(expected)} elements",
        )


# ------------------------------------------------------------------ B.1.6
#
# Stats use a deterministic, implementation-fixed accounting over the chain
# geometry. Node ids are the spatial node indices from the first cache view on
# each leg; the destination compute view contributes world ranks as nodes when
# no cache precedes it. One element is one item of GlobalTensorMeta.itemsize.


def _view_node_ids(view):
    if type(view) in (StorageDTensorsView, ComputeDTensorsView):
        # Storage/compute have no node dimension; stats assign such a leg to a
        # single synthetic node only when it is a cache-free endpoint.
        return None
    mesh = view.d_tensor_table.named_d_tensors[0].d_tensor
    inner = mesh
    while hasattr(inner, "underlying_mesh"):
        inner = inner.underlying_mesh
    if hasattr(inner, "spatial"):
        nodes = np.unique(resolve_cache_mesh(mesh).ravel())
        return [int(n) for n in nodes if n >= 0]
    return None


def _leg_bytes(view) -> int:
    total = 0
    for nd in view.d_tensor_table.named_d_tensors:
        meta = global_tensor_meta_of(nd)
        total += iv_total(owned_intervals(nd, "compute", 0)) * meta.itemsize
    return total


@catch
def get_stat_mem_overhead_of_chain_load(read_chain: DTensorsReadChain):
    # Bytes resident per node: cache payload on cache legs, counted on the
    # cache destination node of each leg.
    return Ok(_stat_overhead(read_chain))


@catch
def get_stat_outbound_traffic_of_chain_load(read_chain: DTensorsReadChain):
    return Ok(_stat_traffic(read_chain, outbound=True))


@catch
def get_stat_inbound_traffic_of_chain_load(read_chain: DTensorsReadChain):
    return Ok(_stat_traffic(read_chain, outbound=False))


def _stat_overhead(chain) -> List[int]:
    acc: dict = {}
    for src, dst in legs_of(chain):
        nodes = _view_node_ids(dst)
        if nodes is None:
            continue
        amount = _leg_bytes(dst)
        for node in nodes:
            acc[node] = acc.get(node, 0) + amount
    n = (max(acc) + 1) if acc else 0
    return [acc.get(i, 0) for i in range(n)]


def _stat_traffic(chain, outbound: bool) -> List[int]:
    # Traffic is counted only on legs whose two endpoints resolve to
    # different node sets; same-node legs are local memory motion.
    acc: dict = {}
    for src, dst in legs_of(chain):
        src_nodes, dst_nodes = _view_node_ids(src), _view_node_ids(dst)
        # Storage legs have no node id; they are attributed to the dst nodes.
        target_nodes = dst_nodes if dst_nodes is not None else src_nodes
        if target_nodes is None:
            continue
        if (
            src_nodes is not None
            and dst_nodes is not None
            and set(src_nodes) == set(dst_nodes)
        ):
            continue
        amount = _leg_bytes(dst)
        chosen = src_nodes if outbound else dst_nodes
        if chosen is None:
            chosen = target_nodes
        for node in chosen:
            acc[node] = acc.get(node, 0) + amount
    n = (max(acc) + 1) if acc else 0
    return [acc.get(i, 0) for i in range(n)]

"""Read chains (design B.1.1 chain_load, B.3.2), plans, and stats.

A read chain describes a storage->compute load through zero to two
node-local caches. Chain types, plans and stats are internal; the
gdtensor.builder namespace exposes user-visible chain products and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, cast

from .builder_mixin import field_hash, make_builder_mixin
from .plan import d_tensors_view_rank_plan_load, validate_rank_load
from .propagation import build_chain_plan, plan_mem, plan_traffic, rank_copy_plans
from .ranges import CopyPlan
from .result import catch, invalid, must_ok
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
        return list(getattr(c.dst, "tensor_ids"))
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


def concrete_chains(read_chain: DTensorsReadChain) -> list:
    """Concrete chains in execution order; Mixed is flattened."""
    t = type(read_chain)
    if t in (
        DirectDTensorsReadChain,
        OneLayerCachedDTensorsReadChain,
        TwoLayerCachedDTensorsReadChain,
    ):
        return [read_chain]
    if t is MixedDTensorsReadChain:
        return [
            seg.read_chain for seg in read_chain.disjoint_tensor_ids_and_read_chain_list
        ]
    raise invalid("read_chain", f"unknown chain type {t.__name__}")


def rank_direct_plan(rank: int, chain: DirectDTensorsReadChain) -> CopyPlan:
    """The storage -> compute plan of a direct chain, validated per rank."""
    plan = must_ok(d_tensors_view_rank_plan_load(rank, chain.dst, chain.src))
    assert isinstance(plan, list)
    validate_rank_load(rank, chain.dst, chain.src, plan)
    return plan


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
    if type(read_chain) is MixedDTensorsReadChain:
        return Ok(
            MixedCopyPlanChain(
                [
                    DisjointTensorIdsAndCopyPlanChain(
                        list(seg.tensor_ids), _plan_concrete(rank, seg.read_chain)
                    )
                    for seg in read_chain.disjoint_tensor_ids_and_read_chain_list
                ]
            )
        )
    concrete_chains(read_chain)
    return Ok(_plan_concrete(rank, read_chain))


def _plan_concrete(rank: int, chain) -> ConcreteCopyPlanChain:
    """Per-rank copy plans derived from the node-level propagation plan.

    The fill legs list the cache segments this rank's drains depend on (its
    closure). A segment shared by several ranks appears in each of their
    plans, but the executor fills it once per node.
    """
    if type(chain) is DirectDTensorsReadChain:
        return DirectCopyPlanChain(rank_direct_plan(rank, chain))
    legs = rank_copy_plans(build_chain_plan(chain), rank)
    if type(chain) is OneLayerCachedDTensorsReadChain:
        return OneLayerCachedCopyPlanChain(legs[0], legs[1])
    return TwoLayerCachedCopyPlanChain(legs[0], legs[1], legs[2])


# ------------------------------------------------------------------ B.1.6
#
# Stats return list[int] indexed by node id, in bytes, from the same
# node-level propagation plan the executor runs:
#   * mem_overhead: per node, the sum over cache layers of the layer budget
#     (the C largest cells that hold demanded bytes). The executor never lets
#     resident bytes exceed it, so the value is a hard bound.
#   * inbound / outbound: bytes on node-crossing edges. Storage -> first layer
#     is inbound of the filling node. Same-node edges carry no traffic. Each
#     segment crosses each edge once.


def _per_node(acc: dict) -> List[int]:
    n = (max(acc) + 1) if acc else 0
    return [int(acc.get(i, 0)) for i in range(n)]


def _cached(read_chain: DTensorsReadChain) -> list:
    return [
        c for c in concrete_chains(read_chain) if type(c) is not DirectDTensorsReadChain
    ]


def _add(acc: dict, part: dict) -> None:
    for node, nbytes in part.items():
        acc[node] = acc.get(node, 0) + nbytes


def get_stat_mem_overhead_of_chain_load(read_chain: DTensorsReadChain) -> List[int]:
    """Cache bytes each node must reserve (bytes). No session, no payload read."""
    acc: dict = {}
    for chain in _cached(read_chain):
        _add(acc, plan_mem(build_chain_plan(chain, with_storage=False)))
    return _per_node(acc)


def get_stat_outbound_traffic_of_chain_load(read_chain: DTensorsReadChain) -> List[int]:
    return _stat_traffic(read_chain, outbound=True)


def get_stat_inbound_traffic_of_chain_load(read_chain: DTensorsReadChain) -> List[int]:
    return _stat_traffic(read_chain, outbound=False)


def _stat_traffic(read_chain: DTensorsReadChain, outbound: bool) -> List[int]:
    acc: dict = {}
    for chain in _cached(read_chain):
        inbound, out = plan_traffic(build_chain_plan(chain, with_storage=False))
        _add(acc, out if outbound else inbound)
    return _per_node(acc)

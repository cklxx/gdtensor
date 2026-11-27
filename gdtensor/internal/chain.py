"""Unified RW chains (design B.1.1, B.3.2), CopyPlanChain, plans, and stats.

One ``DTensorsRWChain`` expresses either direction between storage and compute
through zero to two node-local ``CacheDTensorsView`` layers in data-flow order:
* Read  (Load): ``Storage -> first_cache -> second_cache -> Compute``
* Write (Save): ``Compute -> first_cache -> second_cache -> Storage``
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import List, cast

from .builder_mixin import field_hash, make_builder_mixin
from .plan import (
    d_tensors_view_rank_plan_load,
    d_tensors_view_rank_plan_save,
    validate_rank_load,
    validate_rank_save,
)
from .products import Ok
from .propagation import build_chain_plan, plan_mem, plan_traffic, rank_copy_plans
from .ranges import CopyPlan
from .result import catch, invalid, must_ok
from ..chain_types import Result


# ------------------------------------------------------------------ RW types


DirectDTensorsRWChainBuilderMixin = make_builder_mixin(
    "DirectDTensorsRWChain", [("src", "oneof"), ("dst", "oneof")]
)
OneLayerCachedDTensorsRWChainBuilderMixin = make_builder_mixin(
    "OneLayerCachedDTensorsRWChain",
    [
        ("src", "oneof"),
        ("cache", "object"),
        ("dst", "oneof"),
        ("num_ranks_per_node", "scalar"),
    ],
)
TwoLayerCachedDTensorsRWChainBuilderMixin = make_builder_mixin(
    "TwoLayerCachedDTensorsRWChain",
    [
        ("src", "oneof"),
        ("first_cache", "object"),
        ("second_cache", "object"),
        ("dst", "oneof"),
        ("num_ranks_per_node", "scalar"),
    ],
)
DisjointTensorIdsAndRWChainBuilderMixin = make_builder_mixin(
    "DisjointTensorIdsAndRWChain", [("tensor_ids", "list"), ("rw_chain", "object")]
)
MixedDTensorsRWChainBuilderMixin = make_builder_mixin(
    "MixedDTensorsRWChain",
    [("disjoint_tensor_ids_and_rw_chain_list", "list")],
)


@dataclass(eq=True)
class DirectDTensorsRWChain(DirectDTensorsRWChainBuilderMixin):
    src: object = None
    dst: object = None
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class OneLayerCachedDTensorsRWChain(OneLayerCachedDTensorsRWChainBuilderMixin):
    src: object = None
    cache: object = None
    dst: object = None
    num_ranks_per_node: int = 1
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class TwoLayerCachedDTensorsRWChain(TwoLayerCachedDTensorsRWChainBuilderMixin):
    src: object = None
    first_cache: object = None
    second_cache: object = None
    dst: object = None
    num_ranks_per_node: int = 1
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


ConcreteDTensorsRWChain = (
    DirectDTensorsRWChain
    | OneLayerCachedDTensorsRWChain
    | TwoLayerCachedDTensorsRWChain
)


@dataclass(eq=True)
class DisjointTensorIdsAndRWChain(DisjointTensorIdsAndRWChainBuilderMixin):
    tensor_ids: List[str]
    rw_chain: object = None
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class MixedDTensorsRWChain(MixedDTensorsRWChainBuilderMixin):
    disjoint_tensor_ids_and_rw_chain_list: List[DisjointTensorIdsAndRWChain]
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


DTensorsRWChain = (
    DirectDTensorsRWChain
    | OneLayerCachedDTensorsRWChain
    | TwoLayerCachedDTensorsRWChain
    | MixedDTensorsRWChain
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
    src_to_first_cache: CopyPlan
    first_cache_to_second_cache: CopyPlan
    second_cache_to_dst: CopyPlan
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


_CONCRETE_CHAIN_TYPES = (
    DirectDTensorsRWChain,
    OneLayerCachedDTensorsRWChain,
    TwoLayerCachedDTensorsRWChain,
)
_RW_CHAIN_TYPES = _CONCRETE_CHAIN_TYPES + (MixedDTensorsRWChain,)


def is_rw_chain(chain: object) -> bool:
    return type(chain) in _RW_CHAIN_TYPES


def concrete_direction(chain: object) -> str | None:
    """Return 'read', 'write', or None from endpoint view types."""
    from .views import ComputeDTensorsView, StorageDTensorsView

    st = type(getattr(chain, "src", None))
    dt = type(getattr(chain, "dst", None))
    if st is StorageDTensorsView and dt is ComputeDTensorsView:
        return "read"
    if st is ComputeDTensorsView and dt is StorageDTensorsView:
        return "write"
    return None


def chain_direction(rw_chain: object) -> str | None:
    """Direction of an RW chain ('read', 'write', or None for empty Mixed)."""
    t = type(rw_chain)
    if t in _CONCRETE_CHAIN_TYPES:
        d = concrete_direction(rw_chain)
        if d is None:
            raise invalid(
                "rw_chain",
                "endpoints must connect StorageDTensorsView and ComputeDTensorsView",
            )
        return d
    if t is MixedDTensorsRWChain:
        segs = cast(
            MixedDTensorsRWChain, rw_chain
        ).disjoint_tensor_ids_and_rw_chain_list
        if not segs:
            return None
        dirs = {chain_direction(seg.rw_chain) for seg in segs}
        if len(dirs) != 1:
            raise invalid("rw_chain", "mixed segments have inconsistent directions")
        return dirs.pop()
    raise invalid("rw_chain", f"unknown chain type {t.__name__}")


def _require_direction(rw_chain: object, expected: str) -> str | None:
    if not is_rw_chain(rw_chain):
        raise invalid("rw_chain", f"unknown chain type {type(rw_chain).__name__}")
    d = chain_direction(rw_chain)
    if d is not None and d != expected:
        raise invalid(
            "rw_chain", f"expected {expected}-direction DTensorsRWChain, got {d}"
        )
    return d


def _concrete_tensor_ids(chain) -> List[str]:
    if concrete_direction(chain) == "read":
        return list(getattr(chain.dst, "tensor_ids"))
    if chain.src is not None:
        return list(getattr(chain.src, "tensor_ids"))
    if chain.dst is not None:
        return list(getattr(chain.dst, "tensor_ids"))
    return []


def get_tensor_ids_of_rw_chain(rw_chain: DTensorsRWChain) -> List[str]:
    t = type(rw_chain)
    if t in _CONCRETE_CHAIN_TYPES:
        return _concrete_tensor_ids(rw_chain)
    if t is MixedDTensorsRWChain:
        ids: List[str] = []
        for seg in cast(
            MixedDTensorsRWChain, rw_chain
        ).disjoint_tensor_ids_and_rw_chain_list:
            ids.extend(get_tensor_ids_of_rw_chain(cast(DTensorsRWChain, seg.rw_chain)))
        return ids
    raise invalid("rw_chain", f"unknown chain type {t.__name__}")


def legs_of(rw_chain: DTensorsRWChain):
    """(src, dst) view pairs per leg, data-flow order; Mixed flattened."""
    t = type(rw_chain)
    if t is DirectDTensorsRWChain:
        c = cast(DirectDTensorsRWChain, rw_chain)
        return [(c.src, c.dst)]
    if t is OneLayerCachedDTensorsRWChain:
        c = cast(OneLayerCachedDTensorsRWChain, rw_chain)
        return [(c.src, c.cache), (c.cache, c.dst)]
    if t is TwoLayerCachedDTensorsRWChain:
        c = cast(TwoLayerCachedDTensorsRWChain, rw_chain)
        return [
            (c.src, c.first_cache),
            (c.first_cache, c.second_cache),
            (c.second_cache, c.dst),
        ]
    if t is MixedDTensorsRWChain:
        c = cast(MixedDTensorsRWChain, rw_chain)
        legs = []
        for seg in c.disjoint_tensor_ids_and_rw_chain_list:
            legs.extend(legs_of(cast(DTensorsRWChain, seg.rw_chain)))
        return legs
    raise invalid("rw_chain", f"unknown chain type {t.__name__}")


def concrete_chains(rw_chain: DTensorsRWChain) -> list:
    """Concrete RW chains in execution order; Mixed is flattened."""
    t = type(rw_chain)
    if t in _CONCRETE_CHAIN_TYPES:
        return [rw_chain]
    if t is MixedDTensorsRWChain:
        return [
            seg.rw_chain
            for seg in cast(
                MixedDTensorsRWChain, rw_chain
            ).disjoint_tensor_ids_and_rw_chain_list
        ]
    raise invalid("rw_chain", f"unknown chain type {t.__name__}")


def rank_direct_plan(rank: int, chain: DirectDTensorsRWChain) -> CopyPlan:
    """The storage -> compute plan of a direct read RW chain, validated per rank."""
    plan = must_ok(d_tensors_view_rank_plan_load(rank, chain.dst, chain.src))
    assert isinstance(plan, list)
    validate_rank_load(rank, chain.dst, chain.src, plan)
    return plan


def rank_direct_write_plan(rank: int, chain: DirectDTensorsRWChain) -> CopyPlan:
    """The compute -> storage plan of a direct write RW chain, validated per rank."""
    plan = must_ok(d_tensors_view_rank_plan_save(rank, chain.src, chain.dst))
    assert isinstance(plan, list)
    validate_rank_save(rank, chain.src, chain.dst, plan)
    return plan


# ------------------------------------------------------------------ B.1.2


def validate_rank(rank: int) -> int:
    if (
        isinstance(rank, bool)
        or not isinstance(rank, numbers.Integral)
        or int(rank) < 0
    ):
        raise invalid("rank", f"rank must be non-negative, got {rank!r}")
    return int(rank)


@catch
def d_tensors_view_rank_plan_chain_load(
    rank: int, rw_chain: DTensorsRWChain
) -> Result[CopyPlanChain]:
    from .plan import _check_same_tensors

    _require_direction(rw_chain, "read")
    for src, dst in legs_of(rw_chain):
        _check_same_tensors(src, dst)
    if type(rw_chain) is MixedDTensorsRWChain:
        return Ok(
            MixedCopyPlanChain(
                [
                    DisjointTensorIdsAndCopyPlanChain(
                        list(seg.tensor_ids), _plan_concrete_read(rank, seg.rw_chain)
                    )
                    for seg in rw_chain.disjoint_tensor_ids_and_rw_chain_list
                ]
            )
        )
    concrete_chains(rw_chain)
    return Ok(_plan_concrete_read(rank, rw_chain))


def _plan_concrete_read(rank: int, chain) -> ConcreteCopyPlanChain:
    """Per-rank read copy plans derived from the node-level propagation plan."""
    if type(chain) is DirectDTensorsRWChain:
        return DirectCopyPlanChain(rank_direct_plan(rank, chain))
    legs = rank_copy_plans(build_chain_plan(chain), rank)
    if type(chain) is OneLayerCachedDTensorsRWChain:
        return OneLayerCachedCopyPlanChain(legs[0], legs[1])
    return TwoLayerCachedCopyPlanChain(legs[0], legs[1], legs[2])


@catch
def d_tensors_view_rank_plan_chain_save(
    rank: int, rw_chain: DTensorsRWChain
) -> Result[CopyPlanChain]:
    from .plan import _check_same_tensors

    _require_direction(rw_chain, "write")
    must_ok(rw_chain.check_constraints())
    for src, dst in legs_of(rw_chain):
        _check_same_tensors(src, dst)
    rank = validate_rank(rank)
    if type(rw_chain) is MixedDTensorsRWChain:
        return Ok(
            MixedCopyPlanChain(
                [
                    DisjointTensorIdsAndCopyPlanChain(
                        list(seg.tensor_ids), _plan_concrete_write(rank, seg.rw_chain)
                    )
                    for seg in rw_chain.disjoint_tensor_ids_and_rw_chain_list
                ]
            )
        )
    concrete_chains(rw_chain)
    return Ok(_plan_concrete_write(rank, rw_chain))


def _plan_concrete_write(rank: int, chain) -> ConcreteCopyPlanChain:
    """Per-rank write copy plans derived from the node-level write plan."""
    from .write_propagation import build_write_chain_plan, rank_write_copy_plans

    rank = validate_rank(rank)
    if type(chain) is DirectDTensorsRWChain:
        return DirectCopyPlanChain(rank_direct_write_plan(rank, chain))
    legs = rank_write_copy_plans(build_write_chain_plan(chain), rank)
    if type(chain) is OneLayerCachedDTensorsRWChain:
        return OneLayerCachedCopyPlanChain(legs[0], legs[1])
    return TwoLayerCachedCopyPlanChain(legs[0], legs[1], legs[2])


# ------------------------------------------------------------------ B.1.6


def _per_node(acc: dict) -> List[int]:
    n = (max(acc) + 1) if acc else 0
    return [int(acc.get(i, 0)) for i in range(n)]


def _cached(rw_chain: DTensorsRWChain) -> list:
    return [
        c for c in concrete_chains(rw_chain) if type(c) is not DirectDTensorsRWChain
    ]


def _add(acc: dict, part: dict) -> None:
    for node, nbytes in part.items():
        acc[node] = acc.get(node, 0) + nbytes


def get_stat_mem_overhead_of_rw_chain(rw_chain: DTensorsRWChain) -> List[int]:
    """Cache bytes each node must reserve (bytes). No session, no payload read."""
    from .write_propagation import build_write_chain_plan, plan_write_mem

    chain_direction(rw_chain)
    acc: dict = {}
    for c in _cached(rw_chain):
        if concrete_direction(c) == "write":
            _add(acc, plan_write_mem(build_write_chain_plan(c, with_storage=False)))
        else:
            _add(acc, plan_mem(build_chain_plan(c, with_storage=False)))
    return _per_node(acc)


def get_stat_outbound_traffic_of_rw_chain(rw_chain: DTensorsRWChain) -> List[int]:
    return _stat_traffic(rw_chain, outbound=True)


def get_stat_inbound_traffic_of_rw_chain(rw_chain: DTensorsRWChain) -> List[int]:
    return _stat_traffic(rw_chain, outbound=False)


def _stat_traffic(rw_chain: DTensorsRWChain, outbound: bool) -> List[int]:
    from .write_propagation import build_write_chain_plan, plan_write_traffic

    chain_direction(rw_chain)
    acc: dict = {}
    for c in _cached(rw_chain):
        if concrete_direction(c) == "write":
            inb, out = plan_write_traffic(build_write_chain_plan(c, with_storage=False))
        else:
            inb, out = plan_traffic(build_chain_plan(c, with_storage=False))
        _add(acc, out if outbound else inb)
    return _per_node(acc)

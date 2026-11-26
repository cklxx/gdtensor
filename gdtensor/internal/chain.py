"""Read/write chains (design B.1.1, B.3.2), plans, and stats.

A read chain describes a storage->compute load through zero to two node-local
caches. A write chain describes the symmetric compute->storage save through
zero to two node-local caches. Chain types, plans and stats are internal; the
gdtensor.builder namespace exposes user-visible chain products and helpers.
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
from .propagation import build_chain_plan, plan_mem, plan_traffic, rank_copy_plans
from .ranges import CopyPlan
from .result import catch, invalid, must_ok
from .products import Ok
from ..chain_types import Result

# ------------------------------------------------------------------ read types


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


# ------------------------------------------------------------------ write types


DirectDTensorsWriteChainBuilderMixin = make_builder_mixin(
    "DirectDTensorsWriteChain", [("src", "object"), ("dst", "object")]
)
OneLayerCachedDTensorsWriteChainBuilderMixin = make_builder_mixin(
    "OneLayerCachedDTensorsWriteChain",
    [
        ("src", "object"),
        ("cache", "object"),
        ("dst", "object"),
        ("num_ranks_per_node", "scalar"),
    ],
)
TwoLayerCachedDTensorsWriteChainBuilderMixin = make_builder_mixin(
    "TwoLayerCachedDTensorsWriteChain",
    [
        ("src", "object"),
        ("big_cache", "object"),
        ("small_cache", "object"),
        ("dst", "object"),
        ("num_ranks_per_node", "scalar"),
    ],
)
DisjointTensorIdsAndWriteChainBuilderMixin = make_builder_mixin(
    "DisjointTensorIdsAndWriteChain",
    [("tensor_ids", "list"), ("write_chain", "object")],
)
MixedDTensorsWriteChainBuilderMixin = make_builder_mixin(
    "MixedDTensorsWriteChain",
    [("disjoint_tensor_ids_and_write_chain_list", "list")],
)


@dataclass(eq=True)
class DirectDTensorsWriteChain(DirectDTensorsWriteChainBuilderMixin):
    src: object = None
    dst: object = None
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class OneLayerCachedDTensorsWriteChain(OneLayerCachedDTensorsWriteChainBuilderMixin):
    src: object = None
    cache: object = None
    dst: object = None
    num_ranks_per_node: int = 1
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class TwoLayerCachedDTensorsWriteChain(TwoLayerCachedDTensorsWriteChainBuilderMixin):
    src: object = None
    big_cache: object = None
    small_cache: object = None
    dst: object = None
    num_ranks_per_node: int = 1
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


ConcreteDTensorsWriteChain = (
    DirectDTensorsWriteChain
    | OneLayerCachedDTensorsWriteChain
    | TwoLayerCachedDTensorsWriteChain
)


@dataclass(eq=True)
class DisjointTensorIdsAndWriteChain(DisjointTensorIdsAndWriteChainBuilderMixin):
    tensor_ids: List[str]
    write_chain: object = None
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


@dataclass(eq=True)
class MixedDTensorsWriteChain(MixedDTensorsWriteChainBuilderMixin):
    disjoint_tensor_ids_and_write_chain_list: List[DisjointTensorIdsAndWriteChain]
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


DTensorsWriteChain = (
    DirectDTensorsWriteChain
    | OneLayerCachedDTensorsWriteChain
    | TwoLayerCachedDTensorsWriteChain
    | MixedDTensorsWriteChain
)

DTensorsChain = DTensorsReadChain | DTensorsWriteChain


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


@dataclass(eq=True)
class TwoLayerCachedWriteCopyPlanChain:
    src_to_big_cache: CopyPlan
    big_cache_to_small_cache: CopyPlan
    small_cache_to_dst: CopyPlan
    __hash__ = field_hash


ConcreteWriteCopyPlanChain = (
    DirectCopyPlanChain | OneLayerCachedCopyPlanChain | TwoLayerCachedWriteCopyPlanChain
)


@dataclass(eq=True)
class DisjointTensorIdsAndWriteCopyPlanChain:
    tensor_ids: List[str]
    copy_plan_chain: ConcreteWriteCopyPlanChain
    __hash__ = field_hash


@dataclass(eq=True)
class MixedWriteCopyPlanChain:
    disjoint_tensor_ids_and_copy_plan_chain_list: List[
        DisjointTensorIdsAndWriteCopyPlanChain
    ]
    __hash__ = field_hash


WriteCopyPlanChain = (
    DirectCopyPlanChain
    | OneLayerCachedCopyPlanChain
    | TwoLayerCachedWriteCopyPlanChain
    | MixedWriteCopyPlanChain
)


# ----------------------------------------------------------------- helpers

_WRITE_CHAIN_TYPES = (
    DirectDTensorsWriteChain,
    OneLayerCachedDTensorsWriteChain,
    TwoLayerCachedDTensorsWriteChain,
    MixedDTensorsWriteChain,
)


def is_write_chain(chain: object) -> bool:
    return type(chain) in _WRITE_CHAIN_TYPES


def get_tensor_ids_of_write_chain(write_chain: DTensorsWriteChain) -> List[str]:
    t = type(write_chain)
    if t in (
        DirectDTensorsWriteChain,
        OneLayerCachedDTensorsWriteChain,
        TwoLayerCachedDTensorsWriteChain,
    ):
        c = cast(DirectDTensorsWriteChain, write_chain)
        return list(getattr(c.src, "tensor_ids"))
    if t is MixedDTensorsWriteChain:
        ids: List[str] = []
        for seg in cast(
            MixedDTensorsWriteChain, write_chain
        ).disjoint_tensor_ids_and_write_chain_list:
            ids.extend(
                get_tensor_ids_of_write_chain(cast(DTensorsWriteChain, seg.write_chain))
            )
        return ids
    raise invalid("write_chain", f"unknown chain type {t.__name__}")


def get_tensor_ids_of_read_chain(chain: DTensorsChain) -> List[str]:
    if is_write_chain(chain):
        return get_tensor_ids_of_write_chain(cast(DTensorsWriteChain, chain))
    t = type(chain)
    if t in (
        DirectDTensorsReadChain,
        OneLayerCachedDTensorsReadChain,
        TwoLayerCachedDTensorsReadChain,
    ):
        c = cast(DirectDTensorsReadChain, chain)
        return list(getattr(c.dst, "tensor_ids"))
    if t is MixedDTensorsReadChain:
        ids: List[str] = []
        for seg in cast(
            MixedDTensorsReadChain, chain
        ).disjoint_tensor_ids_and_read_chain_list:
            ids.extend(
                get_tensor_ids_of_read_chain(cast(DTensorsReadChain, seg.read_chain))
            )
        return ids
    raise invalid("read_chain", f"unknown chain type {t.__name__}")


def _write_legs_of(chain: DTensorsWriteChain) -> list:
    t = type(chain)
    if t is DirectDTensorsWriteChain:
        c = cast(DirectDTensorsWriteChain, chain)
        return [(c.src, c.dst)]
    if t is OneLayerCachedDTensorsWriteChain:
        c = cast(OneLayerCachedDTensorsWriteChain, chain)
        return [(c.src, c.cache), (c.cache, c.dst)]
    if t is TwoLayerCachedDTensorsWriteChain:
        c = cast(TwoLayerCachedDTensorsWriteChain, chain)
        return [
            (c.src, c.big_cache),
            (c.big_cache, c.small_cache),
            (c.small_cache, c.dst),
        ]
    c_mix = cast(MixedDTensorsWriteChain, chain)
    legs = []
    for seg in c_mix.disjoint_tensor_ids_and_write_chain_list:
        legs.extend(_write_legs_of(cast(DTensorsWriteChain, seg.write_chain)))
    return legs


def legs_of(chain: DTensorsChain):
    """(src, dst) view pairs per leg, data-flow order; Mixed flattened."""
    if is_write_chain(chain):
        return _write_legs_of(cast(DTensorsWriteChain, chain))
    t = type(chain)
    if t is DirectDTensorsReadChain:
        c = cast(DirectDTensorsReadChain, chain)
        return [(c.src, c.dst)]
    if t is OneLayerCachedDTensorsReadChain:
        c = cast(OneLayerCachedDTensorsReadChain, chain)
        return [(c.src, c.cache), (c.cache, c.dst)]
    if t is TwoLayerCachedDTensorsReadChain:
        c = cast(TwoLayerCachedDTensorsReadChain, chain)
        return [
            (c.src, c.small_cache),
            (c.small_cache, c.big_cache),
            (c.big_cache, c.dst),
        ]
    if t is MixedDTensorsReadChain:
        c = cast(MixedDTensorsReadChain, chain)
        legs = []
        for seg in c.disjoint_tensor_ids_and_read_chain_list:
            legs.extend(legs_of(cast(DTensorsReadChain, seg.read_chain)))
        return legs
    raise invalid("read_chain", f"unknown chain type {t.__name__}")


def concrete_chains(read_chain: DTensorsReadChain) -> list:
    """Concrete read chains in execution order; Mixed is flattened."""
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


def concrete_write_chains(write_chain: DTensorsWriteChain) -> list:
    """Concrete write chains in execution order; Mixed is flattened."""
    t = type(write_chain)
    if t in (
        DirectDTensorsWriteChain,
        OneLayerCachedDTensorsWriteChain,
        TwoLayerCachedDTensorsWriteChain,
    ):
        return [write_chain]
    if t is MixedDTensorsWriteChain:
        return [
            seg.write_chain
            for seg in write_chain.disjoint_tensor_ids_and_write_chain_list
        ]
    raise invalid("write_chain", f"unknown chain type {t.__name__}")


def rank_direct_plan(rank: int, chain: DirectDTensorsReadChain) -> CopyPlan:
    """The storage -> compute plan of a direct read chain, validated per rank."""
    plan = must_ok(d_tensors_view_rank_plan_load(rank, chain.dst, chain.src))
    assert isinstance(plan, list)
    validate_rank_load(rank, chain.dst, chain.src, plan)
    return plan


def rank_direct_write_plan(rank: int, chain: DirectDTensorsWriteChain) -> CopyPlan:
    """The compute -> storage plan of a direct write chain, validated per rank."""
    plan = must_ok(d_tensors_view_rank_plan_save(rank, chain.src, chain.dst))
    assert isinstance(plan, list)
    validate_rank_save(rank, chain.src, chain.dst, plan)
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

    Fill legs list only work the rank may perform: local consumer fills plus
    assigned DN service work. Shared local fills may appear in several plans;
    the coordinator admits each segment once per generation.
    """
    if type(chain) is DirectDTensorsReadChain:
        return DirectCopyPlanChain(rank_direct_plan(rank, chain))
    legs = rank_copy_plans(build_chain_plan(chain), rank)
    if type(chain) is OneLayerCachedDTensorsReadChain:
        return OneLayerCachedCopyPlanChain(legs[0], legs[1])
    return TwoLayerCachedCopyPlanChain(legs[0], legs[1], legs[2])


def validate_rank(rank: int) -> int:
    if (
        isinstance(rank, bool)
        or not isinstance(rank, numbers.Integral)
        or int(rank) < 0
    ):
        raise invalid("rank", f"rank must be non-negative, got {rank!r}")
    return int(rank)


@catch
def d_tensors_view_rank_plan_chain_save(
    rank: int, write_chain: DTensorsWriteChain
) -> Result[WriteCopyPlanChain]:
    from .plan import _check_same_tensors

    if not is_write_chain(write_chain):
        raise invalid("write_chain", f"unknown chain type {type(write_chain).__name__}")
    must_ok(write_chain.check_constraints())
    for src, dst in legs_of(write_chain):
        _check_same_tensors(src, dst)
    rank = validate_rank(rank)
    if type(write_chain) is MixedDTensorsWriteChain:
        return Ok(
            MixedWriteCopyPlanChain(
                [
                    DisjointTensorIdsAndWriteCopyPlanChain(
                        list(seg.tensor_ids),
                        _plan_concrete_write(rank, seg.write_chain),
                    )
                    for seg in write_chain.disjoint_tensor_ids_and_write_chain_list
                ]
            )
        )
    concrete_write_chains(write_chain)
    return Ok(_plan_concrete_write(rank, write_chain))


def _plan_concrete_write(rank: int, chain) -> ConcreteWriteCopyPlanChain:
    """Per-rank copy plans derived from the node-level write-aggregation plan."""
    from .write_propagation import build_write_chain_plan, rank_write_copy_plans

    rank = validate_rank(rank)
    if type(chain) is DirectDTensorsWriteChain:
        return DirectCopyPlanChain(rank_direct_write_plan(rank, chain))
    legs = rank_write_copy_plans(build_write_chain_plan(chain), rank)
    if type(chain) is OneLayerCachedDTensorsWriteChain:
        return OneLayerCachedCopyPlanChain(legs[0], legs[1])
    return TwoLayerCachedWriteCopyPlanChain(legs[0], legs[1], legs[2])


# ------------------------------------------------------------------ B.1.6


def _per_node(acc: dict) -> List[int]:
    n = (max(acc) + 1) if acc else 0
    return [int(acc.get(i, 0)) for i in range(n)]


def _cached(chain: DTensorsChain) -> list:
    if is_write_chain(chain):
        return [
            c
            for c in concrete_write_chains(cast(DTensorsWriteChain, chain))
            if type(c) is not DirectDTensorsWriteChain
        ]
    return [
        c
        for c in concrete_chains(cast(DTensorsReadChain, chain))
        if type(c) is not DirectDTensorsReadChain
    ]


def _add(acc: dict, part: dict) -> None:
    for node, nbytes in part.items():
        acc[node] = acc.get(node, 0) + nbytes


def get_stat_mem_overhead_of_chain_load(chain: DTensorsChain) -> List[int]:
    """Cache bytes each node must reserve (bytes). No session, no payload read."""
    from .write_propagation import build_write_chain_plan, plan_write_mem

    acc: dict = {}
    if is_write_chain(chain):
        for c in _cached(chain):
            _add(acc, plan_write_mem(build_write_chain_plan(c, with_storage=False)))
        return _per_node(acc)
    for c in _cached(chain):
        _add(acc, plan_mem(build_chain_plan(c, with_storage=False)))
    return _per_node(acc)


def get_stat_outbound_traffic_of_chain_load(chain: DTensorsChain) -> List[int]:
    return _stat_traffic(chain, outbound=True)


def get_stat_inbound_traffic_of_chain_load(chain: DTensorsChain) -> List[int]:
    return _stat_traffic(chain, outbound=False)


def _stat_traffic(chain: DTensorsChain, outbound: bool) -> List[int]:
    from .write_propagation import build_write_chain_plan, plan_write_traffic

    acc: dict = {}
    if is_write_chain(chain):
        for c in _cached(chain):
            inb, out = plan_write_traffic(build_write_chain_plan(c, with_storage=False))
            _add(acc, out if outbound else inb)
        return _per_node(acc)
    for c in _cached(chain):
        inbound, out = plan_traffic(build_chain_plan(c, with_storage=False))
        _add(acc, out if outbound else inbound)
    return _per_node(acc)

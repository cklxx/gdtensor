"""Meshes (design B.2.1.2, B.2.2.4, B.3.2) and mesh resolution (internal).

Data classes follow C.1 / C.2: each class ``Foo`` inherits ``FooBuilderMixin``
and has ``check_constraints``. Tagged sizes use ``gdtensor.tag``:
``ArrangedMesh(size=tag.WorldSize(4))``, ``LiteralMesh(ranks=[1, 3],
max=tag.WorldSize(8))``.

Resolution (internal): a mesh resolves to an integer ndarray.
* ComputeMesh -> each cell is a global rank.
* StorageMesh -> each cell is a slot id ``t * W + w``; ``t`` is the tensor-slot
  row, ``w`` is the content-slot column (global rank), ``W`` is the
  content-slot world size. Cells not selected by a LiteralMesh are -1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import GenericAlias
from typing import Any, List, Optional, Tuple

import numpy as np

from .builder_mixin import field_hash, make_builder_mixin
from .result import invalid
from ..tag import (
    ConcurrencyWidth,
    ConstantOne,
    NumNodes,
    TensorCount,
    WorldSize,
)


def _cc(self):
    from .constraints import check_constraints

    return check_constraints(self)


# ---------------------------------------------------------------- B.3.2

ArrangedMeshBuilderMixin = make_builder_mixin("ArrangedMesh", [("size", "scalar")])
LiteralMeshBuilderMixin = make_builder_mixin(
    "LiteralMesh", [("ranks", "list"), ("max", "scalar")]
)
ReshapedMeshBuilderMixin = make_builder_mixin(
    "ReshapedMesh", [("underlying_mesh", "oneof"), ("shape", "list")]
)
PermutedMeshBuilderMixin = make_builder_mixin(
    "PermutedMesh", [("underlying_mesh", "oneof"), ("permutation", "list")]
)
SlicedMeshBuilderMixin = make_builder_mixin(
    "SlicedMesh", [("underlying_mesh", "oneof"), ("slices", "list")]
)
SliceBuilderMixin = make_builder_mixin(
    "Slice", [("start", "nullable"), ("stop", "nullable"), ("step", "nullable")]
)
SlotOuterProductMeshBuilderMixin = make_builder_mixin(
    "SlotOuterProductMesh", [("tensor_slots", "object"), ("content_slots", "oneof")]
)


_hash = field_hash


@dataclass(eq=True)
class ArrangedMesh(ArrangedMeshBuilderMixin):
    size: Any = None  # tagged[tag_name, int]: tag.WorldSize / TensorCount / ConstantOne
    __hash__ = _hash

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


@dataclass(eq=True)
class LiteralMesh(LiteralMeshBuilderMixin):
    ranks: List[int] = field(default_factory=list)
    max: Any = None  # tagged[tag_name, int]
    __hash__ = _hash

    def __post_init__(self) -> None:
        self.ranks = list(self.ranks)

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


@dataclass(eq=True)
class ReshapedMesh(ReshapedMeshBuilderMixin):
    underlying_mesh: Any = None
    shape: List[int] = field(default_factory=list)
    __hash__ = _hash

    def __post_init__(self) -> None:
        self.shape = list(self.shape)

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


@dataclass(eq=True)
class PermutedMesh(PermutedMeshBuilderMixin):
    underlying_mesh: Any = None
    permutation: List[int] = field(default_factory=list)
    __hash__ = _hash

    def __post_init__(self) -> None:
        self.permutation = list(self.permutation)

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


@dataclass(eq=True)
class Slice(SliceBuilderMixin):
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None
    __hash__ = _hash

    def check_constraints(self):
        return _cc(self)


@dataclass(eq=True)
class SlicedMesh(SlicedMeshBuilderMixin):
    underlying_mesh: Any = None
    slices: List[Optional[Slice]] = field(default_factory=list)
    __hash__ = _hash

    def __post_init__(self) -> None:
        self.slices = list(self.slices)

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


# ---------------------------------------------------------------- B.2.2.4


@dataclass(eq=True)
class SlotOuterProductMesh(SlotOuterProductMeshBuilderMixin):
    tensor_slots: Any = None  # ArrangedMesh[ConstantOne] | ArrangedMesh[TensorCount]
    content_slots: Any = None  # ContentSlotsMesh
    __hash__ = _hash

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


ComputeMesh = LiteralMesh | ArrangedMesh | ReshapedMesh | PermutedMesh | SlicedMesh
StorageMesh = SlotOuterProductMesh | ReshapedMesh | SlicedMesh
ContentSlotsMesh = LiteralMesh | ArrangedMesh


# ---------------------------------------------------------------- resolution (internal)


def tag_int(tagged, *tags) -> int:
    if type(tagged) not in tags:
        names = "/".join(t.__name__ for t in tags)
        raise invalid("mesh", f"expected tag {names}, got {tagged!r}")
    return int(tagged.value)


def _apply_transform(mesh, inner: np.ndarray) -> np.ndarray:
    if type(mesh) is ReshapedMesh:
        shape = tuple(mesh.shape)
        if int(np.prod(shape, dtype=np.int64)) != inner.size:
            raise invalid("mesh", f"reshape {inner.shape} -> {shape} changes mesh size")
        return inner.reshape(shape)
    if type(mesh) is PermutedMesh:
        perm = tuple(mesh.permutation)
        if sorted(perm) != list(range(inner.ndim)):
            raise invalid("mesh", f"bad permutation {perm} for mesh ndim {inner.ndim}")
        return inner.transpose(perm)
    if type(mesh) is SlicedMesh:
        if len(mesh.slices) != inner.ndim:
            raise invalid(
                "mesh", f"slices {len(mesh.slices)} != mesh ndim {inner.ndim}"
            )
        idx = tuple(
            slice(None) if s is None else slice(s.start, s.stop, s.step)
            for s in mesh.slices
        )
        out = inner[idx]
        if out.size == 0:
            raise invalid("mesh", "sliced mesh is empty")
        return out
    raise invalid("mesh", f"unknown mesh node {type(mesh).__name__}")


def resolve_compute_mesh(mesh) -> np.ndarray:
    """ndarray of global ranks of a ComputeMesh."""
    if type(mesh) is LiteralMesh:
        tag_int(mesh.max, WorldSize)
        return np.array(mesh.ranks, dtype=np.int64)
    if type(mesh) is ArrangedMesh:
        return np.arange(tag_int(mesh.size, WorldSize), dtype=np.int64)
    if type(mesh) not in (ReshapedMesh, PermutedMesh, SlicedMesh):
        raise invalid("mesh", f"{type(mesh).__name__} is not a ComputeMesh branch")
    return _apply_transform(mesh, resolve_compute_mesh(mesh.underlying_mesh))


def storage_mesh_base(mesh) -> SlotOuterProductMesh:
    while type(mesh) is not SlotOuterProductMesh:
        if type(mesh) not in (ReshapedMesh, SlicedMesh):
            raise invalid("mesh", f"{type(mesh).__name__} is not a StorageMesh branch")
        mesh = mesh.underlying_mesh
    return mesh


def _content_slots(cs) -> Tuple[int, List[int]]:
    if type(cs) is LiteralMesh:
        return tag_int(cs.max, WorldSize), list(cs.ranks)
    if type(cs) is ArrangedMesh:
        w = tag_int(cs.size, WorldSize)
        return w, list(range(w))
    raise invalid("mesh", f"{type(cs).__name__} is not a ContentSlotsMesh branch")


def resolve_storage_mesh(mesh) -> Tuple[np.ndarray, int]:
    """(ndarray of slot ids ``t*W+w`` or -1, W)."""
    if type(mesh) is not SlotOuterProductMesh:
        if type(mesh) not in (ReshapedMesh, SlicedMesh):
            raise invalid("mesh", f"{type(mesh).__name__} is not a StorageMesh branch")
        inner, w_size = resolve_storage_mesh(mesh.underlying_mesh)
        return _apply_transform(mesh, inner), w_size
    if type(mesh.tensor_slots) is not ArrangedMesh:
        raise invalid("mesh", "tensor_slots must be an ArrangedMesh")
    t_size = tag_int(mesh.tensor_slots.size, ConstantOne, TensorCount)
    w_size, w_ranks = _content_slots(mesh.content_slots)
    cells = np.full((t_size, w_size), -1, dtype=np.int64)
    for t in range(t_size):
        for w in w_ranks:
            cells[t, w] = t * w_size + w
    return cells, w_size


def is_single_tensor_storage_mesh(mesh) -> bool:
    return type(storage_mesh_base(mesh).tensor_slots.size) is ConstantOne


def is_multi_tensors_storage_mesh(mesh) -> bool:
    return type(storage_mesh_base(mesh).tensor_slots.size) is TensorCount


def storage_mesh_cells(mesh) -> set:
    """Set of (t, w) cells covered by a storage mesh."""
    cells, w_size = resolve_storage_mesh(mesh)
    return {divmod(int(c), w_size) for c in cells.ravel() if c >= 0}


def mesh_shape(mesh) -> Tuple[int, ...]:
    """Shape of a compute or storage mesh."""
    if type(storage_root(mesh)) is SlotOuterProductMesh:
        return tuple(resolve_storage_mesh(mesh)[0].shape)
    return tuple(resolve_compute_mesh(mesh).shape)


def storage_root(mesh):
    while type(mesh) in (ReshapedMesh, PermutedMesh, SlicedMesh):
        mesh = mesh.underlying_mesh
    return mesh


def mesh_coords(cells: np.ndarray, value: int) -> Optional[Tuple[int, ...]]:
    hit = np.argwhere(cells == value)
    if hit.shape[0] == 0:
        return None
    if hit.shape[0] > 1:
        raise invalid(
            "mesh", f"participant {value} appears {hit.shape[0]} times in mesh"
        )
    return tuple(int(x) for x in hit[0])


# ---------------------------------------------------------------- B.2.3.2

StCacheSlotOuterProductMeshBuilderMixin = make_builder_mixin(
    "StCacheOuterProductMesh",
    [("temporal", "oneof"), ("spatial", "oneof"), ("concurrency", "oneof")],
)


@dataclass(eq=True)
class StCacheOuterProductMesh(StCacheSlotOuterProductMeshBuilderMixin):
    temporal: Any = None  # TemporalCacheMesh
    spatial: Any = None  # SpatioCacheMesh
    concurrency: Any = None  # ArrangedMesh[ConcurrencyWidth]
    __hash__ = _hash

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


# Oneof unions (design B.2.3.2), not products: temporal is an arranged step
# count or a reshape of the spatial branch; spatial is a literal node list or
# a reshape of the temporal branch.
TemporalCacheMesh = ArrangedMesh | ReshapedMesh
SpatioCacheMesh = LiteralMesh | ReshapedMesh

CacheMesh = StCacheOuterProductMesh | PermutedMesh


def _unwrap_cache(v):
    while type(v) is ReshapedMesh:
        v = v.underlying_mesh
    return v


def _cache_base(mesh):
    while type(mesh) is PermutedMesh:
        mesh = mesh.underlying_mesh
    return mesh


def cache_num_steps(mesh) -> int:
    """Temporal step count: the arranged num_steps, or the spatial node count
    when temporal is a reshape of the spatial branch (element count preserved)."""
    base = _unwrap_cache(_cache_base(mesh).temporal)
    if type(base) is ArrangedMesh:
        return int(base.size.value)
    if type(base) is LiteralMesh:
        return len(base.ranks)
    raise invalid("mesh", f"unsupported temporal cache mesh {type(base).__name__}")


def cache_concurrency_width(mesh) -> int:
    c = _cache_base(mesh).concurrency
    tag_int(c.size, ConcurrencyWidth)
    return int(c.size.value)


def resolve_cache_mesh(mesh) -> np.ndarray:
    """ndarray of spatial NODE indices covered by a CacheMesh (shape (N, 1)).

    Temporal and concurrency dimensions add no ranks; they are buffer slots on
    the nodes selected by the spatial dimension. Expanding a node to global
    ranks is done by cache_node_ranks with num_ranks_per_node.
    """
    if type(mesh) is PermutedMesh:
        inner = resolve_cache_mesh(mesh.underlying_mesh)
        return inner.reshape(-1, 1)
    if type(mesh) is not StCacheOuterProductMesh:
        raise invalid("mesh", f"{type(mesh).__name__} is not a CacheMesh branch")
    return _cache_spatial_ranks(mesh.spatial)


def cache_num_nodes(mesh) -> int:
    spatial = _cache_base(mesh).spatial
    while type(spatial) is PermutedMesh:
        spatial = spatial.underlying_mesh
    base = _unwrap_cache(spatial)
    if type(base) is ArrangedMesh:
        # Reshaped-from-temporal carries a NumSteps tag whose count equals the
        # node count by reshape equality; the literal branch carries NumNodes.
        return int(base.size.value)
    if type(base) is LiteralMesh:
        tag_int(base.max, NumNodes)
        return len(base.ranks)
    raise invalid("mesh", f"unsupported spatial cache mesh {type(base).__name__}")


def _cache_spatial_ranks(spatial) -> np.ndarray:
    perm = None
    while type(spatial) is PermutedMesh:
        perm = list(spatial.permutation) if perm is None else perm
        spatial = spatial.underlying_mesh
    base = _unwrap_cache(spatial)
    if type(base) is ArrangedMesh:
        ranks = np.arange(int(base.size.value), dtype=np.int64)
    elif type(base) is LiteralMesh:
        tag_int(base.max, NumNodes)
        ranks = np.array(sorted(base.ranks), dtype=np.int64)
    else:
        raise invalid("mesh", f"unsupported spatial cache mesh {type(base).__name__}")
    if perm is not None:
        ranks = ranks[np.array(perm, dtype=np.int64)]
    return ranks.reshape(-1, 1)


def cache_node_set(mesh) -> list:
    """Sorted spatial node indices a cache mesh covers."""
    return [int(n) for n in np.unique(resolve_cache_mesh(mesh).ravel()) if n >= 0]


def cache_node_ranks(mesh, num_ranks_per_node: int) -> set:
    """Global ranks covered by a cache mesh: node n -> [n*k, (n+1)*k)."""
    if num_ranks_per_node <= 0:
        raise invalid("num_ranks_per_node", "must be positive")
    ranks: set = set()
    for node in cache_node_set(mesh):
        base = node * num_ranks_per_node
        ranks.update(range(base, base + num_ranks_per_node))
    return ranks


def is_cache_mesh_subset_of_mesh(lhs, rhs, num_ranks_per_node: int) -> bool:
    """Every compute rank covered by the cache mesh is also covered by rhs.

    A cache node n covers ranks [n * k, (n + 1) * k), k = num_ranks_per_node.
    Temporal and concurrency dimensions are slots on the same nodes and add no
    ranks. rhs is expanded the same way when it is a cache mesh, and is already
    in ranks when it is a compute mesh.
    """
    lhs_ranks = cache_node_ranks(lhs, num_ranks_per_node)
    if type(_cache_base(rhs)) is StCacheOuterProductMesh or type(rhs) is PermutedMesh:
        rhs_ranks = cache_node_ranks(rhs, num_ranks_per_node)
    else:
        rhs_ranks = set(resolve_compute_mesh(rhs).ravel().tolist())
    return lhs_ranks <= rhs_ranks

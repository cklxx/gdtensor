"""Meshes (DTensorsView B.2.1.2, B.2.2.4, B.3.2).

A mesh resolves to an integer ndarray of participants:

* ComputeMesh   -> each cell is a global rank.
* StorageMesh   -> each cell is a slot id ``t * W + w``; ``t`` is the
  tensor-slot row, ``w`` is the content-slot column (= global rank),
  ``W`` is ``content_slots.size`` (the "world_size" tag).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from .result import ResultError

# ---------------------------------------------------------------- B.3.2


@dataclass(frozen=True)
class ArrangedMesh:
    """ArrangedMesh[tag_name]; ``tag`` is "world_size", "tensor_count" or "constant_one"."""

    size: int
    tag: str = "world_size"

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ResultError(f"ArrangedMesh size must be >= 1, got {self.size}")
        if self.tag == "constant_one" and self.size != 1:
            raise ResultError("ArrangedMesh['constant_one'] must have size 1")


@dataclass(frozen=True)
class Slice:
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    def to_py(self) -> slice:
        return slice(self.start, self.stop, self.step)


@dataclass(frozen=True)
class ReshapedMesh:
    underlying_mesh: "AnyMesh"
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class PermutedMesh:
    underlying_mesh: "AnyMesh"
    permutation: Tuple[int, ...]


@dataclass(frozen=True)
class SlicedMesh:
    """One entry per underlying dim; ``None`` keeps the dim. A Slice keeps the
    dim too (size may shrink); use a length-1 Slice to fix a dim."""

    underlying_mesh: "AnyMesh"
    slices: Tuple[Optional[Slice], ...]


# ---------------------------------------------------------------- B.2.2.4


@dataclass(frozen=True)
class OuterProductMesh:
    tensor_slots: ArrangedMesh  # "constant_one" or "tensor_count"
    content_slots: ArrangedMesh  # "world_size"

    def __post_init__(self) -> None:
        if self.content_slots.tag != "world_size":
            raise ResultError("content_slots must be ArrangedMesh['world_size']")
        if self.tensor_slots.tag not in ("constant_one", "tensor_count"):
            raise ResultError("tensor_slots must be 'constant_one' or 'tensor_count'")


AnyMesh = Union[ArrangedMesh, ReshapedMesh, PermutedMesh, SlicedMesh, OuterProductMesh]

# ComputeMesh = $arranged ArrangedMesh["world_size"] | $reshaped | $permuted | $sliced
ComputeMesh = Union[ArrangedMesh, ReshapedMesh, PermutedMesh, SlicedMesh]
# StorageMesh[T] = $outer_product_mesh | $reshaped_mesh | $sliced_mesh
StorageMesh = Union[OuterProductMesh, ReshapedMesh, SlicedMesh]


def _apply_transform(mesh: AnyMesh, inner: np.ndarray) -> np.ndarray:
    if isinstance(mesh, ReshapedMesh):
        shape = tuple(int(s) for s in mesh.shape)
        if int(np.prod(shape, dtype=np.int64)) != inner.size:
            raise ResultError(f"reshape {inner.shape} -> {shape} changes mesh size")
        return inner.reshape(shape)
    if isinstance(mesh, PermutedMesh):
        perm = tuple(mesh.permutation)
        if sorted(perm) != list(range(inner.ndim)):
            raise ResultError(f"bad permutation {perm} for mesh ndim {inner.ndim}")
        return inner.transpose(perm)
    if isinstance(mesh, SlicedMesh):
        if len(mesh.slices) != inner.ndim:
            raise ResultError(f"slices {len(mesh.slices)} != mesh ndim {inner.ndim}")
        idx = tuple(slice(None) if s is None else s.to_py() for s in mesh.slices)
        out = inner[idx]
        if out.size == 0:
            raise ResultError("sliced mesh is empty")
        return out
    raise ResultError(f"unknown mesh transform {type(mesh).__name__}")


def resolve_compute_mesh(mesh: ComputeMesh) -> np.ndarray:
    """Return the ndarray of global ranks of a ComputeMesh."""
    if isinstance(mesh, ArrangedMesh):
        if mesh.tag != "world_size":
            raise ResultError("ComputeMesh base must be ArrangedMesh['world_size']")
        return np.arange(mesh.size, dtype=np.int64)
    if isinstance(mesh, OuterProductMesh):
        raise ResultError("OuterProductMesh is a StorageMesh, not a ComputeMesh")
    return _apply_transform(mesh, resolve_compute_mesh(mesh.underlying_mesh))


def storage_mesh_base(mesh: StorageMesh) -> OuterProductMesh:
    while not isinstance(mesh, OuterProductMesh):
        if isinstance(mesh, PermutedMesh):
            raise ResultError("StorageMesh has no permuted branch")
        if not isinstance(mesh, (ReshapedMesh, SlicedMesh)):
            raise ResultError(f"bad StorageMesh node {type(mesh).__name__}")
        mesh = mesh.underlying_mesh
    return mesh


def resolve_storage_mesh(mesh: StorageMesh) -> Tuple[np.ndarray, int]:
    """Return (ndarray of slot ids ``t*W+w``, W)."""
    if isinstance(mesh, OuterProductMesh):
        t_size = mesh.tensor_slots.size
        w_size = mesh.content_slots.size
        cells = np.arange(t_size * w_size, dtype=np.int64).reshape(t_size, w_size)
        return cells, w_size
    if isinstance(mesh, PermutedMesh):
        raise ResultError("StorageMesh has no permuted branch")
    inner, w_size = resolve_storage_mesh(mesh.underlying_mesh)
    return _apply_transform(mesh, inner), w_size


def is_single_tensor_storage_mesh(mesh: StorageMesh) -> bool:
    return storage_mesh_base(mesh).tensor_slots.tag == "constant_one"


def is_multi_tensors_storage_mesh(mesh: StorageMesh) -> bool:
    return storage_mesh_base(mesh).tensor_slots.tag == "tensor_count"


def storage_mesh_cells(mesh: StorageMesh) -> set:
    """Set of (t, w) cells covered by a storage mesh."""
    cells, w_size = resolve_storage_mesh(mesh)
    return {divmod(int(c), w_size) for c in cells.ravel()}


def multi_mesh_row(mesh: StorageMesh, tensor_order: int) -> set:
    """Columns of ``multi_tensors_mesh[order:order+1, :]`` (the row slice)."""
    return {w for (t, w) in storage_mesh_cells(mesh) if t == tensor_order}


def mesh_coords(cells: np.ndarray, value: int) -> Optional[Tuple[int, ...]]:
    hit = np.argwhere(cells == value)
    if hit.shape[0] == 0:
        return None
    if hit.shape[0] > 1:
        raise ResultError(f"participant {value} appears {hit.shape[0]} times in mesh")
    return tuple(int(x) for x in hit[0])

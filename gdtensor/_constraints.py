"""check_constraints for data objects (spec 5.7 / B.2.4).

Recursive rules: Object per field; Oneof the active branch; list per element;
dict per value; None skipped. A child error propagates unchanged. Object-level
extra checks are listed in B.2.4.

Sessions and accessors are abstract interfaces and have no check_constraints.
"""
from __future__ import annotations

from typing import Any

from ._mesh import ArrangedMesh, LiteralMesh, SlicedMesh
from .tag import ConstraintViolationError, Ok

_HOOKS = {}


def register_hook(type_name):
    def deco(fn):
        _HOOKS[type_name] = fn
        return fn
    return deco


def _err(obj, constraint, msg):
    return ConstraintViolationError(
        msg, type_name=type(obj).__name__, constraint_name=constraint)


def check_constraints(obj: Any):
    """Recursively validate a data object. Returns Result[None]."""
    if obj is None or isinstance(obj, (int, float, str, bytes, bool)):
        return Ok(None)
    if isinstance(obj, (list, tuple)):
        for x in obj:
            r = check_constraints(x)
            if not r.ok:
                return r
        return Ok(None)
        return Ok(None)
    if isinstance(obj, dict):
        for v in obj.values():
            r = check_constraints(v)
            if not r.ok:
                return r
        return Ok(None)
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses
        for f in dataclasses.fields(obj):
            if not f.init:
                continue
            r = check_constraints(getattr(obj, f.name, None))
            if not r.ok:
                return r
    hook = _HOOKS.get(type(obj).__name__)
    if hook is not None:
        e = hook(obj)
        if e is not None:
            from .tag import Err
            return Err(e)
    return Ok(None)


# ------------------------------------------------------------------ hooks


@register_hook("GlobalTensorMeta")
def _global_tensor_meta(o):
    if not tuple(o.global_shape):
        return _err(o, "global_shape_non_empty", "global_shape is empty")
    if o.itemsize <= 0:
        return _err(o, "itemsize_positive", f"itemsize={o.itemsize}")
    if not o.dtype:
        return _err(o, "dtype_non_empty", "dtype is empty")
    return None


@register_hook("NamedDTensor")
def _named(o):
    if not o.global_tensor_id:
        return _err(o, "global_tensor_id_non_empty", "tensor id is empty")
    return None


@register_hook("NaiveDTensor")
def _naive(o):
    # placements count must equal mesh ndim
    shape = o.mesh.shape if hasattr(o.mesh, "shape") else tuple(
        resolve_shape(o.mesh))
    ndim = len(shape)
    if len(o.placements) != ndim:
        return _err(o, "placements_match_mesh_ndim",
                    f"{len(o.placements)} placements for mesh ndim {ndim}")
    return None


def resolve_shape(mesh):
    from ._mesh import (OuterProductMesh, PermutedMesh, ReshapedMesh, SlicedMesh)
    if isinstance(mesh, (ArrangedMesh, LiteralMesh)):
        return (getattr(mesh, "size", None) or mesh.max_,)
    if isinstance(mesh, OuterProductMesh):
        return resolve_shape(mesh.tensor_slots) + resolve_shape(mesh.content_slots)
    if isinstance(mesh, (ReshapedMesh, SlicedMesh)):
        return mesh.shape if isinstance(mesh, ReshapedMesh) else _sliced_shape(mesh)
    if isinstance(mesh, PermutedMesh):
        return tuple(mesh.underlying_mesh.shape)
    return ()


def _sliced_shape(mesh: SlicedMesh):
    base = tuple(resolve_shape(mesh.underlying_mesh))
    out = []
    for d, s in enumerate(mesh.slices):
        if s is None:
            out.append(base[d])
        else:
            a = s.start or 0
            b = base[d] if s.stop is None else s.stop
            out.append(max(0, b - a))
    return tuple(out)


@register_hook("DTensorTable")
def _table(o):
    seen = set()
    for nd in o.named_d_tensors:
        if nd.global_tensor_id in seen:
            return _err(o, "global_tensor_id_unique",
                        f"duplicate tensor id {nd.global_tensor_id}")
        seen.add(nd.global_tensor_id)
    return None


@register_hook("ArrangedMesh")
def _arranged(o):
    if o.size <= 0:
        return _err(o, "size_positive", f"size={o.size}")
    return None


@register_hook("LiteralMesh")
def _literal(o):
    if not o.ranks:
        return _err(o, "ranks_non_empty", "ranks is empty")
    if len(set(o.ranks)) != len(o.ranks):
        return _err(o, "ranks_unique", "ranks contain duplicates")
    if any(x < 0 or x >= o.max_ for x in o.ranks):
        return _err(o, "ranks_in_range", f"rank outside [0, {o.max_})")
    return None


@register_hook("DisjointStorageMeshList")
def _list(o):
    if not o.disjoint_storage_mesh_list:
        return _err(o, "list_non_empty", "mesh list is empty")
    seen = set()
    from ._mesh import storage_mesh_cells
    for m in o.disjoint_storage_mesh_list:
        c = storage_mesh_cells(m)
        if seen & c:
            return _err(o, "meshes_disjoint", "meshes overlap inside one list")
        seen |= c
    return None


@register_hook("PhysicalLayoutTable")
def _layout(o):
    from ._layout import leader_tensor_component_pos_of
    if o.container_format not in ("safetensors", "distcp"):
        return _err(o, "container_format", f"unknown format {o.container_format}")
    seen_lists = set()
    seen_cells = set()
    for pos, lst in o.tensor_entries.items():
        leader = leader_tensor_component_pos_of(lst)
        if leader in seen_lists:
            continue  # many pos keys map to one list; count it once
        seen_lists.add(leader)
        if leader not in o.file_paths:
            return _err(o, "leader_has_path", f"leader {tuple(leader)} has no path")
        cells = set(lst.cells)
        if seen_cells & cells:
            return _err(o, "meshes_disjoint_across_files", "slot covered by two files")
        seen_cells |= cells
    leaders = seen_lists
    if set(o.file_paths) != leaders:
        return _err(o, "file_paths_keys_are_leaders", "file_paths keys != list leaders")
    if len(set(o.file_paths.values())) != len(o.file_paths):
        return _err(o, "one_path_per_list", "two lists share one path")
    return None


@register_hook("ComputeDTensorsView")
def _compute_view(o):
    if [n.global_tensor_id for n in o.d_tensor_table.named_d_tensors] != list(o.tensor_ids):
        return _err(o, "tensor_ids_match_table", "tensor_ids order differs from table")
    if o.data_accessor is None:
        return _err(o, "data_accessor_non_empty", "data_accessor is None")
    return None


@register_hook("StorageDTensorsView")
def _storage_view(o):
    if [n.global_tensor_id for n in o.d_tensor_table.named_d_tensors] != list(o.tensor_ids):
        return _err(o, "tensor_ids_match_table", "tensor_ids order differs from table")
    table_ids = set(o.tensor_ids)
    for pos in o.physical_layout.tensor_entries:
        if pos.tensor_id not in table_ids:
            return _err(o, "layout_tensor_in_table",
                        f"layout tensor {pos.tensor_id} not in table")
    if o.data_accessor is None or o.begin_save is None or o.begin_load is None:
        return _err(o, "callbacks_non_empty", "data_accessor/begin_* must be set")
    return None

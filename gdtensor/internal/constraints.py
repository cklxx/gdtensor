"""check_constraints for data objects (design 5.7 / B.2.4).

Recursion: Object per field; Oneof the active branch (the value is the branch
class); list per element; dict per value; None skipped. A child error is
returned unchanged. The extra checks below are exactly the B.2.4 table; types
not listed only recurse.

Sessions and accessors are abstract interfaces and have no check_constraints.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .result import ConstraintViolationError, ERROR_TYPES
from ..tag import Box, Err, Ok


def _err(obj, constraint: str, msg: str):
    return Err(ConstraintViolationError(type_name=type(obj).__name__, constraint_name=constraint, msg=msg))


def _recurse(value: Any):
    if value is None or isinstance(value, (int, float, str, bytes, bool)):
        return Ok(None)
    if isinstance(value, (list, tuple)):
        for x in value:
            r = _recurse(x)
            if type(r) is Err:
                return r
        return Ok(None)
    if isinstance(value, dict):
        for x in value.values():
            r = _recurse(x)
            if type(r) is Err:
                return r
        return Ok(None)
    if isinstance(value, Box):
        return _recurse(value.value)
    if dataclasses.is_dataclass(value) and type(value).__name__ in _DATA_TYPES:
        return check_constraints(value)
    return Ok(None)  # opaque payload (callable, accessor, torch tensor)


def check_constraints(obj: Any):
    """Validate a data object and everything it contains. Returns Result[None]."""
    for f in dataclasses.fields(obj):
        r = _recurse(getattr(obj, f.name))
        if type(r) is Err:
            return r
    hook = _EXTRA.get(type(obj).__name__)
    if hook is not None:
        r = hook(obj)
        if r is not None:
            return r
    return Ok(None)


# ------------------------------------------------------------------ extra checks (B.2.4)


def _compute_view(o):
    from .dtensor import tensor_ids_of
    if tensor_ids_of(o.d_tensor_table) != list(o.tensor_ids):
        return _err(o, "tensor_ids_match_table", "tensor_ids order differs from d_tensor_table")
    if o.data_accessor is None:
        return _err(o, "data_accessor_non_nil", "data_accessor is nil")
    return None


def _storage_view(o):
    from .dtensor import tensor_ids_of
    if tensor_ids_of(o.d_tensor_table) != list(o.tensor_ids):
        return _err(o, "tensor_ids_match_table", "tensor_ids order differs from d_tensor_table")
    if o.physical_layout is not None:
        ids = set(o.tensor_ids)
        for pos in o.physical_layout.tensor_entries:
            if pos.tensor_id not in ids:
                return _err(o, "layout_tensor_in_table", f"layout tensor {pos.tensor_id} not in d_tensor_table")
    if o.data_accessor is None or o.begin_save is None or o.begin_load is None:
        return _err(o, "callbacks_non_nil", "data_accessor / begin_save / begin_load must be set")
    return None


def _table(o):
    seen = set()
    for nd in o.named_d_tensors:
        if nd.global_tensor_id in seen:
            return _err(o, "global_tensor_id_unique", f"duplicate tensor id {nd.global_tensor_id}")
        seen.add(nd.global_tensor_id)
    return None


def _named(o):
    if not o.global_tensor_id:
        return _err(o, "global_tensor_id_non_empty", "global_tensor_id is empty")
    return None


def _naive(o):
    from .mesh import mesh_shape
    from ..tag import Shard
    try:
        shape = mesh_shape(o.mesh)
    except Exception as exc:  # noqa: BLE001
        return _err(o, "mesh_valid", str(exc))
    if len(o.placements) != len(shape):
        return _err(o, "placements_match_mesh", f"{len(o.placements)} placements for mesh ndim {len(shape)}")
    nd = len(o.global_tensor_meta.global_shape)
    for p in o.placements:
        if type(p) is Shard and not 0 <= p.value < nd:
            return _err(o, "placements_match_meta", f"Shard axis {p.value} out of range for ndim {nd}")
    return None


def _reshaped(o):
    import math
    if math.prod(o.shape) != math.prod(o.underlying_d_tensor.global_tensor_meta.global_shape):
        return _err(o, "shape_keeps_numel", f"reshape to {o.shape} changes numel")
    return None


def _meta(o):
    if not o.global_shape:
        return _err(o, "global_shape_non_empty", "global_shape is empty")
    if o.itemsize <= 0:
        return _err(o, "itemsize_positive", f"itemsize={o.itemsize}")
    if not o.dtype:
        return _err(o, "dtype_non_empty", "dtype is empty")
    return None


def _layout(o):
    from .layout import leader_tensor_component_pos_of, list_cells, pos_key
    if o.container_format not in ("safetensors", "distcp"):
        return _err(o, "container_format_unique", f"unknown container_format {o.container_format!r}")
    leaders, seen_cells = set(), set()
    for lst in o.tensor_entries.values():
        try:
            leader = leader_tensor_component_pos_of(lst)
            cells = set(list_cells(lst))
        except Exception as exc:  # noqa: BLE001
            return _err(o, "meshes_disjoint", str(exc))
        if leader in leaders:
            continue
        leaders.add(leader)
        if leader not in o.file_paths:
            return _err(o, "leader_in_file_paths", f"leader {pos_key(leader)} has no path")
        if seen_cells & cells:
            return _err(o, "meshes_disjoint", "a slot is covered by two files")
        seen_cells |= cells
    if set(o.file_paths) != leaders:
        return _err(o, "file_paths_keys_are_leaders", "file_paths keys must be list leaders")
    return None


def _list(o):
    from .mesh import storage_mesh_cells
    if not o.disjoint_storage_mesh_list:
        return _err(o, "list_non_empty", "disjoint_storage_mesh_list is empty")
    seen = set()
    for m in o.disjoint_storage_mesh_list:
        c = storage_mesh_cells(m)
        if seen & c:
            return _err(o, "meshes_disjoint", "meshes overlap inside one list")
        seen |= c
    return None


def _arranged(o):
    # StorageMesh content_slots / ComputeMesh ArrangedMesh: size > 0
    if not isinstance(o.size, Box) or not isinstance(o.size.value, int) or o.size.value <= 0:
        return _err(o, "size_positive", f"size={o.size!r}")
    return None


def _literal(o):
    if not o.ranks:
        return _err(o, "ranks_non_empty", "ranks is empty")
    if not isinstance(o.max, Box) or not isinstance(o.max.value, int):
        return _err(o, "max_tagged_int", f"max={o.max!r}")
    if any(x < 0 or x >= o.max.value for x in o.ranks):
        return _err(o, "ranks_in_range", f"rank outside [0, {o.max.value})")
    if len(set(o.ranks)) != len(o.ranks):
        return _err(o, "ranks_unique", "ranks contain duplicates")
    return None


def _error(o):
    if not o.msg:
        return _err(o, "msg_non_empty", "msg is empty")
    return None


_EXTRA = {
    "ComputeDTensorsView": _compute_view,
    "StorageDTensorsView": _storage_view,
    "DTensorTable": _table,
    "NamedDTensor": _named,
    "NaiveDTensor": _naive,
    "ReshapedDTensor": _reshaped,
    "GlobalTensorMeta": _meta,
    "PhysicalLayoutTable": _layout,
    "DisjointStorageMeshList": _list,
    "ArrangedMesh": _arranged,
    "LiteralMesh": _literal,
}
for _cls in ERROR_TYPES:
    _EXTRA[_cls.__name__] = _error

_DATA_TYPES = set(_EXTRA) | {
    "ReshapedMesh", "PermutedMesh", "SlicedMesh", "Slice", "OuterProductMesh",
    "TensorComponentPos", "DTensorsViewMapCore", "RangeFromComputeView", "RangeFromStorageView",
}

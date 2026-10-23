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
from ..tag import Box
from .products import Err, Ok


def _err(obj, constraint: str, msg: str):
    return Err(
        ConstraintViolationError(
            type_name=type(obj).__name__, constraint_name=constraint, msg=msg
        )
    )


def _recurse(value: Any):
    if value is None or isinstance(value, (int, float, str, bytes, bool)):
        return Ok(None)
    if isinstance(value, (list, tuple)):
        return _recurse_values(value)
    if isinstance(value, dict):
        return _recurse_values(value.values())
    if isinstance(value, Box):
        return _recurse(value.value)
    if dataclasses.is_dataclass(value) and type(value).__name__ in _DATA_TYPES:
        return value.check_constraints()
    return Ok(None)  # opaque payload (callable, accessor, torch tensor)


def _recurse_values(values):
    for value in values:
        result = _recurse(value)
        if type(result) is Err:
            return result
    return Ok(None)


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
        return _err(
            o, "tensor_ids_match_table", "tensor_ids order differs from d_tensor_table"
        )
    if o.data_accessor is None:
        return _err(o, "data_accessor_non_nil", "data_accessor is nil")
    return None


def _storage_view(o):
    from .dtensor import tensor_ids_of

    if tensor_ids_of(o.d_tensor_table) != list(o.tensor_ids):
        return _err(
            o, "tensor_ids_match_table", "tensor_ids order differs from d_tensor_table"
        )
    entries = o.physical_layout.tensor_entries if o.physical_layout is not None else {}
    ids = set(o.tensor_ids)
    for pos in entries:
        if pos.tensor_id not in ids:
            return _err(
                o,
                "layout_tensor_in_table",
                f"layout tensor {pos.tensor_id} not in d_tensor_table",
            )
    # Row labels live on the meshes and table order; the list carries no
    # tensor_ids field. build_payload_index validates row count separately.
    if o.data_accessor is None or o.begin_save is None or o.begin_load is None:
        return _err(
            o,
            "callbacks_non_nil",
            "data_accessor / begin_save / begin_load must be set",
        )
    return None


def _table(o):
    seen = set()
    for nd in o.named_d_tensors:
        if nd.global_tensor_id in seen:
            return _err(
                o,
                "global_tensor_id_unique",
                f"duplicate tensor id {nd.global_tensor_id}",
            )
        seen.add(nd.global_tensor_id)
    return None


def _named(o):
    if not o.global_tensor_id:
        return _err(o, "global_tensor_id_non_empty", "global_tensor_id is empty")
    return None


def _naive(o):
    from .mesh import mesh_shape
    from .products import Shard

    try:
        shape = mesh_shape(o.mesh)
    except Exception as exc:  # noqa: BLE001
        return _err(o, "mesh_valid", str(exc))
    if len(o.placements) != len(shape):
        return _err(
            o,
            "placements_match_mesh",
            f"{len(o.placements)} placements for mesh ndim {len(shape)}",
        )
    nd = len(o.global_tensor_meta.global_shape)
    for p in o.placements:
        if type(p) is Shard and not 0 <= p.axis < nd:
            return _err(
                o,
                "placements_match_meta",
                f"Shard axis {p.axis} out of range for ndim {nd}",
            )
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
    from .layout import list_cells

    if o.container_format not in ("safetensors", "distcp"):
        return _err(
            o,
            "container_format_unique",
            f"unknown container_format {o.container_format!r}",
        )
    # Structural invariants that hold without the DTensorTable header order:
    # each distinct physical list has exactly one file_paths key sitting on a
    # component position of that list, and distinct files never share a cell.
    lists: dict[int, object] = {}
    for lst in o.tensor_entries.values():
        lists.setdefault(id(lst), lst)
    error = _check_file_path_keys(o, lists)
    if error is not None:
        return error
    seen_cells = set()
    for lst in lists.values():
        try:
            cells = set(list_cells(lst))
        except Exception as exc:  # noqa: BLE001
            return _err(o, "meshes_disjoint", str(exc))
        if seen_cells & cells:
            return _err(o, "meshes_disjoint", "a slot is covered by two files")
        seen_cells |= cells
    return None


def _check_file_path_keys(o, lists):
    for pos in o.file_paths:
        if pos not in o.tensor_entries:
            msg = f"file_paths key {pos.tensor_id},{pos.rank} is not a tensor entry"
            return _err(o, "file_paths_keys_are_leaders", msg)
    for lst in lists.values():
        keys = [pos for pos in o.file_paths if o.tensor_entries.get(pos) is lst]
        if not keys:
            return _err(o, "list_has_file_path", "a physical list has no file path")
        if len(keys) > 1:
            return _err(
                o,
                "file_paths_keys_are_leaders",
                "a physical list has more than one file path key",
            )
    return None


def _list(o):
    from .mesh import storage_mesh_cells

    if not o.disjoint_storage_mesh_list:
        return _err(o, "list_non_empty", "disjoint_storage_mesh_list is empty")
    try:
        labels_error = _list_row_labels(o)
        cells = [storage_mesh_cells(m) for m in o.disjoint_storage_mesh_list]
    except Exception as exc:
        return _err(o, "meshes_disjoint", str(exc))
    if labels_error is not None:
        return labels_error
    seen = set()
    for c in cells:
        if seen & c:
            return _err(o, "meshes_disjoint", "meshes overlap inside one list")
        seen |= c
    return None


def _list_row_labels(value):
    from .mesh import storage_mesh_base
    from ..tag import TensorCount

    for mesh in value.disjoint_storage_mesh_list:
        tensor_count = storage_mesh_base(mesh).tensor_slots.size
        if type(tensor_count) is not TensorCount:
            return _err(
                value,
                "tensor_slots_tagged",
                "mesh tensor slots must be tagged TensorCount",
            )
    return None


def _arranged(o):
    # StorageMesh content_slots / ComputeMesh ArrangedMesh: size > 0
    if (
        not isinstance(o.size, Box)
        or not isinstance(o.size.value, int)
        or o.size.value <= 0
    ):
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
    "GlobalTensorMeta": _meta,
    "PhysicalLayoutTable": _layout,
    "DisjointStorageMeshList": _list,
    "ArrangedMesh": _arranged,
    "LiteralMesh": _literal,
}
for _cls in ERROR_TYPES:
    _EXTRA[_cls.__name__] = _error

_DATA_TYPES = set(_EXTRA) | {
    "ReshapedDTensor",
    "ReshapedMesh",
    "PermutedMesh",
    "SlicedMesh",
    "Slice",
    "SlotOuterProductMesh",
    "TensorComponentPos",
    "DTensorsViewMapCore",
    "RangeFromComputeView",
    "RangeFromStorageView",
}


# ---------------------------------------------------------------- B.2.3 / B.3.2


def _cache_view(o):
    from .dtensor import tensor_ids_of

    if tensor_ids_of(o.d_tensor_table) != list(o.tensor_ids):
        return _err(
            o, "tensor_ids_match_table", "tensor_ids order differs from d_tensor_table"
        )
    if o.data_accessor is None:
        return _err(o, "data_accessor_non_nil", "data_accessor is nil")
    return None


def _st_cache_mesh(o):
    from .mesh import (
        TemporalCacheMesh,
        SpatioCacheMesh,
        ArrangedMesh,
    )

    t, s, c = o.temporal, o.spatial, o.concurrency
    if type(t) is TemporalCacheMesh:
        if t.size is None or t.reshaped is not None:
            return _err(o, "temporal_one_branch", "temporal must set one branch")
    elif type(s) is SpatioCacheMesh and t.reshaped is s:
        pass
    else:
        return _err(o, "temporal_branch", "temporal must be arranged or reshaped")
    if type(s) is SpatioCacheMesh:
        if (s.ranks is None or s.max is None) or s.reshaped is not None:
            return _err(o, "spatial_one_branch", "spatial must set one branch")
    elif type(t) is TemporalCacheMesh and s.reshaped is t:
        pass
    else:
        return _err(o, "spatial_branch", "spatial must be literal or reshaped")
    if type(c) is not ArrangedMesh:
        return _err(o, "concurrency_arranged", "concurrency must be ArrangedMesh")
    # Placements fixed by dimension: temporal/concurrency shard, spatial replicate.
    # Placements live on the NaiveDTensor whose mesh is this CacheMesh; the mesh
    # constraint only fixes shape validity.
    return None


def _direct_chain(o):
    if list(o.src.tensor_ids) != list(o.dst.tensor_ids):
        return _err(o, "tensor_ids_match", "src and dst tensor_ids differ")
    return None


def _one_layer_chain(o):
    ids = [list(v.tensor_ids) for v in (o.src, o.cache, o.dst)]
    if any(ids[0] != x for x in ids[1:]):
        return _err(o, "tensor_ids_match", "src/cache/dst tensor_ids differ")
    return _cache_subset(o, o.cache, o.dst, o.num_ranks_per_node)


def _two_layer_chain(o):
    ids = [list(v.tensor_ids) for v in (o.src, o.small_cache, o.big_cache, o.dst)]
    if any(ids[0] != x for x in ids[1:]):
        return _err(o, "tensor_ids_match", "all legs tensor_ids differ")
    r1 = _cache_subset(o, o.small_cache, o.big_cache, o.num_ranks_per_node)
    if r1 is not None:
        return r1
    return _cache_subset(o, o.big_cache, o.dst, o.num_ranks_per_node)


def _cache_subset(o, cache, dst, num_ranks_per_node: int):
    from .mesh import is_cache_mesh_subset_of_mesh

    if num_ranks_per_node <= 0:
        return _err(o, "num_ranks_per_node_positive", "must be positive")
    cache_mesh = cache.d_tensor_table.named_d_tensors[0].d_tensor.mesh
    dst_mesh = dst.d_tensor_table.named_d_tensors[0].d_tensor.mesh
    if not is_cache_mesh_subset_of_mesh(cache_mesh, dst_mesh, num_ranks_per_node):
        return _err(o, "cache_subset", "cache ranks are not a subset of dst ranks")
    return None


def _mixed_chain(o):
    ids = [
        x for seg in o.disjoint_tensor_ids_and_read_chain_list for x in seg.tensor_ids
    ]
    if len(ids) != len(set(ids)):
        return _err(o, "tensor_ids_disjoint", "segment tensor_ids overlap")
    return None


def _disjoint_chain(o):
    from .chain import get_tensor_ids_of_read_chain

    if list(o.tensor_ids) != list(get_tensor_ids_of_read_chain(o.read_chain)):
        return _err(
            o, "tensor_ids_match_chain", "tensor_ids must equal chain tensor ids"
        )
    return None


_EXTRA["CacheDTensorsView"] = _cache_view
_EXTRA["StCacheOuterProductMesh"] = _st_cache_mesh
_EXTRA["DirectDTensorsReadChain"] = _direct_chain
_EXTRA["OneLayerCachedDTensorsReadChain"] = _one_layer_chain
_EXTRA["TwoLayerCachedDTensorsReadChain"] = _two_layer_chain
_EXTRA["MixedDTensorsReadChain"] = _mixed_chain
_EXTRA["DisjointTensorIdsAndReadChain"] = _disjoint_chain
_DATA_TYPES.update(
    {
        "CacheMesh",
        "TemporalCacheMesh",
        "SpatioCacheMesh",
        "StCacheOuterProductMesh",
        "DirectDTensorsReadChain",
        "OneLayerCachedDTensorsReadChain",
        "TwoLayerCachedDTensorsReadChain",
        "MixedDTensorsReadChain",
        "DisjointTensorIdsAndReadChain",
    }
)

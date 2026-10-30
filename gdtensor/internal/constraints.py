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

from .result import ConstraintViolationError, ERROR_TYPES, Fail
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

    table_ids = tensor_ids_of(o.d_tensor_table)
    if table_ids != list(o.tensor_ids):
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
    # Every list's row labels must equal the full DTensorTable header order.
    # A standalone list cannot know the table; the view (which holds the table)
    # enforces full-row labels here before any session opens.
    for lst in entries.values():
        if list(lst.tensor_ids) != list(table_ids):
            return _err(
                o,
                "list_labels_match_table",
                "list labels must equal DTensorTable header order",
            )
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
    from .mesh import (
        StCacheOuterProductMesh,
        cache_concurrency_width,
        cache_num_nodes,
        cache_num_steps,
        mesh_shape,
    )
    from .products import Shard

    if type(o.mesh) is StCacheOuterProductMesh:
        # Cache mesh participants are (num_steps x num_nodes x concurrency_width);
        # only the spatial node dimension expands to ranks.
        shape = (
            cache_num_steps(o.mesh),
            cache_num_nodes(o.mesh),
            cache_concurrency_width(o.mesh),
        )
    else:
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
    # Group physical lists by equality: entries pointing at value-equal lists
    # (deepcopy) are the same physical file (approved delta), never by object id.
    lists: dict[Any, Any] = {}
    for lst in o.tensor_entries.values():
        lists.setdefault(lst, lst)
    # Pass 1: labels + geometric cells + file disjointness (first-error wins).
    seen_cells = set()
    for lst in lists.values():
        labels_error = _list_row_labels(lst)
        if labels_error is not None:
            return labels_error
        try:
            cells = set(list_cells(lst))
        except Exception as exc:  # noqa: BLE001
            return _err(o, "meshes_disjoint", str(exc))
        if seen_cells & cells:
            return _err(o, "meshes_disjoint", "a slot is covered by two files")
        seen_cells |= cells
    # Pass 2: file_paths keys vs the true geometric leader of each list.
    return _check_file_path_keys(o, lists)


def _geometric_leader(o, lst):
    """True leader = smallest actual (row, column) cell of the list's meshes.

    Reuses leader_tensor_component_pos_of(lst) (single-argument, real geometry);
    it raises Fail only on label/geometry problems, never recurses into any
    check_constraints. Fail is mapped to a ConstraintViolationError here.
    """
    from .layout import leader_tensor_component_pos_of

    try:
        return leader_tensor_component_pos_of(lst), None
    except Fail as exc:
        return None, _err(o, "list_labels_match_geometry", str(exc.error.msg))


def _check_file_path_keys(o, lists):
    for pos in o.file_paths:
        if pos not in o.tensor_entries:
            msg = f"file_paths key {pos.tensor_id},{pos.rank} is not a tensor entry"
            return _err(o, "file_paths_keys_are_leaders", msg)
    for lst in lists.values():
        labels_error = _list_row_labels(lst)
        if labels_error is not None:
            return labels_error
        leader, err = _geometric_leader(o, lst)
        if err is not None:
            return err
        assert leader is not None
        keys = [pos for pos in o.file_paths if o.tensor_entries.get(pos) == lst]
        if not keys:
            return _err(o, "list_has_file_path", "a physical list has no file path")
        if len(keys) > 1:
            return _err(
                o,
                "file_paths_keys_are_leaders",
                "a physical list has more than one file path key",
            )
        # The true geometric leader cell must itself carry an entry pointing at
        # this list; a layout whose leader cell is missing an entry (only a
        # non-leader row is registered) is rejected, never papered over by
        # taking the minimum registered member.
        if o.tensor_entries.get(leader) != lst:
            return _err(
                o,
                "file_paths_keys_are_leaders",
                f"true leader cell {leader.tensor_id},{leader.rank} "
                f"has no tensor entry for this physical list",
            )
        if keys[0] != leader:
            return _err(
                o,
                "file_paths_keys_are_leaders",
                f"file_paths key {keys[0].tensor_id},{keys[0].rank} "
                f"is not the geometric leader {leader.tensor_id},{leader.rank}",
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
    from .mesh import storage_mesh_base, storage_mesh_cells
    from ..tag import TensorCount

    labels = list(value.tensor_ids)
    if any(not isinstance(x, str) or not x for x in labels):
        return _err(
            value, "labels_non_empty_strings", "labels must be non-empty strings"
        )
    if len(set(labels)) != len(labels):
        return _err(value, "labels_unique", "labels must be unique")
    covered = set()
    for mesh in value.disjoint_storage_mesh_list:
        tensor_count = storage_mesh_base(mesh).tensor_slots.size
        if type(tensor_count) is not TensorCount:
            return _err(
                value,
                "tensor_slots_tagged",
                "mesh tensor slots must be tagged TensorCount",
            )
        for t, _ in storage_mesh_cells(mesh):
            covered.add(t)
    missing = [t for t in sorted(covered) if t >= len(labels)]
    if missing:
        return _err(value, "labels_cover_rows", f"row {missing[0]} has no label")
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
    from .dtensor import naive_of, tensor_ids_of
    from .mesh import cache_concurrency_width, cache_node_set

    if tensor_ids_of(o.d_tensor_table) != list(o.tensor_ids):
        return _err(
            o, "tensor_ids_match_table", "tensor_ids order differs from d_tensor_table"
        )
    # The plan executor (propagation._view_nodes / _view_width) needs one node
    # set and one width per view; reject a mixed view before any plan builds.
    sets = {
        tuple(cache_node_set(naive_of(nd).mesh))
        for nd in o.d_tensor_table.named_d_tensors
    }
    if len(sets) != 1:
        return _err(
            o,
            "cache_node_set_consistent",
            "all tensors of a cache view must cover one node set",
        )
    widths = {
        cache_concurrency_width(naive_of(nd).mesh)
        for nd in o.d_tensor_table.named_d_tensors
    }
    if len(widths) != 1:
        return _err(
            o,
            "cache_width_consistent",
            "all tensors of a cache view must share one width",
        )
    if o.data_accessor is None:
        return _err(o, "data_accessor_non_nil", "data_accessor is nil")
    return None


def _st_cache_mesh(o):
    from .mesh import (
        ArrangedMesh,
        LiteralMesh,
        ReshapedMesh,
        _unwrap_cache,
    )

    t, s, c = o.temporal, o.spatial, o.concurrency
    if type(t) not in (ArrangedMesh, ReshapedMesh):
        return _err(
            o, "temporal_branch", "temporal must be ArrangedMesh or ReshapedMesh"
        )
    if type(s) not in (ArrangedMesh, LiteralMesh, ReshapedMesh):
        return _err(
            o,
            "spatial_branch",
            "spatial must be ArrangedMesh, LiteralMesh or ReshapedMesh",
        )
    if type(c) is not ArrangedMesh:
        return _err(o, "concurrency_arranged", "concurrency must be ArrangedMesh")
    # Exactly one anchor branch; the other dimension reshape-wraps it. Two
    # flat branches (both ArrangedMesh) are legal: the executor reads node
    # count from the spatial branch and step count from the temporal branch.
    tb, sb = _unwrap_cache(t), _unwrap_cache(s)
    anchored = (
        (type(t) is ArrangedMesh and type(s) in (ArrangedMesh, LiteralMesh))
        or (type(s) is ReshapedMesh and type(tb) is ArrangedMesh)
        or (type(t) is ReshapedMesh and type(sb) is LiteralMesh)
    )
    if not anchored:
        return _err(
            o,
            "temporal_spatial_anchor",
            "one of temporal/spatial anchors (Arranged/Literal) and the other reshapes it",
        )
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
        "CacheDTensorsView",
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

"""gdtensor.builder: user-facing product types, view map, and helpers (C.3 / C.4).

Implementations live in internal modules; this namespace re-exports exactly
what table C.3 / C.4 names. The compute/torch bridge five functions are at
package top level (5.6).
"""

from __future__ import annotations

from typing import Callable as _Callable, Optional as _Optional
from dataclasses import dataclass as _dataclass

from .internal.builder_mixin import (
    field_hash as _field_hash,
    make_builder_mixin as _make_builder_mixin,
)

from .internal.products import Err, Ok, Partial, Replicate, Shard
from .internal.products import (
    ErrBuilderMixin,
    OkBuilderMixin,
    PartialBuilderMixin,
    ReplicateBuilderMixin,
    ShardBuilderMixin,
)
from .internal.dtensor import (
    DTensorTable,
    DTensorTableBuilderMixin,
    GenericDTensor,
    GlobalTensorMeta,
    GlobalTensorMetaBuilderMixin,
    NaiveDTensor,
    NaiveDTensorBuilderMixin,
    NamedDTensor,
    NamedDTensorBuilderMixin,
    Placement,
    ReshapedDTensor,
    ReshapedDTensorBuilderMixin,
    global_tensor_meta_of,
    tensor_order_value_of,
)
from .internal.layout import (
    ContainerFormat,
    DisjointStorageMeshList,
    DisjointStorageMeshListBuilderMixin,
    PhysicalLayoutTable,
    PhysicalLayoutTableBuilderMixin,
    TensorComponentPos,
    TensorComponentPosBuilderMixin,
)
from .internal.chain import (
    ConcreteDTensorsReadChain,
    DTensorsReadChain,
    DirectDTensorsReadChain,
    DisjointTensorIdsAndReadChain,
    MixedDTensorsReadChain,
    OneLayerCachedDTensorsReadChain,
    TwoLayerCachedDTensorsReadChain,
    legs_of,
)
from .internal.mesh import (
    ArrangedMesh,
    ArrangedMeshBuilderMixin,
    CacheMesh,
    ComputeMesh,
    ContentSlotsMesh,
    LiteralMesh,
    LiteralMeshBuilderMixin,
    SlotOuterProductMesh,
    SlotOuterProductMeshBuilderMixin,
    PermutedMesh,
    PermutedMeshBuilderMixin,
    ReshapedMesh,
    ReshapedMeshBuilderMixin,
    Slice,
    SliceBuilderMixin,
    SlicedMesh,
    SlicedMeshBuilderMixin,
    SpatioCacheMesh,
    StCacheOuterProductMesh,
    StorageMesh,
    TemporalCacheMesh,
    is_cache_mesh_subset_of_mesh,
)
from .internal.views import CacheDTensorsView
from .internal.result import (
    CheckFailedError,
    CheckFailedErrorBuilderMixin,
    ConstraintViolationError,
    ConstraintViolationErrorBuilderMixin,
    GDTensorIOError,
    GDTensorIOErrorBuilderMixin,
    InvalidArgumentError,
    InvalidArgumentErrorBuilderMixin,
    NotFoundError,
    NotFoundErrorBuilderMixin,
)
from .internal.views import (
    ComputeDTensorsView,
    ComputeDTensorsViewBuilderMixin,
    DTensorsView,
    StorageDTensorsView,
    StorageDTensorsViewBuilderMixin,
)


def file_path_of(layout, pos):
    from .internal.layout import file_path_of as _file_path_of

    return _file_path_of(layout, pos)


def leader_tensor_component_pos_of(disjoint_storage_meshes):
    from .internal.layout import (
        leader_tensor_component_pos_of as _leader,
    )

    return _leader(disjoint_storage_meshes)


# ------------------------------------------------------------------ 5.5 view map


DTensorsViewMapCoreBuilderMixin = _make_builder_mixin(
    "DTensorsViewMapCore",
    [
        ("convert_each_compute_view", "nullable"),
        ("convert_each_storage_view", "nullable"),
        ("convert_each_cache_view", "nullable"),
        ("convert_each_storage_load_session", "nullable"),
        ("convert_each_storage_save_session", "nullable"),
    ],
)


@_dataclass(eq=True)
class DTensorsViewMapCore(DTensorsViewMapCoreBuilderMixin):
    convert_each_compute_view: _Optional[_Callable] = None
    convert_each_storage_view: _Optional[_Callable] = None
    convert_each_cache_view: _Optional[_Callable] = None
    convert_each_storage_load_session: _Optional[_Callable] = None
    convert_each_storage_save_session: _Optional[_Callable] = None
    __hash__ = _field_hash

    def check_constraints(self):
        from .internal.constraints import check_constraints

        return check_constraints(self)


def _result_of(value):

    if type(value) in (Ok, Err):
        return value
    return Err(
        InvalidArgumentError(
            argument_name="callback", msg="callback must return a Result"
        )
    )


def _map_table(table, convert):

    if convert is None:
        return Ok(table)
    named = []
    for nd in table.named_d_tensors:
        result = _result_of(convert(nd.d_tensor))
        if type(result) is Err:
            return result
        named.append(NamedDTensor(nd.global_tensor_id, result.value, nd.alias))
    return Ok(DTensorTable(named))


def _close_failed_conversion(session, close):
    try:
        getattr(session, close)()
    except Exception:
        pass  # Preserve the conversion failure when cleanup also fails.


def _map_begin(begin, convert, close):

    if convert is None:
        return begin

    def mapped(view, rank):
        result = _result_of(begin(view, rank))
        if type(result) is Err:
            return result
        session = result.value
        try:
            result = _result_of(convert(session))
        except BaseException:
            _close_failed_conversion(session, close)
            raise
        if type(result) is Err:
            _close_failed_conversion(session, close)
        return result

    return mapped


def d_tensors_view_map(origin, map_core: DTensorsViewMapCore):

    from .internal.views import CacheDTensorsView

    if type(origin) is CacheDTensorsView:
        convert = map_core.convert_each_cache_view
        if convert is None:
            return Ok(origin)
        table = _map_table(origin.d_tensor_table, convert)
        if type(table) is Err:
            return table
        return Ok(
            CacheDTensorsView(origin.tensor_ids, table.value, origin.data_accessor)
        )
    if type(origin) not in (ComputeDTensorsView, StorageDTensorsView):
        return Err(
            InvalidArgumentError(
                argument_name="origin", msg=f"unknown view type {type(origin).__name__}"
            )
        )
    if type(origin) is ComputeDTensorsView:
        convert = map_core.convert_each_compute_view
        if convert is None:
            return Ok(origin)
        table = _map_table(origin.d_tensor_table, convert)
        if type(table) is Err:
            return table
        return Ok(
            ComputeDTensorsView(origin.tensor_ids, table.value, origin.data_accessor)
        )

    convert = map_core.convert_each_storage_view
    save_fn = map_core.convert_each_storage_save_session
    load_fn = map_core.convert_each_storage_load_session
    if convert is None and save_fn is None and load_fn is None:
        return Ok(origin)
    table = _map_table(origin.d_tensor_table, convert)
    if type(table) is Err:
        return table
    return Ok(
        StorageDTensorsView(
            origin.tensor_ids,
            table.value,
            origin.data_accessor,
            origin.physical_layout,
            _map_begin(origin.begin_save, save_fn, "abort"),
            _map_begin(origin.begin_load, load_fn, "finish"),
        )
    )


__all__ = [
    # views / tables / tensors
    "DTensorsView",
    "ComputeDTensorsView",
    "StorageDTensorsView",
    "DTensorsViewMapCore",
    "d_tensors_view_map",
    "DTensorTable",
    "NamedDTensor",
    "GenericDTensor",
    "NaiveDTensor",
    "ReshapedDTensor",
    "GlobalTensorMeta",
    "Placement",
    "ComputeMesh",
    "StorageMesh",
    "SlotOuterProductMesh",
    "ContentSlotsMesh",
    "CacheMesh",
    "StCacheOuterProductMesh",
    "TemporalCacheMesh",
    "SpatioCacheMesh",
    "CacheDTensorsView",
    "DTensorsReadChain",
    "ConcreteDTensorsReadChain",
    "DirectDTensorsReadChain",
    "OneLayerCachedDTensorsReadChain",
    "TwoLayerCachedDTensorsReadChain",
    "MixedDTensorsReadChain",
    "DisjointTensorIdsAndReadChain",
    "is_cache_mesh_subset_of_mesh",
    "legs_of",
    "LiteralMesh",
    "ArrangedMesh",
    "ReshapedMesh",
    "PermutedMesh",
    "SlicedMesh",
    "Slice",
    "PhysicalLayoutTable",
    "DisjointStorageMeshList",
    "ContainerFormat",
    "TensorComponentPos",
    # result / placement products (B.2.4.7 / B.3.4)
    "Ok",
    "Err",
    "Shard",
    "Replicate",
    "Partial",
    # errors
    "ConstraintViolationError",
    "CheckFailedError",
    "NotFoundError",
    "InvalidArgumentError",
    "GDTensorIOError",
    # helpers (C.4)
    "global_tensor_meta_of",
    "tensor_order_value_of",
    "file_path_of",
    "leader_tensor_component_pos_of",
    "ComputeDTensorsViewBuilderMixin",
    "new_compute_d_tensors_view",
    "StorageDTensorsViewBuilderMixin",
    "new_slot_outer_product_mesh",
    "new_storage_d_tensors_view",
    "DTensorsViewMapCoreBuilderMixin",
    "new_d_tensors_view_map_core",
    "DTensorTableBuilderMixin",
    "new_d_tensor_table",
    "NamedDTensorBuilderMixin",
    "new_named_d_tensor",
    "NaiveDTensorBuilderMixin",
    "new_naive_d_tensor",
    "ReshapedDTensorBuilderMixin",
    "new_reshaped_d_tensor",
    "GlobalTensorMetaBuilderMixin",
    "new_global_tensor_meta",
    "SlotOuterProductMeshBuilderMixin",
    "LiteralMeshBuilderMixin",
    "new_literal_mesh",
    "ArrangedMeshBuilderMixin",
    "new_arranged_mesh",
    "ReshapedMeshBuilderMixin",
    "new_reshaped_mesh",
    "PermutedMeshBuilderMixin",
    "new_permuted_mesh",
    "SlicedMeshBuilderMixin",
    "new_sliced_mesh",
    "SliceBuilderMixin",
    "new_slice",
    "PhysicalLayoutTableBuilderMixin",
    "new_physical_layout_table",
    "DisjointStorageMeshListBuilderMixin",
    "new_disjoint_storage_mesh_list",
    "TensorComponentPosBuilderMixin",
    "new_tensor_component_pos",
    "ConstraintViolationErrorBuilderMixin",
    "new_constraint_violation_error",
    "CheckFailedErrorBuilderMixin",
    "new_check_failed_error",
    "NotFoundErrorBuilderMixin",
    "new_not_found_error",
    "InvalidArgumentErrorBuilderMixin",
    "new_invalid_argument_error",
    "GDTensorIOErrorBuilderMixin",
    "new_gd_tensor_io_error",
    "new_ok",
    "new_err",
    "new_shard",
    "new_replicate",
    "new_partial",
    "OkBuilderMixin",
    "ErrBuilderMixin",
    "ShardBuilderMixin",
    "ReplicateBuilderMixin",
    "PartialBuilderMixin",
]


def _factory[T](cls: type[T], name: str) -> _Callable[[], T]:
    def create() -> T:
        return cls()

    create.__name__ = name
    return create


new_compute_d_tensors_view = _factory(ComputeDTensorsView, "new_compute_d_tensors_view")
new_storage_d_tensors_view = _factory(StorageDTensorsView, "new_storage_d_tensors_view")
new_d_tensors_view_map_core = _factory(
    DTensorsViewMapCore, "new_d_tensors_view_map_core"
)
new_d_tensor_table = _factory(DTensorTable, "new_d_tensor_table")
new_named_d_tensor = _factory(NamedDTensor, "new_named_d_tensor")
new_naive_d_tensor = _factory(NaiveDTensor, "new_naive_d_tensor")
new_reshaped_d_tensor = _factory(ReshapedDTensor, "new_reshaped_d_tensor")
new_global_tensor_meta = _factory(GlobalTensorMeta, "new_global_tensor_meta")
new_slot_outer_product_mesh = _factory(
    SlotOuterProductMesh, "new_slot_outer_product_mesh"
)
new_literal_mesh = _factory(LiteralMesh, "new_literal_mesh")
new_arranged_mesh = _factory(ArrangedMesh, "new_arranged_mesh")
new_reshaped_mesh = _factory(ReshapedMesh, "new_reshaped_mesh")
new_permuted_mesh = _factory(PermutedMesh, "new_permuted_mesh")
new_sliced_mesh = _factory(SlicedMesh, "new_sliced_mesh")
new_slice = _factory(Slice, "new_slice")
new_physical_layout_table = _factory(PhysicalLayoutTable, "new_physical_layout_table")
new_disjoint_storage_mesh_list = _factory(
    DisjointStorageMeshList, "new_disjoint_storage_mesh_list"
)
new_tensor_component_pos = _factory(TensorComponentPos, "new_tensor_component_pos")
new_constraint_violation_error = _factory(
    ConstraintViolationError, "new_constraint_violation_error"
)
new_check_failed_error = _factory(CheckFailedError, "new_check_failed_error")
new_not_found_error = _factory(NotFoundError, "new_not_found_error")
new_invalid_argument_error = _factory(
    InvalidArgumentError, "new_invalid_argument_error"
)
new_gd_tensor_io_error = _factory(GDTensorIOError, "new_gd_tensor_io_error")
new_ok = _factory(Ok, "new_ok")
new_err = _factory(Err, "new_err")
new_shard = _factory(Shard, "new_shard")
new_replicate = _factory(Replicate, "new_replicate")
new_partial = _factory(Partial, "new_partial")

"""gdtensor.builder: user-facing product types, view map, and helpers (C.3 / C.4).

Implementations live in internal modules; this namespace re-exports exactly
what table C.3 / C.4 names. The compute/torch bridge five functions are at
package top level (5.6).
"""
from __future__ import annotations

from typing import Callable, Optional

from .internal.dtensor import (
    DTensorTable, GenericDTensor, GlobalTensorMeta, NaiveDTensor, NamedDTensor,
    Placement, ReshapedDTensor, global_tensor_meta_of, named_of, tensor_order_value_of,
)
from .internal.layout import (
    ContainerFormat, DisjointStorageMeshList, PhysicalLayoutTable,
    TensorComponentPos, file_path_of, leader_tensor_component_pos_of,
)
from .internal.mesh import (
    ArrangedMesh, ComputeMesh, ContentSlotsMesh, LiteralMesh, OuterProductMesh,
    PermutedMesh, ReshapedMesh, Slice, SlicedMesh, StorageMesh,
)
from .internal.result import (
    CheckFailedError, ConstraintViolationError, ERROR_TYPES,
    GDTensorIOError, InvalidArgumentError, NotFoundError,
)
from .internal.views import ComputeDTensorsView, DTensorsView, StorageDTensorsView


# ------------------------------------------------------------------ 5.6 bridge (top-level names)


def get_tensor_ids(view):
    # compute_d_tensors_view_get_tensor_ids
    from .tag import Ok
    return Ok(list(view.tensor_ids))


def get_gd_tensor(view, tensor_id: str):
    # compute_d_tensors_view_get_gd_tensor
    from .internal.result import catch
    @catch
    def fn():
        nd = named_of(view.d_tensor_table, tensor_id)
        return nd.d_tensor
    return fn()


def make_gd_tensor(d_tensor):
    # gd_tensor_make: torch.DTensor -> GenericDTensor[ComputeMesh]
    from .internal import torch_bridge
    return torch_bridge.gd_tensor_make(d_tensor)


def get_d_tensor(gd_tensor):
    # gd_tensor_get_d_tensor: torch.DTensor or nil
    from .tag import Ok
    dt = gd_tensor if type(gd_tensor) is NaiveDTensor else gd_tensor.underlying_d_tensor
    return Ok(getattr(dt, "torch_d_tensor", None))


def reset_d_tensor(gd_tensor, d_tensor) -> None:
    # gd_tensor_reset_d_tensor: replace the underlying torch.DTensor
    from .internal.result import invalid
    from .tag import Err
    target = gd_tensor if type(gd_tensor) is NaiveDTensor else gd_tensor.underlying_d_tensor
    if type(target) is not NaiveDTensor:
        raise invalid("gd_tensor", "not a GenericDTensor")
    target.torch_d_tensor = d_tensor


# ------------------------------------------------------------------ 5.5 view map


class DTensorsViewMapCore:
    """Four optional callbacks. A nil tensor callback means identity; a nil
    session callback means skip that session."""

    def __init__(
        self,
        convert_each_compute_view: Optional[Callable] = None,
        convert_each_storage_view: Optional[Callable] = None,
        convert_each_storage_load_session: Optional[Callable] = None,
        convert_each_storage_save_session: Optional[Callable] = None,
    ) -> None:
        self.convert_each_compute_view = convert_each_compute_view
        self.convert_each_storage_view = convert_each_storage_view
        self.convert_each_storage_load_session = convert_each_storage_load_session
        self.convert_each_storage_save_session = convert_each_storage_save_session


def _ok_value(v, name):
    from .tag import Err, Ok
    if v is None:
        return None
    if type(v) is Ok:
        return v.value
    if type(v) is Err:
        return v
    # plain return is accepted as success
    return v


def d_tensors_view_map(origin, map_core: DTensorsViewMapCore):
    from .tag import Ok, Err

    def _result_of(v):
        if type(v) in (Ok, Err):
            return v
        return Ok(v)

    if type(origin) is ComputeDTensorsView:
        fn = map_core.convert_each_compute_view
        if fn is None:
            return Ok(origin)
        named = []
        for nd in origin.d_tensor_table.named_d_tensors:
            r = _result_of(fn(nd.d_tensor))
            if type(r) is Err:
                return r
            named.append(NamedDTensor(nd.global_tensor_id, r.value, nd.alias))
        return Ok(ComputeDTensorsView(origin.tensor_ids, DTensorTable(named), origin.data_accessor))

    if type(origin) is StorageDTensorsView:
        fn = map_core.convert_each_storage_view
        if fn is None:
            return Ok(origin)
        named = []
        for nd in origin.d_tensor_table.named_d_tensors:
            r = _result_of(fn(nd.d_tensor))
            if type(r) is Err:
                return r
            named.append(NamedDTensor(nd.global_tensor_id, r.value, nd.alias))
        # session callbacks are applied by the caller when sessions open; the
        # view constructor itself does no I/O.
        return Ok(StorageDTensorsView(
            origin.tensor_ids, DTensorTable(named), origin.data_accessor,
            origin.physical_layout, origin.begin_save, origin.begin_load))

    return Err(InvalidArgumentError(argument_name="origin",
                                    msg=f"unknown view type {type(origin).__name__}"))


Error = tuple(ERROR_TYPES)

__all__ = [
    # views / tables / tensors
    "DTensorsView", "ComputeDTensorsView", "StorageDTensorsView",
    "DTensorsViewMapCore", "d_tensors_view_map",
    "DTensorTable", "NamedDTensor", "GenericDTensor", "NaiveDTensor", "ReshapedDTensor",
    "GlobalTensorMeta", "Placement",
    "ComputeMesh", "StorageMesh", "OuterProductMesh", "ContentSlotsMesh",
    "LiteralMesh", "ArrangedMesh", "ReshapedMesh", "PermutedMesh", "SlicedMesh", "Slice",
    "PhysicalLayoutTable", "DisjointStorageMeshList", "ContainerFormat", "TensorComponentPos",
    # errors
    "ConstraintViolationError", "CheckFailedError", "NotFoundError",
    "InvalidArgumentError", "GDTensorIOError", "Error",
    # helpers (C.4)
    "global_tensor_meta_of", "tensor_order_value_of", "file_path_of",
    "leader_tensor_component_pos_of",
    # bridge five (also at package top level)
    "get_tensor_ids", "get_gd_tensor", "make_gd_tensor", "get_d_tensor", "reset_d_tensor",
]

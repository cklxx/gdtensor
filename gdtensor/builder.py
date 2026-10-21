"""Compute <-> torch.DTensor bridge and the view-map constructor (spec 5.5/5.6).

torch is an optional dependency. When it is absent, make_gd_tensor raises
InvalidArgumentError; the metadata-only lite path still works.

Short names per C.3 live at package top level (get_tensor_ids,
get_gd_tensor, make_gd_tensor, get_d_tensor, reset_d_tensor).
"""
from __future__ import annotations

from typing import Callable, Optional

from .dtensor import DTensorTable, NamedDTensor, NaiveDTensor, ReshapedDTensor, named_of
from .result import catch
from .tag import (
    InvalidArgumentError, NotFoundError, Ok, Replicate, Shard,
)
from .views import ComputeDTensorsView, StorageDTensorsView


# ------------------------------------------------------------------ 5.6 bridge


def get_tensor_ids(view: ComputeDTensorsView):
    return Ok(list(view.tensor_ids))


@catch
def get_gd_tensor(view: ComputeDTensorsView, tensor_id: str):
    # B.1.4: returns the GenericDTensor (the .d_tensor of the NamedDTensor).
    try:
        return named_of(view.d_tensor_table, tensor_id).d_tensor
    except Exception as exc:
        if "not in DTensorTable" in str(exc):
            raise NotFoundError(str(exc), key_name="tensor_id", key=tensor_id) from None
        raise


def _torch_bridge():
    from . import torch_bridge as tb
    return tb


@catch
def make_gd_tensor(d_tensor, tensor_id=None, p=None, stage: int = 0, placements=None):
    """gd_tensor_make: torch.DTensor -> gdtensor GenericDTensor (spec 5.6).

    torch is optional. tensor_id/p/stage/placements name the gdtensor tensor
    and topology for mesh mapping; without torch returns InvalidArgumentError.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        raise InvalidArgumentError(
            "make_gd_tensor requires torch (torch.distributed.tensor)",
            argument_name="d_tensor",
        ) from None
    from .torch_bridge import make_gd_tensor as _impl
    if tensor_id is None:
        tensor_id = getattr(d_tensor, "name", None) or "tensor"
    return _impl(d_tensor, tensor_id, p, stage, placements)


def get_d_tensor(gd_tensor):
    """Underlying torch.DTensor, or None when metadata-only."""
    if isinstance(gd_tensor, ReshapedDTensor):
        gd_tensor = gd_tensor.naive
    if isinstance(gd_tensor, NaiveDTensor):
        return Ok(getattr(gd_tensor, "torch_d_tensor", None))
    raise InvalidArgumentError("not a GenericDTensor", argument_name="gd_tensor")


def reset_d_tensor(gd_tensor, d_tensor) -> None:
    """Replace the underlying torch.DTensor. Meta/placements/mesh must match."""
    if isinstance(gd_tensor, ReshapedDTensor):
        gd_tensor = gd_tensor.naive
    if not isinstance(gd_tensor, NaiveDTensor):
        raise InvalidArgumentError("not a GenericDTensor", argument_name="gd_tensor")
    try:
        object.__setattr__(gd_tensor, "torch_d_tensor", d_tensor)
    except Exception as exc:  # frozen dataclass
        raise InvalidArgumentError(str(exc), argument_name="d_tensor") from None


# ------------------------------------------------------------------ 5.5 view map


class DTensorsViewMapCore:
    """Four optional callbacks. None means identity (tensors) or skip (sessions)."""

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


@catch
def d_tensors_view_map(origin, map_core: DTensorsViewMapCore):
    """Convert a view in place structurally; no range selection, no I/O."""
    if isinstance(origin, ComputeDTensorsView):
        fn = map_core.convert_each_compute_view
        if fn is None:
            return origin
        new = []
        for nd in origin.d_tensor_table.named_d_tensors:
            out = fn(nd.d_tensor)
            if out is None:
                raise InvalidArgumentError("compute view callback returned nil", argument_name="map_core")
            new.append(NamedDTensor(nd.global_tensor_id, out, nd.alias))
        table = DTensorTable(new)
        return ComputeDTensorsView(origin.tensor_ids, table, origin.data_accessor)
    if isinstance(origin, StorageDTensorsView):
        fn = map_core.convert_each_storage_view
        if fn is None:
            return origin
        new = []
        for nd in origin.d_tensor_table.named_d_tensors:
            out = fn(nd.d_tensor)
            if out is None:
                raise InvalidArgumentError("storage view callback returned nil", argument_name="map_core")
            new.append(NamedDTensor(nd.global_tensor_id, out, nd.alias))
        table = DTensorTable(new)
        # sessions are only converted when a callback is supplied; no I/O here
        return StorageDTensorsView(
            origin.tensor_ids, table, origin.data_accessor, origin.physical_layout,
            origin.begin_save, origin.begin_load,
        )
    raise InvalidArgumentError(f"unknown view type {type(origin).__name__}", argument_name="origin")





# ------------------------------------------------------------------ C.3 re-exports
# The builder namespace is the single import surface for every product type,
# mesh, layout, typed error, and helper listed in design table C.3. The
# implementations live in their own modules; this block only re-exports them.
from .dtensor import (  # noqa: E402
    DTensorTable, GenericDTensor, GlobalTensorMeta, NaiveDTensor, NamedDTensor,
    ReshapedDTensor, global_tensor_meta_of, tensor_order_value_of,
)
from .layout import (  # noqa: E402
    ContainerFormat, DisjointStorageMeshList, PhysicalLayoutTable,
    TensorComponentPos, file_path_of, leader_tensor_component_pos_of,
)
from .mesh import (  # noqa: E402
    ArrangedMesh, ComputeMesh, ContentSlotsMesh, LiteralMesh, OuterProductMesh,
    PermutedMesh, ReshapedMesh, Slice, SlicedMesh, StorageMesh,
)
from .tag import (  # noqa: E402
    CheckFailedError, ConstraintViolationError, GDTensorIOError,
    InvalidArgumentError, NotFoundError, Placement,
)
from .views import ComputeDTensorsView, DTensorsView, StorageDTensorsView  # noqa: E402

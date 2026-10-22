"""Implement the five compute/torch functions from design B.1.4."""

from __future__ import annotations

import numpy as np

from .dtensor import GlobalTensorMeta, NaiveDTensor, named_of
from .mesh import (
    ArrangedMesh,
    LiteralMesh,
    ReshapedMesh,
    resolve_compute_mesh,
    storage_root,
)
from .result import catch, invalid
from .views import ComputeDTensorsView
from ..tag import Partial, Replicate, Shard, WorldSize

_DTYPE_NAMES = {
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float64": "F64",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.int8": "I8",
    "torch.uint8": "U8",
    "torch.bool": "BOOL",
}


def get_tensor_ids(view: ComputeDTensorsView) -> list[str]:
    if type(view) is not ComputeDTensorsView:
        raise ValueError("view must be a ComputeDTensorsView")
    return list(view.tensor_ids)


@catch
def get_gd_tensor(view: ComputeDTensorsView, tensor_id: str):
    if type(view) is not ComputeDTensorsView:
        raise invalid("view", "view must be a ComputeDTensorsView")
    return named_of(view.d_tensor_table, tensor_id).d_tensor


def _placement(placement):
    from torch.distributed.tensor import (
        Partial as TorchPartial,
        Replicate as TorchReplicate,
        Shard as TorchShard,
    )

    if isinstance(placement, TorchShard):
        return Shard(placement.dim)
    if isinstance(placement, TorchReplicate):
        return Replicate()
    if isinstance(placement, TorchPartial):
        return Partial()
    raise ValueError(f"unsupported torch placement: {placement!r}")


def make_gd_tensor(d_tensor):
    from torch.distributed.tensor import DTensor
    from torch.distributed import get_world_size

    if not isinstance(d_tensor, DTensor):
        raise ValueError("d_tensor is not a torch.DTensor")
    dtype = _DTYPE_NAMES.get(str(d_tensor.dtype))
    if dtype is None:
        raise ValueError(f"unsupported torch dtype: {d_tensor.dtype}")
    mesh = d_tensor.device_mesh
    ranks = mesh.mesh.detach().cpu().reshape(-1).tolist()
    return NaiveDTensor(
        GlobalTensorMeta(list(d_tensor.shape), dtype, d_tensor.element_size()),
        [_placement(pl) for pl in d_tensor.placements],
        ReshapedMesh(LiteralMesh(ranks, WorldSize(get_world_size())), list(mesh.shape)),
        torch_d_tensor=d_tensor,
    )


def _underlying(gd_tensor):
    target = (
        gd_tensor
        if type(gd_tensor) is NaiveDTensor
        else getattr(gd_tensor, "underlying_d_tensor", None)
    )
    if type(target) is not NaiveDTensor:
        raise ValueError("gd_tensor is not a GenericDTensor")
    root = storage_root(target.mesh)
    if not (
        (type(root) is LiteralMesh and type(root.max) is WorldSize)
        or (type(root) is ArrangedMesh and type(root.size) is WorldSize)
    ):
        raise ValueError("gd_tensor requires a ComputeMesh")
    return target


def get_d_tensor(gd_tensor):
    return _underlying(gd_tensor).torch_d_tensor


def reset_d_tensor(gd_tensor, d_tensor) -> None:
    target = _underlying(gd_tensor)
    replacement = make_gd_tensor(d_tensor)
    old_meta, new_meta = target.global_tensor_meta, replacement.global_tensor_meta
    if old_meta.global_shape != new_meta.global_shape:
        raise ValueError("shape differs")
    if (old_meta.dtype, old_meta.itemsize) != (new_meta.dtype, new_meta.itemsize):
        raise ValueError("dtype differs")
    if target.placements != replacement.placements:
        raise ValueError("placements differ")
    if not np.array_equal(
        resolve_compute_mesh(target.mesh), resolve_compute_mesh(replacement.mesh)
    ):
        raise ValueError("mesh rank grid differs")
    target.torch_d_tensor = d_tensor

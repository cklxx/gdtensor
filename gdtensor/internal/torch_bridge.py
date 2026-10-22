"""gdtensor <-> torch.DTensor bridge (spec 5.6 / B.1.4).

Real integration with torch.distributed.tensor:

* gd_tensor_make(d_tensor)
    The design bridge (B.1.4): data/meta/placements/mesh from the torch tensor.
* gd_for_layout(d_tensor, tensor_id, p, stage, placements)
    Test-framework helper: wrap a torch.DTensor for a virtual (pp,dp,tp)
    topology that differs from the single-process torch device mesh.
* fabric_from_torch(view/table, tensors)
    Fill a ComputeFabric for every gdtensor rank from the torch global values,
    so gdtensor save / load / reshard operate on torch-produced bytes.
* torch_from_fabric(...)
    Gather owned rank buffers back to a global torch tensor and distribute it
    to a torch.DTensor on a torch device mesh.

Single-process note: torch is initialized with one process here. The bridge
reads torch's global tensor (full_tensor) and derives every rank's owned
intervals with the gdtensor ownership model, so the full plan/IO path is
exercised without a multi-process torch job.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dtensor import (
    DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor,
    global_tensor_meta_of, named_of, nbytes_of, owned_intervals,
)
from .result import Fail, InvalidArgumentError, catch
from ..tag import Ok, Replicate, Shard
from .views import ComputeFabric

_TORCH_DTYPE_TO_NAME = {
    "torch.float32": "F32", "torch.float": "F32",
    "torch.float16": "F16", "torch.half": "F16",
    "torch.bfloat16": "BF16",
    "torch.float64": "F64", "torch.double": "F64",
    "torch.int32": "I32", "torch.int": "I32",
    "torch.int64": "I64", "torch.long": "I64",
    "torch.int8": "I8", "torch.uint8": "U8", "torch.bool": "BOOL",
}


def _torch():
    import torch
    return torch


def _has_torch():
    import importlib.util
    return importlib.util.find_spec("torch") is not None


# ------------------------------------------------------------------ mapping


def torch_placement_to_gd(pl):
    name = type(pl).__name__
    if name == "Shard":
        return Shard(int(pl.dim))
    if name == "Replicate":
        return Replicate()
    if name == "_Partial":
        from ..tag import Partial
        return Partial()
    raise Fail(InvalidArgumentError(argument_name="placements", msg=f"unsupported torch placement {pl!r}"))


def parallel_config_from_mesh(device_mesh) -> Tuple[int, int, int]:
    """Map a torch device mesh shape to (pp, dp, tp) by dim names."""
    shape = tuple(int(x) for x in device_mesh.shape)
    names = tuple(getattr(device_mesh, "mesh_dim_names", None) or ())
    if names and len(names) == len(shape):
        idx = {n: i for i, n in enumerate(names)}
        pp = shape[idx["pp"]] if "pp" in idx else 1
        dp = shape[idx["dp"]] if "dp" in idx else 1
        tp = shape[idx["tp"]] if "tp" in idx else 1
        if pp * dp * tp != int(np.prod(shape)):
            raise Fail(InvalidArgumentError(argument_name="device_mesh", msg=f"mesh dims {names} are not exactly pp/dp/tp"))
        return pp, dp, tp
    # unnamed: single dim -> dp
    if len(shape) == 1:
        return 1, shape[0], 1
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    if len(shape) == 3:
        return shape
    raise Fail(InvalidArgumentError(argument_name="device_mesh", msg=f"cannot map mesh shape {shape} to (pp,dp,tp)"))


def gd_compute_mesh(p, stage: int):
    """SlicedMesh fixing the PP stage, matching models.compute_mesh."""
    from .models import compute_mesh
    return compute_mesh(p, stage)


def _dtype_name(dt) -> Tuple[str, int]:
    name = _TORCH_DTYPE_TO_NAME.get(str(dt.dtype), str(dt.dtype).upper().replace("TORCH.", ""))
    return name, int(dt.element_size())


# ------------------------------------------------------------------ make


def gd_for_layout(d_tensor, tensor_id: str, p=None, stage: int = 0, placements=None):
    """Test-framework helper (not a design API): wrap a torch.DTensor for a
    virtual (pp,dp,tp) layout. The single-process torch job holds the global
    values; gdtensor then plans a sharded save over that virtual topology."""
    return _make(d_tensor, tensor_id, p, stage, placements)


def gd_tensor_make(d_tensor):
    """gd_tensor_make (design B.1.4): torch.DTensor -> GenericDTensor.

    Data, global meta, placements and ComputeMesh all come from the torch
    tensor. No topology arguments. Returns Result[GenericDTensor].
    """
    from .result import catch
    if not _has_torch():
        from ..tag import Err
        return Err(InvalidArgumentError(
            argument_name="d_tensor",
            msg="torch bridge requires torch (torch.distributed.tensor)",
        ))

    @catch
    def fn():
        return _make(d_tensor, getattr(d_tensor, "name", None) or "tensor").value

    return fn()


def _make(d_tensor, tensor_id: str, p=None, stage: int = 0, placements=None):
    import torch
    dt = d_tensor
    if not hasattr(dt, "device_mesh") or not hasattr(dt, "placements"):
        raise Fail(InvalidArgumentError(argument_name="d_tensor", msg="input is not a torch.DTensor"))
    if p is None:
        from .models import ParallelConfig
        pp, dp, tp = parallel_config_from_mesh(dt.device_mesh)
        p = ParallelConfig(tp=tp, dp=dp, pp=pp)
    dtype, itemsize = _dtype_name(dt)
    meta = GlobalTensorMeta(tuple(int(x) for x in dt.shape), dtype, itemsize)
    if placements is None:
        placements = tuple(torch_placement_to_gd(pl) for pl in dt.placements)
        # pad placements to (pp, dp, tp): torch meshes without pp/dp are replicated
        if len(placements) < 3:
            placements = (Replicate(),) * (3 - len(placements)) + placements
    return Ok(NaiveDTensor(meta, tuple(placements), gd_compute_mesh(p, stage), torch_d_tensor=dt))


# ------------------------------------------------------------------ fabric fill


def global_bytes_of_dtensor(d_tensor) -> np.ndarray:
    """Deterministic global element bytes of a torch.DTensor (uint8 view)."""
    torch = _torch()
    full = d_tensor.full_tensor().detach().cpu().contiguous()
    return np.frombuffer(full.view(torch.uint8).numpy().tobytes(), dtype=np.uint8)


def fabric_from_torch(table: DTensorTable, tensors: Dict[str, object]) -> ComputeFabric:
    """Populate every rank's owned buffers from torch.DTensor global values."""
    fab = ComputeFabric(table).allocate()
    for tid, dt in tensors.items():
        nd = named_of(table, tid)
        isz = global_tensor_meta_of(nd).itemsize
        glob = global_bytes_of_dtensor(dt)
        for r in list(fab.buffers.keys()):
            iv = fab.intervals(r, tid)
            if iv.size == 0:
                continue
            elems = np.concatenate([np.arange(a, b) for a, b in iv])
            buf = glob[(elems[:, None] * isz + np.arange(isz)[None, :]).ravel()]
            fab.buffers[r][tid] = np.asarray(buf, dtype=np.uint8)
    return fab


# ------------------------------------------------------------------ gather back


def gather_global_bytes(table: DTensorTable, fabric: ComputeFabric, tensor_id: str) -> np.ndarray:
    """Reconstruct the global uint8 byte array from all rank buffers."""
    nd = named_of(table, tensor_id)
    isz = global_tensor_meta_of(nd).itemsize
    nbytes = nbytes_of(global_tensor_meta_of(nd))
    out = np.zeros(nbytes, dtype=np.uint8)
    placed = np.zeros(nbytes, dtype=bool)
    for r in list(fabric.buffers.keys()):
        iv = fabric.intervals(r, tensor_id)
        if iv.size == 0 or tensor_id not in fabric.buffers.get(r, {}):
            continue
        buf = fabric.buffers[r][tensor_id]
        off = 0
        for a, b in iv:
            n = (b - a) * isz
            seg = np.arange(a * isz, b * isz)
            if placed[seg].any():
                raise Fail(InvalidArgumentError(argument_name="global_range", msg=f"rank {r} overlaps others for {tensor_id}"))
            out[seg] = buf[off:off + n]
            placed[seg] = True
            off += n
    if not placed.all():
        raise Fail(InvalidArgumentError(argument_name="global_range", msg=f"{tensor_id}: not every element loaded back"))
    return out


def torch_from_fabric(table: DTensorTable, fabric: ComputeFabric, tensor_id: str,
                      device_mesh=None, placements=None):
    """Reconstruct a torch.DTensor from a loaded gdtensor fabric."""
    torch = _torch()
    nd = named_of(table, tensor_id)
    meta = global_tensor_meta_of(nd)
    raw = gather_global_bytes(table, fabric, tensor_id)
    t = torch.frombuffer(bytes(raw), dtype=getattr(torch, _TORCH_TO_NATIVE.get(meta.dtype, "float32"))) \
        if meta.dtype in _TORCH_TO_NATIVE else _from_bytes(torch, raw, meta)
    t = t.reshape(tuple(meta.global_shape)).clone()
    if device_mesh is None:
        return t  # metadata path with no torch mesh returns a plain tensor
    from torch.distributed.tensor import distribute_tensor
    pls = placements if placements is not None else _torch_placements_from_gd(nd)
    return distribute_tensor(t, device_mesh, pls)


_TORCH_TO_NATIVE = {
    "F32": "float32", "F16": "float16", "BF16": "bfloat16", "F64": "float64",
    "I32": "int32", "I64": "int64", "I8": "int8", "U8": "uint8",
}


def _from_bytes(torch, raw: np.ndarray, meta):
    # default: treat unknown dtypes as uint8 bytes
    return torch.frombuffer(bytes(raw), dtype=torch.uint8)


def _torch_placements_from_gd(nd: NamedDTensor):
    from torch.distributed.tensor import Replicate as TRep, Shard as TShard
    torch = _torch()
    out = []
    for pl in naive_of(nd).placements:
        if type(pl) is Shard:
            out.append(TShard(pl.value))
        else:
            out.append(TRep())
    return tuple(out)

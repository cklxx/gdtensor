"""End-to-end torch DTensor -> gdtensor -> checkpoint -> torch tensor.

Single torch process holds global values; gdtensor plans and writes a sharded
safetensors checkpoint over a virtual 4-rank DP topology, loads it back, and
rebuilds an equal torch tensor. This is the concrete proof the bridge is
"through", not just declared.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", minversion="2.0")
import torch.distributed as dist  # noqa: E402
from torch.distributed.tensor import (  # noqa: E402
    DTensor, Replicate as TRep, Shard as TShard, distribute_tensor, init_device_mesh,
)

import safetensors
from conftest import assert_all_ok  # noqa: E402
from gdtensor.builder import make_gd_tensor  # noqa: E402
from gdtensor._dtensor import DTensorTable, NamedDTensor  # noqa: E402
from gdtensor._models import ParallelConfig, layout_builder, storage_table  # noqa: E402
from gdtensor._runner import load_threads, save_threads  # noqa: E402
from gdtensor.tag import Replicate, Shard  # noqa: E402
from gdtensor._torch_bridge import fabric_from_torch, torch_from_fabric  # noqa: E402
from gdtensor._views import ComputeFabric  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1")


@pytest.fixture(scope="module", autouse=True)
def _pg():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29579")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize("shape,pls", [
    ((12, 8), (Replicate(), Shard(0), Replicate())),     # row shard
    ((8, 12), (Replicate(), Shard(1), Replicate())),    # col shard (strided)
    ((16,), (Replicate(), Shard(0), Replicate())),      # 1-D tensor
])
def test_torch_dtensor_roundtrip(tmp_path, shape, pls):
    p = ParallelConfig(tp=1, dp=4, pp=1)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    g = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    dt = distribute_tensor(g, mesh, [TRep()])
    gd = make_gd_tensor(dt, "w", p=p, placements=pls).unwrap()
    ct = DTensorTable([NamedDTensor("w", gd)])
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")

    root = str(tmp_path / "ckpt"); os.makedirs(root)
    fab = fabric_from_torch(ct, {"w": dt})
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])
    # official safetensors must decode the same element count as the torch tensor
    n_elem = 0
    for f in sorted(x for x in os.listdir(root) if x.endswith(".safetensors")):
        decoded = safetensors.deserialize(open(os.path.join(root, f), "rb").read())
        # deserialize yields (key, tensor); tolerate nested dict shapes
        for item in decoded:
            ten = item[1]
            n_elem += int(ten.numel()) if hasattr(ten, "numel") else int(
                np.prod(ten["shape"]))
    assert n_elem == int(np.prod(shape))

    ctl = DTensorTable([NamedDTensor("w", gd)])
    fab2 = ComputeFabric(ctl).allocate()
    assert_all_ok(load_threads(ctl, st, lay, root, fab2, p.world)[0])
    rebuilt = torch_from_fabric(ctl, fab2, "w")
    assert tuple(rebuilt.shape) == shape
    assert torch.equal(rebuilt, g)


def test_two_tensor_table_roundtrip(tmp_path):
    p = ParallelConfig(tp=1, dp=2, pp=1)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    a = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    b = torch.arange(24, 48, dtype=torch.float32).reshape(3, 8)
    dta = distribute_tensor(a, mesh, [TRep()])
    dtb = distribute_tensor(b, mesh, [TRep()])
    pls = (Replicate(), Shard(0), Replicate())
    gda = make_gd_tensor(dta, "a", p=p, placements=pls).unwrap()
    gdb = make_gd_tensor(dtb, "b", p=p, placements=pls).unwrap()
    ct = DTensorTable([NamedDTensor("a", gda), NamedDTensor("b", gdb)])
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    root = str(tmp_path / "ckpt2"); os.makedirs(root)
    fab = fabric_from_torch(ct, {"a": dta, "b": dtb})
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])
    ctl = DTensorTable([NamedDTensor("a", gda), NamedDTensor("b", gdb)])
    fab2 = ComputeFabric(ctl).allocate()
    assert_all_ok(load_threads(ctl, st, lay, root, fab2, p.world)[0])
    assert torch.equal(torch_from_fabric(ctl, fab2, "a"), a)
    assert torch.equal(torch_from_fabric(ctl, fab2, "b"), b)

"""Real gdtensor <-> torch.DTensor bridge tests.

Run only on an interpreter with torch (skips otherwise). Uses a single-process
torch.distributed group: torch DTensor global values are fed through the whole
gdtensor ownership / save / load path and rebuilt into a torch tensor.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", minversion="2.0")
from torch.distributed.tensor import (  # noqa: E402
    Replicate as TRep, Shard as TShard, distribute_tensor, init_device_mesh,
)

import torch.distributed as dist  # noqa: E402

from conftest import assert_all_ok  # noqa: E402
from gdtensor.builder import get_d_tensor, get_gd_tensor  # noqa: E402
from gdtensor.internal.dtensor import DTensorTable, NamedDTensor  # noqa: E402
from gdtensor_sim.models import ParallelConfig, layout_builder, storage_table  # noqa: E402
from gdtensor_sim.runner import save_threads  # noqa: E402
from gdtensor.internal.torch_bridge import (  # noqa: E402
    fabric_from_torch, gather_global_bytes, global_bytes_of_dtensor,
    parallel_config_from_mesh, torch_from_fabric, torch_placement_to_gd,
)
from gdtensor.tag import Replicate, Shard  # noqa: E402
from gdtensor.internal.torch_bridge import gd_for_layout  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1 (torch distributed)")


@pytest.fixture(scope="module", autouse=True)
def _pg():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29567")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _mesh(shape, names):
    return init_device_mesh("cpu", shape, mesh_dim_names=names)


def test_placement_and_mesh_mapping():
    assert type(torch_placement_to_gd(TShard(1))) is Shard
    assert torch_placement_to_gd(TShard(1)).value == 1
    assert type(torch_placement_to_gd(TRep())) is Replicate
    m = _mesh((1,), ("dp",))
    assert parallel_config_from_mesh(m) == (1, 1, 1)


def test_parallel_config_from_named_mesh():
    # shape->(pp,dp,tp) by dim names; exercised with a 1-rank each mesh
    m = _mesh((1, 1, 1), ("pp", "dp", "tp"))
    assert parallel_config_from_mesh(m) == (1, 1, 1)


def test_gd_for_layout_from_torch_dtensor():
    p = ParallelConfig(tp=1, dp=4, pp=1)
    mesh = _mesh((1,), ("dp",))
    g = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    dt = distribute_tensor(g, mesh, [TRep()])  # global values on the 1 torch rank
    r = gd_for_layout(dt, "w", p=p)  # gdtensor owns the 4-way shard plan
    assert r.ok
    nd = r.unwrap()
    assert list(nd.global_tensor_meta.global_shape) == [8, 4]
    assert nd.global_tensor_meta.dtype == "F32"
    assert nd.global_tensor_meta.itemsize == 4
    # default: torch placements map 1:1 (Replicate stays Replicate)
    assert type(nd.placements[-1]) is Replicate
    # explicit placements override: request a dp shard
    nd2 = gd_for_layout(dt, "w", p=p,
                         placements=(Replicate(), Shard(0), Replicate())).unwrap()
    assert type(nd2.placements[1]) is Shard
    assert get_d_tensor(nd).unwrap() is dt


def test_roundtrip_through_safetensors(tmp_path):
    """torch DTensor -> gdtensor fabric -> save -> load -> torch tensor."""
    from gdtensor_sim.runner import load_threads

    p = ParallelConfig(tp=1, dp=4, pp=1)
    mesh = _mesh((1,), ("dp",))
    g = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    dt = distribute_tensor(g, mesh, [TRep()])
    # one torch process holds global values; gdtensor shards over 4 dp ranks
    gd = gd_for_layout(dt, "w", p=p,
                        placements=(Replicate(), Shard(0), Replicate())).unwrap()
    ct = DTensorTable([NamedDTensor("w", gd)])
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")

    root = str(tmp_path / "ckpt"); os.makedirs(root)
    fab = fabric_from_torch(ct, {"w": dt})
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])

    ctl = DTensorTable([NamedDTensor("w", gd)])
    fab2 = type(fab)(ctl).allocate()
    assert_all_ok(load_threads(ctl, st, lay, root, fab2, p.world)[0])
    rebuilt = torch_from_fabric(ctl, fab2, "w")  # plain torch tensor
    assert rebuilt.shape == g.shape
    assert torch.equal(rebuilt, g)


def test_full_tensor_bytes_equivalence():
    mesh = _mesh((1,), ("dp",))
    g = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    dt = distribute_tensor(g, mesh, [TRep()])
    raw = global_bytes_of_dtensor(dt)
    expect = np.frombuffer(g.contiguous().view(torch.uint8).numpy().tobytes(), dtype=np.uint8)
    assert np.array_equal(raw, expect)


def test_gather_requires_full_coverage():
    # an empty fabric has no buffers -> gather must fail rather than produce zeros
    p = ParallelConfig(tp=1, dp=4, pp=1)
    mesh = _mesh((1,), ("dp",))
    g = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    dt = distribute_tensor(g, mesh, [TRep()])
    gd = gd_for_layout(dt, "w", p=p).unwrap()
    ct = DTensorTable([NamedDTensor("w", gd)])
    from gdtensor.internal.views import ComputeFabric
    empty = ComputeFabric(ct).allocate()
    from gdtensor.internal.result import Fail
    with pytest.raises(Fail, match="overlaps others|not every element loaded back"):
        gather_global_bytes(ct, empty, "w")


def test_get_gd_tensor_roundtrip_id():
    p = ParallelConfig(tp=1, dp=2, pp=1)
    mesh = _mesh((1,), ("dp",))
    g = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    dt = distribute_tensor(g, mesh, [TRep()])
    gd = gd_for_layout(dt, "x", p=p).unwrap()
    table = DTensorTable([NamedDTensor("x", gd)])
    from gdtensor_sim.runner import compute_view
    from gdtensor.internal.views import ComputeFabric
    fab = ComputeFabric(table).allocate()
    view = compute_view(table, fab, 0)
    got = get_gd_tensor(view, "x").unwrap()
    assert got is gd
    missing = get_gd_tensor(view, "nope")
    assert not missing.ok

"""Native torch bridge tests through the documented top-level functions."""

import os

import pytest

import gdtensor
import gdtensor.builder as gb
import gdtensor.tag as gt

pytestmark = pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)


@pytest.fixture(scope="module")
def torch_mesh(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from torch.distributed import destroy_process_group, init_process_group
    from torch.distributed.device_mesh import DeviceMesh

    path = tmp_path_factory.mktemp("bridge") / "pg"
    init_process_group("gloo", init_method="file://" + str(path), rank=0, world_size=1)
    yield torch, DeviceMesh("cpu", [0], mesh_dim_names=("dp",))
    destroy_process_group()


@pytest.mark.parametrize(
    "dtype,name,size",
    [
        ("float32", "F32", 4),
        ("float64", "F64", 8),
        ("float16", "F16", 2),
        ("bfloat16", "BF16", 2),
        ("int64", "I64", 8),
        ("int32", "I32", 4),
        ("int8", "I8", 1),
        ("uint8", "U8", 1),
        ("bool", "BOOL", 1),
    ],
)
def test_make_preserves_torch_metadata_and_identity(torch_mesh, dtype, name, size):
    torch, mesh = torch_mesh
    from torch.distributed.tensor import DTensor, Shard

    local = torch.arange(15).reshape(3, 5).to(getattr(torch, dtype))
    dt = DTensor.from_local(
        local, mesh, [Shard(1)], shape=local.shape, stride=local.stride()
    )
    gd = gdtensor.make_gd_tensor(dt)
    assert type(gd) is gb.NaiveDTensor
    assert gd.global_tensor_meta == gb.GlobalTensorMeta([3, 5], name, size)
    assert gd.placements == [gt.Shard(1)]
    assert gd.mesh.shape == [1] and gd.mesh.underlying_mesh.ranks == [0]
    assert gdtensor.get_d_tensor(gd) is dt
    assert type(gd.check_constraints()) is gt.Ok


def test_bridge_ids_lookup_and_metadata_only_values():
    gd = gb.NaiveDTensor(
        gb.GlobalTensorMeta([3], "F32", 4),
        [gt.Replicate()],
        gb.ArrangedMesh(gt.WorldSize(1)),
    )
    cv = gb.ComputeDTensorsView(
        ["b", "a"],
        gb.DTensorTable([gb.NamedDTensor("b", gd), gb.NamedDTensor("a", gd)]),
        None,
    )
    ids = gdtensor.get_tensor_ids(cv)
    assert type(ids) is list and ids == ["b", "a"]
    ids.clear()
    assert cv.tensor_ids == ["b", "a"]
    result = gdtensor.get_gd_tensor(cv, "b")
    assert type(result) is gt.Ok and result.value is gd
    result = gdtensor.get_gd_tensor(cv, "missing")
    assert type(result) is gt.Err and type(result.value) is gb.NotFoundError
    assert gdtensor.get_d_tensor(gd) is None


def test_reset_updates_naive_and_reshaped_wrapper(torch_mesh):
    torch, mesh = torch_mesh
    from torch.distributed.tensor import DTensor, Replicate

    first = DTensor.from_local(torch.arange(6).float(), mesh, [Replicate()])
    second = first + 10
    gd = gdtensor.make_gd_tensor(first)
    reshaped = gb.ReshapedDTensor(gd, [2, 3])
    assert gdtensor.reset_d_tensor(reshaped, second) is None
    assert gdtensor.get_d_tensor(reshaped) is second
    assert gb.global_tensor_meta_of(gb.NamedDTensor("w", reshaped)).global_shape == [
        2,
        3,
    ]


@pytest.mark.parametrize("change", ["shape", "dtype", "placements"])
def test_reset_rejects_changes_without_losing_original(torch_mesh, change):
    torch, mesh = torch_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    first = DTensor.from_local(torch.arange(6).float(), mesh, [Replicate()])
    gd = gdtensor.make_gd_tensor(first)
    if change == "shape":
        bad = DTensor.from_local(torch.ones(3, 2), mesh, [Replicate()])
    elif change == "dtype":
        bad = first.double()
    else:
        bad = DTensor.from_local(torch.arange(6).float(), mesh, [Shard(0)])
    with pytest.raises(
        ValueError, match=change + (" differ" if change == "placements" else " differs")
    ):
        gdtensor.reset_d_tensor(gd, bad)
    assert gdtensor.get_d_tensor(gd) is first


def test_make_rejects_plain_tensor(torch_mesh):
    torch, _ = torch_mesh
    with pytest.raises(ValueError, match="not a torch.DTensor"):
        gdtensor.make_gd_tensor(torch.ones(2))


def test_product_equality_does_not_execute_torch_element_comparisons(torch_mesh):
    torch, mesh = torch_mesh
    from torch.distributed.tensor import DTensor, Replicate

    tensor = DTensor.from_local(torch.arange(6).float(), mesh, [Replicate()])
    a, b = gdtensor.make_gd_tensor(tensor), gdtensor.make_gd_tensor(tensor)
    assert a == b and hash(a) == hash(b)
    c = gdtensor.make_gd_tensor(tensor.clone())
    assert a != c

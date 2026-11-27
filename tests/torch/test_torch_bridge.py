"""Native torch bridge tests through the documented top-level functions."""

import ctypes
import os

import numpy as np
import pytest

import gdtensor
import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_conn.formats import make_storage_view
from gdtensor_conn.iobackend import FileIO
from tests.support.oracle import TorchAccessor

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
    assert gd.placements == [gb.Shard(1)]
    assert gd.mesh.shape == [1] and gd.mesh.underlying_mesh.ranks == [0]
    assert gdtensor.get_d_tensor(gd) is dt
    assert type(gd.check_constraints()) is gb.Ok


def test_bridge_ids_lookup_and_metadata_only_values():
    gd = gb.NaiveDTensor(
        gb.GlobalTensorMeta([3], "F32", 4),
        [gb.Replicate()],
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
    assert type(result) is gb.Ok and result.value is gd
    result = gdtensor.get_gd_tensor(cv, "missing")
    assert type(result) is gb.Err and type(result.value) is gb.NotFoundError
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


@pytest.mark.parametrize("change", ["shape", "dtype", "placements", "mesh"])
def test_reset_rejects_changes_without_losing_original(torch_mesh, change):
    torch, mesh = torch_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    first = DTensor.from_local(torch.arange(6).float(), mesh, [Replicate()])
    gd = gdtensor.make_gd_tensor(first)
    if change == "shape":
        bad = DTensor.from_local(torch.ones(3, 2), mesh, [Replicate()])
    elif change == "dtype":
        bad = first.double()
    elif change == "placements":
        bad = DTensor.from_local(torch.arange(6).float(), mesh, [Shard(0)])
    else:
        gd.set_mesh(gb.ArrangedMesh(gt.WorldSize(2)))
        bad = first
    with pytest.raises(ValueError, match=change + r".*differ"):
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


class _BridgeMemCache(ga.DTensorDataAccessor):
    def __init__(self, nbytes: int) -> None:
        self.buf = bytearray(nbytes)

    def read_to_buffer(self, global_range, read_to_buffer):
        arr = (ctypes.c_char * global_range.size).from_buffer(
            self.buf, global_range.global_tensor_offset
        )
        return read_to_buffer(
            ctypes.addressof(arr) if global_range.size else 0, global_range.size
        )

    def write_from_buffer(self, global_range, write_from_buffer):
        arr = (ctypes.c_char * global_range.size).from_buffer(
            self.buf, global_range.global_tensor_offset
        )
        return write_from_buffer(
            ctypes.addressof(arr) if global_range.size else 0, global_range.size
        )


def _bridge_cache_view(spatial_pl, acc):
    cm = gb.StCacheOuterProductMesh(
        gb.ArrangedMesh(gt.NumSteps(2)),
        gb.LiteralMesh([0], gt.NumNodes(1)),
        gb.ArrangedMesh(gt.ConcurrencyWidth(1)),
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([6], "F32", 4),
        [gb.Shard(0), spatial_pl, gb.Shard(0)],
        cm,
    )
    return gb.CacheDTensorsView(["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), acc)


def test_cached_rw_chain_with_live_torch_d_tensor(torch_mesh, tmp_path, monkeypatch):
    coord = tmp_path / "coord"
    coord.mkdir()
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(coord))
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "file")
    torch, mesh = torch_mesh
    from torch.distributed.tensor import DTensor, Replicate

    src_local = torch.arange(6, dtype=torch.float32)
    dst_local = torch.zeros(6, dtype=torch.float32)
    src_gd = gdtensor.make_gd_tensor(DTensor.from_local(src_local, mesh, [Replicate()]))
    dst_gd = gdtensor.make_gd_tensor(DTensor.from_local(dst_local, mesh, [Replicate()]))
    pts = {"w": np.arange(6)}
    cv_src = gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", src_gd)]), None
    )
    cv_src.data_accessor = TorchAccessor(cv_src, pts, torch)
    cv_dst = gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", dst_gd)]), None
    )
    cv_dst.data_accessor = TorchAccessor(cv_dst, pts, torch)

    sm = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    st = gb.DTensorTable(
        [
            gb.NamedDTensor(
                "w",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([6], "F32", 4),
                    [gb.Replicate(), gb.Replicate()],
                    sm,
                ),
            )
        ]
    )
    grp = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(1))
            )
        ],
        ["w"],
    )
    pos = gb.TensorComponentPos("w", 0)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {pos: grp}, {pos: "model.safetensors"}
    )
    sv = make_storage_view(["w"], st, layout, str(tmp_path / "ckpt"), io=FileIO())

    wchain = gb.OneLayerCachedDTensorsRWChain(
        src=cv_src,
        cache=_bridge_cache_view(gb.Shard(0), _BridgeMemCache(24)),
        dst=sv,
        num_ranks_per_node=1,
    )
    assert type(wchain.check_constraints()) is gb.Ok
    assert type(gio.chain_save(0, wchain)) is gb.Ok

    rchain = gb.OneLayerCachedDTensorsRWChain(
        src=sv,
        cache=_bridge_cache_view(gb.Replicate(), _BridgeMemCache(24)),
        dst=cv_dst,
        num_ranks_per_node=1,
    )
    assert type(rchain.check_constraints()) is gb.Ok
    assert type(gio.chain_load(0, rchain)) is gb.Ok
    assert torch.equal(dst_local, src_local)

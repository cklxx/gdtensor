"""Every non-internal public symbol, exercised through its public namespace.

Rule of this file: the system under test is imported only from the five
public namespaces (gdtensor / .io / .builder / .session / .access / .tag).
Internal modules appear only to build fixtures (model tables, fabrics,
format views) because the design places builders outside the core.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import verify_fabric, verify_safetensors_dir  # noqa: E402

import gdtensor  # noqa: E402
import gdtensor.io as gio  # noqa: E402
import gdtensor.builder as gb  # noqa: E402
import gdtensor.session as gs  # noqa: E402
import gdtensor.access as ga  # noqa: E402
import gdtensor.tag as gt  # noqa: E402

# fixtures only (not the system under test)
from gdtensor.internal.dtensor import tensor_ids_of  # noqa: E402
from gdtensor_sim.models import (  # noqa: E402
    ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table,
)
from gdtensor.internal.formats import make_storage_view  # noqa: E402
from gdtensor_sim.runner import compute_view  # noqa: E402
from gdtensor.internal.views import ComputeFabric  # noqa: E402
from gdtensor_sim.coord import ThreadCoordinator  # noqa: E402
from gdtensor.internal.result import Fail  # noqa: E402

M = ModelConfig(layers=1, hidden=8, ffn=16, vocab=32)


def _world(root, p, layout="single", mode="hf", fmt="safetensors", coord=False):
    ct = compute_table(M, p)
    st = storage_table(ct, p, mode)
    lay = layout_builder(layout, st, p, fmt)
    sv = make_storage_view(tensor_ids_of(st), st, lay, root,
                           coord=ThreadCoordinator(p.world) if coord else None)
    return ct, st, lay, sv


def _run(world, fn):
    with ThreadPoolExecutor(max_workers=world) as ex:
        return list(ex.map(fn, range(world)))


# ------------------------------------------------------------------ gdtensor.tag


def test_make_box_subclass_is_tag_box():
    Foo = gt.make_box_subclass("Foo")
    assert issubclass(Foo, gt.Box)
    assert Foo(7).value == 7
    # custom tags discriminate by type and compare/hash by value
    Bar = gt.make_box_subclass("Bar")
    assert Foo(1) != Bar(1) and Foo(1) == Foo(1)
    assert hash(Foo(1)) == hash(Foo(1))
    # box subscription yields a parameterized generic alias
    assert Foo[int] is not None


def test_tag_box_repr_and_generic_alias():
    assert repr(gt.WorldSize(2)) == "WorldSize(2)"
    assert gt.Shard[0] is not None  # box subscription is a valid type alias


def test_design_tags_exist_with_correct_types():
    for t in (gt.Ok, gt.Err, gt.Shard, gt.Replicate, gt.Partial,
              gt.ConstantOne, gt.TensorCount, gt.WorldSize):
        assert issubclass(t, gt.Box)
    assert type(gt.Replicate()) is gt.Replicate
    assert gt.Shard(2).value == 2


# ------------------------------------------------------------------ top-level bridge


def test_get_tensor_ids_returns_ok_order():
    p = ParallelConfig(tp=2, dp=2, pp=1)
    ct = compute_table(M, p)
    cv = compute_view(ct, None, 0)
    r = gdtensor.get_tensor_ids(cv)
    assert type(r) is gt.Ok and r.value == tensor_ids_of(ct)


def test_get_gd_tensor_ok_and_not_found():
    p = ParallelConfig(tp=1, dp=1, pp=1)
    ct = compute_table(M, p)
    cv = compute_view(ct, None, 0)
    tid = tensor_ids_of(ct)[0]
    hit = gdtensor.get_gd_tensor(cv, tid)
    assert type(hit) is gt.Ok and hit.value.global_tensor_meta.global_shape
    miss = gdtensor.get_gd_tensor(cv, "does-not-exist")
    assert type(miss) is gt.Err and type(miss.value) is gb.NotFoundError


def test_get_d_tensor_metadata_only_is_none():
    p = ParallelConfig(tp=1, dp=1, pp=1)
    ct = compute_table(M, p)
    gd = ct.named_d_tensors[0].d_tensor
    r = gdtensor.get_d_tensor(gd)
    assert type(r) is gt.Ok and r.value is None


def test_make_gd_tensor_rejects_non_dtensor():
    r = gdtensor.make_gd_tensor(object())
    assert type(r) is gt.Err and type(r.value) is gb.InvalidArgumentError


def test_reset_d_tensor_wrong_target_raises():
    with pytest.raises(Fail):
        gdtensor.reset_d_tensor(object(), object())


def test_result_alias_covers_ok_and_error_tuple():
    # Result[T] = Ok[T] | Err[Error]; Error is the five-type product tuple.
    assert isinstance(gt.Err(gb.NotFoundError(key_name="k", key="v", msg="m")).value,
                      gdtensor.Error)


# ------------------------------------------------------------------ gdtensor.builder helpers


def test_tensor_order_value_of_ok_and_err():
    p = ParallelConfig(tp=1, dp=1, pp=1)
    ct = compute_table(M, p)
    ids = tensor_ids_of(ct)
    assert gb.tensor_order_value_of(ct, ids[1]).value == 1
    bad = gb.tensor_order_value_of(ct, "missing")
    assert type(bad) is gt.Err and type(bad.value) is gb.NotFoundError


def test_file_path_and_leader_helpers():
    p = ParallelConfig(tp=2, dp=1, pp=1)
    _, st, lay, _ = _world("/unused", p, "single")
    leader = gb.leader_tensor_component_pos_of(
        lay.tensor_entries[next(iter(lay.tensor_entries))])
    assert type(leader) is gb.TensorComponentPos
    r = gb.file_path_of(lay, leader)
    assert type(r) is gt.Ok and r.value.endswith(".safetensors")
    ghost = gb.TensorComponentPos("missing", 0)
    missing = gb.file_path_of(lay, ghost)
    assert type(missing) is gt.Err and type(missing.value) is gb.NotFoundError


# ------------------------------------------------------------------ d_tensors_view_map


def test_view_map_compute_none_is_identity():
    p = ParallelConfig(tp=2, dp=1, pp=1)
    ct = compute_table(M, p)
    cv = compute_view(ct, ComputeFabric(ct).allocate(), 0)
    assert gb.d_tensors_view_map(cv, gb.DTensorsViewMapCore()).value is cv


def test_view_map_storage_none_is_identity(tmp_path):
    p = ParallelConfig(tp=1, dp=1, pp=1)
    _, st, lay, sv = _world(str(tmp_path), p, "single")
    assert gb.d_tensors_view_map(sv, gb.DTensorsViewMapCore()).value is sv


def test_view_map_compute_conversion_ok():
    p = ParallelConfig(tp=2, dp=1, pp=1)
    ct = compute_table(M, p)
    cv = compute_view(ct, ComputeFabric(ct).allocate(), 0)
    out = gb.d_tensors_view_map(cv, gb.DTensorsViewMapCore(
        convert_each_compute_view=lambda t: t)).value
    assert type(out) is gb.ComputeDTensorsView
    assert tensor_ids_of(out.d_tensor_table) == tensor_ids_of(ct)


def test_view_map_conversion_error_short_circuits():
    p = ParallelConfig(tp=2, dp=1, pp=1)
    ct = compute_table(M, p)
    cv = compute_view(ct, ComputeFabric(ct).allocate(), 0)
    err = gb.GDTensorIOError(file_path="f", operation="x", msg="boom")

    def fail(_t):
        return gt.Err(err)

    r = gb.d_tensors_view_map(cv, gb.DTensorsViewMapCore(convert_each_compute_view=fail))
    assert type(r) is gt.Err and r.value is err


def test_view_map_unknown_origin_is_invalid_argument():
    r = gb.d_tensors_view_map(object(), gb.DTensorsViewMapCore())
    assert type(r) is gt.Err and type(r.value) is gb.InvalidArgumentError


# ------------------------------------------------------------------ public product types


def test_public_mesh_and_format_types_are_usable():
    # tagged sizes and mesh product types construct through the public surface
    a = gb.ArrangedMesh(gt.WorldSize(4))
    assert a.get_size().value == 4
    lit = gb.LiteralMesh([0, 2], gt.WorldSize(4))
    assert lit.get_ranks() == [0, 2]
    perm = gb.PermutedMesh(lit, [1, 0])
    assert perm.get_permutation() == [1, 0]
    assert type(perm.check_constraints()) is gt.Ok
    # ContainerFormat is the documented string alias; PhysicalLayoutTable uses it
    _, _, lay, _ = _world("/unused", ParallelConfig(tp=1, dp=1, pp=1))
    assert lay.get_container_format() in ("safetensors", "distcp")
    # mesh / placement / generic aliases are exported as public type names
    import typing
    for alias in (gb.ComputeMesh, gb.StorageMesh, gb.ContentSlotsMesh,
                  gb.Placement, gb.GenericDTensor):
        assert typing.get_args(alias)


# ------------------------------------------------------------------ gdtensor.io (E2E)


def test_public_rank_save_and_load_roundtrip(tmp_path):
    p = ParallelConfig(tp=2, dp=2, pp=1)
    root = str(tmp_path / "ck"); os.makedirs(root)
    ct, st, lay, sv = _world(root, p, "single", coord=True)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    save_res = _run(p.world, lambda r: gio.rank_save(r, compute_view(ct, fab, r), sv))
    assert all(type(x) is gt.Ok for x in save_res), save_res
    verify_safetensors_dir(root, st)

    _, _, _, sv2 = _world(root, p, "single")
    fab2 = ComputeFabric(ct).allocate()
    load_res = _run(p.world, lambda r: gio.rank_load(r, compute_view(ct, fab2, r), sv2))
    assert all(type(x) is gt.Ok for x in load_res), load_res
    verify_fabric(fab2, ct)


def test_public_rank_reshard_compute_to_compute():
    p1 = ParallelConfig(tp=2, dp=2, pp=1)
    p2 = ParallelConfig(tp=4, dp=1, pp=1)
    ct = compute_table(M, p1)
    ct2 = compute_table(M, p2)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    fab2 = ComputeFabric(ct2).allocate()
    res = _run(p2.world,
               lambda r: gio.rank_reshard(r, compute_view(ct, fab, r), compute_view(ct2, fab2, r)))
    assert all(type(x) is gt.Ok for x in res), res
    verify_fabric(fab2, ct2)


def test_public_rank_save_bad_root_returns_error(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("x")  # parent is a file: session cannot create the root
    p = ParallelConfig(tp=1, dp=1, pp=1)
    ct, _, _, sv = _world(str(blocked / "nested"), p, "single", coord=True)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    r = gio.rank_save(0, compute_view(ct, fab, 0), sv)
    assert type(r) is gt.Err
    assert isinstance(r.value, gdtensor.Error)


# ------------------------------------------------------------------ sessions / access ABCs


def test_session_and_access_abc_are_abstract():
    with pytest.raises(TypeError):
        gs.StorageSaveSession()
    with pytest.raises(TypeError):
        gs.StorageLoadSession()
    with pytest.raises(TypeError):
        ga.DTensorDataAccessor()


def test_minimal_concrete_sessions_and_accessor():
    class Save(gs.StorageSaveSession):
        def get_file_offset(self, pos):
            return 0

        def init_data_accessor(self):
            return gt.Ok(None)

        def commit(self):
            return gt.Ok(None)

        def abort(self):
            return gt.Ok(None)

    class Load(gs.StorageLoadSession):
        def get_file_offset(self, pos):
            return 0

        def init_data_accessor(self):
            return gt.Ok(None)

        def finish(self):
            return gt.Ok(None)

    class Acc(ga.DTensorDataAccessor):
        def read_to_buffer(self, global_range, cb):
            return gt.Ok(cb(0, 0))

        def write_from_buffer(self, global_range, cb):
            return gt.Ok(cb(0, 0))

    assert type(Save().commit()) is gt.Ok
    assert type(Load().finish()) is gt.Ok
    # callback contract is (buffer_ptr, buffer_len) -> Result
    assert type(Acc().read_to_buffer(None, lambda ptr, n: None)) is gt.Ok


# ------------------------------------------------------------------ reset via real torch (gated)


torch = pytest.importorskip("torch", minversion="2.0")
if os.environ.get("GDTENSOR_TORCH_TESTS"):
    import torch.distributed as _dist  # noqa: E402
    from torch.distributed.tensor import (  # noqa: E402
        Replicate as TRep, distribute_tensor, init_device_mesh,
    )

    @pytest.fixture(scope="module", autouse=True)
    def _pg_public():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29581")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not _dist.is_initialized():
            _dist.init_process_group(backend="gloo", rank=0, world_size=1)
        yield
        if _dist.is_initialized():
            _dist.destroy_process_group()

    def test_public_reset_d_tensor_through_public_bridge():
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        b = torch.arange(12, 24, dtype=torch.float32).reshape(3, 4)
        dta = distribute_tensor(a, mesh, [TRep()])
        dtb = distribute_tensor(b, mesh, [TRep()])
        gd = gdtensor.make_gd_tensor(dta).value
        assert gdtensor.get_d_tensor(gd).value is dta
        gdtensor.reset_d_tensor(gd, dtb)
        assert gdtensor.get_d_tensor(gd).value is dtb

    def test_public_reset_rejects_changed_shape():
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        dta = distribute_tensor(torch.arange(12, dtype=torch.float32).reshape(3, 4),
                                mesh, [TRep()])
        dtbad = distribute_tensor(torch.arange(6, dtype=torch.float32).reshape(2, 3),
                                  mesh, [TRep()])
        gd = gdtensor.make_gd_tensor(dta).value
        with pytest.raises(Fail):
            gdtensor.reset_d_tensor(gd, dtbad)

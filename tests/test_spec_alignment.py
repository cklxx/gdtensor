"""Spec alignment tests (design 1513-line revision).

Covers:
* tag Box classes (Shard/Replicate/Partial, Ok/Err) and typed Error branches;
* tagged sizes: ArrangedMesh(size=tag.*), LiteralMesh(ranks, max=tag.WorldSize);
* check_constraints recursion and the B.2.4 extra checks;
* d_tensors_view_map identity and conversion (5.5);
* compute bridge short names (5.6);
* session/access abstract base surface.
"""
from __future__ import annotations

import pytest

from gdtensor import get_d_tensor, get_gd_tensor, get_tensor_ids  # noqa: F401
from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import DTensorsViewMapCore, d_tensors_view_map
from gdtensor.internal.constraints import check_constraints
from gdtensor.internal.dtensor import DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor, tensor_ids_of
from gdtensor.internal.formats import make_storage_view
from gdtensor.internal.mesh import (
    ArrangedMesh, LiteralMesh, OuterProductMesh, ReshapedMesh, Slice, SlicedMesh,
    resolve_compute_mesh, resolve_storage_mesh, storage_mesh_cells,
)
from gdtensor_sim.models import (
    ModelConfig, ParallelConfig, compute_table, layout_builder, storage_table,
)
from gdtensor_sim.runner import compute_view
from gdtensor.session import StorageLoadSession, StorageSaveSession
from gdtensor.builder import (
    CheckFailedError, ConstraintViolationError, GDTensorIOError,
    InvalidArgumentError, NotFoundError,
)
from gdtensor.tag import (
    Box, ConstantOne, Err, Ok, Replicate, Shard, TensorCount, WorldSize,
)
from gdtensor.internal.views import ComputeFabric

M = ModelConfig(layers=1, hidden=8, ffn=12, vocab=16)


# ------------------------------------------------------------------ tags


def test_tag_box_semantics():
    assert Shard(0).value == 0
    assert type(Replicate()) is Replicate and Replicate().value is None
    assert Ok(3).value == 3
    assert type(Err("x")) is Err and Err("x").value == "x"
    assert isinstance(Replicate(), Box) and isinstance(Shard(2), Box)
    # tags of the same class/value are equal and hashable (they sit in fields)
    assert Shard(0) == Shard(0) and hash(Shard(0)) == hash(Shard(0))


def test_tag_discrimination_is_type_based():
    assert type(Shard(0)) is Shard
    assert type(Replicate()) is not Shard
    assert type(Ok(1)) is Ok and type(Err(1)) is Err


def test_tagged_int_tags_are_plain_boxes():
    assert WorldSize(4).value == 4
    assert ConstantOne(1).value == 1
    assert TensorCount(3).value == 3
    assert type(WorldSize(4)) is WorldSize


def test_typed_error_branches_are_products():
    e = NotFoundError(key_name="tensor_id", key="x", msg="no id")
    assert e.key == "x" and e.msg == "no id"
    assert CheckFailedError(check_name="covering", msg="gap").check_name == "covering"
    assert InvalidArgumentError(argument_name="tensor_ids", msg="bad").argument_name == "tensor_ids"
    assert GDTensorIOError(file_path="f", operation="read", msg="io").operation == "read"
    c = ConstraintViolationError(type_name="T", constraint_name="k", msg="c")
    assert c.type_name == "T" and c.constraint_name == "k"
    assert e == NotFoundError(key_name="tensor_id", key="x", msg="no id")


# ------------------------------------------------------------------ LiteralMesh


def test_literal_mesh_compute_ranks():
    m = LiteralMesh([1, 3, 5], WorldSize(8))
    assert resolve_compute_mesh(m).tolist() == [1, 3, 5]


@pytest.mark.parametrize("ranks,bound", [((), 4), ((1, 1), 4), ((-1,), 4), ((4,), 4)])
def test_literal_mesh_rejects_bad_ranks(ranks, bound):
    # duplicates/range/empty are B.2.4 failures of check_constraints
    r = check_constraints(LiteralMesh(list(ranks), WorldSize(bound)))
    assert type(r) is Err
    assert type(r.value) is ConstraintViolationError


def test_literal_mesh_as_compute_base_with_slice():
    lit = LiteralMesh([0, 2, 4, 6], WorldSize(8))
    sliced = SlicedMesh(ReshapedMesh(lit, [4]), [Slice(1, 3)])
    assert resolve_compute_mesh(sliced).tolist() == [2, 4]


def test_literal_content_slots_roundtrip():
    base = OuterProductMesh(ArrangedMesh(ConstantOne(1)), LiteralMesh([0, 2], WorldSize(4)))
    cells = storage_mesh_cells(base)
    assert cells == {(0, 0), (0, 2)}
    table, w = resolve_storage_mesh(base)
    assert w == 4
    assert table.ravel().tolist() == [0, -1, 2, -1]


# ------------------------------------------------------------------ check_constraints


def test_check_constraints_valid_objects():
    p = ParallelConfig(tp=2, dp=2, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    assert type(ct.check_constraints()) is Ok
    assert type(lay.check_constraints()) is Ok


def test_check_constraints_bad_meta():
    bad = GlobalTensorMeta([4], "U8", 0)
    r = bad.check_constraints()
    assert type(r) is Err and type(r.value) is ConstraintViolationError
    assert r.value.constraint_name == "itemsize_positive"


def test_duplicate_id_rejected_at_check():
    meta = GlobalTensorMeta([4], "U8", 1)
    mesh = ArrangedMesh(WorldSize(1))
    a = NamedDTensor("x", NaiveDTensor(meta, [Replicate()], mesh))
    b = NamedDTensor("x", NaiveDTensor(meta, [Replicate()], mesh))
    t = DTensorTable([a, b])
    r = t.check_constraints()
    assert type(r) is Err and r.value.constraint_name == "global_tensor_id_unique"


def test_literal_mesh_rules_via_check():
    assert type(check_constraints(LiteralMesh([0, 1], WorldSize(2)))) is Ok
    assert type(check_constraints(LiteralMesh([0, 0], WorldSize(2)))) is Err
    assert type(check_constraints(LiteralMesh([], WorldSize(2)))) is Err
    assert type(check_constraints(LiteralMesh([2], WorldSize(2)))) is Err


# ------------------------------------------------------------------ view map


def test_view_map_none_is_identity():
    p = ParallelConfig(tp=2, dp=1, pp=1)
    ct = compute_table(M, p)
    fab = ComputeFabric(ct).allocate()
    cv = compute_view(ct, fab, 0)
    out = d_tensors_view_map(cv, DTensorsViewMapCore()).value
    assert out is cv


def test_view_map_compute_conversion():
    p = ParallelConfig(tp=2, dp=1, pp=1)
    ct = compute_table(M, p)
    fab = ComputeFabric(ct).allocate()
    cv = compute_view(ct, fab, 0)
    out = d_tensors_view_map(cv, DTensorsViewMapCore(convert_each_compute_view=lambda t: t)).value
    assert [n.global_tensor_id for n in out.d_tensor_table.named_d_tensors] == tensor_ids_of(ct)


# ------------------------------------------------------------------ bridge


def test_bridge_get_tensor_ids_and_gd_tensor():
    p = ParallelConfig(tp=2, dp=1, pp=1)
    ct = compute_table(M, p)
    fab = ComputeFabric(ct).allocate()
    cv = compute_view(ct, fab, 0)
    ids = get_tensor_ids(cv).value
    assert ids == tensor_ids_of(ct)
    gd = get_gd_tensor(cv, ids[0]).value
    assert gd.global_tensor_meta.global_shape
    missing = get_gd_tensor(cv, "nope")
    assert type(missing) is Err and type(missing.value) is NotFoundError


def test_make_gd_tensor_requires_torch():
    import gdtensor
    r = gdtensor.make_gd_tensor(object())
    assert type(r) is Err and type(r.value) is InvalidArgumentError


def test_get_and_reset_d_tensor_metadata_only():
    p = ParallelConfig(tp=1, dp=1, pp=1)
    ct = compute_table(M, p)
    gd = ct.named_d_tensors[0].d_tensor
    assert get_d_tensor(gd).value is None


# ------------------------------------------------------------------ sessions/access ABCs


def test_concrete_sessions_are_spec_subclasses(tmp_path):
    import os
    from conftest import assert_all_ok
    from gdtensor_sim.models import fill_pattern
    from gdtensor_sim.runner import save_threads
    from gdtensor.tag import Ok
    p = ParallelConfig(tp=1, dp=1, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    root = str(tmp_path / "ckpt"); os.makedirs(root)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])
    sv = make_storage_view(tensor_ids_of(st), st, lay, root)
    assert isinstance(sv.begin_load(sv, 0).value, StorageLoadSession)
    root2 = str(tmp_path / "ckpt2"); os.makedirs(root2)
    sv2 = make_storage_view(tensor_ids_of(st), st, lay, root2)
    s = sv2.begin_save(sv2, 0).value
    assert isinstance(s, StorageSaveSession)
    assert isinstance(sv.data_accessor, DTensorDataAccessor)
    assert type(s.abort()) is Ok

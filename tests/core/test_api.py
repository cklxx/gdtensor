"""Public products and helpers, without internal fixtures or Result patches."""

import abc
import dataclasses
import types
import typing

import pytest

import gdtensor
import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.session as gs
import gdtensor.tag as gt
from tests.support.memory import Memory, compute_view, storage_view


def test_result_and_error_are_type_aliases():
    assert isinstance(gdtensor.Error, types.UnionType)
    assert set(typing.get_args(gdtensor.Error)) == {
        gb.ConstraintViolationError,
        gb.CheckFailedError,
        gb.NotFoundError,
        gb.InvalidArgumentError,
        gb.GDTensorIOError,
    }
    assert isinstance(gdtensor.Result, typing.TypeAliasType)
    assert typing.get_origin(gdtensor.Result[int]) is gdtensor.Result
    assert typing.get_args(gdtensor.Result[int]) == (int,)
    # Result[T] = Ok[T] | Err; introspect the union structure without dynamic
    # subscripting type params at runtime.
    body = gdtensor.Result.__value__
    parts = typing.get_args(body)
    assert {typing.get_origin(p) for p in parts} == {gb.Ok, gb.Err}


def test_scalar_tags_are_distinct_boxes_and_products_are_dataclasses():
    Foo, Bar = gt.make_box_subclass("Foo"), gt.make_box_subclass("Bar")
    assert issubclass(Foo, gt.Box)
    assert Foo(1) == Foo(1) and Foo(1) != Bar(1)
    assert hash(Foo(1)) == hash(Foo(1))
    assert repr(gt.WorldSize(2)) == "WorldSize(2)"
    for tag in (gt.ConstantOne, gt.TensorCount, gt.WorldSize):
        assert issubclass(tag, gt.Box)
    assert gb.Shard(2).axis == 2 and gb.Shard(2) == gb.Shard(2)
    assert gb.Replicate() == gb.Replicate() and gb.Replicate() != gb.Partial()
    assert hash(gb.Replicate()) == hash(gb.Replicate())
    assert not hasattr(gb.Replicate(), "value")


def test_result_products_are_not_boxes():
    # Negative issubclass against Box in isolation, so the positive Result
    # structure checks below are not narrowed to Never by the type checker.
    assert not issubclass(gb.Ok, gt.Box)
    assert not issubclass(gb.Shard, gt.Box)
    assert not issubclass(gb.Replicate, gt.Box)
    assert not issubclass(gb.Partial, gt.Box)


def test_result_union_structure():
    parts = typing.get_args(gdtensor.Result.__value__)
    assert {typing.get_origin(p) for p in parts} == {gb.Ok, gb.Err}
    assert isinstance(gb.NotFoundError(msg="missing"), gdtensor.Error)
    assert not hasattr(gb.Ok, "unwrap") and not hasattr(gb.Ok, "ok")
    assert not hasattr(gb.Err, "error")


def test_tensor_lookup_helpers_and_missing_id():
    table = gb.DTensorTable([gb.NamedDTensor("b"), gb.NamedDTensor("a")])
    assert gb.tensor_order_value_of(table, "a") == gb.Ok(1)
    missing = gb.tensor_order_value_of(table, "missing")
    assert type(missing) is gb.Err and type(missing.value) is gb.NotFoundError


def test_file_path_and_leader_helpers():
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(2))
            )
        ],
        ["a", "b"],
    )
    leader = gb.leader_tensor_component_pos_of(group)
    assert leader == gb.TensorComponentPos("a", 0)
    pos = gb.TensorComponentPos("a", 1)
    a0 = gb.TensorComponentPos("a", 0)
    layout = gb.PhysicalLayoutTable(
        "safetensors",
        {a0: group, pos: group},
        {a0: "model.safetensors"},
    )
    assert gb.file_path_of(layout, pos) == gb.Ok("model.safetensors")
    missing = gb.file_path_of(layout, gb.TensorComponentPos("missing", 0))
    assert type(missing) is gb.Err and type(missing.value) is gb.NotFoundError


def test_meshes_and_type_aliases():
    mesh = gb.PermutedMesh(
        gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(4)), [2, 2]), [1, 0]
    )
    assert type(mesh.check_constraints()) is gb.Ok
    literal = gb.LiteralMesh([3, 1], gt.WorldSize(4))
    assert literal.get_ranks() == [3, 1]
    assert type(literal.check_constraints()) is gb.Ok
    assert gb.ContainerFormat is str
    for alias in (
        gb.DTensorsView,
        gb.ComputeMesh,
        gb.StorageMesh,
        gb.ContentSlotsMesh,
        gb.Placement,
        gb.GenericDTensor,
    ):
        assert isinstance(alias, types.UnionType)


@pytest.mark.parametrize("name", sorted(n for n in gb.__all__ if n.startswith("new_")))
def test_product_factories_create_independent_defaults(name):
    factory = getattr(gb, name)
    first, second = factory(), factory()
    assert type(first) is type(second) and first is not second
    assert dataclasses.is_dataclass(first)
    mixin = getattr(gb, type(first).__name__ + "BuilderMixin")
    assert isinstance(first, mixin)
    for field in dataclasses.fields(first):
        assert hasattr(first, "get_" + field.name) and hasattr(
            first, "set_" + field.name
        )
        value = getattr(first, field.name)
        if isinstance(value, (list, dict)):
            assert value is not getattr(second, field.name)
    assert callable(first.check_constraints)


def test_product_hash_ignores_dict_insertion_order():
    a, b = gb.TensorComponentPos("a", 0), gb.TensorComponentPos("b", 1)
    left = gb.PhysicalLayoutTable("safetensors", {}, {a: "a", b: "b"})
    right = gb.PhysicalLayoutTable("safetensors", {}, {b: "b", a: "a"})
    assert left == right and hash(left) == hash(right)


def test_session_and_accessor_interfaces_are_abstract():
    for interface in (
        ga.DTensorDataAccessor,
        gs.StorageSaveSession,
        gs.StorageLoadSession,
    ):
        assert issubclass(interface, abc.ABC)
        with pytest.raises(TypeError):
            interface()
        assert not hasattr(interface, "check_constraints")


def test_bridge_plain_values_and_result_lookup():
    gd = gb.NaiveDTensor(
        gb.GlobalTensorMeta([2], "F32", 4),
        [gb.Replicate()],
        gb.ArrangedMesh(gt.WorldSize(1)),
    )
    view = gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", gd)]), None
    )
    ids = gdtensor.get_tensor_ids(view)
    assert type(ids) is list and ids == ["w"]
    ids.clear()
    assert view.tensor_ids == ["w"]
    assert gdtensor.get_d_tensor(gd) is None
    result = gdtensor.get_gd_tensor(view, "w")
    assert type(result) is gb.Ok and result.value is gd


def test_oneof_fields_can_be_cleared():
    tensor = gb.new_named_d_tensor()
    tensor.set_d_tensor(gb.new_naive_d_tensor())
    tensor.clear_d_tensor()
    assert tensor.get_d_tensor() is None
    naive = gb.new_naive_d_tensor()
    naive.set_mesh(gb.ArrangedMesh(gt.WorldSize(1)))
    naive.clear_mesh()
    assert naive.get_mesh() is None
    mesh = gb.new_reshaped_mesh()
    mesh.set_underlying_mesh(gb.ArrangedMesh(gt.WorldSize(1)))
    mesh.clear_underlying_mesh()
    assert mesh.get_underlying_mesh() is None


@pytest.mark.parametrize("name", sorted(n for n in gb.__all__ if n.startswith("new_")))
def test_builder_methods_operate_on_the_named_field(name):
    value = getattr(gb, name)()
    for field in dataclasses.fields(value):
        original = getattr(value, field.name)
        assert getattr(value, "get_" + field.name)() is original
        if isinstance(original, list):
            singular = (
                field.name[:-1]
                if field.name.endswith("s") and field.name != "alias"
                else field.name
            )
            getattr(value, "add_" + singular)("element")
            assert getattr(value, "mutable_" + field.name)() == ["element"]
            getattr(value, "clear_" + field.name)()
            assert getattr(value, field.name) == []
        elif isinstance(original, dict):
            getattr(value, "mutable_" + field.name)()["key"] = "value"
            assert getattr(value, field.name) == {"key": "value"}
            getattr(value, "clear_" + field.name)()
            assert getattr(value, field.name) == {}
        replacement = object()
        getattr(value, "set_" + field.name)(replacement)
        assert getattr(value, "get_" + field.name)() is replacement


def test_compute_bridge_rejects_storage_views_and_tensors():
    view = storage_view(Memory(range(8)), [])
    with pytest.raises(ValueError, match="ComputeDTensorsView"):
        gdtensor.get_tensor_ids(view)
    result = gdtensor.get_gd_tensor(view, "w")
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError
    tensor = view.d_tensor_table.named_d_tensors[0].d_tensor
    for value in (tensor, gb.ReshapedDTensor(tensor, [2, 4])):
        with pytest.raises(ValueError, match="ComputeMesh"):
            gdtensor.get_d_tensor(value)
        with pytest.raises(ValueError, match="ComputeMesh"):
            gdtensor.reset_d_tensor(value, object())


def test_file_path_missing_leader_returns_the_documented_error():
    view = storage_view(Memory(range(8)), [])
    view.physical_layout.mutable_file_paths().clear()
    result = gb.file_path_of(view.physical_layout, gb.TensorComponentPos("w", 0))
    assert type(result) is gb.Err and type(result.value) is gb.NotFoundError
    assert result.value.key_name == "file_paths" and result.value.msg


def test_top_level_tensor_lookup_reports_missing_id_without_torch():
    result = gdtensor.get_gd_tensor(compute_view(Memory(range(8))), "missing")
    assert type(result) is gb.Err and type(result.value) is gb.NotFoundError
    assert result.value.key_name == "tensor_id" and result.value.key == "missing"


@pytest.mark.parametrize(
    "name",
    [
        "NaiveDTensor",
        "ReshapedDTensor",
        "NamedDTensor",
        "DTensorTable",
        "ArrangedMesh",
        "LiteralMesh",
        "ReshapedMesh",
        "PermutedMesh",
        "SlicedMesh",
        "SlotOuterProductMesh",
    ],
)
def test_parameterized_products_preserve_annotation_arguments(name):
    cls = getattr(gb, name)
    annotation = cls[int]
    assert typing.get_origin(annotation) is cls
    assert typing.get_args(annotation) == (int,)

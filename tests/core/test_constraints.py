"""Validate the documented product constraints through their public methods."""

import pytest

import gdtensor.builder as gb
import gdtensor.tag as gt
from tests.support.memory import Memory, compute_view, storage_view


@pytest.mark.parametrize(
    "value,type_name",
    [
        (gb.GlobalTensorMeta([], "F32", 4), "GlobalTensorMeta"),
        (gb.GlobalTensorMeta([2], "", 4), "GlobalTensorMeta"),
        (gb.GlobalTensorMeta([2], "F32", 0), "GlobalTensorMeta"),
        (gb.ArrangedMesh(gt.WorldSize(0)), "ArrangedMesh"),
        (gb.LiteralMesh([], gt.WorldSize(2)), "LiteralMesh"),
        (gb.LiteralMesh([0, 0], gt.WorldSize(2)), "LiteralMesh"),
        (gb.LiteralMesh([2], gt.WorldSize(2)), "LiteralMesh"),
        (gb.NamedDTensor(""), "NamedDTensor"),
        (gb.DTensorTable([gb.NamedDTensor("a"), gb.NamedDTensor("a")]), "DTensorTable"),
        (gb.DisjointStorageMeshList(), "DisjointStorageMeshList"),
        (gb.ArrangedMesh(gt.WorldSize(-1)), "ArrangedMesh"),
        (gb.LiteralMesh([-1], gt.WorldSize(2)), "LiteralMesh"),
    ],
)
def test_documented_invalid_products(value, type_name):
    result = value.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError
    assert result.value.type_name == type_name
    assert result.value.constraint_name and result.value.msg


@pytest.mark.parametrize(
    "name",
    [
        "ConstraintViolationError",
        "CheckFailedError",
        "NotFoundError",
        "InvalidArgumentError",
        "GDTensorIOError",
    ],
)
def test_error_product_constraints(name):
    value = getattr(gb, name)()
    result = value.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError
    value.set_msg("detail")
    assert type(value.check_constraints()) is gt.Ok


def test_nested_constraint_error_is_returned_unchanged():
    view = compute_view(Memory(range(8)))
    meta = view.d_tensor_table.named_d_tensors[0].d_tensor.global_tensor_meta
    error = gb.ConstraintViolationError(
        type_name="GlobalTensorMeta", constraint_name="injected", msg="bad meta"
    )
    meta.check_constraints = lambda: gt.Err(error)
    result = view.check_constraints()
    assert type(result) is gt.Err and result.value is error


@pytest.mark.parametrize("change", ["ids", "accessor", "placements", "shard_axis"])
def test_compute_view_constraints(change):
    view = compute_view(Memory(range(8)))
    if change == "ids":
        view.tensor_ids = ["missing"]
    elif change == "accessor":
        view.data_accessor = None
    elif change == "placements":
        view.d_tensor_table.named_d_tensors[0].d_tensor.placements = []
    else:
        view.d_tensor_table.named_d_tensors[0].d_tensor.placements = [gt.Shard(1)]
    result = view.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError


@pytest.mark.parametrize("field", ["data_accessor", "begin_save", "begin_load"])
def test_storage_requires_callbacks(field):
    view = storage_view(Memory(range(8)), [])
    setattr(view, field, None)
    result = view.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError


def test_physical_layout_rejects_overlap_and_missing_leader():
    base = gb.OuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList([base, base], ["w"])
    result = group.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError
    group = gb.DisjointStorageMeshList([base], ["w"])
    pos = gb.TensorComponentPos("w", 0)
    layout = gb.PhysicalLayoutTable("safetensors", {pos: group}, {})
    result = layout.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError


def test_naive_equality_and_hash_include_the_torch_handle():
    meta, mesh = gb.GlobalTensorMeta([2], "F32", 4), gb.ArrangedMesh(gt.WorldSize(1))
    payload = object()
    a = gb.NaiveDTensor(meta, [gt.Replicate()], mesh, torch_d_tensor=payload)
    b = gb.NaiveDTensor(meta, [gt.Replicate()], mesh, torch_d_tensor=payload)
    assert a == b and hash(a) == hash(b)
    b.set_torch_d_tensor(object())
    assert a != b


@pytest.mark.parametrize("labels", [[], ["first"], ["first", "second", "extra"]])
def test_physical_list_requires_one_label_per_tensor_row(labels):
    mesh = gb.OuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    result = gb.DisjointStorageMeshList([mesh], labels).check_constraints()
    assert type(result) is gt.Err
    assert result.value.constraint_name == "tensor_ids_match_tensor_count"


def test_physical_list_labels_follow_table_order():
    view = storage_view(Memory(range(8)), [])
    group = next(iter(view.physical_layout.tensor_entries.values()))
    group.tensor_ids = ["another"]
    view.physical_layout.file_paths = {gb.TensorComponentPos("another", 0): "memory"}
    assert type(group.check_constraints()) is gt.Ok
    result = view.check_constraints()
    assert type(result) is gt.Err
    assert result.value.constraint_name == "tensor_ids_match_table"


def test_file_leader_uses_table_row_order_instead_of_lookup_tensor():
    mesh = gb.OuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList([mesh], ["z_first", "a_second"])
    leader = gb.TensorComponentPos("z_first", 0)
    queried = gb.TensorComponentPos("a_second", 1)
    layout = gb.PhysicalLayoutTable("safetensors", {queried: group}, {leader: "shared"})
    assert type(group.check_constraints()) is gt.Ok
    assert gb.leader_tensor_component_pos_of(group) == leader
    assert gb.file_path_of(layout, queried) == gt.Ok("shared")


def test_physical_lists_with_the_same_leader_cannot_hide_overlap():
    base = gb.OuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    first = gb.DisjointStorageMeshList(
        [gb.SlicedMesh(base, [None, gb.Slice(0, 1)])], ["w"]
    )
    second = gb.DisjointStorageMeshList([base], ["w"])
    p0, p1 = gb.TensorComponentPos("w", 0), gb.TensorComponentPos("w", 1)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {p0: first, p1: second}, {p0: "shared"}
    )
    result = layout.check_constraints()
    assert type(result) is gt.Err and result.value.constraint_name == "meshes_disjoint"


def test_valid_storage_view_and_equal_physical_lists_pass_constraints():
    view = storage_view(Memory(range(8)), [])
    assert type(view.check_constraints()) is gt.Ok
    original = next(iter(view.physical_layout.tensor_entries.values()))
    copied = gb.DisjointStorageMeshList(
        original.disjoint_storage_mesh_list, original.tensor_ids
    )
    view.physical_layout.tensor_entries[gb.TensorComponentPos("w", 0)] = copied
    assert type(view.check_constraints()) is gt.Ok


@pytest.mark.parametrize("transform", ["reshape", "empty_slice", "zero_step"])
def test_physical_list_returns_constraint_error_for_invalid_mesh(transform):
    base = gb.OuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    if transform == "reshape":
        mesh = gb.ReshapedMesh(base, [3])
    elif transform == "empty_slice":
        mesh = gb.SlicedMesh(base, [None, gb.Slice(0, 0)])
    else:
        mesh = gb.SlicedMesh(base, [None, gb.Slice(step=0)])
    group = gb.DisjointStorageMeshList([mesh], ["w"])
    result = group.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError
    assert result.value.type_name == "DisjointStorageMeshList"
    assert result.value.constraint_name == "meshes_disjoint" and result.value.msg
    pos = gb.TensorComponentPos("w", 0)
    layout = gb.PhysicalLayoutTable("safetensors", {pos: group}, {pos: "memory"})
    assert layout.check_constraints() == result
    print(
        f"[CONSTRAINTS][EXPECTED_ERROR] invalid_mesh={transform} error={result.value}",
        flush=True,
    )


@pytest.mark.parametrize(
    "change,type_name,constraint",
    [
        ("ids", "StorageDTensorsView", "tensor_ids_match_table"),
        ("unknown_tensor", "StorageDTensorsView", "layout_tensor_in_table"),
        ("format", "PhysicalLayoutTable", "container_format_unique"),
        ("extra_leader", "PhysicalLayoutTable", "file_paths_keys_are_leaders"),
        ("content_size", "ArrangedMesh", "size_positive"),
    ],
)
def test_storage_constraint_boundaries(change, type_name, constraint):
    view = storage_view(Memory(range(8)), [])
    layout = view.physical_layout
    if change == "ids":
        view.set_tensor_ids(["unknown"])
    elif change == "unknown_tensor":
        group = layout.tensor_entries.pop(gb.TensorComponentPos("w", 0))
        layout.tensor_entries[gb.TensorComponentPos("unknown", 0)] = group
    elif change == "format":
        layout.set_container_format("mixed")
    elif change == "extra_leader":
        layout.file_paths[gb.TensorComponentPos("w", 1)] = "extra"
    else:
        group = next(iter(layout.tensor_entries.values()))
        group.disjoint_storage_mesh_list[0].content_slots.set_size(gt.WorldSize(0))
    result = view.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError
    assert result.value.type_name == type_name
    assert result.value.constraint_name == constraint and result.value.msg


def test_different_file_lists_cannot_cover_the_same_cell():
    base = gb.OuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    first = gb.DisjointStorageMeshList([base], ["w"])
    second = gb.DisjointStorageMeshList(
        [gb.SlicedMesh(base, [None, gb.Slice(1, 2)])], ["w"]
    )
    p0, p1 = gb.TensorComponentPos("w", 0), gb.TensorComponentPos("w", 1)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {p0: first, p1: second}, {p0: "first", p1: "second"}
    )
    result = layout.check_constraints()
    assert type(result) is gt.Err and type(result.value) is gb.ConstraintViolationError
    assert result.value.constraint_name == "meshes_disjoint"

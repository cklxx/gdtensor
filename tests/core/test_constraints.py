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
    assert type(result) is gb.Err and type(result.value) is gb.ConstraintViolationError
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
    assert type(result) is gb.Err and type(result.value) is gb.ConstraintViolationError
    value.set_msg("detail")
    assert type(value.check_constraints()) is gb.Ok


def test_nested_constraint_error_is_returned_unchanged():
    view = compute_view(Memory(range(8)))
    meta = view.d_tensor_table.named_d_tensors[0].d_tensor.global_tensor_meta
    error = gb.ConstraintViolationError(
        type_name="GlobalTensorMeta", constraint_name="injected", msg="bad meta"
    )
    meta.check_constraints = lambda: gb.Err(error)
    result = view.check_constraints()
    assert type(result) is gb.Err and result.value is error


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
        view.d_tensor_table.named_d_tensors[0].d_tensor.placements = [gb.Shard(1)]
    result = view.check_constraints()
    assert type(result) is gb.Err and type(result.value) is gb.ConstraintViolationError


@pytest.mark.parametrize("field", ["data_accessor", "begin_save", "begin_load"])
def test_storage_requires_callbacks(field):
    view = storage_view(Memory(range(8)), [])
    setattr(view, field, None)
    result = view.check_constraints()
    assert type(result) is gb.Err and type(result.value) is gb.ConstraintViolationError


def test_physical_layout_rejects_overlapping_lists():
    base = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList([base, base])
    result = group.check_constraints()
    assert type(result) is gb.Err and type(result.value) is gb.ConstraintViolationError


def test_naive_equality_and_hash_include_the_torch_handle():
    meta, mesh = gb.GlobalTensorMeta([2], "F32", 4), gb.ArrangedMesh(gt.WorldSize(1))
    payload = object()
    a = gb.NaiveDTensor(meta, [gb.Replicate()], mesh, torch_d_tensor=payload)
    b = gb.NaiveDTensor(meta, [gb.Replicate()], mesh, torch_d_tensor=payload)
    assert a == b and hash(a) == hash(b)
    b.set_torch_d_tensor(object())
    assert a != b


def test_file_leader_and_path_resolution_by_list_identity():
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList([mesh], ["a_second", "b_other"])
    leader = gb.leader_tensor_component_pos_of(group)
    assert leader == gb.TensorComponentPos("a_second", 0)
    a0 = gb.TensorComponentPos("a_second", 0)
    queried = gb.TensorComponentPos("a_second", 1)
    layout = gb.PhysicalLayoutTable(
        "safetensors",
        {a0: group, queried: group},
        {a0: "shared"},
    )
    assert type(group.check_constraints()) is gb.Ok
    assert leader.rank == 0  # geometric leader; tensor_id needs table context
    assert gb.file_path_of(layout, queried) == gb.Ok("shared")
    missing = gb.file_path_of(layout, gb.TensorComponentPos("absent", 0))
    assert type(missing) is gb.Err and type(missing.value) is gb.NotFoundError


def test_list_without_tensor_ids_label_cannot_resolve_leader():
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList([mesh])  # no tensor_ids
    raised = False
    try:
        gb.leader_tensor_component_pos_of(group)
    except Exception as e:
        raised = type(e).__name__ == "Fail"
    assert raised


def test_reverse_order_labels_are_rejected_before_session():
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    group = gb.DisjointStorageMeshList([mesh], ["other", "w"])  # reversed
    view = storage_view(Memory(range(8)), [])
    view.d_tensor_table = gb.DTensorTable(
        [
            gb.NamedDTensor("w", view.d_tensor_table.named_d_tensors[0].d_tensor),
            gb.NamedDTensor("other", view.d_tensor_table.named_d_tensors[0].d_tensor),
        ]
    )
    view.set_tensor_ids(["w", "other"])
    pos = gb.TensorComponentPos("w", 0)
    view.physical_layout.set_tensor_entries({pos: group})
    view.physical_layout.set_file_paths({pos: "m"})
    raised = False
    try:
        view.check_constraints()
    except Exception as e:
        raised = "labels order" in str(e)
    assert raised


def test_physical_lists_with_the_same_leader_cannot_hide_overlap():
    base = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    first = gb.DisjointStorageMeshList([gb.SlicedMesh(base, [None, gb.Slice(0, 1)])])
    second = gb.DisjointStorageMeshList([base])
    p0, p1 = gb.TensorComponentPos("w", 0), gb.TensorComponentPos("w", 1)
    layout = gb.PhysicalLayoutTable(
        "safetensors",
        {p0: first, p1: second},
        {p0: "first", p1: "second"},
    )
    result = layout.check_constraints()
    assert type(result) is gb.Err and result.value.constraint_name == "meshes_disjoint"


def test_valid_storage_view_and_equal_physical_lists_pass_constraints():
    view = storage_view(Memory(range(8)), [])
    assert type(view.check_constraints()) is gb.Ok
    original = next(iter(view.physical_layout.tensor_entries.values()))
    copied = gb.DisjointStorageMeshList(
        list(original.disjoint_storage_mesh_list), list(original.tensor_ids)
    )
    view.physical_layout.tensor_entries[gb.TensorComponentPos("w", 0)] = copied
    assert type(view.check_constraints()) is gb.Ok


@pytest.mark.parametrize("transform", ["reshape", "empty_slice", "zero_step"])
def test_physical_list_returns_constraint_error_for_invalid_mesh(transform):
    base = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    if transform == "reshape":
        mesh = gb.ReshapedMesh(base, [3])
    elif transform == "empty_slice":
        mesh = gb.SlicedMesh(base, [None, gb.Slice(0, 0)])
    else:
        mesh = gb.SlicedMesh(base, [None, gb.Slice(step=0)])
    group = gb.DisjointStorageMeshList([mesh])
    result = group.check_constraints()
    assert type(result) is gb.Err and type(result.value) is gb.ConstraintViolationError
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
        old_path = layout.file_paths.pop(gb.TensorComponentPos("w", 0))
        layout.file_paths[gb.TensorComponentPos("unknown", 0)] = old_path
    elif change == "format":
        layout.set_container_format("mixed")
    elif change == "extra_leader":
        layout.file_paths[gb.TensorComponentPos("w", 1)] = "extra"
    else:
        group = next(iter(layout.tensor_entries.values()))
        group.disjoint_storage_mesh_list[0].content_slots.set_size(gt.WorldSize(0))
    result = view.check_constraints()
    assert type(result) is gb.Err and type(result.value) is gb.ConstraintViolationError
    assert result.value.type_name == type_name
    assert result.value.constraint_name == constraint and result.value.msg


def test_different_file_lists_cannot_cover_the_same_cell():
    base = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    first = gb.DisjointStorageMeshList([base])
    second = gb.DisjointStorageMeshList([gb.SlicedMesh(base, [None, gb.Slice(1, 2)])])
    p0, p1 = gb.TensorComponentPos("w", 0), gb.TensorComponentPos("w", 1)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {p0: first, p1: second}, {p0: "first", p1: "second"}
    )
    result = layout.check_constraints()
    assert type(result) is gb.Err and type(result.value) is gb.ConstraintViolationError
    assert result.value.constraint_name == "meshes_disjoint"

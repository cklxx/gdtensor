"""Public-API regression tests for the df1db8c audit findings A-F1..A-F5.

Every test drives only public builder/entry points and asserts on the Result
shape returned by public check_constraints / rank_save. No private helpers.
"""

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.memory import Memory, storage_view


# ------------------------------------------------------------------ A-F3: Err payload


def test_err_with_none_value_is_constraint_error():
    result = gb.Err(None).check_constraints()
    assert type(result) is gb.Err
    assert type(result.value) is gb.ConstraintViolationError
    assert result.value.constraint_name == "value_non_empty"


def test_err_with_non_error_value_is_constraint_error():
    result = gb.Err(42).check_constraints()
    assert type(result) is gb.Err
    assert type(result.value) is gb.ConstraintViolationError
    assert result.value.constraint_name == "value_non_empty"


def test_err_with_valid_error_payload_passes():
    result = gb.Err(
        gb.InvalidArgumentError(argument_name="x", msg="bad")
    ).check_constraints()
    assert type(result) is gb.Ok


def test_err_with_valid_but_empty_msg_still_fails():
    result = gb.Err(
        gb.InvalidArgumentError(argument_name="x", msg="")
    ).check_constraints()
    assert type(result) is gb.Err
    assert result.value.constraint_name == "msg_non_empty"


# ------------------------------------------------- A-F1: file_paths key must be leader


def _two_row_storage_layout():
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    group = gb.DisjointStorageMeshList([mesh], ["w", "u"])
    return group


def test_non_leader_file_paths_key_is_constraint_error():
    group = _two_row_storage_layout()
    p_w = gb.TensorComponentPos("w", 0)
    p_u = gb.TensorComponentPos("u", 0)
    layout = gb.PhysicalLayoutTable(
        "safetensors",
        {p_w: group, p_u: group},
        {p_u: "mem"},  # keyed on non-leader
    )
    result = layout.check_constraints()
    assert type(result) is gb.Err
    assert type(result.value) is gb.ConstraintViolationError
    assert result.value.constraint_name == "file_paths_keys_are_leaders"
    assert "not the geometric leader" in result.value.msg


def test_leader_file_paths_key_passes():
    group = _two_row_storage_layout()
    p_w = gb.TensorComponentPos("w", 0)
    p_u = gb.TensorComponentPos("u", 0)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {p_w: group, p_u: group}, {p_w: "mem"}
    )
    assert type(layout.check_constraints()) is gb.Ok


def test_missing_true_leader_entry_is_rejected_not_min_member():
    # List covers geometric leader cell (w,0) but ONLY the non-leader entry
    # (u,0) is registered. min(registered members) would wrongly pick (u,0);
    # the real geometric leader (w,0) having no entry must be an error.
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    group = gb.DisjointStorageMeshList([mesh], ["w", "u"])
    p_u = gb.TensorComponentPos("u", 0)
    layout = gb.PhysicalLayoutTable("safetensors", {p_u: group}, {p_u: "mem"})
    result = layout.check_constraints()
    assert type(result) is gb.Err
    assert result.value.constraint_name == "file_paths_keys_are_leaders"
    assert "true leader cell w,0 has no tensor entry" in result.value.msg


# ------------------------------ A-F2: equal-but-distinct (deepcopy) lists = one file


def test_equal_distinct_lists_share_one_file_path_pass_constraints():
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    list_a = gb.DisjointStorageMeshList([mesh], ["w", "u"])
    list_b = gb.DisjointStorageMeshList(
        list(list_a.disjoint_storage_mesh_list), list(list_a.tensor_ids)
    )
    p_w = gb.TensorComponentPos("w", 0)
    p_u = gb.TensorComponentPos("u", 0)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {p_w: list_a, p_u: list_b}, {p_w: "shared.bin"}
    )
    assert list_a == list_b and list_a is not list_b
    result = layout.check_constraints()
    assert type(result) is gb.Ok, result
    assert gb.file_path_of(layout, p_w).value == "shared.bin"
    assert gb.file_path_of(layout, p_u).value == "shared.bin"


# -------------------------------------------------- A-F5: list label validation


def test_duplicate_labels_rejected_with_accurate_name():
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList([mesh], ["w", "w"])
    result = group.check_constraints()
    assert type(result) is gb.Err
    assert result.value.constraint_name == "labels_unique"


def test_empty_label_rejected_with_accurate_name():
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList([mesh], [""])
    result = group.check_constraints()
    assert type(result) is gb.Err
    assert result.value.constraint_name == "labels_non_empty_strings"


def test_row_without_label_rejected():
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(2)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    group = gb.DisjointStorageMeshList([mesh], ["w"])  # row 1 has no label
    result = group.check_constraints()
    assert type(result) is gb.Err
    assert result.value.constraint_name == "labels_cover_rows"


def test_standalone_list_cannot_know_full_table_n():
    # Contract pin: a standalone DisjointStorageMeshList has no table reference,
    # so it only validates label shape (non-empty/unique) and that every covered
    # row carries a label. Extra labels beyond the mesh extent are NOT rejected
    # at list level; exact full-table N + order is enforced by the storage
    # view hook (which holds the table), never by the list alone.
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList([mesh], ["w", "extra_row"])
    assert type(group.check_constraints()) is gb.Ok


# ----------------------------- A-F4: view label order returns Err, never raises


def test_view_label_order_mismatch_returns_err_not_raise():
    group = _two_row_storage_layout()
    view = storage_view(Memory(range(8)), [])
    view.d_tensor_table = gb.DTensorTable(
        [
            gb.NamedDTensor("w", view.d_tensor_table.named_d_tensors[0].d_tensor),
            gb.NamedDTensor("u", view.d_tensor_table.named_d_tensors[0].d_tensor),
        ]
    )
    view.set_tensor_ids(["w", "u"])
    pos = gb.TensorComponentPos("w", 0)
    view.physical_layout.set_tensor_entries({pos: group})  # labels already ["w","u"]
    assert type(view.check_constraints()) is gb.Ok

    bad = gb.DisjointStorageMeshList(list(group.disjoint_storage_mesh_list), ["u", "w"])
    bad_pos = gb.TensorComponentPos("u", 0)  # geometric leader of reversed labels
    view.physical_layout.set_tensor_entries({bad_pos: bad})
    view.physical_layout.set_file_paths({bad_pos: "m"})
    raised = False
    try:
        result = view.check_constraints()
    except Exception:
        raised = True
    assert not raised, "public check_constraints must return Result, never raise"
    assert type(result) is gb.Err
    assert result.value.constraint_name == "list_labels_match_table"


# ------------------------- IO regression guard: valid layout still round-trips


def test_valid_storage_view_still_roundtrips_via_rank_save():
    from tests.support.memory import compute_view

    mem = Memory(range(8))
    view = storage_view(mem, [])
    result = gio.rank_save(0, compute_view(mem), view)
    assert type(result) is gb.Ok

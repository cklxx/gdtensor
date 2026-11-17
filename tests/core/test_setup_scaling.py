"""Setup hashes a shared mesh list once per call and stays near linear.

Entries of one file share one ``DisjointStorageMeshList`` object. Hashing it
walks every mesh, so hashing once per entry made setup grow with
ranks x tensors x meshes. Nothing may be cached across calls: builders are
mutable, and a later call must see every edit.
"""

import copy
import sys
from typing import Any

import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_conn.models import build_layout
from tests.support.setup_plans import Trace, compute_view, storage_view


def _layout(rows=16):
    names = [f"tensor-{row}" for row in range(rows)]
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(rows)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    group = gb.DisjointStorageMeshList([mesh], names)
    positions = [gb.TensorComponentPos(name, 0) for name in names]
    layout = gb.PhysicalLayoutTable(
        "distcp", dict.fromkeys(positions, group), {positions[0]: "file"}
    )
    return layout, group


def _count_hashes(monkeypatch):
    hashes = []
    original = gb.DisjointStorageMeshList.__hash__

    def counted(value):
        hashes.append(id(value))
        return original(value)

    monkeypatch.setattr(gb.DisjointStorageMeshList, "__hash__", counted)
    return hashes


def test_shared_list_is_hashed_once_per_call(monkeypatch):
    layout, group = _layout(128)
    hashes = _count_hashes(monkeypatch)
    assert type(layout.check_constraints()) is gb.Ok
    assert hashes == [id(group)]


def test_edit_between_calls_is_hashed_again_and_seen(monkeypatch):
    layout, group = _layout(128)
    hashes = _count_hashes(monkeypatch)
    assert type(layout.check_constraints()) is gb.Ok
    before = hash(group)
    hashes.clear()
    old = next(iter(layout.tensor_entries))
    renamed = gb.TensorComponentPos("new-name", 0)
    group.tensor_ids[0] = renamed.tensor_id
    layout.tensor_entries[renamed] = layout.tensor_entries.pop(old)
    layout.file_paths[renamed] = layout.file_paths.pop(old)
    assert type(layout.check_constraints()) is gb.Ok
    assert hashes == [id(group)]
    assert hash(group) != before


def test_invalid_edit_between_calls_is_rejected():
    layout, group = _layout()
    assert type(layout.check_constraints()) is gb.Ok
    group.tensor_ids[-1] = ""
    result = layout.check_constraints()
    assert type(result) is gb.Err
    assert result.value.constraint_name == "labels_non_empty_strings"
    group.tensor_ids[-1] = "tensor-15"
    assert type(layout.check_constraints()) is gb.Ok


def test_equal_distinct_lists_are_one_file(monkeypatch):
    layout, first = _layout()
    second = copy.deepcopy(first)
    last = list(layout.tensor_entries)[-1]
    layout.tensor_entries[last] = second
    hashes = _count_hashes(monkeypatch)
    assert type(layout.check_constraints()) is gb.Ok
    assert sorted(hashes) == sorted([id(first), id(second)])
    paths = {gb.file_path_of(layout, pos).value for pos in layout.tensor_entries}
    assert paths == {"file"}


def test_unequal_copy_is_a_second_file_and_needs_its_own_path():
    layout, first = _layout()
    second = copy.deepcopy(first)
    second.tensor_ids[-1] = "other"
    last = list(layout.tensor_entries)[-1]
    layout.tensor_entries[last] = second
    assert type(layout.check_constraints()) is gb.Err


# ------------------------------------------------------------------ plans


def _tables(rows, world):
    """Sharded rows, with every fourth row a scalar stored by one rank."""
    compute, storage = [], []
    all_ranks = gb.ReshapedMesh(
        gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
        ),
        [world],
    )
    for row in range(rows):
        scalar = row % 4 == 3
        meta = gb.GlobalTensorMeta([1] if scalar else [world * 2, 3], "F32", 4)
        place = gb.Replicate() if scalar else gb.Shard(0)
        writer = gb.Slice(row % world, row % world + 1)
        mesh = gb.SlicedMesh(all_ranks, [writer]) if scalar else all_ranks
        live = gb.NaiveDTensor(meta, [place], gb.ArrangedMesh(gt.WorldSize(world)))
        compute.append(gb.NamedDTensor(f"t{row}", live))
        storage.append(gb.NamedDTensor(f"t{row}", gb.NaiveDTensor(meta, [place], mesh)))
    return gb.DTensorTable(compute), gb.DTensorTable(storage)


def _file_namer(shared):
    if shared:
        return lambda t, w, nd: "model.bin"
    return lambda t, w, nd: f"rank{w}.bin"


def _save_plan(compute, storage, layout, rank=0):
    trace = Trace()
    view = storage_view(storage, layout, trace)
    result = gio.rank_save(rank, compute_view(compute, trace), view)
    assert type(result) is gb.Ok, result
    return trace.ranges


def test_plan_follows_a_layout_edit_between_operations():
    compute, storage = _tables(8, 2)
    layout = build_layout(storage, 2, "safetensors", _file_namer(False))
    first = _save_plan(compute, storage, layout)
    assert {r[1] for r in first if r[0] == "storage"} == {"rank0.bin"}
    for pos in list(layout.file_paths):
        layout.file_paths[pos] = layout.file_paths[pos].replace("rank", "moved")
    second = _save_plan(compute, storage, layout)
    assert {r[1] for r in second if r[0] == "storage"} == {"moved0.bin"}
    assert [r[2:] for r in first] == [r[2:] for r in second]


def test_plan_follows_a_mesh_list_edit_between_operations():
    compute, storage = _tables(8, 2)
    layout = build_layout(storage, 2, "safetensors", _file_namer(True))
    assert len(_save_plan(compute, storage, layout)) > 0
    group = next(iter(layout.tensor_entries.values()))
    # Drop the mesh that covers the first rows: rank 0 loses their storage.
    group.disjoint_storage_mesh_list.pop(0)
    trace = Trace()
    view = storage_view(storage, layout, trace)
    result = gio.rank_save(0, compute_view(compute, trace), view)
    assert type(result) is gb.Err
    assert trace.ranges == []


# ------------------------------------------------------------------ scaling


class _LineCounter:
    """Count Python lines executed inside the core.

    A deterministic stand-in for setup time: it does not depend on the
    machine or its load, only on how much Python work the core does.
    """

    def __init__(self) -> None:
        self.lines = 0

    def local(self, frame: Any, event: str, arg: Any) -> Any:
        self.lines += event == "line"
        return self.local

    def tracer(self, frame: Any, event: str, arg: Any) -> Any:
        inside = "/gdtensor/internal/" in frame.f_code.co_filename
        return self.local if inside else None


def _line_events(run) -> int:
    counter = _LineCounter()
    previous = sys.gettrace()
    sys.settrace(counter.tracer)
    try:
        run()
    finally:
        sys.settrace(previous)
    return counter.lines


def _setup_events(rows, world, shared):
    compute, storage = _tables(rows, world)
    layout = build_layout(storage, world, "safetensors", _file_namer(shared))

    def run():
        assert type(layout.check_constraints()) is gb.Ok
        assert len(_save_plan(compute, storage, layout)) > 0

    return _line_events(run)


@pytest.mark.parametrize("shared", [False, True], ids=["per_rank", "shared_file"])
def test_setup_work_grows_near_linearly_with_tensors(shared):
    small = _setup_events(24, 6, shared)
    large = _setup_events(96, 6, shared)
    # Four times the tensors: linear is 4x. Hashing per entry or filling the
    # slot grid in Python loops each gave 10x to 15x here.
    assert large < 8 * small, (small, large)


def test_setup_work_grows_near_linearly_with_ranks_times_tensors():
    small = _setup_events(24, 2, False)
    large = _setup_events(96, 6, False)
    # Twelve times ranks x tensors: linear is 12x. The old setup gave 50x,
    # and either old behavior alone gave more than 35x.
    assert large < 20 * small, (small, large)

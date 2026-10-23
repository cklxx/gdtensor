"""Observe internal planning only through public IO and accessor callbacks."""

import copy

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.memory import Memory, compute_view, storage_view


class TraceMemory(Memory):
    def __init__(self, data):
        super().__init__(data)
        self.ranges = []

    def _call(self, rng, callback):
        if hasattr(rng, "global_tensor_offset"):
            address = ("compute", rng.global_tensor_id, rng.global_tensor_offset)
        else:
            address = ("storage", rng.file_path, rng.contiguous_payload_offset)
        self.ranges.append((*address, rng.size))
        return super()._call(rng, callback)


def _compute_columns(memory):
    view = compute_view(memory)
    tensor = view.d_tensor_table.named_d_tensors[0].d_tensor
    tensor.set_global_tensor_meta(gb.GlobalTensorMeta([2, 5], "I16", 2))
    tensor.set_mesh(gb.LiteralMesh([2, 0, 3, 1], gt.WorldSize(4)))
    tensor.set_placements([gt.Shard(1)])
    return view


def _storage_rows(memory, events):
    view = storage_view(memory, events)
    tensor = view.d_tensor_table.named_d_tensors[0].d_tensor
    tensor.set_global_tensor_meta(gb.GlobalTensorMeta([2, 5], "I16", 2))
    tensor.set_mesh(
        gb.OuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(3))
        )
    )
    tensor.set_placements([gt.Replicate(), gt.Shard(0)])
    group = gb.DisjointStorageMeshList(
        [
            gb.OuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(3))
            )
        ],
        ["w"],
    )
    view.physical_layout.set_tensor_entries(
        {gb.TensorComponentPos("w", rank): group for rank in (0, 1)}
    )
    return view


def _run(operation, rank, src, dst):
    if operation == "save":
        return gio.rank_save(rank, src, dst)
    if operation == "load":
        return gio.rank_load(rank, dst, src)
    return gio.rank_reshard(rank, src, dst)


def _expected_bytes(operation, rank):
    # 2x5 I16, column chunks of 2 elements on ranks [2, 0, 3, 1].
    columns = {0: [4, 5, 6, 7, 14, 15, 16, 17], 2: [0, 1, 2, 3, 10, 11, 12, 13]}
    columns[3] = [8, 9, 18, 19]
    # Storage row chunks: first row on rank 0, second row on rank 1.
    rows = {0: list(range(10)), 1: list(range(10, 20))}
    return (rows if operation in ("cs", "ss") else columns).get(rank, [])


def _accessed_bytes(memory):
    return [
        byte
        for _, _, offset, size in memory.ranges
        for byte in range(offset, offset + size)
    ]


@pytest.mark.parametrize("operation", ["save", "load", "cc"])
@pytest.mark.parametrize("rank", range(4))
def test_permuted_mesh_changes_owners_without_changing_tensor_bytes(operation, rank):
    source, target, events = TraceMemory(range(20)), TraceMemory([0xA5] * 20), []
    src = (
        _storage_rows(source, events)
        if operation == "load"
        else _compute_columns(source)
    )
    dst = (
        _storage_rows(target, events)
        if operation == "save"
        else _compute_columns(target)
    )
    permuted = src if operation == "save" else dst
    tensor = permuted.d_tensor_table.named_d_tensors[0].d_tensor
    tensor.set_mesh(
        gb.PermutedMesh(
            gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(4)), [2, 2]), [1, 0]
        )
    )
    tensor.set_placements([gt.Shard(0), gt.Shard(1)])
    # Mesh [[0, 2], [1, 3]]; row chunks have 1 row, column chunks have 3 columns.
    expected = {
        0: list(range(6)),
        1: list(range(10, 16)),
        2: list(range(6, 10)),
        3: list(range(16, 20)),
    }[rank]
    assert type(src.check_constraints()) is type(dst.check_constraints()) is gt.Ok
    result = _run(operation, rank, src, dst)
    assert type(result) is gt.Ok, result
    assert (
        sorted(_accessed_bytes(source)) == sorted(_accessed_bytes(target)) == expected
    )
    wanted = np.full(20, 0xA5, dtype=np.uint8)
    wanted[expected] = np.array(expected, dtype=np.uint8)
    np.testing.assert_array_equal(target.data, wanted)
    np.testing.assert_array_equal(source.data, np.arange(20, dtype=np.uint8))
    print(
        f"[PLAN][PERMUTED_MESH][PASS] operation={operation} rank={rank} expected_bytes={expected}",
        flush=True,
    )


@pytest.mark.parametrize(
    ("selection", "owners"),
    [(gb.Slice(0, 4, 2), [0, 2]), (gb.Slice(3, None, -1), [3, 2, 1, 0])],
    ids=["step_two", "reverse"],
)
def test_sliced_mesh_step_preserves_save_and_load_ownership(selection, owners):
    source, saved, events = TraceMemory(range(8)), TraceMemory([0xA5] * 8), []
    cv, sv = compute_view(source), storage_view(saved, events)
    tensor = cv.d_tensor_table.named_d_tensors[0].d_tensor
    tensor.set_mesh(gb.SlicedMesh(gb.ArrangedMesh(gt.WorldSize(4)), [selection]))
    tensor.set_placements([gt.Shard(0)])
    assert type(cv.check_constraints()) is gt.Ok
    chunk = 8 // len(owners)
    for rank in range(4):
        expected = (
            list(range(owners.index(rank) * chunk, (owners.index(rank) + 1) * chunk))
            if rank in owners
            else []
        )
        source.ranges.clear()
        saved.ranges.clear()
        result = gio.rank_save(rank, cv, sv)
        assert type(result) is gt.Ok, result
        assert (
            sorted(_accessed_bytes(source))
            == sorted(_accessed_bytes(saved))
            == expected
        )
        target = TraceMemory([0xA5] * 8)
        loaded = gb.ComputeDTensorsView(cv.tensor_ids, cv.d_tensor_table, target)
        result = gio.rank_load(rank, loaded, sv)
        assert type(result) is gt.Ok, result
        assert sorted(_accessed_bytes(target)) == expected
        wanted = np.full(8, 0xA5, dtype=np.uint8)
        wanted[expected] = np.array(expected, dtype=np.uint8)
        np.testing.assert_array_equal(target.data, wanted)
        print(
            f"[PLAN][SLICED_MESH][PASS] step={selection.step} rank={rank} expected_bytes={expected}",
            flush=True,
        )
    np.testing.assert_array_equal(saved.data, np.arange(8, dtype=np.uint8))


@pytest.mark.parametrize("operation", ["save", "load", "cc", "cs", "sc", "ss"])
@pytest.mark.parametrize("rank", range(5))
def test_public_io_plans_discontiguous_bytes_and_empty_ranks(operation, rank):
    events = []
    source, target = TraceMemory(range(20)), TraceMemory([0xA5] * 20)
    src = (
        _storage_rows(source, events)
        if operation in ("load", "sc", "ss")
        else _compute_columns(source)
    )
    dst = (
        _storage_rows(target, events)
        if operation in ("save", "cs", "ss")
        else _compute_columns(target)
    )
    schema = copy.deepcopy((src.d_tensor_table, dst.d_tensor_table))
    assert type(src.check_constraints()) is gt.Ok
    assert type(dst.check_constraints()) is gt.Ok
    result = _run(operation, rank, src, dst)
    assert type(result) is gt.Ok, result
    expected = _expected_bytes(operation, rank)
    assert sorted(_accessed_bytes(source)) == expected
    assert sorted(_accessed_bytes(target)) == expected
    assert [r[3] for r in source.ranges] == [r[3] for r in target.ranges]
    wanted = np.full(20, 0xA5, dtype=np.uint8)
    wanted[expected] = np.array(expected, dtype=np.uint8)
    np.testing.assert_array_equal(target.data, wanted)
    np.testing.assert_array_equal(source.data, np.arange(20, dtype=np.uint8))
    assert schema == (src.d_tensor_table, dst.d_tensor_table)
    print(
        f"[PLAN][PUBLIC_IO][PASS] operation={operation} rank={rank} "
        f"expected_bytes={expected} reads={source.ranges} writes={target.ranges}",
        flush=True,
    )


@pytest.mark.parametrize("operation", ["load", "sc", "ss"])
def test_source_gap_fails_in_plan_before_any_session_or_access(operation):
    events = []
    source, target = TraceMemory(range(20)), TraceMemory([0xA5] * 20)
    src = _storage_rows(source, events)
    dst = (
        _storage_rows(target, events) if operation == "ss" else _compute_columns(target)
    )
    group = next(iter(src.physical_layout.tensor_entries.values()))
    base = group.disjoint_storage_mesh_list[0]
    group.set_disjoint_storage_mesh_list([gb.SlicedMesh(base, [None, gb.Slice(0, 1)])])
    del src.physical_layout.tensor_entries[gb.TensorComponentPos("w", 1)]
    result = _run(operation, 1 if operation == "ss" else 2, src, dst)
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError
    assert (
        result.value.argument_name == "src_view"
        and "no source covers" in result.value.msg
    )
    assert not events and not source.ranges and not target.ranges
    np.testing.assert_array_equal(target.data, [0xA5] * 20)
    print(
        f"[PLAN][EXPECTED_ERROR] operation={operation} source_gap=second_row "
        "sessions_opened=0 accessor_calls=0 target_unchanged=True",
        flush=True,
    )


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("rank", range(5))
def test_packed_storage_offsets_differ_from_global_offsets(operation, rank):
    # Storage columns concatenate [cols 0:2], [cols 2:4], [col 4], then an empty slot.
    packed = [0, 1, 2, 3, 10, 11, 12, 13, 4, 5, 6, 7, 14, 15, 16, 17, 8, 9, 18, 19]
    events = []
    compute = TraceMemory(range(20) if operation == "save" else [0xA5] * 20)
    storage = TraceMemory(packed if operation == "load" else [0xA5] * 20)
    cv, sv = _compute_columns(compute), _storage_rows(storage, events)
    tensor = sv.d_tensor_table.named_d_tensors[0].d_tensor
    tensor.mesh.content_slots.set_size(gt.WorldSize(4))
    tensor.set_placements([gt.Replicate(), gt.Shard(1)])
    group = next(iter(sv.physical_layout.tensor_entries.values()))
    group.disjoint_storage_mesh_list[0].content_slots.set_size(gt.WorldSize(4))
    sv.physical_layout.tensor_entries[gb.TensorComponentPos("w", 2)] = group
    assert type(sv.check_constraints()) is gt.Ok
    result = (
        gio.rank_save(rank, cv, sv)
        if operation == "save"
        else gio.rank_load(rank, cv, sv)
    )
    assert type(result) is gt.Ok, result
    global_bytes = _expected_bytes(operation, rank)
    storage_bytes = sorted(packed.index(byte) for byte in global_bytes)
    assert sorted(_accessed_bytes(compute)) == global_bytes
    assert sorted(_accessed_bytes(storage)) == storage_bytes
    target, indices, values = (
        (storage, storage_bytes, np.array(packed)[storage_bytes])
        if operation == "save"
        else (compute, global_bytes, global_bytes)
    )
    expected = np.full(20, 0xA5, dtype=np.uint8)
    expected[indices] = np.array(values, dtype=np.uint8)
    np.testing.assert_array_equal(target.data, expected)
    print(
        f"[PLAN][PACKED_OFFSETS][PASS] operation={operation} rank={rank} "
        f"global_bytes={global_bytes} payload_bytes={storage_bytes} "
        f"compute_ranges={compute.ranges} storage_ranges={storage.ranges}",
        flush=True,
    )


def test_replicated_save_writes_each_tensor_once_in_table_order():
    ids = ["z_first", "a_second", "m_third", "b_fourth", "y_fifth"]
    source, target, events = TraceMemory(range(5)), TraceMemory([0xA5] * 15), []
    cv, sv = compute_view(source), storage_view(target, events)
    compute_table, storage_table, entries = [], [], {}
    group = gb.DisjointStorageMeshList(
        [
            gb.OuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(5)), gb.ArrangedMesh(gt.WorldSize(1))
            )
        ],
        ids,
    )
    for size, tid in enumerate(ids, 1):
        meta = gb.GlobalTensorMeta([size], "U8", 1)
        compute_mesh = gb.LiteralMesh([4, 1, 6], gt.WorldSize(7))
        compute_table.append(
            gb.NamedDTensor(tid, gb.NaiveDTensor(meta, [gt.Replicate()], compute_mesh))
        )
        storage_table.append(
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    meta,
                    [gt.Replicate()] * 2,
                    sv.d_tensor_table.named_d_tensors[0].d_tensor.mesh,
                ),
            )
        )
        entries[gb.TensorComponentPos(tid, 0)] = group
    cv.set_tensor_ids(ids)
    cv.set_d_tensor_table(gb.DTensorTable(compute_table))
    sv.set_tensor_ids(ids)
    sv.set_d_tensor_table(gb.DTensorTable(storage_table))
    sv.physical_layout.set_tensor_entries(dict(reversed(list(entries.items()))))
    sv.physical_layout.set_file_paths({gb.TensorComponentPos(ids[0], 0): "memory"})
    assert type(cv.check_constraints()) is type(sv.check_constraints()) is gt.Ok
    for rank in (6, 0, 4, 1):
        result = gio.rank_save(rank, cv, sv)
        assert type(result) is gt.Ok, result
    assert sorted(source.ranges) == sorted(
        ("compute", tid, 0, size) for size, tid in enumerate(ids, 1)
    )
    assert sorted(target.ranges) == [
        ("storage", "memory", offset, size)
        for offset, size in zip([0, 1, 3, 6, 10], range(1, 6))
    ]
    expected = [byte for size in range(1, 6) for byte in range(size)]
    np.testing.assert_array_equal(target.data, expected)
    print(
        f"[PLAN][REPLICATED_SAVE][PASS] ranks=[4,1,6] ids={ids} "
        f"writes={target.ranges} payload={target.data.tolist()} each_tensor_written_once=True",
        flush=True,
    )

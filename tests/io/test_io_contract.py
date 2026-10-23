"""Public sessions and accessors need no internal index or bind methods."""

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.compat import rank_reshard
import gdtensor.tag as gt
from tests.support.memory import Memory, compute_view, storage_view


@pytest.mark.parametrize(
    "operation,side",
    [
        ("save", "destination"),
        ("load", "source"),
        ("ss", "source"),
        ("ss", "destination"),
    ],
)
@pytest.mark.parametrize("failure", ["err", "exception"])
def test_begin_failure_does_not_copy_and_only_closes_opened_sessions(
    operation, side, failure
):
    events = []
    error = gb.GDTensorIOError(operation="begin", msg="injected begin failure")
    source, target = Memory(range(8)), Memory([0xA5] * 8)
    src = storage_view(source, events) if operation != "save" else compute_view(source)
    dst = storage_view(target, events) if operation != "load" else compute_view(target)
    failing = src if side == "source" else dst
    method = "begin_load" if side == "source" else "begin_save"

    def begin(view, rank):
        events.append(method)
        if failure == "exception":
            raise OSError(error.msg)
        return gb.Err(error)

    setattr(failing, method, begin)
    if operation == "save":
        result = gio.rank_save(0, src, dst)
    elif operation == "load":
        result = gio.rank_load(0, dst, src)
    else:
        result = rank_reshard(0, src, dst)
    assert type(result) is gb.Err and type(result.value) is gb.GDTensorIOError
    assert error.msg in result.value.msg
    if failure == "err":
        assert result.value is error
    expected = (
        ["begin_load", "init_load", "begin_save", "finish"]
        if operation == "ss" and side == "destination"
        else [method]
    )
    assert events == expected
    np.testing.assert_array_equal(target.data, [0xA5] * 8)
    np.testing.assert_array_equal(source.data, range(8))


@pytest.mark.parametrize("direction", ["save", "load", "cc", "cs", "sc", "ss"])
def test_custom_sessions_and_accessors(direction):
    events = []
    source, destination = Memory(range(8)), Memory([0xA5] * 8)
    src = (
        storage_view(source, events)
        if direction in ("load", "sc", "ss")
        else compute_view(source)
    )
    dst = (
        storage_view(destination, events)
        if direction in ("save", "cs", "ss")
        else compute_view(destination)
    )
    if direction == "save":
        result = gio.rank_save(0, src, dst)
    elif direction == "load":
        result = gio.rank_load(0, dst, src)
    else:
        result = rank_reshard(0, src, dst)
    assert type(result) is gb.Ok, result
    np.testing.assert_array_equal(destination.data, source.data)
    assert (
        events.count("init_save")
        == events.count("commit")
        == int(direction in ("save", "cs", "ss"))
    )
    assert (
        events.count("init_load")
        == events.count("finish")
        == int(direction in ("load", "sc", "ss"))
    )
    assert "abort" not in events


@pytest.mark.parametrize("operation", ["save", "load", "reshard"])
@pytest.mark.parametrize("failure", ["init", "read", "write"])
def test_io_failure_closes_sessions(operation, failure):
    events = []
    error = gb.GDTensorIOError(operation=failure, msg="injected failure")
    source = Memory(range(8), error if failure == "read" else None)
    dest = Memory([0xA5] * 8, error if failure == "write" else None)
    if operation == "save":
        result = gio.rank_save(
            0,
            compute_view(source),
            storage_view(dest, events, init_error=error if failure == "init" else None),
        )
        expected = ["begin_save", "init_save", "abort"]
    elif operation == "load":
        result = gio.rank_load(
            0,
            compute_view(dest),
            storage_view(
                source, events, init_error=error if failure == "init" else None
            ),
        )
        expected = ["begin_load", "init_load", "finish"]
    else:
        result = rank_reshard(
            0,
            storage_view(source, events),
            storage_view(dest, events, init_error=error if failure == "init" else None),
        )
        expected = [
            "begin_load",
            "init_load",
            "begin_save",
            "init_save",
            "abort",
            "finish",
        ]
    assert type(result) is gb.Err and result.value is error, result
    assert events == expected
    np.testing.assert_array_equal(dest.data, [0xA5] * 8)


@pytest.mark.parametrize("side", ["read", "write"])
@pytest.mark.parametrize("length", [0, 4, 12])
def test_io_rejects_wrong_buffer_size(side, length):
    source = Memory(range(8), length=length if side == "read" else None)
    destination = Memory([0xA5] * 8, length=length if side == "write" else None)
    result = rank_reshard(0, compute_view(source), compute_view(destination))
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError, (
        result
    )
    if side == "read" or length != 4:
        np.testing.assert_array_equal(destination.data, [0xA5] * 8)


@pytest.mark.parametrize("side", ["read", "write"])
def test_io_rejects_accessor_that_skips_callback(side, monkeypatch):
    source, destination = Memory(range(8)), Memory([0xA5] * 8)
    if side == "read":
        monkeypatch.setattr(source, "read_to_buffer", lambda rng, callback: gb.Ok(None))
    else:
        monkeypatch.setattr(
            destination, "write_from_buffer", lambda rng, callback: gb.Ok(None)
        )
    result = rank_reshard(0, compute_view(source), compute_view(destination))
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError
    np.testing.assert_array_equal(destination.data, [0xA5] * 8)


def test_load_preserves_copy_error_when_finish_also_fails():
    error = gb.GDTensorIOError(operation="read", msg="read failed")
    closing = gb.GDTensorIOError(operation="finish", msg="finish failed")
    events = []
    result = gio.rank_load(
        0,
        compute_view(Memory([0] * 8)),
        storage_view(Memory(range(8), error), events, finish_error=closing),
    )
    assert type(result) is gb.Err and result.value is error
    assert events == ["begin_load", "init_load", "finish"]


@pytest.mark.parametrize("operation", ["load", "reshard"])
def test_io_reports_finish_error(operation):
    error = gb.GDTensorIOError(operation="finish", msg="finish failed")
    events = []
    src = storage_view(Memory(range(8)), events, finish_error=error)
    dst = compute_view(Memory([0] * 8))
    result = (
        gio.rank_load(0, dst, src) if operation == "load" else rank_reshard(0, src, dst)
    )
    assert type(result) is gb.Err and result.value is error
    assert events == ["begin_load", "init_load", "finish"]


def test_reshard_finishes_source_on_unknown_commit():
    error = gb.GDTensorIOError(operation="commit", msg="commit outcome unknown")
    events = []
    result = rank_reshard(
        0,
        storage_view(Memory(range(8)), events),
        storage_view(Memory([0] * 8), events, commit_error=error),
    )
    assert type(result) is gb.Err and result.value is error
    assert events == [
        "begin_load",
        "init_load",
        "begin_save",
        "init_save",
        "commit",
        "finish",
    ]


@pytest.mark.parametrize("operation", ["save", "load", "reshard"])
@pytest.mark.parametrize("mismatch", ["ids", "shape", "dtype", "itemsize"])
def test_io_rejects_schema_mismatch_before_opening_session(operation, mismatch):
    events = []
    cv = compute_view(Memory(range(8)))
    sv = storage_view(Memory(range(8)), events)
    if mismatch == "ids":
        cv.tensor_ids = ["different"]
    else:
        meta = cv.d_tensor_table.named_d_tensors[0].d_tensor.global_tensor_meta
        if mismatch == "shape":
            meta.global_shape = [2, 4]
        elif mismatch == "dtype":
            meta.dtype = "I8"
        else:
            meta.itemsize = 2
    if operation == "save":
        result = gio.rank_save(0, cv, sv)
    elif operation == "load":
        result = gio.rank_load(0, cv, sv)
    else:
        result = rank_reshard(0, sv, cv)
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError, (
        result
    )
    assert not events


class ChunkedMemory(Memory):
    def __init__(self, data, chunks):
        super().__init__(data)
        self.chunks = chunks

    def _call(self, rng, callback):
        offset = (
            rng.global_tensor_offset
            if hasattr(rng, "global_tensor_offset")
            else rng.contiguous_payload_offset
        )
        return _apply_chunks(self.data, offset, self.chunks, callback)


def _apply_chunks(data, offset, chunks, callback):
    cursor = offset
    for size in chunks:
        result = callback(data[cursor : cursor + size].ctypes.data, size)
        if type(result) is gb.Err:
            return result
        cursor += size
    return gb.Ok(None)


@pytest.mark.parametrize("read_chunks", [[8], [3, 5], [0, 1, 0, 7]])
@pytest.mark.parametrize("write_chunks", [[8], [2, 3, 3], [0, 4, 0, 4]])
@pytest.mark.parametrize("direction", ["save", "load", "cc", "cs", "sc", "ss"])
def test_chunked_callbacks_use_independent_read_and_write_boundaries(
    direction, read_chunks, write_chunks
):
    source = ChunkedMemory(range(8), read_chunks)
    dest = ChunkedMemory([0xA5] * 8, write_chunks)
    events = []
    src = (
        storage_view(source, events)
        if direction in ("load", "sc", "ss")
        else compute_view(source)
    )
    dst = (
        storage_view(dest, events)
        if direction in ("save", "cs", "ss")
        else compute_view(dest)
    )
    if direction == "save":
        result = gio.rank_save(0, src, dst)
    elif direction == "load":
        result = gio.rank_load(0, dst, src)
    else:
        result = rank_reshard(0, src, dst)
    assert type(result) is gb.Ok, result
    np.testing.assert_array_equal(dest.data, source.data)


@pytest.mark.parametrize("side", ["read", "write"])
@pytest.mark.parametrize("chunks", [[3, 4], [4, 5], [-1], []])
def test_chunked_callbacks_reject_invalid_total(side, chunks):
    source = ChunkedMemory(range(8), chunks if side == "read" else [8])
    dest = ChunkedMemory([0xA5] * 8, chunks if side == "write" else [8])
    result = rank_reshard(0, compute_view(source), compute_view(dest))
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError, (
        result
    )
    if side == "read":
        np.testing.assert_array_equal(dest.data, [0xA5] * 8)


@pytest.mark.parametrize("operation", [gio.rank_save, gio.rank_load])
@pytest.mark.parametrize("types", ["swapped", "both_compute", "both_storage"])
def test_load_and_save_reject_wrong_view_domains(operation, types):
    events = []
    compute, storage = (
        compute_view(Memory(range(8))),
        storage_view(Memory(range(8)), events),
    )
    args = {
        "swapped": (storage, compute),
        "both_compute": (compute, compute),
        "both_storage": (storage, storage),
    }[types]
    result = operation(0, *args)
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError
    assert not events


@pytest.mark.parametrize("side", ["source", "destination"])
def test_reshard_rejects_unknown_view_type_before_opening_sessions(side):
    events = []
    storage = storage_view(Memory(range(8)), events)
    args = (object(), storage) if side == "source" else (storage, object())
    result = rank_reshard(0, *args)
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError
    assert not events


@pytest.mark.parametrize("operation", ["save", "load", "reshard"])
def test_io_keeps_the_primary_error_when_terminal_cleanup_raises(operation):
    events = []
    error = gb.GDTensorIOError(operation="copy", msg="primary read error")
    source = storage_view(Memory(range(8), error), events)
    target = storage_view(Memory([0] * 8), events)
    begin_save, begin_load = target.begin_save, source.begin_load
    assert begin_save is not None and begin_load is not None

    def failing_abort(view, rank):
        session = begin_save(view, rank).value
        session.abort = cleanup
        return gb.Ok(session)

    def failing_finish(view, rank):
        session = begin_load(view, rank).value
        session.finish = cleanup
        return gb.Ok(session)

    def cleanup():
        events.append("cleanup_raises")
        raise RuntimeError("secondary cleanup error")

    target.begin_save = failing_abort
    source.begin_load = failing_finish
    if operation == "save":
        result = gio.rank_save(0, compute_view(Memory(range(8), error)), target)
    elif operation == "load":
        result = gio.rank_load(0, compute_view(Memory([0] * 8)), source)
    else:
        result = rank_reshard(0, source, target)
    assert type(result) is gb.Err and result.value is error
    assert events.count("cleanup_raises") == (2 if operation == "reshard" else 1)


@pytest.mark.parametrize("partial", [False, True])
def test_save_rejects_nonempty_tensor_without_complete_physical_storage(partial):
    events = []
    view = storage_view(Memory([0] * 8), events)
    if partial:
        dt = view.d_tensor_table.named_d_tensors[0].d_tensor
        dt.mesh = gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
        dt.placements = [gb.Replicate(), gb.Shard(0)]
        base = gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
        group = gb.DisjointStorageMeshList(
            [gb.SlicedMesh(base, [None, gb.Slice(0, 1)])],
        )
        view.physical_layout.tensor_entries = {gb.TensorComponentPos("w", 0): group}
    else:
        view.physical_layout.tensor_entries.clear()
        view.physical_layout.file_paths.clear()
    result = gio.rank_save(0, compute_view(Memory(range(8))), view)
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError
    assert not events


def test_save_does_not_infer_a_file_for_a_missing_component_entry():
    events = []
    view = storage_view(Memory([0] * 8), events)
    dt = view.d_tensor_table.named_d_tensors[0].d_tensor
    dt.mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    dt.placements = [gb.Replicate(), gb.Shard(0)]
    group = next(iter(view.physical_layout.tensor_entries.values()))
    group.disjoint_storage_mesh_list = [
        gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
    ]
    missing = gb.TensorComponentPos("w", 1)
    assert type(gb.file_path_of(view.physical_layout, missing)) is gb.Err
    result = gio.rank_save(0, compute_view(Memory(range(8))), view)
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError
    assert result.value.argument_name == "tensor_entries"
    assert not events

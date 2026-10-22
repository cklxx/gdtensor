"""Public sessions and accessors need no internal index or bind methods."""

import numpy as np
import pytest

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.session as gs
import gdtensor.tag as gt


class Memory(ga.DTensorDataAccessor):
    def __init__(self, data, error=None, length=None):
        self.data = np.array(data, dtype=np.uint8)
        self.error, self.length = error, length

    def _call(self, rng, callback):
        if self.error is not None:
            return gt.Err(self.error)
        offset = (
            rng.global_tensor_offset
            if hasattr(rng, "global_tensor_offset")
            else rng.contiguous_payload_offset
        )
        data = self.data[offset : offset + rng.size]
        return callback(
            data.ctypes.data, data.nbytes if self.length is None else self.length
        )

    def read_to_buffer(self, global_range, read_to_buffer):
        return self._call(global_range, read_to_buffer)

    def write_from_buffer(self, global_range, write_from_buffer):
        return self._call(global_range, write_from_buffer)


class Save(gs.StorageSaveSession):
    def __init__(self, memory, events, init_error=None, commit_error=None):
        self.memory, self.events = memory, events
        self.init_error, self.commit_error = init_error, commit_error

    def get_file_offset(self, pos):
        return gt.Ok(0)

    def init_data_accessor(self):
        self.events.append("init_save")
        return gt.Err(self.init_error) if self.init_error else gt.Ok(self.memory)

    def commit(self):
        self.events.append("commit")
        return gt.Err(self.commit_error) if self.commit_error else gt.Ok(None)

    def abort(self):
        self.events.append("abort")
        return gt.Ok(None)


class Load(gs.StorageLoadSession):
    def __init__(self, memory, events, init_error=None, finish_error=None):
        self.memory, self.events = memory, events
        self.init_error, self.finish_error = init_error, finish_error

    def get_file_offset(self, pos):
        return gt.Ok(0)

    def init_data_accessor(self):
        self.events.append("init_load")
        return gt.Err(self.init_error) if self.init_error else gt.Ok(self.memory)

    def finish(self):
        self.events.append("finish")
        return gt.Err(self.finish_error) if self.finish_error else gt.Ok(None)


def _compute(memory):
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([8], "U8", 1),
        [gt.Replicate()],
        gb.ArrangedMesh(gt.WorldSize(1)),
    )
    return gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), memory
    )


def _storage(memory, events, *, init_error=None, commit_error=None, finish_error=None):
    mesh = gb.OuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    dt = gb.NaiveDTensor(gb.GlobalTensorMeta([8], "U8", 1), [gt.Replicate()] * 2, mesh)
    group = gb.DisjointStorageMeshList(
        [
            gb.OuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(1))
            )
        ],
        ["w"],
    )
    pos = gb.TensorComponentPos("w", 0)
    layout = gb.PhysicalLayoutTable("safetensors", {pos: group}, {pos: "memory"})

    def begin_save(view, rank):
        events.append("begin_save")
        return gt.Ok(Save(memory, events, init_error, commit_error))

    def begin_load(view, rank):
        events.append("begin_load")
        return gt.Ok(Load(memory, events, init_error, finish_error))

    # Executors must use the accessor returned by the session.
    decoy = Memory([0] * 8, gb.GDTensorIOError(operation="decoy", msg="wrong accessor"))
    return gb.StorageDTensorsView(
        ["w"],
        gb.DTensorTable([gb.NamedDTensor("w", dt)]),
        decoy,
        layout,
        begin_save,
        begin_load,
    )


@pytest.mark.parametrize("direction", ["save", "load", "cc", "cs", "sc", "ss"])
def test_custom_sessions_and_accessors(direction):
    events = []
    source, destination = Memory(range(8)), Memory([0xA5] * 8)
    src = (
        _storage(source, events)
        if direction in ("load", "sc", "ss")
        else _compute(source)
    )
    dst = (
        _storage(destination, events)
        if direction in ("save", "cs", "ss")
        else _compute(destination)
    )
    if direction == "save":
        result = gio.rank_save(0, src, dst)
    elif direction == "load":
        result = gio.rank_load(0, dst, src)
    else:
        result = gio.rank_reshard(0, src, dst)
    assert type(result) is gt.Ok, result
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
            _compute(source),
            _storage(dest, events, init_error=error if failure == "init" else None),
        )
        expected = ["begin_save", "init_save", "abort"]
    elif operation == "load":
        result = gio.rank_load(
            0,
            _compute(dest),
            _storage(source, events, init_error=error if failure == "init" else None),
        )
        expected = ["begin_load", "init_load", "finish"]
    else:
        result = gio.rank_reshard(
            0,
            _storage(source, events),
            _storage(dest, events, init_error=error if failure == "init" else None),
        )
        expected = [
            "begin_load",
            "init_load",
            "begin_save",
            "init_save",
            "abort",
            "finish",
        ]
    assert type(result) is gt.Err and result.value is error, result
    assert events == expected
    np.testing.assert_array_equal(dest.data, [0xA5] * 8)


@pytest.mark.parametrize("side", ["read", "write"])
@pytest.mark.parametrize("length", [0, 4, 12])
def test_io_rejects_wrong_buffer_size(side, length):
    source = Memory(range(8), length=length if side == "read" else None)
    destination = Memory([0xA5] * 8, length=length if side == "write" else None)
    result = gio.rank_reshard(0, _compute(source), _compute(destination))
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError, (
        result
    )
    if side == "read" or length != 4:
        np.testing.assert_array_equal(destination.data, [0xA5] * 8)


@pytest.mark.parametrize("side", ["read", "write"])
def test_io_rejects_accessor_that_skips_callback(side, monkeypatch):
    source, destination = Memory(range(8)), Memory([0xA5] * 8)
    if side == "read":
        monkeypatch.setattr(source, "read_to_buffer", lambda rng, callback: gt.Ok(None))
    else:
        monkeypatch.setattr(
            destination, "write_from_buffer", lambda rng, callback: gt.Ok(None)
        )
    result = gio.rank_reshard(0, _compute(source), _compute(destination))
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError
    np.testing.assert_array_equal(destination.data, [0xA5] * 8)


def test_load_preserves_copy_error_when_finish_also_fails():
    error = gb.GDTensorIOError(operation="read", msg="read failed")
    closing = gb.GDTensorIOError(operation="finish", msg="finish failed")
    events = []
    result = gio.rank_load(
        0,
        _compute(Memory([0] * 8)),
        _storage(Memory(range(8), error), events, finish_error=closing),
    )
    assert type(result) is gt.Err and result.value is error
    assert events == ["begin_load", "init_load", "finish"]


@pytest.mark.parametrize("operation", ["load", "reshard"])
def test_io_reports_finish_error(operation):
    error = gb.GDTensorIOError(operation="finish", msg="finish failed")
    events = []
    src = _storage(Memory(range(8)), events, finish_error=error)
    dst = _compute(Memory([0] * 8))
    result = (
        gio.rank_load(0, dst, src)
        if operation == "load"
        else gio.rank_reshard(0, src, dst)
    )
    assert type(result) is gt.Err and result.value is error
    assert events == ["begin_load", "init_load", "finish"]


def test_reshard_finishes_source_on_unknown_commit():
    error = gb.GDTensorIOError(operation="commit", msg="commit outcome unknown")
    events = []
    result = gio.rank_reshard(
        0,
        _storage(Memory(range(8)), events),
        _storage(Memory([0] * 8), events, commit_error=error),
    )
    assert type(result) is gt.Err and result.value is error
    assert events == [
        "begin_load",
        "init_load",
        "begin_save",
        "init_save",
        "commit",
        "finish",
    ]


@pytest.mark.parametrize("convert_tensors", [False, True])
def test_view_map_converts_sessions_when_opened(convert_tensors):
    events = []
    memory = Memory(range(8))
    origin = _storage(memory, events)

    def convert_save(session):
        events.append("convert_save")
        return gt.Ok(session)

    def convert_load(session):
        events.append("convert_load")
        return gt.Ok(session)

    result = gb.d_tensors_view_map(
        origin,
        gb.DTensorsViewMapCore(
            convert_each_storage_view=(lambda t: gt.Ok(t)) if convert_tensors else None,
            convert_each_storage_save_session=convert_save,
            convert_each_storage_load_session=convert_load,
        ),
    )
    assert type(result) is gt.Ok and not events
    mapped = result.value
    assert type(gio.rank_save(0, _compute(Memory(range(8))), mapped)) is gt.Ok
    assert type(gio.rank_load(0, _compute(Memory([0] * 8)), mapped)) is gt.Ok
    assert events == [
        "begin_save",
        "convert_save",
        "init_save",
        "commit",
        "begin_load",
        "convert_load",
        "init_load",
        "finish",
    ]


def test_view_map_preserves_begin_error_without_calling_converter():
    events = []
    error = gb.GDTensorIOError(operation="open", msg="open failed")
    origin = _storage(Memory(range(8)), events)
    origin.begin_load = lambda view, rank: gt.Err(error)
    mapped = gb.d_tensors_view_map(
        origin,
        gb.DTensorsViewMapCore(
            convert_each_storage_load_session=lambda session: events.append("convert")
        ),
    ).value
    result = gio.rank_load(0, _compute(Memory([0] * 8)), mapped)
    assert type(result) is gt.Err and result.value is error
    assert not events


@pytest.mark.parametrize("operation", ["save", "load"])
def test_view_map_closes_session_on_conversion_error(operation):
    events = []
    error = gb.GDTensorIOError(operation="convert", msg="conversion failed")
    origin = _storage(Memory(range(8)), events)
    core = gb.DTensorsViewMapCore(
        **{
            "convert_each_storage_" + operation + "_session": lambda session: gt.Err(
                error
            )
        }
    )
    mapped = gb.d_tensors_view_map(origin, core).value
    cv = _compute(Memory(range(8)))
    result = (
        gio.rank_save(0, cv, mapped)
        if operation == "save"
        else gio.rank_load(0, cv, mapped)
    )
    assert type(result) is gt.Err and result.value is error
    assert events == [
        "begin_" + operation,
        "abort" if operation == "save" else "finish",
    ]


@pytest.mark.parametrize("operation", ["save", "load", "reshard"])
@pytest.mark.parametrize("mismatch", ["ids", "shape", "dtype", "itemsize"])
def test_io_rejects_schema_mismatch_before_opening_session(operation, mismatch):
    events = []
    cv = _compute(Memory(range(8)))
    sv = _storage(Memory(range(8)), events)
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
        result = gio.rank_reshard(0, sv, cv)
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError, (
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
        if type(result) is gt.Err:
            return result
        cursor += size
    return gt.Ok(None)


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
        _storage(source, events)
        if direction in ("load", "sc", "ss")
        else _compute(source)
    )
    dst = (
        _storage(dest, events) if direction in ("save", "cs", "ss") else _compute(dest)
    )
    if direction == "save":
        result = gio.rank_save(0, src, dst)
    elif direction == "load":
        result = gio.rank_load(0, dst, src)
    else:
        result = gio.rank_reshard(0, src, dst)
    assert type(result) is gt.Ok, result
    np.testing.assert_array_equal(dest.data, source.data)


@pytest.mark.parametrize("side", ["read", "write"])
@pytest.mark.parametrize("chunks", [[3, 4], [4, 5], [-1], []])
def test_chunked_callbacks_reject_invalid_total(side, chunks):
    source = ChunkedMemory(range(8), chunks if side == "read" else [8])
    dest = ChunkedMemory([0xA5] * 8, chunks if side == "write" else [8])
    result = gio.rank_reshard(0, _compute(source), _compute(dest))
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError, (
        result
    )
    if side == "read":
        np.testing.assert_array_equal(dest.data, [0xA5] * 8)


@pytest.mark.parametrize("operation", [gio.rank_save, gio.rank_load])
@pytest.mark.parametrize("types", ["swapped", "both_compute", "both_storage"])
def test_load_and_save_reject_wrong_view_domains(operation, types):
    events = []
    compute, storage = _compute(Memory(range(8))), _storage(Memory(range(8)), events)
    args = {
        "swapped": (storage, compute),
        "both_compute": (compute, compute),
        "both_storage": (storage, storage),
    }[types]
    result = operation(0, *args)
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError
    assert not events


@pytest.mark.parametrize("side", ["source", "destination"])
def test_reshard_rejects_unknown_view_type_before_opening_sessions(side):
    events = []
    storage = _storage(Memory(range(8)), events)
    args = (object(), storage) if side == "source" else (storage, object())
    result = gio.rank_reshard(0, *args)
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError
    assert not events


@pytest.mark.parametrize("operation", ["save", "load"])
def test_view_map_uses_the_replacement_session(operation):
    events = []
    original_memory, replacement_memory = Memory([91] * 8), Memory(range(8))
    origin = _storage(original_memory, events)
    replacements = []

    def convert(session):
        events.append("convert_" + operation)
        replacement = (
            Save(replacement_memory, events)
            if operation == "save"
            else Load(replacement_memory, events)
        )
        replacements.append(replacement)
        return gt.Ok(replacement)

    core = gb.DTensorsViewMapCore(
        **{"convert_each_storage_" + operation + "_session": convert}
    )
    mapped = gb.d_tensors_view_map(origin, core).value
    assert not events and not replacements
    compute_memory = Memory(range(8)) if operation == "save" else Memory([0] * 8)
    result = getattr(gio, "rank_" + operation)(0, _compute(compute_memory), mapped)
    assert type(result) is gt.Ok and len(replacements) == 1
    assert events == [
        "begin_" + operation,
        "convert_" + operation,
        "init_" + operation,
        "commit" if operation == "save" else "finish",
    ]
    np.testing.assert_array_equal(original_memory.data, [91] * 8)
    np.testing.assert_array_equal(compute_memory.data, range(8))
    np.testing.assert_array_equal(replacement_memory.data, range(8))


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("failure", ["err", "exception", "invalid_result"])
@pytest.mark.parametrize("cleanup_failure", ["err", "exception"])
def test_view_map_conversion_failure_survives_cleanup_failure(
    operation, failure, cleanup_failure
):
    events = []
    error = gb.GDTensorIOError(operation="convert", msg="primary conversion error")
    cleanup_error = gb.GDTensorIOError(operation="cleanup", msg="secondary error")
    origin = _storage(Memory(range(8)), events)
    close = "abort" if operation == "save" else "finish"
    session = (
        Save(Memory(range(8)), events)
        if operation == "save"
        else Load(Memory(range(8)), events)
    )

    def cleanup():
        events.append(close)
        if cleanup_failure == "exception":
            raise RuntimeError("secondary error")
        return gt.Err(cleanup_error)

    def convert(opened):
        assert opened is session
        if failure == "exception":
            raise RuntimeError("primary conversion error")
        return gt.Err(error) if failure == "err" else None

    setattr(session, close, cleanup)
    setattr(origin, "begin_" + operation, lambda view, rank: gt.Ok(session))
    core = gb.DTensorsViewMapCore(
        **{"convert_each_storage_" + operation + "_session": convert}
    )
    mapped = gb.d_tensors_view_map(origin, core).value
    assert not events
    result = getattr(gio, "rank_" + operation)(0, _compute(Memory(range(8))), mapped)
    assert type(result) is gt.Err and result.value is not cleanup_error
    if failure == "err":
        assert result.value is error
    elif failure == "exception":
        assert result.value.msg == "RuntimeError: primary conversion error"
    else:
        assert type(result.value) is gb.InvalidArgumentError
    assert events == [close]


@pytest.mark.parametrize("operation", ["save", "load"])
def test_view_map_without_callbacks_preserves_factories(operation):
    origin = _storage(Memory(range(8)), [])
    mapped = gb.d_tensors_view_map(origin, gb.DTensorsViewMapCore()).value
    assert mapped is origin
    assert getattr(mapped, "begin_" + operation) is getattr(
        origin, "begin_" + operation
    )


def test_view_map_stops_at_first_tensor_error_and_preserves_ids_and_aliases():
    origin = _compute(Memory(range(8)))
    first = origin.d_tensor_table.named_d_tensors[0]
    first.alias = ["weight"]
    second = gb.NamedDTensor("second", first.d_tensor, ["other"])
    origin.d_tensor_table.named_d_tensors.append(second)
    origin.tensor_ids.append("second")
    calls = []
    error = gb.InvalidArgumentError("convert", "stop")

    def convert(dt):
        calls.append(dt)
        return gt.Err(error)

    core = gb.DTensorsViewMapCore(convert_each_compute_view=convert)
    result = gb.d_tensors_view_map(origin, core)
    assert result == gt.Err(error) and len(calls) == 1
    core.convert_each_compute_view = lambda dt: gt.Ok(dt)
    mapped = gb.d_tensors_view_map(origin, core).value
    assert mapped.tensor_ids == ["w", "second"]
    assert [nd.alias for nd in mapped.d_tensor_table.named_d_tensors] == [
        ["weight"],
        ["other"],
    ]
    assert mapped.data_accessor is origin.data_accessor


@pytest.mark.parametrize("operation", ["save", "load", "reshard"])
def test_io_keeps_the_primary_error_when_terminal_cleanup_raises(operation):
    events = []
    error = gb.GDTensorIOError(operation="copy", msg="primary read error")
    source = _storage(Memory(range(8), error), events)
    target = _storage(Memory([0] * 8), events)
    begin_save, begin_load = target.begin_save, source.begin_load
    assert begin_save is not None and begin_load is not None

    def failing_abort(view, rank):
        session = begin_save(view, rank).value
        session.abort = cleanup
        return gt.Ok(session)

    def failing_finish(view, rank):
        session = begin_load(view, rank).value
        session.finish = cleanup
        return gt.Ok(session)

    def cleanup():
        events.append("cleanup_raises")
        raise RuntimeError("secondary cleanup error")

    target.begin_save = failing_abort
    source.begin_load = failing_finish
    if operation == "save":
        result = gio.rank_save(0, _compute(Memory(range(8), error)), target)
    elif operation == "load":
        result = gio.rank_load(0, _compute(Memory([0] * 8)), source)
    else:
        result = gio.rank_reshard(0, source, target)
    assert type(result) is gt.Err and result.value is error
    assert events.count("cleanup_raises") == (2 if operation == "reshard" else 1)


@pytest.mark.parametrize("partial", [False, True])
def test_save_rejects_nonempty_tensor_without_complete_physical_storage(partial):
    events = []
    view = _storage(Memory([0] * 8), events)
    if partial:
        dt = view.d_tensor_table.named_d_tensors[0].d_tensor
        dt.mesh = gb.OuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
        dt.placements = [gt.Replicate(), gt.Shard(0)]
        base = gb.OuterProductMesh(
            gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
        group = gb.DisjointStorageMeshList(
            [gb.SlicedMesh(base, [None, gb.Slice(0, 1)])], ["w"]
        )
        view.physical_layout.tensor_entries = {gb.TensorComponentPos("w", 0): group}
    else:
        view.physical_layout.tensor_entries.clear()
        view.physical_layout.file_paths.clear()
    result = gio.rank_save(0, _compute(Memory(range(8))), view)
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError
    assert not events


def test_save_does_not_infer_a_file_for_a_missing_component_entry():
    events = []
    view = _storage(Memory([0] * 8), events)
    dt = view.d_tensor_table.named_d_tensors[0].d_tensor
    dt.mesh = gb.OuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    dt.placements = [gt.Replicate(), gt.Shard(0)]
    group = next(iter(view.physical_layout.tensor_entries.values()))
    group.disjoint_storage_mesh_list = [
        gb.OuterProductMesh(
            gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
    ]
    missing = gb.TensorComponentPos("w", 1)
    assert type(gb.file_path_of(view.physical_layout, missing)) is gt.Err
    result = gio.rank_save(0, _compute(Memory(range(8))), view)
    assert type(result) is gt.Err and type(result.value) is gb.InvalidArgumentError
    assert result.value.argument_name == "tensor_entries"
    assert not events

"""Builder view mapping and deferred session conversion through public APIs."""

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.memory import Load, Memory, Save, compute_view, storage_view


@pytest.mark.parametrize("branch", ["compute", "storage"])
def test_view_map_selects_its_domain_and_keeps_origin_unchanged(branch):
    events, calls = [], []
    origin = (
        compute_view(Memory(range(8)))
        if branch == "compute"
        else storage_view(Memory(range(8)), events)
    )
    original = origin.d_tensor_table.named_d_tensors[0].d_tensor
    replacement = gb.NaiveDTensor(
        original.global_tensor_meta, original.placements, original.mesh
    )

    def convert(tensor):
        calls.append(tensor)
        return gb.Ok(replacement)

    def wrong_domain(tensor):
        pytest.fail("view map invoked the other domain's converter")

    core = gb.DTensorsViewMapCore(
        convert_each_compute_view=convert if branch == "compute" else wrong_domain,
        convert_each_storage_view=convert if branch == "storage" else wrong_domain,
    )
    result = gb.d_tensors_view_map(origin, core)
    assert type(result) is gb.Ok and calls == [original] and not events
    mapped = result.value
    assert mapped is not None
    assert mapped.d_tensor_table.named_d_tensors[0].d_tensor is replacement
    assert origin.d_tensor_table.named_d_tensors[0].d_tensor is original
    assert mapped.tensor_ids == origin.tensor_ids
    assert mapped.data_accessor is origin.data_accessor
    if branch == "storage":
        assert mapped.physical_layout is origin.physical_layout
        assert mapped.begin_save is origin.begin_save
        assert mapped.begin_load is origin.begin_load


@pytest.mark.parametrize("convert_tensors", [False, True])
def test_view_map_converts_sessions_when_opened(convert_tensors):
    events = []
    memory = Memory(range(8))
    origin = storage_view(memory, events)

    def convert_save(session):
        events.append("convert_save")
        return gb.Ok(session)

    def convert_load(session):
        events.append("convert_load")
        return gb.Ok(session)

    result = gb.d_tensors_view_map(
        origin,
        gb.DTensorsViewMapCore(
            convert_each_storage_view=(lambda t: gb.Ok(t)) if convert_tensors else None,
            convert_each_storage_save_session=convert_save,
            convert_each_storage_load_session=convert_load,
        ),
    )
    assert type(result) is gb.Ok and not events
    mapped = result.value
    assert type(gio.rank_save(0, compute_view(Memory(range(8))), mapped)) is gb.Ok
    assert type(gio.rank_load(0, compute_view(Memory([0] * 8)), mapped)) is gb.Ok
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
    origin = storage_view(Memory(range(8)), events)
    origin.begin_load = lambda view, rank: gb.Err(error)
    mapped = gb.d_tensors_view_map(
        origin,
        gb.DTensorsViewMapCore(
            convert_each_storage_load_session=lambda session: events.append("convert")
        ),
    ).value
    result = gio.rank_load(0, compute_view(Memory([0] * 8)), mapped)
    assert type(result) is gb.Err and result.value is error
    assert not events


@pytest.mark.parametrize("operation", ["save", "load"])
def test_view_map_closes_session_on_conversion_error(operation):
    events = []
    error = gb.GDTensorIOError(operation="convert", msg="conversion failed")
    origin = storage_view(Memory(range(8)), events)
    core = gb.DTensorsViewMapCore(
        **{
            "convert_each_storage_" + operation + "_session": lambda session: gb.Err(
                error
            )
        }
    )
    mapped = gb.d_tensors_view_map(origin, core).value
    cv = compute_view(Memory(range(8)))
    result = (
        gio.rank_save(0, cv, mapped)
        if operation == "save"
        else gio.rank_load(0, cv, mapped)
    )
    assert type(result) is gb.Err and result.value is error
    assert events == [
        "begin_" + operation,
        "abort" if operation == "save" else "finish",
    ]


@pytest.mark.parametrize("operation", ["save", "load"])
def test_view_map_uses_the_replacement_session(operation):
    events = []
    original_memory, replacement_memory = Memory([91] * 8), Memory(range(8))
    origin = storage_view(original_memory, events)
    replacements = []

    def convert(session):
        events.append("convert_" + operation)
        replacement = (
            Save(replacement_memory, events)
            if operation == "save"
            else Load(replacement_memory, events)
        )
        replacements.append(replacement)
        return gb.Ok(replacement)

    core = gb.DTensorsViewMapCore(
        **{"convert_each_storage_" + operation + "_session": convert}
    )
    mapped = gb.d_tensors_view_map(origin, core).value
    assert not events and not replacements
    compute_memory = Memory(range(8)) if operation == "save" else Memory([0] * 8)
    result = getattr(gio, "rank_" + operation)(0, compute_view(compute_memory), mapped)
    assert type(result) is gb.Ok and len(replacements) == 1
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
    origin = storage_view(Memory(range(8)), events)
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
        return gb.Err(cleanup_error)

    def convert(opened):
        assert opened is session
        if failure == "exception":
            raise RuntimeError("primary conversion error")
        return gb.Err(error) if failure == "err" else None

    setattr(session, close, cleanup)
    setattr(origin, "begin_" + operation, lambda view, rank: gb.Ok(session))
    core = gb.DTensorsViewMapCore(
        **{"convert_each_storage_" + operation + "_session": convert}
    )
    mapped = gb.d_tensors_view_map(origin, core).value
    assert not events
    result = getattr(gio, "rank_" + operation)(
        0, compute_view(Memory(range(8))), mapped
    )
    assert type(result) is gb.Err and result.value is not cleanup_error
    if failure == "err":
        assert result.value is error
    elif failure == "exception":
        assert result.value.msg == "RuntimeError: primary conversion error"
    else:
        assert type(result.value) is gb.InvalidArgumentError
    assert events == [close]


@pytest.mark.parametrize("operation", ["save", "load"])
def test_view_map_without_callbacks_preserves_factories(operation):
    origin = storage_view(Memory(range(8)), [])
    mapped = gb.d_tensors_view_map(origin, gb.DTensorsViewMapCore()).value
    assert mapped is origin
    assert getattr(mapped, "begin_" + operation) is getattr(
        origin, "begin_" + operation
    )


def test_view_map_stops_at_first_tensor_error_and_preserves_ids_and_aliases():
    origin = compute_view(Memory(range(8)))
    first = origin.d_tensor_table.named_d_tensors[0]
    first.alias = ["weight"]
    second = gb.NamedDTensor("second", first.d_tensor, ["other"])
    origin.d_tensor_table.named_d_tensors.append(second)
    origin.tensor_ids.append("second")
    calls = []
    error = gb.InvalidArgumentError("convert", "stop")

    def convert(dt):
        calls.append(dt)
        return gb.Err(error)

    core = gb.DTensorsViewMapCore(convert_each_compute_view=convert)
    result = gb.d_tensors_view_map(origin, core)
    assert result == gb.Err(error) and len(calls) == 1
    core.convert_each_compute_view = lambda dt: gb.Ok(dt)
    mapped = gb.d_tensors_view_map(origin, core).value
    assert mapped is not None
    assert mapped.tensor_ids == ["w", "second"]
    assert [nd.alias for nd in mapped.d_tensor_table.named_d_tensors] == [
        ["weight"],
        ["other"],
    ]
    assert mapped.data_accessor is origin.data_accessor


def test_view_map_rejects_unknown_origin():
    result = gb.d_tensors_view_map(object(), gb.DTensorsViewMapCore())
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError


def test_view_map_requires_result_callbacks():
    gd = gb.NaiveDTensor(
        gb.GlobalTensorMeta([2], "F32", 4),
        [gb.Replicate()],
        gb.ArrangedMesh(gt.WorldSize(1)),
    )
    view = gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", gd)]), None
    )
    result = gb.d_tensors_view_map(
        view, gb.DTensorsViewMapCore(convert_each_compute_view=lambda tensor: tensor)
    )
    assert type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError

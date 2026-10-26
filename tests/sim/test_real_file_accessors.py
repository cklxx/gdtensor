"""Real file accessors: callback IO, close/reopen, and corruption oracles."""

import ctypes
import json
import os
import struct
import zlib
from types import SimpleNamespace
from typing import TypeVar, cast

import numpy as np
import pytest
import safetensors.numpy

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor import Result
from gdtensor_sim.formats import make_storage_view
from gdtensor_sim.iobackend import FileIO
from tests.support.cache_oracle import storage_schema
from tests.support.oracle import ArrayFabric


def _fixture(root, fmt, size=17):
    st, layout = storage_schema(["w"], size)
    layout.container_format = fmt
    path = "model.safetensors" if fmt == "safetensors" else "__0_0.distcp"
    layout.set_file_paths({gb.TensorComponentPos("w", 0): path})
    ct = gb.DTensorTable(
        [
            gb.NamedDTensor(
                "w",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([size], "U8", 1),
                    [gb.Replicate()],
                    gb.ArrangedMesh(gt.WorldSize(1)),
                ),
            )
        ]
    )
    values = {"w": np.arange(size, dtype=np.uint8)}
    points = {0: {"w": np.arange(size)}}
    io = FileIO()
    sv = make_storage_view(["w"], st, layout, str(root), io, dcp_mode="process")
    return ct, st, layout, values, points, sv, io, path


def _take(dst):
    def read(ptr, size):
        dst.extend(ctypes.string_at(ptr, size))
        return gb.Ok(None)

    return read


def _give(data):
    def write(ptr, size):
        assert size == len(data)
        ctypes.memmove(ptr, data, size)
        return gb.Ok(None)

    return write


T = TypeVar("T")


def _ok(result: Result[T]) -> T:
    assert type(result) is gb.Ok, result
    # Ok also permits the None payload of commit/finish; T records the caller's
    # Result contract while the assertion checks the runtime variant.
    return cast(T, result.value)


def _begin(view, saving):
    if saving:
        assert view.begin_save is not None
        return _ok(view.begin_save(view, 0))
    assert view.begin_load is not None
    return _ok(view.begin_load(view, 0))


def test_accessor_is_abstract_and_file_session_implements_callbacks(tmp_path):
    assert ga.DTensorDataAccessor.__abstractmethods__ == {
        "read_to_buffer",
        "write_from_buffer",
    }
    ct, st, layout, values, points, sv, io, path = _fixture(tmp_path, "safetensors")
    session = _begin(sv, True)
    accessor = _ok(session.init_data_accessor())
    assert isinstance(accessor, ga.DTensorDataAccessor)
    rng = SimpleNamespace(file_path=path, contiguous_payload_offset=0, size=17)
    _ok(accessor.write_from_buffer(rng, _give(values["w"].tobytes())))
    _ok(session.commit())
    assert any(r.op == "pwrite" and r.size == 17 for r in io.tracer.records)
    reader = make_storage_view(["w"], st, layout, str(tmp_path), FileIO())
    session = _begin(reader, False)
    accessor = _ok(session.init_data_accessor())
    got = bytearray()
    _ok(accessor.read_to_buffer(rng, _take(got)))
    _ok(session.finish())
    assert bytes(got) == values["w"].tobytes()
    np.testing.assert_array_equal(
        safetensors.numpy.load_file(tmp_path / path)["w"], values["w"]
    )


def test_file_callback_error_does_not_write_payload(tmp_path):
    _, _, _, _, _, sv, io, path = _fixture(tmp_path, "safetensors")
    session = _begin(sv, True)
    accessor = _ok(session.init_data_accessor())
    rng = SimpleNamespace(file_path=path, contiguous_payload_offset=0, size=17)
    before = len(io.tracer.records)
    error = gb.GDTensorIOError(operation="callback", msg="intentional callback error")
    result = accessor.write_from_buffer(rng, lambda ptr, size: gb.Err(error))
    assert type(result) is gb.Err and result.value is error
    assert len(io.tracer.records) == before
    _ok(session.abort())


def test_reopened_file_short_read_returns_error_without_callback(tmp_path):
    ct, st, layout, values, points, sv, _, path = _fixture(tmp_path, "safetensors")
    _ok(gio.rank_save(0, ArrayFabric(ct, points, values).view(0), sv))
    full = tmp_path / path
    full.write_bytes(full.read_bytes()[:-1])
    reader = make_storage_view(["w"], st, layout, str(tmp_path), FileIO())
    session = _begin(reader, False)
    calls = []
    accessor = _ok(session.init_data_accessor())
    rng = SimpleNamespace(file_path=path, contiguous_payload_offset=0, size=17)
    result = accessor.read_to_buffer(rng, lambda ptr, size: calls.append(size))
    assert type(result) is gb.Err
    assert "short read" in result.value.msg
    assert not calls
    _ok(session.finish())


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_public_io_persists_reopens_and_independently_decodes_files(tmp_path, fmt):
    ct, st, layout, values, points, sv, _, path = _fixture(tmp_path, fmt)
    source = ArrayFabric(ct, points, values)
    _ok(gio.rank_save(0, source.view(0), sv))
    files = sorted(p for p in tmp_path.iterdir() if p.is_file())
    assert files and not list(tmp_path.glob("*.tmp"))
    decoded = _decode(tmp_path, fmt)
    want = values["w"].tobytes()
    assert decoded == want
    assert zlib.crc32(decoded) == zlib.crc32(want)
    reader = make_storage_view(["w"], st, layout, str(tmp_path), FileIO())
    loaded = ArrayFabric(ct, points)
    _ok(gio.rank_load(0, loaded.view(0), reader))
    loaded.assert_values(values)
    print(
        f"[REAL_FILE] format={fmt} path={tmp_path} files={[p.name for p in files]} checks=close_reopen,crc,arange PASS",
        flush=True,
    )


def _decode(root, fmt):
    if fmt == "safetensors":
        return safetensors.numpy.load_file(root / "model.safetensors")["w"].tobytes()
    import torch
    from torch.distributed.checkpoint import load

    target = {"w": torch.empty(17, dtype=torch.uint8)}
    load(target, checkpoint_id=root, no_dist=True)
    return target["w"].numpy().tobytes()


def test_file_corruption_is_detected_by_crc_and_arange(tmp_path):
    ct, st, layout, values, points, sv, _, path = _fixture(tmp_path, "safetensors")
    _ok(gio.rank_save(0, ArrayFabric(ct, points, values).view(0), sv))
    full = tmp_path / path
    with full.open("r+b") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        offset = 8 + n + header["w"]["data_offsets"][0]
        f.seek(offset)
        f.write(b"\xff")
        f.flush()
        os.fsync(f.fileno())
    decoded = _decode(tmp_path, "safetensors")
    assert zlib.crc32(decoded) != zlib.crc32(values["w"].tobytes())
    reader = make_storage_view(["w"], st, layout, str(tmp_path), FileIO())
    loaded = ArrayFabric(ct, points)
    _ok(gio.rank_load(0, loaded.view(0), reader))
    with pytest.raises(AssertionError):
        loaded.assert_values(values)

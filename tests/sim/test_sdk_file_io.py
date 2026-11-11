"""Safetensors filesystem SDK integration with an in-memory client."""

from __future__ import annotations

import ctypes
import threading
from types import SimpleNamespace
from typing import TypeVar, cast

import numpy as np
import pytest
import safetensors.numpy

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor import Result
from gdtensor_sim.coord import ThreadCoordinator
from gdtensor_sim.formats import COMMIT_MARKER_NAME, make_storage_view
from gdtensor_sim.iobackend import SdkFileIO
from gdtensor_sim.models import ParallelConfig, layout_builder, storage_table
from tests.support.cache_oracle import storage_schema
from tests.support.oracle import ArrayFabric
from tests.support.schemas import run_ranks, tensor_schema


def _synchronized(method):
    def run(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return run


class MemoryClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytearray] = {}
        self.versions: dict[str, str] = {}
        self.history: dict[tuple[str, str], bytes] = {}
        self.synced: set[str] = set()
        self.events: list[tuple] = []
        self.counter = 0
        self.max_write: int | None = None
        self.zero_write = False
        self.short_read = False
        self.fail_publish: str | None = None
        self._lock = threading.RLock()

    @_synchronized
    def _set(self, key: str, data) -> None:
        self.counter += 1
        version = f"v{self.counter}"
        raw = bytes(data)
        self.objects[key] = bytearray(raw)
        self.versions[key] = version
        self.history[(key, version)] = raw

    @_synchronized
    def metadata(self, key: str) -> tuple[int, str]:
        if key not in self.objects:
            raise OSError(f"missing {key}")
        return len(self.objects[key]), self.versions[key]

    @_synchronized
    def read_range(self, key: str, offset: int, size: int, version: str) -> bytes:
        try:
            raw = self.history[(key, version)]
        except KeyError:
            raise OSError(f"missing version {key}@{version}") from None
        if self.short_read and size:
            size -= 1
        return raw[offset : offset + size]

    @_synchronized
    def create_staging(self, key: str, size: int) -> None:
        current = self.objects.get(key)
        if current is not None and len(current) == size:
            return
        raw = bytearray(current or b"")
        del raw[size:]
        raw.extend(b"\0" * (size - len(raw)))
        self._set(key, raw)

    @_synchronized
    def write_range(self, key: str, offset: int, data: memoryview) -> int | None:
        if self.zero_write:
            return 0
        raw = bytes(data)
        size = len(raw) if self.max_write is None else min(len(raw), self.max_write)
        self.objects[key][offset : offset + size] = raw[:size]
        self._set(key, self.objects[key])
        return size

    @_synchronized
    def sync(self, key: str) -> None:
        self.events.append(("sync", key))
        self.synced.add(key)

    @_synchronized
    def publish(self, temporary_key: str, final_key: str) -> None:
        self.events.append(("publish", temporary_key, final_key))
        if final_key == self.fail_publish:
            raise OSError(f"intentional publish failure: {final_key}")
        if temporary_key not in self.synced:
            raise OSError(f"unsynced publication: {temporary_key}")
        self._set(final_key, self.objects[temporary_key])
        self.discard(temporary_key)

    @_synchronized
    def discard(self, key: str) -> None:
        self.objects.pop(key, None)
        self.versions.pop(key, None)
        self.synced.discard(key)

    @_synchronized
    def close_rank(self, rank: int, prefix: str | None = None) -> None:
        self.events.append(("close", rank, prefix))


T = TypeVar("T")


def _ok(result: Result[T]) -> T:
    assert type(result) is gb.Ok, result
    return cast(T, result.value)


def _fixture(root: str, client: MemoryClient, size: int = 17):
    st, layout = storage_schema(["w"], size)
    layout.container_format = "safetensors"
    path = "model.safetensors"
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
    io = SdkFileIO(client)
    sv = make_storage_view(["w"], st, layout, root, io)
    return ct, st, layout, values, points, sv, path


def _take(dst):
    def read(ptr, size):
        dst.extend(ctypes.string_at(ptr, size))
        return gb.Ok(None)

    return read


def test_sdk_rank_save_load_and_direct_chain_load() -> None:
    root = "sdk://test/checkpoint"
    client = MemoryClient()
    ct, st, layout, values, points, sv, path = _fixture(root, client)
    source = ArrayFabric(ct, points, values)
    _ok(gio.rank_save(0, source.view(0), sv))

    final = f"{root}/{path}"
    decoded = safetensors.numpy.load(bytes(client.objects[final]))
    np.testing.assert_array_equal(decoded["w"], values["w"])
    assert f"{root}/{COMMIT_MARKER_NAME}" in client.objects
    assert not [key for key in client.objects if key.endswith(".tmp")]
    publications = [event for event in client.events if event[0] == "publish"]
    assert publications[-1][2] == f"{root}/{COMMIT_MARKER_NAME}"

    for use_chain in (False, True):
        loaded = ArrayFabric(ct, points)
        reader = make_storage_view(["w"], st, layout, root, SdkFileIO(client))
        if use_chain:
            chain = gb.DirectDTensorsReadChain(src=reader, dst=loaded.view(0))
            _ok(gio.chain_load(0, chain))
        else:
            _ok(gio.rank_load(0, loaded.view(0), reader))
        loaded.assert_values(values)


def test_sdk_two_ranks_save_and_load_one_shared_shard() -> None:
    root = "sdk://test/two-ranks"
    p = (1, 1, 2)
    ct, points, values = tensor_schema(p)
    parallel = ParallelConfig(pp=p[0], dp=p[1], tp=p[2])
    table = storage_table(ct, parallel, "mirror")
    layout = layout_builder("single", table, parallel, "safetensors", 3)
    tensor_ids = [nd.global_tensor_id for nd in table.named_d_tensors]
    client = MemoryClient()
    storage = make_storage_view(
        tensor_ids,
        table,
        layout,
        root,
        SdkFileIO(client),
        coord=ThreadCoordinator(parallel.world),
    )
    source = ArrayFabric(ct, points, values)
    run_ranks(2, lambda rank: gio.rank_save(rank, source.view(rank), storage))

    assert len([key for key in client.objects if key.endswith(".safetensors")]) == 1
    assert f"{root}/{COMMIT_MARKER_NAME}" in client.objects
    assert not [key for key in client.objects if key.endswith(".tmp")]
    loaded = ArrayFabric(ct, points)
    reader = make_storage_view(tensor_ids, table, layout, root, SdkFileIO(client))
    run_ranks(2, lambda rank: gio.rank_load(rank, loaded.view(rank), reader))
    loaded.assert_values(values)


def test_sdk_read_is_pinned_and_identity_detects_replacement() -> None:
    root = "sdk://test/identity"
    client = MemoryClient()
    ct, st, layout, values, points, sv, path = _fixture(root, client)
    _ok(gio.rank_save(0, ArrayFabric(ct, points, values).view(0), sv))
    reader = make_storage_view(["w"], st, layout, root, SdkFileIO(client))
    assert reader.begin_load is not None
    session = _ok(reader.begin_load(reader, 0))
    accessor = _ok(session.init_data_accessor())
    first_identity = accessor.cache_identity()

    final = f"{root}/{path}"
    client.write_range(final, 8, memoryview(b"changed"))
    got = bytearray()
    rng = SimpleNamespace(file_path=path, contiguous_payload_offset=0, size=17)
    _ok(accessor.read_to_buffer(rng, _take(got)))
    assert bytes(got) == values["w"].tobytes()
    with pytest.raises(OSError, match="source path was replaced"):
        accessor.cache_identity()
    assert first_identity != accessor.io.identity(0, final, False)
    _ok(session.finish())


def test_sdk_short_io_and_multi_rank_staging_writes() -> None:
    client = MemoryClient()
    client.max_write = 2
    io = SdkFileIO(client)
    staging, final = "sdk://bytes.tmp", "sdk://bytes"
    io.create_staging(staging, 8)
    io.pwrite(0, staging, 0, b"abcd")
    io.create_staging(staging, 8)
    io.pwrite(1, staging, 4, b"efgh")
    io.sync(staging)
    io.publish(staging, final)
    assert io.pread(2, final, 0, 8) == b"abcdefgh"

    client.short_read = True
    with pytest.raises(OSError, match="short read"):
        SdkFileIO(client).pread(0, final, 0, 8)
    client.short_read = False
    client.zero_write = True
    with pytest.raises(OSError, match="invalid SDK write count"):
        io.pwrite(0, final, 0, b"x")


def test_sdk_callback_failure_aborts_only_staging() -> None:
    root = "sdk://test/abort"
    client = MemoryClient()
    _, _, _, _, _, sv, path = _fixture(root, client)
    assert sv.begin_save is not None
    session = _ok(sv.begin_save(sv, 0))
    accessor = _ok(session.init_data_accessor())
    rng = SimpleNamespace(file_path=path, contiguous_payload_offset=0, size=17)
    error = gb.GDTensorIOError(operation="callback", msg="intentional")
    before = list(client.events)
    result = accessor.write_from_buffer(rng, lambda ptr, size: gb.Err(error))
    assert type(result) is gb.Err and result.value is error
    assert client.events == before
    _ok(session.abort())
    assert client.objects == {}


def test_sdk_marker_failure_keeps_commit_unknown_payloads() -> None:
    root = "sdk://test/unknown"
    client = MemoryClient()
    marker = f"{root}/{COMMIT_MARKER_NAME}"
    client.fail_publish = marker
    ct, _, _, values, points, sv, path = _fixture(root, client)
    result = gio.rank_save(0, ArrayFabric(ct, points, values).view(0), sv)
    assert type(result) is gb.Err
    assert f"{root}/{path}" in client.objects
    assert marker not in client.objects
    assert f"{marker}.tmp" in client.objects

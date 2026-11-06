"""File cache regression: source paths, short writes, corruption and oracle gaps."""

import os
from dataclasses import replace

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from gdtensor_sim.file_cache import (
    Device,
    DiskNodeCache,
    DiskStorage,
    SparseRankBuffers,
    pwrite_all,
)
from tests.support import cache_shapes as shapes
from tests.support.cluster_sim import Cluster
from tests.support.golden_oracle import verify_parts


def build(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    cluster = Cluster(1, 1)
    cluster.rank = 0
    values = {"w": np.array([7, 8, 255, 9], np.uint8)}
    device = Device(1, 0, 0)
    src = DiskStorage(cluster, values, tmp_path, device)
    cache = DiskNodeCache(
        cluster, {"w": 4}, lambda _r: 0, str(tmp_path / "cache"), device
    )
    dst = SparseRankBuffers(cluster)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=shapes._storage_view(src, ["w"], {"w": 4}, 1),
        cache=shapes._cache_view(cache, ["w"], {"w": 4}, 1, [0], 1),
        dst=shapes._compute_view(dst, ["w"], {"w": 4}, 1, {"w": False}),
    )
    return chain, src, cache, dst, values


def load(chain, dst, values):
    dst.parts.clear()
    result = gio.chain_load(0, chain)
    assert type(result) is gb.Ok, result
    verify_parts(dst.parts, values, 1, {"w": False})


def changed_source(tmp_path, src, values, chain, change):
    if change == "different_path":
        other = tmp_path / "checkpoint2"
        other.mkdir()
        source2 = DiskStorage(src.cluster, values, other, src.device)
        return source2, replace(
            chain, src=shapes._storage_view(source2, ["w"], {"w": 4}, 1)
        )
    with open(src.path, "r+b") as file:
        file.write(values["w"].tobytes())
    return src, chain


def corrupt(cache, path, fault):
    if fault == "checksum":
        cache.integrity[next(iter(cache.integrity))] = b"bad"
        return
    with open(path, "r+b") as file:
        if fault == "truncate":
            file.truncate(1)
        else:
            file.write(b"bad!")


@pytest.mark.parametrize("change", ["same_path", "different_path"])
def test_source_file_identity_prevents_stale_hit(tmp_path, monkeypatch, change):
    chain, src, cache, dst, values = build(tmp_path, monkeypatch)
    try:
        load(chain, dst, values)
        before = sum(src.cluster.storage_in.values())
        load(chain, dst, values)
        assert sum(src.cluster.storage_in.values()) == before
        values["w"] ^= np.uint8(255)
        source2, chain = changed_source(tmp_path, src, values, chain, change)
        load(chain, dst, values)
        assert sum(src.cluster.storage_in.values()) == before + 4
        if source2 is not src:
            source2.close()
    finally:
        src.close()
        cache.close()


@pytest.mark.parametrize("fault", ["flip", "truncate", "checksum"])
def test_corrupt_cache_fails_before_delivery(tmp_path, monkeypatch, fault):
    chain, src, cache, dst, values = build(tmp_path, monkeypatch)
    try:
        load(chain, dst, values)
        path = next((tmp_path / "cache").rglob("cache0.bin"))
        corrupt(cache, path, fault)
        dst.parts.clear()
        before = sum(src.cluster.storage_in.values())
        result = gio.chain_load(0, chain)
        assert type(result) is gb.Err, result
        assert not dst.parts
        assert sum(src.cluster.storage_in.values()) == before
    finally:
        src.close()
        cache.close()


def test_zero_short_write_never_publishes(tmp_path, monkeypatch):
    chain, src, cache, dst, values = build(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(os, "pwrite", lambda *_a: 0)
        result = gio.chain_load(0, chain)
        assert type(result) is gb.Err
        assert not cache.mem and not cache.fills and not dst.parts
    finally:
        src.close()
        cache.close()


def test_partial_write_is_completed(tmp_path, monkeypatch):
    path = tmp_path / "partial"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    write = os.pwrite
    monkeypatch.setattr(os, "pwrite", lambda f, b, o: write(f, b[:1], o))
    try:
        pwrite_all(fd, b"abcd", 0)
        assert os.pread(fd, 4, 0) == b"abcd"
    finally:
        os.close(fd)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_source_change_during_load_returns_error(tmp_path, monkeypatch, depth):
    chain, src, cache, dst, _values = build(tmp_path, monkeypatch)
    if depth == 0:
        chain = gb.DirectDTensorsReadChain(src=chain.src, dst=chain.dst)
    if depth == 2:
        chain = gb.TwoLayerCachedDTensorsReadChain(
            src=chain.src,
            dst=chain.dst,
            small_cache=chain.cache,
            big_cache=chain.cache,
            num_ranks_per_node=1,
        )
    original = src.read_to_buffer

    def changing(global_range, read_to_buffer):
        result = original(global_range, read_to_buffer)
        with open(src.path, "r+b") as file:
            file.write(b"new!")
        return result

    monkeypatch.setattr(src, "read_to_buffer", changing)
    try:
        assert type(gio.chain_load(0, chain)) is gb.Err
    finally:
        src.close()
        cache.close()


def test_restart_without_validity_index_refills(tmp_path, monkeypatch):
    chain, src, cache, dst, values = build(tmp_path, monkeypatch)
    try:
        load(chain, dst, values)
        cache.close()
        new = DiskNodeCache(src.cluster, {"w": 4}, lambda _r: 0, cache.root, src.device)
        assert isinstance(chain.cache, gb.CacheDTensorsView)
        chain = replace(chain, cache=replace(chain.cache, data_accessor=new))
        before = sum(src.cluster.storage_in.values())
        load(chain, dst, values)
        assert sum(src.cluster.storage_in.values()) == before + 4
        new.close()
    finally:
        src.close()


def test_subset_checksum_read_records_physical_amplification(tmp_path, monkeypatch):
    chain, src, cache, dst, values = build(tmp_path, monkeypatch)
    try:
        load(chain, dst, values)
        namespace, node, tid, off, size = next(iter(cache.integrity))
        request = type(
            "Range",
            (),
            dict(
                source_identity=src.cache_identity(),
                cache_node=node,
                global_tensor_id=tid,
                global_tensor_offset=off + 1,
                size=1,
            ),
        )()
        before = cache.physical_read_bytes
        assert type(cache.read_to_buffer(request, lambda *_a: gb.Ok(None))) is gb.Ok
        assert cache.physical_read_bytes - before == size
    finally:
        src.close()
        cache.close()


def test_parallel_nodes_do_not_share_file_io_lock(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    cluster = Cluster(2, 1)
    acc = DiskNodeCache(cluster, {"w": 4}, lambda r: r, str(tmp_path), Device(2, 0, 0))
    barrier = threading.Barrier(2, timeout=3)
    write = os.pwrite
    monkeypatch.setattr(os, "pwrite", lambda *a: (barrier.wait(), write(*a))[1])
    errors = []

    def one(node):
        cluster.rank = node
        request = SimpleNamespace(
            source_identity="same",
            cache_node=node,
            global_tensor_id="w",
            global_tensor_offset=0,
            size=4,
        )
        try:
            assert (
                type(acc.write_from_buffer(request, lambda *_a: gb.Ok(None))) is gb.Ok
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=one, args=(n,)) for n in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        assert not errors and not any(t.is_alive() for t in threads)
    finally:
        acc.close()


def test_dcp_metadata_change_invalidates_identity(tmp_path):
    from types import SimpleNamespace
    from gdtensor_sim.dcp_backend import DcpLoadSession

    metadata = tmp_path / ".metadata"
    metadata.write_bytes(b"original")
    session = DcpLoadSession(
        0, str(tmp_path), gb.DTensorTable(), None, SimpleNamespace(files={})
    )
    before = session.cache_identity()
    assert session.full is None
    metadata.write_bytes(b"changed-metadata")
    assert session.cache_identity() != before


def test_public_plan_refreshes_inplace_layout(tmp_path):
    class Recorder:
        def __init__(self):
            self.paths = []

        def read_to_buffer(self, global_range, read_to_buffer):
            self.paths.append(global_range.file_path)
            data = np.arange(global_range.size, dtype=np.uint8)
            return read_to_buffer(data.ctypes.data, data.nbytes)

    class Session:
        def __init__(self, accessor):
            self.accessor = accessor

        def init_data_accessor(self):
            return gb.Ok(self.accessor)

        def finish(self):
            return gb.Ok(None)

    cluster = Cluster(1, 1)
    cluster.rank = 0
    recorder = Recorder()
    src = shapes._storage_view(recorder, ["w"], {"w": 4}, 1)
    src.begin_load = lambda *_a: gb.Ok(Session(recorder))
    dst = shapes._compute_view(
        SparseRankBuffers(cluster), ["w"], {"w": 4}, 1, {"w": False}
    )
    assert type(gio.rank_load(0, dst, src)) is gb.Ok
    dst.data_accessor.parts.clear()
    src.physical_layout.file_paths[gb.TensorComponentPos("w", 0)] = "new.bin"
    assert type(gio.rank_load(0, dst, src)) is gb.Ok
    assert recorder.paths == ["model.bin", "new.bin"]


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_hundred_node_file_coordinator_chain(tmp_path, monkeypatch, depth):
    import threading
    import time

    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "file")
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    cluster = Cluster(100, 1)
    data = {"w": np.random.default_rng(91).integers(0, 256, 16002, np.uint8)}
    src = shapes.FileStorage(cluster, data)
    dst = SparseRankBuffers(cluster)
    source = shapes._storage_view(src, ["w"], {"w": 16002}, 100)
    compute = shapes._compute_view(dst, ["w"], {"w": 16002}, 100, {"w": False})
    caches = [
        DiskNodeCache(
            cluster,
            {"w": 16002},
            lambda r: r,
            str(tmp_path / f"layer-{i}"),
            Device(100, 0, 0),
        )
        for i in range(depth)
    ]
    views = [
        shapes._cache_view(c, ["w"], {"w": 16002}, 2, list(range(100)), 2)
        for c in caches
    ]
    chain = gb.DirectDTensorsReadChain(src=source, dst=compute)
    if depth == 1:
        chain = gb.OneLayerCachedDTensorsReadChain(
            src=source, dst=compute, cache=views[0]
        )
    if depth == 2:
        chain = gb.TwoLayerCachedDTensorsReadChain(
            src=source, dst=compute, small_cache=views[0], big_cache=views[1]
        )
    errors = []

    def one(rank):
        cluster.rank = rank
        try:
            result = gio.chain_load(rank, chain)
            assert type(result) is gb.Ok, result
        except BaseException as exc:
            errors.append(exc)

    try:
        for _round in range(2):
            dst.parts.clear()
            _hundred_threads(one, time, threading)
            assert not errors
            verify_parts(dst.parts, data, 100, {"w": False})
    finally:
        for acc in caches:
            acc.close()


def _hundred_threads(one, time, threading):
    threads = [threading.Thread(target=one, args=(r,), daemon=True) for r in range(100)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 30
    for thread in threads:
        thread.join(max(0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads)


@pytest.mark.parametrize(
    "parts",
    [
        {("w", 0): [7, 8], ("w", 1): [8], ("w", 3): [9]},
        {("w", 0): [7, 8, 255, 9], ("w", 5): [91]},
        {("w", 0): [7, 8, 255, 9], ("unknown", 0): [92]},
    ],
)
def test_oracle_rejects_overlap_gap_and_extra(parts):
    buffers = {0: {k: np.array(v, np.uint8) for k, v in parts.items()}}
    with pytest.raises(AssertionError):
        verify_parts(
            buffers, {"w": np.array([7, 8, 255, 9], np.uint8)}, 1, {"w": False}
        )

"""File cache regression: source paths, short writes, corruption and oracle gaps."""

import os
import sys
import threading
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

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
from tests.support.cluster_sim import Cluster, NodeCache
from tests.support.golden_oracle import verify_parts

internal_api = sys.modules[gio.chain_load.__module__]


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


def test_same_root_instances_do_not_share_coordination(tmp_path, monkeypatch):
    chain, src, cache, dst, values = build(tmp_path, monkeypatch)
    new = DiskNodeCache(src.cluster, {"w": 4}, lambda _r: 0, cache.root, src.device)
    try:
        assert isinstance(chain.cache, gb.CacheDTensorsView)
        other = replace(chain, cache=replace(chain.cache, data_accessor=new))
        assert internal_api._slots(other) is not internal_api._slots(chain)
        before = sum(src.cluster.storage_in.values())
        load(other, dst, values)
        assert sum(src.cluster.storage_in.values()) == before + 4
    finally:
        src.close()
        cache.close()
        new.close()


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

        def unexpected_probe(_range):
            raise AssertionError("cache read repeated its presence probe")

        monkeypatch.setattr(cache, "has_bytes", unexpected_probe)
        assert type(cache.read_to_buffer(request, lambda *_a: gb.Ok(None))) is gb.Ok
        assert cache.physical_read_bytes - before == size
    finally:
        src.close()
        cache.close()


def test_hot_batch_load_and_checksum_failure(tmp_path, monkeypatch):
    chain, src, cache, dst, values = build(tmp_path, monkeypatch)
    try:
        load(chain, dst, values)
        original = cache.read_bytes_batch
        batches = []

        def counting(ranges):
            batches.append(ranges)
            return original(ranges)

        monkeypatch.setattr(cache, "read_bytes_batch", counting)
        load(chain, dst, values)
        assert len(batches) == 1 and len(batches[0]) == 1
        cache.integrity[next(iter(cache.integrity))] = b"bad"
        dst.parts.clear()
        result = gio.chain_load(0, chain)
        assert type(result) is gb.Err
        assert not dst.parts
    finally:
        src.close()
        cache.close()


def _cold_parent_case(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path / "coord"))
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    world = 4
    cluster = Cluster(world, 1)
    sizes = {f"t{i}": 64 for i in range(4)}
    rng = np.random.default_rng(73)
    data = {
        tid: rng.integers(0, 256, size, dtype=np.uint8) for tid, size in sizes.items()
    }
    src = shapes.FileStorage(cluster, data)
    dst = SparseRankBuffers(cluster)
    caches = [
        DiskNodeCache(
            cluster,
            sizes,
            lambda r: r,
            str(tmp_path / f"cache-{layer}"),
            Device(world, 0, 0),
        )
        for layer in range(2)
    ]
    views = [
        shapes._cache_view(caches[0], list(sizes), sizes, 1, [0, 1], 1),
        shapes._cache_view(caches[1], list(sizes), sizes, 1, list(range(world)), 1),
    ]
    chain = gb.TwoLayerCachedDTensorsReadChain(
        src=shapes._storage_view(src, list(sizes), sizes, world),
        small_cache=views[0],
        big_cache=views[1],
        dst=shapes._compute_view(
            dst, list(sizes), sizes, world, dict.fromkeys(sizes, False)
        ),
        num_ranks_per_node=1,
    )
    return chain, cluster, caches, dst, data


def _thread_results(chain, cluster, world):
    import threading
    import time

    results = {}

    def one(rank):
        cluster.rank = rank
        try:
            results[rank] = gio.chain_load(rank, chain)
        except BaseException as exc:
            results[rank] = exc

    threads = [
        threading.Thread(target=one, args=(r,), daemon=True) for r in range(world)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 30
    for thread in threads:
        thread.join(max(0, deadline - time.monotonic()))
    assert not any(thread.is_alive() for thread in threads), "chain_load hung"
    assert len(results) == world
    return results


def test_non_all_fit_hot_uses_batch_reads(tmp_path, monkeypatch):
    chain, cluster, caches, dst, data = _cold_parent_case(tmp_path, monkeypatch)
    original_init = internal_api._RankRun.__init__

    def non_all_fit(run, *args):
        original_init(run, *args)
        run._all_fit = False

    monkeypatch.setattr(internal_api._RankRun, "__init__", non_all_fit)
    try:
        results = _thread_results(chain, cluster, 4)
        assert all(type(result) is gb.Ok for result in results.values()), results
        terminal = caches[1]
        original_batch = terminal.read_bytes_batch
        original_scalar = terminal.read_to_buffer
        batches, scalar_reads = [], []

        def counting_batch(ranges):
            batches.append(ranges)
            return original_batch(ranges)

        def counting_scalar(global_range, callback):
            scalar_reads.append(global_range)
            return original_scalar(global_range, callback)

        monkeypatch.setattr(terminal, "read_bytes_batch", counting_batch)
        monkeypatch.setattr(terminal, "read_to_buffer", counting_scalar)
        dst.parts.clear()
        results = _thread_results(chain, cluster, 4)
        assert all(type(result) is gb.Ok for result in results.values()), results
        verify_parts(dst.parts, data, 4, dict.fromkeys(data, False))
        assert batches and not scalar_reads
    finally:
        for cache in caches:
            cache.close()


def _corrupt_second_batch_range(cache, original, corrupted, seen):
    lock = threading.Lock()

    def corrupt(ranges):
        if len(ranges) < 2 or corrupted:
            return
        target = ranges[1]
        node, namespace = cache._ends(target)[1:]
        tid = target.global_tensor_id
        off, size = target.global_tensor_offset, target.size
        hits = [
            key
            for key in cache.integrity
            if key[:3] == (namespace, node, tid)
            and key[3] <= off
            and off + size <= key[3] + key[4]
        ]
        assert len(hits) == 1
        cache.integrity[hits[0]] = b"bad"
        corrupted.append(cache.cluster.rank)

    def wrapped(ranges):
        seen.append(len(ranges))
        with lock:
            corrupt(ranges)
        return original(ranges)

    return wrapped


@pytest.mark.parametrize("mode", ["batch", "no_batch", "reserve_fallback", "bounded"])
def test_cold_parent_batch_load_and_fallback(tmp_path, monkeypatch, mode):
    chain, cluster, caches, dst, data = _cold_parent_case(tmp_path, monkeypatch)
    parent = caches[0]
    api = sys.modules[gio.chain_load.__module__]
    if mode == "bounded":
        monkeypatch.setattr(api, "_DRAIN_BATCH_BYTES", 32)
    plan = api._slots(chain).plan
    parent_ranges = sum(
        len(seg.parents)
        for segments in plan.segments[1].values()
        for seg in segments.values()
    )
    batches, scalar_reads = [], []
    original_batch = parent.read_bytes_batch
    original_scalar = parent.read_to_buffer

    def counting_batch(ranges):
        assert api._slots(chain).inflight > 0
        batches.append(ranges)
        return original_batch(ranges)

    def counting_scalar(global_range, callback):
        scalar_reads.append(global_range)
        return original_scalar(global_range, callback)

    original_reserve = api._RankRun._reserve
    failed = []

    def fail_one_batch(self, layer, node, segs):
        if layer == 1 and len(segs) > 1 and not failed:
            failed.append((node, len(segs)))
            return False
        return original_reserve(self, layer, node, segs)

    if mode == "no_batch":
        monkeypatch.setattr(parent, "read_bytes_batch", None)
    else:
        monkeypatch.setattr(parent, "read_bytes_batch", counting_batch)
    monkeypatch.setattr(parent, "read_to_buffer", counting_scalar)
    if mode == "reserve_fallback":
        monkeypatch.setattr(api._RankRun, "_reserve", fail_one_batch)
    try:
        results = _thread_results(chain, cluster, 4)
        assert all(type(result) is gb.Ok for result in results.values()), results
        verify_parts(dst.parts, data, 4, dict.fromkeys(data, False))
        assert sum(map(len, batches)) + len(scalar_reads) == parent_ranges
        if mode in ("batch", "bounded"):
            assert 0 < len(batches) < parent_ranges
            assert not scalar_reads
        if mode == "bounded":
            assert len(batches) > 4
            assert all(sum(r.size for r in batch) <= 32 for batch in batches)
        if mode == "no_batch":
            assert len(scalar_reads) == parent_ranges
        if mode == "reserve_fallback":
            assert len(failed) == 1 and len(scalar_reads) == failed[0][1]
    finally:
        for cache in caches:
            cache.close()


def test_cold_parent_checksum_failure_publishes_no_batch_bytes(tmp_path, monkeypatch):
    chain, cluster, caches, dst, data = _cold_parent_case(tmp_path, monkeypatch)
    parent, terminal = caches
    api = sys.modules[gio.chain_load.__module__]
    try:
        results = _thread_results(chain, cluster, 4)
        assert all(type(result) is gb.Ok for result in results.values()), results
        terminal.clear()
        dst.parts.clear()
        fill_count = len(terminal.fills)
        original_batch = parent.read_bytes_batch
        corrupted, seen = [], []
        monkeypatch.setattr(
            parent,
            "read_bytes_batch",
            _corrupt_second_batch_range(parent, original_batch, corrupted, seen),
        )
        results = _thread_results(chain, cluster, 4)
        assert corrupted, (seen, results)
        errors = [result.value for result in results.values() if type(result) is gb.Err]
        assert errors and all(error == errors[0] for error in errors), results
        assert type(results[corrupted[0]]) is not gb.Ok
        node = corrupted[0]
        assert node not in dst.parts
        assert not [fill for fill in terminal.fills[fill_count:] if fill[1] == node]
        state = api._slots(chain).node(1, node)
        assert not state.slots and state.resident == 0
        assert api._slots(chain).inflight == 0
    finally:
        for cache in caches:
            cache.close()


def test_parent_ack_failure_fails_waiting_peers(tmp_path, monkeypatch):
    chain, cluster, caches, dst, _data = _cold_parent_case(tmp_path, monkeypatch)
    api = sys.modules[gio.chain_load.__module__]
    monkeypatch.setattr(api._SlotTable, "in_memory", False)
    original = api._RankRun._ack_parents_locked

    def fail_parent_ack(self, layer, node, seg):
        if layer == 1:
            raise OSError("injected parent ACK failure")
        return original(self, layer, node, seg)

    monkeypatch.setattr(api._RankRun, "_ack_parents_locked", fail_parent_ack)
    try:
        results = _thread_results(chain, cluster, 4)
        errors = [result.value for result in results.values() if type(result) is gb.Err]
        assert len(errors) == 4 and all(error == errors[0] for error in errors), results
        assert not dst.parts
        table = api._slots(chain)
        assert table.inflight == 0
        assert all(node.resident >= 0 for node in table.nodes.values())
    finally:
        for cache in caches:
            cache.close()


def _check_partial_batch_round(
    first, probes, scalar_probes, new_fills, read_batches, missing
):
    if first:
        assert False in probes and scalar_probes
        assert len(new_fills) == 1
        assert new_fills[0][1:3] == missing[1:3]
    else:
        assert probes and all(probes)
        assert not scalar_probes and not new_fills
    assert len(read_batches) == 8
    assert all(sum(r.size for r in batch) <= 32 for batch in read_batches)


def test_warm_partial_batch_miss_refills_only_missing_segment(tmp_path, monkeypatch):
    chain, cluster, caches, dst, data = _cold_parent_case(tmp_path, monkeypatch)
    terminal = caches[1]
    api = sys.modules[gio.chain_load.__module__]
    monkeypatch.setattr(api, "_DRAIN_BATCH_BYTES", 32)
    try:
        results = _thread_results(chain, cluster, 4)
        assert all(type(result) is gb.Ok for result in results.values()), results
        missing = next(key for key in terminal.mem if key[1] == 0)
        del terminal.mem[missing]
        original_batch = terminal.has_bytes_batch
        original_scalar = terminal.has_bytes
        original_read = terminal.read_bytes_batch
        probes, scalar_probes, read_batches = [], [], []

        def counting_batch(ranges):
            result = original_batch(ranges)
            probes.append(result)
            return result

        def counting_scalar(global_range):
            scalar_probes.append(global_range)
            return original_scalar(global_range)

        def counting_read(ranges):
            read_batches.append(ranges)
            return original_read(ranges)

        monkeypatch.setattr(terminal, "has_bytes_batch", counting_batch)
        monkeypatch.setattr(terminal, "has_bytes", counting_scalar)
        monkeypatch.setattr(terminal, "read_bytes_batch", counting_read)
        source = cast(Any, chain).src.data_accessor
        for round_index in range(2):
            dst.parts.clear()
            fill_count = len(terminal.fills)
            source_reads = len(source.reads)
            probes.clear()
            scalar_probes.clear()
            read_batches.clear()
            results = _thread_results(chain, cluster, 4)
            assert all(type(result) is gb.Ok for result in results.values()), results
            verify_parts(dst.parts, data, 4, dict.fromkeys(data, False))
            assert len(source.reads) == source_reads
            new_fills = terminal.fills[fill_count:]
            _check_partial_batch_round(
                round_index == 0,
                probes,
                scalar_probes,
                new_fills,
                read_batches,
                missing,
            )
    finally:
        for cache in caches:
            cache.close()


def test_hot_checksum_failure_publishes_no_multi_range_batch(tmp_path, monkeypatch):
    chain, cluster, caches, dst, _data = _cold_parent_case(tmp_path, monkeypatch)
    terminal = caches[1]
    api = sys.modules[gio.chain_load.__module__]
    corrupted, seen, accounting = [], [], []
    original = _corrupt_second_batch_range(
        terminal, terminal.read_bytes_batch, corrupted, seen
    )

    def measured(ranges):
        rank = terminal.cluster.rank
        before = sum(read[0] == rank for read in terminal.reads)
        try:
            return original(ranges)
        finally:
            after = sum(read[0] == rank for read in terminal.reads)
            accounting.append(after - before) if len(ranges) >= 2 else None

    try:
        results = _thread_results(chain, cluster, 4)
        assert all(type(result) is gb.Ok for result in results.values()), results
        dst.parts.clear()
        monkeypatch.setattr(terminal, "read_bytes_batch", measured)
        results = _thread_results(chain, cluster, 4)
        assert corrupted and max(seen) >= 2, (seen, results)
        assert 1 in accounting, accounting
        node = corrupted[0]
        assert type(results[node]) is not gb.Ok
        assert node not in dst.parts
        assert api._slots(chain).inflight == 0
    finally:
        for cache in caches:
            cache.close()


@pytest.mark.parametrize("path", ["hot", "cold_parent"])
@pytest.mark.parametrize("malformation", ["missing_item", "short_payload"])
def test_bad_batch_results_publish_nothing(tmp_path, monkeypatch, path, malformation):
    chain, cluster, caches, dst, _data = _cold_parent_case(tmp_path, monkeypatch)
    api = sys.modules[gio.chain_load.__module__]
    terminal = caches[1]
    target = caches[0] if path == "cold_parent" else terminal
    original = target.read_bytes_batch

    def malformed(ranges):
        data = list(original(ranges))
        if malformation == "missing_item":
            return data[:-1]
        data[1] = data[1][:-1]
        return data

    try:
        results = _thread_results(chain, cluster, 4)
        assert all(type(result) is gb.Ok for result in results.values()), results
        dst.parts.clear()
        fill_count = len(terminal.fills)
        if path == "cold_parent":
            terminal.clear()
        monkeypatch.setattr(target, "read_bytes_batch", malformed)
        results = _thread_results(chain, cluster, 4)
        errors = [result.value for result in results.values() if type(result) is gb.Err]
        assert len(errors) == 4 and all(error == errors[0] for error in errors), results
        assert not dst.parts
        assert len(terminal.fills) == fill_count
        table = api._slots(chain)
        assert table.inflight == 0
        assert all(
            slot["filled"]
            for state in table.nodes.values()
            for slot in state.slots.values()
        )
    finally:
        for cache in caches:
            cache.close()


def test_cold_parent_write_failure_cleans_remaining_reservations(tmp_path, monkeypatch):
    chain, cluster, caches, _dst, _data = _cold_parent_case(tmp_path, monkeypatch)
    terminal = caches[1]
    api = sys.modules[gio.chain_load.__module__]
    original_write = terminal.write_from_buffer
    lock = threading.Lock()
    writes = []

    def fail_second_write(global_range, callback):
        node = global_range.cache_node
        with lock:
            writes.append(node)
            count = writes.count(node)
        if node == 0 and count == 2:
            return gb.Err(gb.GDTensorIOError(operation="write", msg="injected"))
        return original_write(global_range, callback)

    monkeypatch.setattr(terminal, "write_from_buffer", fail_second_write)
    try:
        results = _thread_results(chain, cluster, 4)
        assert type(results[0]) is not gb.Ok
        state = api._slots(chain).node(1, 0)
        assert len(state.completed) == 1
        assert len([fill for fill in terminal.fills if fill[1] == 0]) == 1
        assert all(slot["filled"] for slot in state.slots.values())
        assert state.resident == sum(key[2] for key in state.slots)
        assert api._slots(chain).inflight == 0
    finally:
        for cache in caches:
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


def _direct_cache(tmp_path, size=4):
    cluster = Cluster(1, 1)
    cluster.rank = 0
    cache = DiskNodeCache(
        cluster, {"w": size}, lambda _rank: 0, str(tmp_path), Device(1, 0, 0)
    )
    return cluster, cache


def _cache_request(identity="source", off=0, size=4):
    return SimpleNamespace(
        source_identity=identity,
        cache_node=0,
        global_tensor_id="w",
        global_tensor_offset=off,
        size=size,
    )


@dataclass
class _ReprIdentity:
    value: int

    def __repr__(self):
        return "same repr"


def test_cache_identity_content_prevents_repr_collision(tmp_path):
    _cluster, cache = _direct_cache(tmp_path)
    try:
        first, second = _ReprIdentity(1), _ReprIdentity(2)
        internal_api._write(cache, _cache_request(first), b"abcd")
        assert internal_api._read(cache, _cache_request(first)) == b"abcd"
        with pytest.raises(OSError, match="never published"):
            internal_api._read(cache, _cache_request(second))
    finally:
        cache.close()


def test_overlapping_write_refreshes_combined_integrity_record(tmp_path):
    _cluster, cache = _direct_cache(tmp_path)
    try:
        internal_api._write(cache, _cache_request(), b"abcd")
        internal_api._write(cache, _cache_request(off=1, size=2), b"XY")
        assert internal_api._read(cache, _cache_request()) == b"aXYd"
        assert internal_api._read(cache, _cache_request(off=1, size=2)) == b"XY"
    finally:
        cache.close()


def test_adjacent_integrity_records_cover_one_read(tmp_path):
    _cluster, cache = _direct_cache(tmp_path, size=8)
    try:
        internal_api._write(cache, _cache_request(), b"abcd")
        internal_api._write(cache, _cache_request(off=4), b"efgh")
        request = _cache_request(size=8)
        assert cache.has_bytes(request)
        assert internal_api._read(cache, request) == b"abcdefgh"
        assert cache.read_bytes_batch([request]) == [b"abcdefgh"]
    finally:
        cache.close()


def test_memory_cache_read_rejects_a_different_source():
    cluster = Cluster(1, 1)
    cluster.rank = 0
    cache = NodeCache(cluster, {"w": 4}, lambda _rank: 0)
    internal_api._write(cache, _cache_request("first"), b"abcd")
    called = []

    def receive(ptr, size):
        called.append((ptr, size))
        return gb.Ok(None)

    result = cache.read_to_buffer(_cache_request("second"), receive)
    assert type(result) is gb.Err
    assert not called


def test_dcp_metadata_payload_change_invalidates_identity(tmp_path):
    import pickle
    from gdtensor_sim.dcp_backend import DcpLoadSession

    payload = tmp_path / "__1_0.distcp"
    payload.write_bytes(b"original")
    metadata = tmp_path / ".metadata"
    metadata.write_bytes(
        pickle.dumps(
            SimpleNamespace(
                storage_data={0: SimpleNamespace(relative_path=payload.name)}
            )
        )
    )
    session = DcpLoadSession(
        0, str(tmp_path), gb.DTensorTable(), None, SimpleNamespace(files={})
    )
    before = session.cache_identity()
    assert session.full is None
    payload.write_bytes(b"changed-payload")
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

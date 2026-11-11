"""Thread-coordinator scaling regressions.

The thread coordinator must not wake every waiter for state changes that
make no waiter predicate true, and it must batch one rank's drain scan into
a constant number of lock passes. Both tests are deterministic: the first
counts version bumps, the second blocks one fill past the wait limit and
relies on the in-flight poll, not on wall-clock progress.
"""

from __future__ import annotations

import copy
import gc
import pickle
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.cache_shapes import GUARD, _build, _wants


class _ReducedInt(int):
    tag: str

    def __new__(cls, value, tag):
        result = super().__new__(cls, value)
        result.tag = tag
        return result

    def __eq__(self, other):
        return int(self) == int(other) and self.tag == getattr(other, "tag", None)

    def __getstate__(self):
        return None

    def __reduce_ex__(self, _protocol):
        return type(self), (int(self), self.tag)


@dataclass(init=False, eq=False)
class _ReducedList(list):
    tag: str

    def __init__(self, values, tag):
        list.__init__(self, values)
        self.tag = tag

    def __reduce_ex__(self, _protocol):
        return type(self), (list(self), self.tag)


class _ReducedSet(set):
    order: list

    def __reduce_ex__(self, _protocol):
        return type(self), (self.order,)


class _StringReduction:
    def __reduce_ex__(self, _protocol):
        return "missing_global"


class _FixedReprInt(int):
    def __repr__(self):
        return "fixed"


class _TaggedInt(int):
    tag: str

    def __new__(cls, value, tag):
        result = super().__new__(cls, value)
        result.tag = tag
        return result

    def __eq__(self, other):
        return int(self) == int(other) and self.tag == getattr(other, "tag", None)


class _TaggedList(list):
    tag: str

    def __eq__(self, other):
        return list(self) == list(other) and self.tag == getattr(other, "tag", None)


class _CountedInt(int):
    calls = 0

    def __getstate__(self):
        type(self).calls += 1
        return None


def _api():
    return sys.modules[gio.chain_load.__module__]


def _reset_outputs(built):
    for tensors in built["dst"].buffers.values():
        for buf in tensors.values():
            buf[:] = GUARD


def _run_threads(built, world):
    _reset_outputs(built)
    cluster = built["cluster"]
    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    threads = [
        threading.Thread(target=one, args=(r,), daemon=True) for r in range(world)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 30
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "chain_load hung"
    bad = {r: v.value for r, v in results.items() if type(v) is not gb.Ok}
    assert len(results) == world and not bad, bad
    return results


def _check_outputs(built, ranks=None):
    for rank, tensors in built["dst"].buffers.items():
        if ranks is not None and rank not in ranks:
            continue
        for tid, got in tensors.items():
            lo, hi = _wants(got.size, built["world"], built["replicate"][tid], rank)
            want = np.full(got.size, GUARD, np.uint8)
            want[lo:hi] = built["data"][tid][lo:hi]
            np.testing.assert_array_equal(got, want, err_msg=f"rank {rank} {tid}")


def _run_serial(built, ranks):
    ranks = tuple(ranks)
    _reset_outputs(built)
    results = {}
    for rank in ranks:
        built["cluster"].rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])
    assert all(type(result) is gb.Ok for result in results.values()), results
    _check_outputs(built, set(ranks))


def _case(sizes):
    return dict(
        seed=0,
        k=1,
        nodes=4,
        layers=[[0, 1], [0, 1, 2, 3]],
        steps=[1, 1],
        width=[1, 1],
        sizes=sizes,
        replicate=[False] * len(sizes),
        coordinator="thread",
        bandwidth=0.0,
        repeat=1,
    )


def test_bounded_batches_preserve_ranges_and_cap(monkeypatch):
    api = _api()
    monkeypatch.setattr(api, "_DRAIN_BATCH_BYTES", 4)
    assert list(api._bounded_batches([3, 1, 2, 4, 5], lambda size: size)) == [
        [3, 1],
        [2],
        [4],
        [5],
    ]


def test_thread_coordinator_bump_count_stays_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    api = _api()
    built = _build(_case([8] * 64))
    assert type(built["chain"].check_constraints()) is gb.Ok
    plan = api._slots(built["chain"]).plan
    drains = sum(len(pieces) for pieces in plan.drains.values())

    count = 0
    lock = threading.Lock()
    original = api._SlotTable.bump

    def counting(self):
        nonlocal count
        with lock:
            count += 1
        return original(self)

    monkeypatch.setattr(api._SlotTable, "bump", counting)
    _run_threads(built, built["world"])
    _check_outputs(built)
    # Joins, fill completions, seals and slot releases are O(segments);
    # per-piece scans and in-flight marks must not add a bump per piece.
    assert count <= 4 * drains + 2 * built["world"] + 16, (count, drains)


def test_loop_preserves_step_version_for_wait(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    built = _build(_case([16]))
    table = api._slots(built["chain"])
    run = object.__new__(api._RankRun)
    run.t, run.gens, run.services, run.wait = table, {}, [], 1
    versions = []
    before = table.version

    def step(_pending):
        with table.cond:
            table.bump()
        return False

    def block(version, _deadline, _pending):
        versions.append(version)
        raise StopIteration

    run._step = step
    run._block = block
    with pytest.raises(StopIteration):
        run._loop([object()])
    # A peer release during _step must remain visible to the wait predicate.
    assert versions == [before]


def test_cold_parent_probes_stay_near_segment_count(tmp_path, monkeypatch):
    """Four child nodes share one parent node, and every parent write is
    blocked for a window. While no parent segment can complete, each cold
    parent segment must be probed on the DN a small number of times, not
    once per waiting rank: the shared absent memo answers the other ranks
    for the rest of the generation."""
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    api = _api()
    case = _case([16] * 8)
    case["layers"] = [[0], [0, 1, 2, 3]]
    built = _build(case)
    assert type(built["chain"].check_constraints()) is gb.Ok
    plan = api._slots(built["chain"]).plan
    parent_segments = len(plan.segments[0][0])

    cache = built["caches"][0]
    probes = 0
    lock = threading.Lock()
    original_probe = cache.has_bytes

    def counting(global_range):
        nonlocal probes
        with lock:
            probes += 1
        return original_probe(global_range)

    cache.has_bytes = counting
    entered, release = threading.Event(), threading.Event()
    original_write = cache.write_from_buffer

    def gated(global_range, write_from_buffer):
        entered.set()
        assert release.wait(timeout=15), "test never released the parent fills"
        return original_write(global_range, write_from_buffer)

    cache.write_from_buffer = gated
    cluster = built["cluster"]
    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    threads = [
        threading.Thread(target=one, args=(r,), daemon=True)
        for r in range(built["world"])
    ]
    for thread in threads:
        thread.start()
    try:
        assert entered.wait(timeout=15), "no parent fill reached the DN write"
        time.sleep(2.0)
    finally:
        release.set()
    deadline = time.monotonic() + 30
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "chain_load hung"
    bad = {r: v.value for r, v in results.items() if type(v) is not gb.Ok}
    assert len(results) == built["world"] and not bad, bad
    _check_outputs(built)
    assert probes <= parent_segments + built["world"], (probes, parent_segments)


def test_all_fit_hot_generation_never_reads_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    built = _build(_case([16] * 16))
    _run_threads(built, built["world"])
    _check_outputs(built)

    parent = built["caches"][0]

    def forbidden(*_args):
        raise AssertionError("terminal-hit load read parent payload")

    parent.read_to_buffer = forbidden

    terminal = built["caches"][1]
    original_probe = terminal.has_bytes
    arrived, release, lock = set(), threading.Event(), threading.Lock()

    def gated_probe(global_range):
        with lock:
            arrived.add(built["cluster"].rank)
            ready = len(arrived) == built["world"]
        if ready:
            release.set()
        assert release.wait(timeout=15), "not every rank reached preflight"
        return original_probe(global_range)

    terminal.has_bytes = gated_probe
    source_reads = len(built["src"].reads)
    _run_threads(built, built["world"])
    _check_outputs(built)
    assert len(built["src"].reads) == source_reads
    table = _api()._slots(built["chain"])
    assert all(
        not state.slots and state.resident == 0
        for (layer, _node), state in table.nodes.items()
        if layer == 0
    )


def test_all_fit_single_rank_terminal_hit_ignores_cold_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    case = _case([16] * 4)
    case.update(nodes=1, layers=[[0], [0]])
    built = _build(case)
    _run_threads(built, 1)
    source_reads = len(built["src"].reads)
    built["caches"][0].mem.clear()
    _run_threads(built, 1)
    _check_outputs(built)
    assert len(built["src"].reads) == source_reads


def test_all_fit_partial_batch_miss_refills_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    built = _build(_case([16] * 16))
    _run_threads(built, built["world"])
    terminal = built["caches"][1]
    del terminal.mem[next(iter(terminal.mem))]
    _run_threads(built, built["world"])
    _check_outputs(built)


def test_all_fit_hot_owner_adopts_parent_for_cold_peer(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "1")
    built = _build(_case([16] * 16))
    _run_threads(built, built["world"])
    plan = _api()._slots(built["chain"]).plan
    key = ("t0", 8, 4)
    assert (0, 0, key) in plan.services[0]
    assert plan.fill_ranks[0, 0, key] == (0,)
    del built["caches"][1].mem[2, "t0"]
    _run_serial(built, (0, 2))


def test_all_fit_empty_drains_still_run_upstream_service(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "1")
    case = _case([1])
    case["nodes"] = 2
    case["layers"] = [[1], [0, 1]]
    built = _build(case)
    plan = _api()._slots(built["chain"]).plan
    assert not plan.drains[1]
    assert any(layer == 0 for layer, _node, _key in plan.services[1])
    _run_threads(built, 2)


def test_all_fit_terminal_service_miss_runs_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "1")
    case = _case([16] * 2)
    case["layers"] = [[0, 1]]
    built = _build(case)
    _run_threads(built, built["world"])
    plan = _api()._slots(built["chain"]).plan
    key = ("t0", 8, 4)
    assert (0, 0, key) in plan.services[0]
    assert 2 not in plan.fill_ranks[0, 0, key]
    built["caches"][0].mem[0, "t0"][1][8:12] = False
    _run_serial(built, (0, 2))


def test_all_fit_serial_calls_keep_service_path(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    built = _build(_case([16] * 8))
    for _round in range(2):
        _run_serial(built, range(built["world"]))


def test_all_fit_hot_calls_allow_delayed_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "5")
    built = _build(_case([16] * 8))
    _run_threads(built, built["world"])
    terminal = built["caches"][1]
    original_probe = terminal.has_bytes
    early_ready, lock = threading.Event(), threading.Lock()
    arrived = set()

    def probe(global_range):
        rank = built["cluster"].rank
        if rank not in (0, 1):
            return original_probe(global_range)
        with lock:
            arrived.add(rank)
            ready = len(arrived) == 2
        if ready:
            early_ready.set()
        return original_probe(global_range)

    def forbidden(*_args):
        raise AssertionError("a delayed full cohort read parent payload")

    terminal.has_bytes = probe
    parent = built["caches"][0]
    parent.read_to_buffer = forbidden
    _reset_outputs(built)
    results = {}

    def one(rank):
        built["cluster"].rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    threads = [threading.Thread(target=one, args=(r,), daemon=True) for r in (0, 1)]
    for thread in threads:
        thread.start()
    assert early_ready.wait(timeout=5), "early ranks never reached preflight"
    time.sleep(0.2)
    late = [threading.Thread(target=one, args=(r,), daemon=True) for r in (2, 3)]
    for thread in late:
        thread.start()
    threads.extend(late)
    deadline = time.monotonic() + 30
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(thread.is_alive() for thread in threads), "chain_load hung"
    assert len(results) == built["world"]
    assert all(type(result) is gb.Ok for result in results.values()), results
    _check_outputs(built)


def test_all_fit_partial_rank_hot_calls_do_not_wait_world(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "5")
    built = _build(_case([16] * 8))
    _run_threads(built, built["world"])
    source_reads = len(built["src"].reads)
    results = {}

    def one(rank):
        built["cluster"].rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    started = time.monotonic()
    _reset_outputs(built)
    threads = [threading.Thread(target=one, args=(r,), daemon=True) for r in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    elapsed = time.monotonic() - started
    assert not any(thread.is_alive() for thread in threads)
    assert elapsed < 3, elapsed
    assert all(type(result) is gb.Ok for result in results.values()), results
    assert len(built["src"].reads) == source_reads
    for rank in (0, 1):
        for tid, got in built["dst"].buffers[rank].items():
            lo, hi = _wants(got.size, built["world"], built["replicate"][tid], rank)
            want = np.full(got.size, GUARD, np.uint8)
            want[lo:hi] = built["data"][tid][lo:hi]
            np.testing.assert_array_equal(got, want, err_msg=f"rank {rank} {tid}")


def test_all_fit_preflight_error_releases_cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    built = _build(_case([16] * 8))
    _run_threads(built, built["world"])

    terminal = built["caches"][1]
    original_probe = terminal.has_bytes
    arrived, release, lock = set(), threading.Event(), threading.Lock()

    def failing_probe(global_range):
        rank = built["cluster"].rank
        if rank == 0:
            assert release.wait(timeout=15), "peers never reached preflight"
            raise OSError("injected terminal probe failure")
        with lock:
            arrived.add(rank)
            ready = len(arrived) == built["world"] - 1
        if ready:
            release.set()
        return original_probe(global_range)

    terminal.has_bytes = failing_probe
    cluster, results = built["cluster"], {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    started = time.monotonic()
    threads = [
        threading.Thread(target=one, args=(r,), daemon=True)
        for r in range(built["world"])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert not any(thread.is_alive() for thread in threads)
    assert time.monotonic() - started < 5, results
    assert len(results) == built["world"]
    assert type(results[0]) is not gb.Ok


@pytest.mark.parametrize("coordinator", ["file", "thread"])
def test_slow_filler_start_without_bump_never_false_wait(
    coordinator, tmp_path, monkeypatch
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", coordinator)
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "1")
    built = _build(_case([16] * 4))
    assert type(built["chain"].check_constraints()) is gb.Ok

    cache = built["caches"][0]
    original = cache.write_from_buffer
    entered, release = threading.Event(), threading.Event()
    claimed = threading.Lock()
    state = {"blocked": False}

    def gated(global_range, write_from_buffer):
        with claimed:
            first = not state["blocked"]
            state["blocked"] = True
        if first:
            entered.set()
            assert release.wait(timeout=15), "test never released the filler"
        return original(global_range, write_from_buffer)

    cache.write_from_buffer = gated
    cluster = built["cluster"]
    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    threads = [
        threading.Thread(target=one, args=(r,), daemon=True)
        for r in range(built["world"])
    ]
    for thread in threads:
        thread.start()
    try:
        assert entered.wait(timeout=15), "no fill reached the first cache write"
        # Hold the in-flight fill past the 1 s wait limit. Peers must treat
        # inflight > 0 as progress and poll, not raise cache_wait.
        time.sleep(2.0)
    finally:
        release.set()
    deadline = time.monotonic() + 30
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "chain_load hung"
    bad = {r: v.value for r, v in results.items() if type(v) is not gb.Ok}
    assert len(results) == built["world"] and not bad, bad
    _check_outputs(built)


def test_all_fit_probe_marks_inflight(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "5")
    built = _build(_case([16] * 8))
    _run_threads(built, built["world"])
    terminal = built["caches"][1]
    original = terminal.has_bytes
    entered, release = threading.Event(), threading.Event()

    def gated(ranges):
        entered.set()
        assert release.wait(timeout=15), "test never released the probe"
        return all(original(global_range) for global_range in ranges)

    terminal.has_bytes_batch = gated
    _reset_outputs(built)
    results = {}

    def load():
        built["cluster"].rank = 0
        results[0] = gio.chain_load(0, built["chain"])

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    assert entered.wait(timeout=15), "hot rank did not probe the terminal cache"
    assert _api()._slots(built["chain"]).inflight > 0
    release.set()
    thread.join(timeout=15)
    assert not thread.is_alive(), "chain_load hung"
    assert type(results[0]) is gb.Ok, results
    assert _api()._slots(built["chain"]).inflight == 0
    _check_outputs(built, {0})


def test_all_fit_probe_stays_inflight_until_completion_is_published(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "5")
    api = _api()
    built = _build(_case([16] * 8))
    _run_threads(built, built["world"])
    chain = built["chain"]
    table = api._slots(chain)
    fill = next(fill for fill in built["caches"][0].fills if fill[1] == 0)
    key = (fill[2], fill[3], fill[4])
    assert key in table.plan.segments[0][0]
    piece = (0, key, key[1], key[2])
    with table.cond:
        state = table.node(0, 0)
        state.slots.clear()
        state.completed.clear()
        state.resident = 0
        table._absent_memo.clear()

    probe = object.__new__(api._RankRun)
    probe.t, probe.plan, probe.rank = table, table.plan, 0
    probe.caches, probe.storage = [chain.small_cache], chain.src
    probe.source_identity = chain.src.data_accessor.cache_identity()
    probe._absent, probe.owner = set(), None
    assert probe.source_identity is not None
    built["cluster"].rank = 0
    assert chain.small_cache.data_accessor.has_bytes(probe._range(key, 0)), (
        key,
        probe.source_identity,
        built["caches"][0].identities,
    )
    waiter = object.__new__(api._RankRun)
    waiter.t, waiter.rank, waiter.gens, waiter.wait = table, 2, {(0, 0): 0}, 0.2

    waiting, probed, release = threading.Event(), threading.Event(), threading.Event()
    original_check = waiter._version_checked
    original_inflight = probe._inflight

    def checked():
        waiting.set()
        return original_check()

    def gated(delta):
        original_inflight(delta)
        if delta < 0:
            probed.set()
            assert release.wait(timeout=5), "probe completion was not released"

    waiter._version_checked = checked
    probe._inflight = gated
    wait_errors, probe_errors, probe_results = [], [], []

    def wait():
        try:
            waiter._wait_layer(0, [(0, 0, key)])
        except BaseException as exc:
            wait_errors.append(exc)

    def run_probe():
        built["cluster"].rank = 0
        try:
            probe_results.append(probe._classify_drains([piece], 0, seal=False))
        except BaseException as exc:
            probe_errors.append(exc)

    wait_thread = threading.Thread(target=wait, daemon=True)
    probe_thread = threading.Thread(target=run_probe, daemon=True)
    wait_thread.start()
    assert waiting.wait(timeout=5), "waiter did not start"
    probe_thread.start()
    try:
        assert probed.wait(timeout=5), "probe did not finish"
        time.sleep(0.4)
    finally:
        release.set()
    wait_thread.join(timeout=5)
    probe_thread.join(timeout=5)
    assert not wait_thread.is_alive() and not probe_thread.is_alive()
    assert not wait_errors, (wait_errors, probe_results, probe_errors, state.completed)
    assert not probe_errors, probe_errors
    assert probe_results == [[piece]]


def test_all_fit_wait_deadline_tracks_any_completed_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    built = _build(_case([16] * 8))
    assert type(built["chain"].check_constraints()) is gb.Ok
    table = api._slots(built["chain"])
    keys = sorted(table.plan.segments[0][0])[:2]
    items = [(0, 0, key) for key in keys]
    run = object.__new__(api._RankRun)
    run.t, run.rank, run.gens, run.wait = table, 0, {(0, 0): 0}, 2
    entered, errors = threading.Event(), []
    original_check = run._version_checked

    def checked():
        entered.set()
        return original_check()

    run._version_checked = checked

    def wait():
        try:
            run._wait_layer(0, items)
        except BaseException as exc:
            errors.append(exc)

    def complete(key):
        seg = table.plan.segments[0][0][key]
        with table.cond:
            state = table.node(0, 0)
            assert state.reserve([seg])
            state.slots[key]["filled"] = True
            state.completed.add(key)
            table.bump()

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    assert entered.wait(timeout=5)
    time.sleep(1.0)
    complete(keys[1])
    time.sleep(1.5)
    complete(keys[0])
    thread.join(timeout=5)
    assert not thread.is_alive(), "layer wait hung"
    assert not errors, errors


def test_independent_memory_caches_do_not_share_coordination(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    first, second = _build(_case([16])), _build(_case([16]))
    assert api._slots(first["chain"]) is not api._slots(second["chain"])


def test_parent_batch_entry_failure_releases_reservations(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    table = api._slots(_build(_case([16] * 4))["chain"])
    seg = next(iter(table.plan.segments[1][0].values()))
    state = table.node(1, 0)
    assert state.reserve([seg])
    run = object.__new__(api._RankRun)
    run.t = table

    def entry_error(_delta):
        raise OSError("injected parent-batch entry failure")

    run._inflight = entry_error
    with pytest.raises(OSError, match="parent-batch entry"):
        run._fill_parent_batch(1, 0, [seg], None)
    assert not state.slots
    assert state.error is not None


def test_coordination_table_lives_until_the_last_view_is_gone(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    case = _case([16])
    case.update(layers=[[0, 1, 2, 3]], steps=[1], width=[1])
    built = _build(case)
    table = api._slots(built["chain"])
    cache_key = next(key for key, value in api._view_slots.items() if value is table)
    first_view = built["chain"].cache
    second_chain = replace(built["chain"], cache=replace(first_view))
    assert second_chain.cache == first_view and second_chain.cache is not first_view
    assert api._slots(second_chain) is table

    del built, first_view
    gc.collect()
    assert api._view_slots.get(cache_key) is table
    del second_chain
    gc.collect()
    assert cache_key not in api._view_slots
    assert cache_key not in api._view_users


def test_chain_fingerprint_ignores_equal_object_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    chain = _build(_case([16]))["chain"]
    layout = copy.deepcopy(chain.src.physical_layout)
    layout.tensor_entries = {
        key: copy.deepcopy(value) for key, value in layout.tensor_entries.items()
    }
    table = copy.deepcopy(chain.src.d_tensor_table)
    other = replace(
        chain,
        src=replace(chain.src, d_tensor_table=table, physical_layout=layout),
    )
    assert table == chain.src.d_tensor_table and table is not chain.src.d_tensor_table
    assert other.src.physical_layout == chain.src.physical_layout
    assert len({id(value) for value in layout.tensor_entries.values()}) == len(
        layout.tensor_entries
    )
    assert api._chain_fingerprint(other) == api._chain_fingerprint(chain)
    assert api._slots(other) is api._slots(chain)


def test_slots_share_active_fingerprint_and_refresh_afterwards(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    chain = _build(_case([16]))["chain"]
    original_fingerprint = api._chain_fingerprint
    original_entry = api._fingerprint_entry
    fingerprint_started = threading.Event()
    all_entered = threading.Event()
    release = threading.Event()
    calls, tables, errors = [], [], []

    def measured_fingerprint(value):
        calls.append(value)
        fingerprint_started.set()
        assert release.wait(timeout=5)
        return original_fingerprint(value)

    def measured_entry(value):
        entry, leader = original_entry(value)
        all_entered.set() if entry.users == 4 else None
        return entry, leader

    def load_slots():
        try:
            tables.append(api._slots(chain))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(api, "_chain_fingerprint", measured_fingerprint)
    monkeypatch.setattr(api, "_fingerprint_entry", measured_entry)
    threads = [threading.Thread(target=load_slots, daemon=True) for _ in range(4)]
    for thread in threads:
        thread.start()
    assert fingerprint_started.wait(timeout=5)
    assert all_entered.wait(timeout=5)
    release.set()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert len(calls) == 1
    assert len(tables) == 4 and all(table is tables[0] for table in tables)
    assert not api._FINGERPRINTS
    assert api._slots(chain) is tables[0]
    assert len(calls) == 2
    assert not api._FINGERPRINTS


def test_content_digest_distinguishes_primitive_subclass_values():
    api = _api()
    assert _FixedReprInt(1) != _FixedReprInt(2)
    assert api._content_digest(_FixedReprInt(1)) != api._content_digest(
        _FixedReprInt(2)
    )


def test_content_digest_distinguishes_primitive_subclass_state():
    api = _api()
    first, second = _TaggedInt(1, "a"), _TaggedInt(1, "b")
    assert not first == second
    assert api._content_digest(first) != api._content_digest(second)
    assert api._content_digest(first) == api._content_digest(_TaggedInt(1, "a"))


def test_content_digest_includes_primitive_reduce_arguments():
    first, second = _ReducedInt(1, "a"), _ReducedInt(1, "b")
    assert not first == second
    assert pickle.dumps(first, protocol=5) != pickle.dumps(second, protocol=5)
    assert _api()._content_digest(first) != _api()._content_digest(second)


def test_content_digest_ignores_generic_object_aliases():
    shared = ["x"]
    first = SimpleNamespace(children=[shared, shared])
    second = SimpleNamespace(children=[["x"], ["x"]])
    assert first == second
    assert pickle.dumps(first, protocol=5) != pickle.dumps(second, protocol=5)
    assert _api()._content_digest(first) == _api()._content_digest(second)


def test_content_digest_includes_container_subclass_state():
    first, second = _TaggedList([1]), _TaggedList([1])
    first.tag, second.tag = "a", "b"
    assert not first == second
    assert _api()._content_digest(first) != _api()._content_digest(second)


def test_content_digest_includes_dataclass_container_reduction():
    first, second = _ReducedList([1], "x"), _ReducedList([2], "x")
    assert not first == second
    assert pickle.dumps(first, protocol=5) != pickle.dumps(second, protocol=5)
    assert _api()._content_digest(first) != _api()._content_digest(second)

    first, second = _ReducedList([1], "a"), _ReducedList([1], "b")
    assert _api()._content_digest(first) != _api()._content_digest(second)


def test_content_digest_normalizes_unordered_subclass_reductions():
    first = defaultdict(list, [("a", [1]), ("b", [2])])
    second = defaultdict(list, [("b", [2]), ("a", [1])])
    assert first == second
    assert pickle.dumps(first, protocol=5) != pickle.dumps(second, protocol=5)
    assert _api()._content_digest(first) == _api()._content_digest(second)

    first_set, second_set = _ReducedSet([1, 2]), _ReducedSet([1, 2])
    first_set.order, second_set.order = [1, 2], [2, 1]
    assert first_set == second_set
    assert pickle.dumps(first_set, protocol=5) != pickle.dumps(second_set, protocol=5)
    assert _api()._content_digest(first_set) == _api()._content_digest(second_set)


def test_content_digest_supports_large_integers():
    api = _api()
    assert api._content_digest(10**4300) != api._content_digest(10**4300 + 1)
    assert api._content_digest(_ReducedInt(10**4300, "a")) != api._content_digest(
        _ReducedInt(10**4300, "b")
    )


def test_content_digest_rejects_unserializable_reductions():
    class Malformed:
        def __reduce_ex__(self, _protocol):
            return None, ()

    class LongReduction:
        def __reduce_ex__(self, _protocol):
            return list, (), None, None, None, None, None

    api = _api()
    with pytest.raises(ValueError, match="not serializable"):
        api._content_digest(lambda: None)
    with pytest.raises(ValueError, match="not serializable"):
        api._content_digest(Malformed())
    with pytest.raises(ValueError, match="not serializable"):
        api._content_digest(LongReduction())
    with pytest.raises(ValueError, match="not serializable"):
        api._content_digest(_StringReduction())


def test_content_digest_bounds_shared_subtree_expansion():
    value = _CountedInt(1)
    for _ in range(8):  # pyrefly: ignore[non-convergent-recursion]
        value = [value, value]
    other = copy.deepcopy(value)
    _CountedInt.calls = 0
    api = _api()
    digest = api._content_digest(value)
    assert _CountedInt.calls == 1
    identity_payload = api._content_digest.__globals__["_identity_payload"]
    assert len(identity_payload(value)) == 32
    assert api._content_digest(other) == digest


def test_content_digest_supports_cycles_without_retaining_aliases():
    value = []
    value.append(value)
    other = copy.deepcopy(value)
    different: list[object] = [1]
    different.append(different)
    api = _api()
    assert api._content_digest(other) == api._content_digest(value)
    assert api._content_digest(different) != api._content_digest(value)


def test_unserializable_source_identity_is_an_invalid_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))

    class Unserializable:
        def __reduce_ex__(self, _protocol):
            raise TypeError("identity cannot be serialized")

    built = _build(_case([16]))
    built["chain"].src.data_accessor.cache_identity = lambda: Unserializable()
    built["cluster"].rank = 0
    result = gio.chain_load(0, built["chain"])
    assert type(result) is gb.Err
    assert getattr(result.value, "argument_name", None) == "cache_identity", result
    assert "not serializable" in result.value.msg
    assert not _api()._FINGERPRINTS

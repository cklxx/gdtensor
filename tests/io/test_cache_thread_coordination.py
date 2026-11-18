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

    def counting(self, ranks=None):
        nonlocal count
        with lock:
            count += 1
        return original(self, ranks)

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


def test_loop_refreshes_deadline_on_peer_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    built = _build(_case([16]))
    table = api._slots(built["chain"])
    run = object.__new__(api._RankRun)
    run.t, run.gens, run.services, run.wait = table, {}, [], 10
    deadlines = []

    def block(_version, deadline, _pending):
        deadlines.append(deadline)
        if len(deadlines) > 1:
            raise StopIteration
        with table.cond:
            table.bump()
        return True

    run._step = lambda _pending: False
    run._block = block
    with pytest.raises(StopIteration):
        run._loop([object()])
    assert len(deadlines) == 2
    assert deadlines[1] > deadlines[0]


def test_in_memory_inflight_counter_avoids_global_state_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    table = api._slots(_build(_case([16]))["chain"])
    run = object.__new__(api._RankRun)
    run.t, run.owner = table, None
    table.cond = None
    barrier = threading.Barrier(8, timeout=3)
    errors = []

    def increment():
        try:
            barrier.wait()
            _ = [run._inflight(1) for _ in range(1000)]
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=increment) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not errors and not any(thread.is_alive() for thread in threads)
    assert table.inflight == 8000
    run._inflight(-8000)
    assert table.inflight == 0


def test_targeted_bump_notifies_only_selected_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    table = _api()._slots(_build(_case([16]))["chain"])
    first, second = list(table.plan.ranks)[:2]
    calls = []
    monkeypatch.setattr(table.cond, "notify_all", lambda: calls.append("all"))
    for rank in (first, second):
        monkeypatch.setitem(
            table.rank_conds,
            rank,
            SimpleNamespace(notify_all=lambda rank=rank: calls.append(rank)),
        )
    with table.cond:
        table.bump({first})
    assert calls == [first]


def test_all_fit_unrelated_version_does_not_leave_bounded_wait(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    built = _build(_case([16]))
    table = api._slots(built["chain"])
    run = object.__new__(api._RankRun)
    run.t, run.last, run.services, run._all_fit = table, 0, [], True
    run.wait_cond = table.cond
    piece = table.plan.drains[0][0]
    calls = []

    def unrelated_notification(predicate, timeout):
        table.version += 1
        calls.append(timeout)
        assert not predicate()
        return False

    def legacy_wait(*_args):
        raise AssertionError("an unrelated version used the legacy wake-up path")

    monkeypatch.setattr(table.cond, "wait_for", unrelated_notification)
    run._wait_deadline = legacy_wait
    assert run._block_locked(0, time.monotonic() + 1, [piece])
    assert len(calls) == 1 and 0 < calls[0] <= 0.05


def test_all_fit_wait_ignores_completed_service_behind_blocked_head(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    built = _build(_case([16]))
    run = object.__new__(_api()._RankRun)
    first, second = ("t0", 0, 4), ("t0", 4, 4)
    run.t = _api()._slots(built["chain"])
    run.last = 0
    run.services = [(0, 0, first), (0, 0, second)]
    run._all_fit = True
    run.t.node(0, 0).completed.add(second)
    assert not run._all_fit_ready([])
    run.t.node(0, 0).completed.add(first)
    assert run._all_fit_ready([])


def test_all_fit_wait_wakes_for_service_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    table = api._slots(_build(_case([16]))["chain"])
    run = object.__new__(api._RankRun)
    node = next(iter(table.plan.segments[1]))
    key = next(iter(table.plan.segments[1][node]))
    seg = table.plan.segments[1][node][key]
    pnode = table.plan.parent[1][node]
    run.t, run.last, run.services, run._all_fit = table, 1, [(1, node, key)], True
    run.plan = table.plan
    assert not run._all_fit_ready([])
    for pkey, _off, _size in seg.parents:
        table.node(0, pnode).completed.add(pkey)
    assert run._all_fit_ready([])
    table.node(1, node).slots[key] = {"filled": False}
    assert not run._all_fit_ready([])


def test_all_fit_wait_predicate_wakes_for_relevant_fill(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    built = _build(_case([16]))
    table = api._slots(built["chain"])
    run = object.__new__(api._RankRun)
    run.t, run.last, run.services, run._all_fit = table, 0, [], True
    run.wait_cond = table.cond
    piece = table.plan.drains[0][0]

    def relevant_fill(predicate, timeout=None):
        table.node(0, piece[0]).completed.add(piece[1])
        assert predicate()
        return True

    monkeypatch.setattr(table.cond, "wait_for", relevant_fill)
    assert run._block_locked(0, time.monotonic() + 1, [piece])


def test_node_generation_tracks_reentry_across_partial_cohorts():
    state = _api()._NodeState(10, {0, 1, 2})

    def enter_and_leave(rank, generation):
        assert state.join(rank) == generation
        state.active -= 1

    enter_and_leave(0, 0)
    enter_and_leave(1, 0)
    enter_and_leave(0, 1)
    enter_and_leave(2, 1)
    # Rank 1's second call matches the current greatest call; it is a late
    # participant in generation 1, not a newer generation.
    enter_and_leave(1, 1)


def test_node_generation_advances_for_a_rank_with_newer_calls():
    state = _api()._NodeState(10, {0, 1})

    def enter_and_leave(rank, generation):
        assert state.join(rank) == generation
        state.active -= 1

    enter_and_leave(0, 0)
    enter_and_leave(0, 1)
    enter_and_leave(1, 1)
    enter_and_leave(1, 2)
    enter_and_leave(0, 3)


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
    original_parent_probe = parent.has_bytes
    parent_probes = 0

    def count_parent_probe(global_range):
        nonlocal parent_probes
        parent_probes += 1
        return original_parent_probe(global_range)

    parent.has_bytes = count_parent_probe

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
    assert parent_probes > 0
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

    def forbidden_parent_probe(*_args):
        raise AssertionError("single-rank terminal-hit load probed the parent")

    built["caches"][0].has_bytes = forbidden_parent_probe
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


def test_all_fit_cold_uses_static_layer_loop():
    run = object.__new__(_api()._RankRun)
    run.source_identity = "source"
    run.rank = 0
    run.plan = SimpleNamespace(ranks=[0, 1], drains={0: []})
    run.services = []
    run.last = 1
    run.closure = [(0, 0, ("t0", 0, 4))]
    run._can_fill = lambda *_args: True
    run._classify_drains = lambda *_args, **_kwargs: []
    calls: list[object] = []
    run._run_all_fit_cold = lambda pending: calls.append(("static", pending))
    run._drain_hot = lambda *_args: calls.append("drain")
    run._run_all_fit()
    assert calls == [("static", [])]


def test_cold_node_deduplicates_parent_waits_and_batches_fill():
    run = object.__new__(_api()._RankRun)
    child_keys = [("t1", 0, 4), ("t1", 4, 4)]
    parent_keys = [("t0", 0, 4), ("t0", 4, 4)]
    state = SimpleNamespace(completed=set(), slots={})
    run.t = SimpleNamespace(
        cond=threading.Condition(), node=lambda _layer, _node: state
    )
    run.plan = SimpleNamespace(
        parent={1: {1: 0}},
        segments={
            1: {
                1: {
                    child_keys[0]: SimpleNamespace(
                        parents=[(parent_keys[0], 0, 4), (parent_keys[1], 0, 4)]
                    ),
                    child_keys[1]: SimpleNamespace(parents=[(parent_keys[1], 0, 4)]),
                }
            }
        },
    )
    run._classify_drains = lambda *_args: []
    run._has_parent_batch = lambda _layer: True
    waits, fills = [], []
    run._wait_cold_slot = lambda _layer, _node, key: waits.append(key)
    run._ensure_items = lambda items: fills.append(items)
    items = [(1, 1, key) for key in child_keys]
    run._fill_cold_node(1, 1, items)
    assert waits == parent_keys
    assert fills == [items]


def test_lower_layer_service_precedes_child_dependency_scan():
    run = object.__new__(_api()._RankRun)
    key = ("t0", 0, 4)
    calls = []
    run._has_parent_batch = lambda layer: layer > 0
    run._ensure = lambda layer, *_args: calls.append(layer) or True

    def unexpected_child_scan(_items):
        raise AssertionError("child dependencies scanned before lower-layer progress")

    run._ensure_parent_batches = unexpected_child_scan
    assert run._ensure_items([(0, 0, key), (1, 0, key)])
    assert calls == [0]


def test_parent_batch_capability_is_checked_once_per_item():
    run = object.__new__(_api()._RankRun)
    key = ("t0", 0, 4)
    items = [(0, 0, key), (1, 0, key)]
    calls = []

    def has_parent_batch(layer):
        calls.append(("capability", layer))
        return layer > 0

    def ensure(layer, *_args):
        calls.append(("ensure", layer))
        return False

    run._has_parent_batch = has_parent_batch
    run._ensure = ensure
    run._ensure_parent_batches = lambda batched: (
        calls.append(("batch", batched)) or False
    )
    assert not run._ensure_items(items)
    assert calls == [
        ("capability", 0),
        ("ensure", 0),
        ("capability", 1),
        ("batch", [items[1]]),
    ]


def test_all_fit_service_step_checks_only_queue_head():
    run = object.__new__(_api()._RankRun)
    first = (0, 0, ("t0", 0, 4))
    second = (0, 0, ("t0", 4, 4))
    run._all_fit = True
    run.services = [first, second]
    run._completed = lambda *_args: False
    calls = []
    run._ensure_items = lambda items: calls.append(items) or True
    assert run._service_step()
    assert calls == [[first]]


def test_all_fit_service_progress_precedes_pending_classification():
    run = object.__new__(_api()._RankRun)
    run._all_fit = True
    calls = []
    run._service_step = lambda: calls.append("service") or True

    def unexpected_classification(_pending, _layer):
        raise AssertionError("pending was classified before service progress")

    run._classify_drains = unexpected_classification
    assert run._step([object()])
    assert calls == ["service"]


def test_single_batch_probe_miss_does_not_probe_scalar_again():
    run = object.__new__(_api()._RankRun)
    key = ("t0", 0, 4)
    run.plan = SimpleNamespace(segments={0: {0: {key: object()}}})
    run._absent = set()
    run._absent_memo_fp = lambda *_args: None
    run._absent_memo_store = lambda *_args: None
    run._range = lambda key, node: (node, key)

    def unexpected_scalar_probe(*_args):
        raise AssertionError("a single batch miss was probed again")

    run._probe_one = unexpected_scalar_probe
    segs = []
    run._probe_node_batch([(0, key)], object(), segs, 0, lambda _ranges: False)
    assert segs == []
    assert run._absent == {(0, 0, key)}


def test_shared_absent_memo_suppresses_repeated_probe():
    run = object.__new__(_api()._RankRun)
    key = ("t0", 0, 4)
    item = (0, 0, key)
    run._absent = set()
    run.t = SimpleNamespace(
        _in_mem=True,
        _absent_memo={item: 4},
        node=lambda _layer, _node: SimpleNamespace(gen=4),
    )
    piece = (0, key, 0, 4)
    seen, probes = set(), []
    run._collect_probe(piece, set(), seen, probes, 0)
    assert probes == []
    assert run._absent == {item}

    run._absent.clear()
    run.t._absent_memo[item] = 3
    seen.clear()
    run._collect_probe(piece, set(), seen, probes, 0)
    assert probes == [(0, key)]
    assert run._absent == set()


def test_parent_readiness_checks_each_parent_once():
    run = object.__new__(_api()._RankRun)
    keys = [("t0", 0, 4), ("t0", 4, 4)]
    state = SimpleNamespace(completed={keys[0]}, slots={keys[1]: {"filled": True}})
    run.t = SimpleNamespace(
        cond=threading.Condition(), node=lambda _layer, _node: state
    )
    run.plan = SimpleNamespace(
        parent={1: {1: 0}},
        segments={0: {0: {key: SimpleNamespace(key=key) for key in keys}}},
    )
    run._adopt = lambda *_args: None
    run._parent_blocks = {}
    calls = []
    original = run._slot_filled

    def checked(*args):
        calls.append(args)
        return original(*args)

    run._slot_filled = checked
    seg = SimpleNamespace(key=("t1", 0, 8), parents=[(key, 0, 4) for key in keys])
    assert run._parents_ready(1, 1, seg)
    assert calls == [(0, 0, key) for key in keys]


def test_parent_readiness_skips_unchanged_nonowned_block():
    run = object.__new__(_api()._RankRun)
    keys = [("t0", 0, 4), ("t0", 4, 4)]
    state = SimpleNamespace(completed=set(), slots={})
    run.t = SimpleNamespace(
        cond=threading.Condition(), node=lambda _layer, _node: state
    )
    run.plan = SimpleNamespace(
        parent={1: {1: 0}},
        segments={0: {0: {key: SimpleNamespace(key=key) for key in keys}}},
    )
    run._all_fit, run._parent_blocks = True, {}
    probes = []
    original = run._slot_filled

    def checked(*args):
        probes.append(args)
        return original(*args)

    run._slot_filled = checked
    adoptions, ownership = [], []
    run._adopt = lambda *_args: adoptions.append(_args)
    run._can_fill = lambda *_args: ownership.append(_args) or False
    seg = SimpleNamespace(key=("t1", 0, 8), parents=[(key, 0, 4) for key in keys])
    assert not run._parents_ready(1, 1, seg)
    counts = len(adoptions), len(ownership)
    assert counts == (2, 1)
    probe_count = len(probes)
    parents = seg.parents
    seg.parents = None
    assert not run._parents_ready(1, 1, seg)
    seg.parents = parents
    assert (len(adoptions), len(ownership)) == counts
    assert len(probes) == probe_count
    state.completed.add(keys[0])
    assert not run._parents_ready(1, 1, seg)
    assert len(adoptions) == 3


def test_non_owner_hot_adoption_precedes_fill_ownership():
    run = object.__new__(_api()._RankRun)
    key = ("t0", 0, 4)
    run.plan = SimpleNamespace(segments={0: {0: {key: object()}}})
    calls = []
    run._room = lambda *_args: True
    run._adopt = lambda *_args: calls.append("adopt") or True

    def forbidden(*_args):
        raise AssertionError("fill ownership was checked before hot adoption")

    run._can_fill = forbidden
    assert run._prepare_fill(0, 0, key) is None
    assert calls == ["adopt"]


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
    probe.source_digest = api._content_digest(probe.source_identity)
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
    waiter.last = 0
    waiter.services = []
    waiter._all_fit = True
    waiter.wait_cond = table.cond

    waiting, probed, release = threading.Event(), threading.Event(), threading.Event()
    original_check = waiter._version_checked
    original_inflight = probe._inflight
    original_reserve = state.reserve
    reservations = []

    def record_reserve(segs):
        reservations.extend(segs)
        return original_reserve(segs)

    state.reserve = record_reserve

    def checked():
        waiting.set()
        return original_check()

    def gated(delta):
        if delta < 0:
            assert key in state.completed, (
                "in-flight ended before completion publication"
            )
        original_inflight(delta)
        if delta < 0:
            probed.set()
            assert release.wait(timeout=5), "probe completion was not released"

    waiter._version_checked = checked
    probe._inflight = gated
    wait_errors, probe_errors, probe_results = [], [], []

    def wait():
        try:
            version = waiter._version_checked()
            waiter._block(version, time.monotonic() + waiter.wait, [piece])
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
    assert not reservations
    assert key not in state.slots and state.resident == 0


def test_independent_memory_caches_do_not_share_coordination(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    first, second = _build(_case([16])), _build(_case([16]))
    assert api._slots(first["chain"]) is not api._slots(second["chain"])


def test_slots_fallback_identity_changes_coordination_table(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    api = _api()
    chain = _build(_case([16]))["chain"]
    first = api._slots(chain)
    monkeypatch.setattr(chain.src.data_accessor, "cache_identity", lambda: "changed")
    assert api._slots(chain) is not first


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


def test_cached_load_reuses_fingerprint_and_identity_per_generation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    built = _build(_case([16] * 4))
    api = _api()
    original_fingerprint = api._chain_fingerprint
    original_entry = api._fingerprint_entry
    original_identity = built["src"].cache_identity
    counts = {"fingerprint": 0, "identity": 0}
    lock = threading.Lock()
    all_entered = threading.Event()

    def measured_entry(chain):
        entry, leader = original_entry(chain)
        if entry.users == built["world"]:
            all_entered.set()
        return entry, leader

    def count_fingerprint(chain):
        with lock:
            counts["fingerprint"] += 1
        assert all_entered.wait(timeout=5)
        return original_fingerprint(chain)

    def count_identity():
        with lock:
            counts["identity"] += 1
        return original_identity()

    monkeypatch.setattr(api, "_chain_fingerprint", count_fingerprint)
    monkeypatch.setattr(api, "_fingerprint_entry", measured_entry)
    monkeypatch.setattr(built["src"], "cache_identity", count_identity)
    for generation in range(1, 3):
        all_entered.clear()
        _run_threads(built, built["world"])
        _check_outputs(built)
        assert counts == {
            "fingerprint": generation,
            "identity": 2 * generation * built["world"],
        }
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

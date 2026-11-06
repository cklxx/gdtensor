"""Thread-coordinator scaling regressions.

The thread coordinator must not wake every waiter for state changes that
make no waiter predicate true, and it must batch one rank's drain scan into
a constant number of lock passes. Both tests are deterministic: the first
counts version bumps, the second blocks one fill past the wait limit and
relies on the in-flight poll, not on wall-clock progress.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.cache_shapes import GUARD, _build, _wants


def _api():
    return sys.modules[gio.chain_load.__module__]


def _run_threads(built, world):
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


def _check_outputs(built):
    for rank, tensors in built["dst"].buffers.items():
        for tid, got in tensors.items():
            lo, hi = _wants(got.size, built["world"], built["replicate"][tid], rank)
            want = np.full(got.size, GUARD, np.uint8)
            want[lo:hi] = built["data"][tid][lo:hi]
            np.testing.assert_array_equal(got, want, err_msg=f"rank {rank} {tid}")


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

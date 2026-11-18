"""Detect capacity/ACK mutants through public IO and real on-disk slots."""

import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.numpy

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_conn.formats import make_storage_view
from tests.support.cache_oracle import (
    FileCache,
    cache_view,
    compute_table,
    storage_schema,
)
from tests.support.oracle import ArrayFabric


_runtime = sys.modules[gio.chain_load.__module__]
_NodeState = _runtime._NodeState
Fail = _runtime.Fail


@pytest.fixture(autouse=True)
def isolated_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path / "coord"))
    (tmp_path / "coord").mkdir()
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "2")


def _chain(root, names, exclusive, steps=1):
    table = compute_table(names, exclusive=exclusive)
    points = {
        rank: {name: _points(rank, i, exclusive) for i, name in enumerate(names)}
        for rank in range(2)
    }
    values = {
        name: np.arange(4, dtype=np.uint8) + i * 16 for i, name in enumerate(names)
    }
    st, layout = storage_schema(names)
    dense = gb.DTensorTable(
        [
            gb.NamedDTensor(
                name,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([4], "U8", 1),
                    [gb.Replicate()],
                    gb.ArrangedMesh(gt.WorldSize(1)),
                ),
            )
            for name in names
        ]
    )
    all_points = {0: {name: np.arange(4) for name in names}}
    full = ArrayFabric(dense, all_points, values)
    saved = make_storage_view(names, st, layout, str(root / "checkpoint"))
    assert type(gio.rank_save(0, full.view(0), saved)) is gb.Ok
    decoded = safetensors.numpy.load_file(root / "checkpoint" / "model.safetensors")
    for name in names:
        np.testing.assert_array_equal(decoded[name], values[name])
    expected = {
        name: {r: pts[name] for r, pts in points.items() if len(pts[name])}
        for name in names
    }
    cache = FileCache(root / "cache", expected)
    dst = ArrayFabric(table, points)
    stored = make_storage_view(names, st, layout, str(root / "checkpoint"))
    chain = gb.OneLayerCachedDTensorsReadChain(
        stored, cache_view(names, cache, steps=steps), dst.view(0), num_ranks_per_node=2
    )
    return chain, dst, cache, values


def _points(rank, i, exclusive):
    if exclusive:
        return np.arange(4) if rank == i else np.array([], dtype=np.int64)
    return np.arange(rank * 2, (rank + 1) * 2)


def _run(chain, dst, cache, rank):
    cache.rank = rank
    concrete = gb.OneLayerCachedDTensorsReadChain(
        chain.src, chain.cache, dst.view(rank), num_ranks_per_node=2
    )
    return gio.chain_load(rank, concrete)


def _outputs(dst, values):
    dst.assert_values(values)
    for rank, tensors in dst.buffers.items():
        for name, data in tensors.items():
            indices = dst.points[rank][name]
            wanted = values[name].reshape(-1)[indices].tobytes()
            assert zlib.crc32(data.tobytes()) == zlib.crc32(wanted)


def test_cell_budget_sums_all_tensors_of_a_cell(tmp_path):
    """C=1 cell budget is the (step, slot) cell summed over every tensor.
    'a' and 'b' share cell (0, 0): both may be resident together, and the
    full read chain completes without the budget pinning one tensor out."""
    chain, dst, cache, values = _chain(tmp_path, ["a", "b"], True)
    release = threading.Event()
    cache.release, cache.block_rank = release, 0
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        first = pool.submit(_run, chain, dst, cache, 0)
        assert cache.entered.wait(5), "rank0 did not reach file-cache read"
        second = pool.submit(_run, chain, dst, cache, 1)
        # While rank0 is blocked on its file-cache read, 'a' (4B) is resident.
        # The (0, 0) cell budget sums both tensors (4+4=8B), so the concurrent
        # fill of 'b' is admitted: two files coexist (peak 2), unlike the old
        # per-tensor budget that pinned it out.
        assert cache.second_write.wait(1), "cell budget should admit the second tensor"
        assert cache.peak == 2, cache.writes
        release.set()
        assert type(first.result(timeout=10)) is gb.Ok
        assert type(second.result(timeout=10)) is gb.Ok
    finally:
        release.set()
        pool.shutdown(wait=True)
    assert len(cache.writes) == 2
    assert not cache.slots
    _outputs(dst, values)


def test_ack_releases_slot_only_after_every_cohort_member(tmp_path, monkeypatch):
    """Shared file slot: first ACK must not delete the slot needed by rank1."""
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "0.5")
    chain, dst, cache, values = _chain(tmp_path, ["w"], False)
    assert type(_run(chain, dst, cache, 0)) is gb.Ok
    assert len(cache.slots) == 1
    assert type(_run(chain, dst, cache, 1)) is gb.Ok
    assert len(cache.writes) == 1, "early release forced a second disk fill"
    assert not cache.slots
    _outputs(dst, values)


def test_node_state_ack_matches_exact_piece_model():
    key = ("w", 0, 4)
    pieces = {(("rank", 0), 0, 2), (("rank", 1), 2, 2)}
    owed = set(pieces)
    seg = SimpleNamespace(key=key, size=4, owed=lambda: set(owed))
    state = _NodeState(4, {0, 1})
    assert state.reserve([seg])
    model = set(owed)
    sequence = [
        (("rank", 9), 0, 4),
        (("rank", 0), 0, 2),
        (("rank", 0), 0, 2),
        (("rank", 9), 2, 2),
        (("rank", 1), 2, 2),
    ]
    for piece in sequence:
        known = piece in model
        model.discard(piece)
        expected_release = known and not model
        assert state.ack(key, piece) == expected_release
        assert state.slots.get(key, {}).get("owed", set()) == model
        assert state.resident == (0 if expected_release else 4)
    with pytest.raises(Fail) as caught:
        state.ack(key, pieces.pop())
    assert caught.value.error.check_name == "cache_ack"
    assert state.resident == 0


def test_completed_segment_cannot_be_reserved_again():
    key = ("w", 0, 4)
    seg = SimpleNamespace(key=key, size=4, owed=lambda: {(("rank", 0), 0, 4)})
    state = _NodeState(4, {0})
    state.completed.add(key)
    assert not state.reserve([seg])
    assert state.resident == 0
    assert not state.slots


def test_previous_generation_child_does_not_ack_current_parent():
    key = ("w", 0, 4)
    consumer = ("seg", 1, 1, key)
    piece = (consumer, 0, 4)
    seg = SimpleNamespace(key=key, node=0, consumers={consumer: [(0, 4)]})
    parent = _NodeState(4, set())
    parent.started = 10
    child = _NodeState(4, set())
    child.completed.add(key)
    child.completed_at[key] = 5
    states = {(0, 0): parent, (1, 1): child}
    run = object.__new__(_runtime._RankRun)
    run.t = SimpleNamespace(node=lambda layer, node: states[layer, node])
    reserved = SimpleNamespace(key=key, size=4, owed=lambda: {piece})
    assert parent.reserve([reserved])
    run._settle_completed_children(0, 0, seg)
    assert parent.slots[key]["owed"] == {piece}
    assert parent.resident == 4


def test_unknown_ack_does_not_release_consumer_free_slot():
    key = ("w", 0, 4)
    seg = SimpleNamespace(key=key, size=4, owed=set)
    state = _NodeState(4, set())
    assert state.reserve([seg])
    assert not state.ack(key, (("rank", 9), 0, 4))
    assert state.resident == 4
    assert key in state.slots

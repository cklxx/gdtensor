"""Detect capacity/ACK mutants through public IO and real on-disk slots."""

import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import safetensors.numpy

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_sim.formats import make_storage_view
from tests.support.cache_oracle import (
    FileCache,
    cache_view,
    compute_table,
    storage_schema,
)
from tests.support.oracle import ArrayFabric


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


def test_ack_releases_slot_only_after_every_cohort_member(tmp_path):
    """Shared file slot: first ACK must not delete the slot needed by rank1."""
    chain, dst, cache, values = _chain(tmp_path, ["w"], False)
    assert type(_run(chain, dst, cache, 0)) is gb.Ok
    assert len(cache.slots) == 1
    assert type(_run(chain, dst, cache, 1)) is gb.Ok
    assert len(cache.writes) == 1, "early release forced a second disk fill"
    assert not cache.slots
    _outputs(dst, values)

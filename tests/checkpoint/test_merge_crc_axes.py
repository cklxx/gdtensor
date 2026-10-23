"""Multi-checkpoint merge with CRC and arange across PP/DP/EP/TP.

Checkpoints A, A+3, A+9 (int64 arange, 11 tensors incl. 5 experts, an
optimizer reshaped tensor, a single element and an empty tensor) are saved
on a source (PP, DP, EP, TP) grid into per-rank multi-file layouts, repacked
into the other format on a merge grid, averaged by the caller, saved again
and consolidated into one file on a third grid.

Each storage boundary checks payload CRC, exact bytes and one stored copy
per element against a dense oracle. A negative case swaps two expert
payloads of equal size across files; CRC and loaded values must both fail.
Ranks are threads; numeric merging is the caller's arithmetic.
"""

import zlib

import numpy as np
import pytest

import gdtensor.io as gio
from tests.support.compat import rank_reshard
import gdtensor.builder as gb
from gdtensor_sim.coord import ThreadCoordinator
from tests.support import axes
from tests.support.checkpoint import file_bytes
from tests.support.oracle import ArrayFabric
from tests.support.schemas import payload_regions, run_ranks
from tests.support.storage import FileStorage

# (label, source grid, merge grid, final grid); each crosses EP.
CASES = [
    ("EP", (1, 1, 3, 1), (1, 1, 2, 1), (1, 1, 4, 1)),
    ("EP_TP", (1, 1, 2, 2), (1, 1, 3, 2), (1, 1, 1, 3)),
    ("PP_EP", (2, 1, 2, 1), (3, 1, 2, 1), (2, 1, 3, 1)),
    ("DP_EP_TP", (1, 2, 2, 2), (1, 3, 2, 1), (1, 1, 2, 3)),
    ("PP_DP_EP_TP", (2, 2, 2, 2), (2, 1, 3, 2), (3, 1, 2, 1)),
]
SHIFTS = (0, 3, 9)


def _storage(root, topology, fmt, kind, *, save=False):
    table, points, _, stages = axes.schema(topology, storage=True)
    world = int(np.prod(topology))
    coord = ThreadCoordinator(world) if save else None
    return FileStorage(
        root, table, axes.flat_points(points), world, fmt, kind, stages, coord
    ).view


def _verify(label, root, topology, fmt, values, *, files=None):
    _, points, _, _ = axes.schema(topology, storage=True)
    seen = {tid: np.zeros(v.size, dtype=np.int64) for tid, v in values.items()}
    payloads = {}
    for tid, slot, _, data, path, _ in payload_regions(root, fmt):
        indices = np.arange(values[tid].size) if slot is None else points[slot][tid]
        expected = values[tid].ravel()[indices].copy().tobytes()
        assert zlib.crc32(data) == zlib.crc32(expected), (label, tid, slot)
        assert data == expected, (label, tid, slot)
        seen[tid][indices] += 1
        payloads.setdefault(path.name, bytearray()).extend(data)
    for tid, counts in seen.items():
        np.testing.assert_array_equal(counts, np.ones_like(counts), err_msg=tid)
    if files is not None:
        assert len(payloads) == files, (label, sorted(payloads))
    crcs = {n: f"{zlib.crc32(d):08x}" for n, d in sorted(payloads.items())[:4]}
    print(
        f"[CRC+ARANGE][AXES][{label}][PASS] grid={topology} format={fmt} "
        f"files={len(payloads)} crc_sample={crcs}",
        flush=True,
    )


def _with_shift(values, shift):
    return {tid: v + shift for tid, v in values.items()}


def _save(root, topology, fmt, values):
    table, points, _, _ = axes.schema(topology)
    source = ArrayFabric(table, points, values)
    view = _storage(root, topology, fmt, "rank_multi", save=True)
    run_ranks(int(np.prod(topology)), lambda r: gio.rank_save(r, source.view(r), view))
    _verify("SAVE", root, topology, fmt, values)
    return _storage(root, topology, fmt, "rank_multi")


def _repack_and_load(root, stored, topology, fmt, values):
    target = _storage(root, topology, fmt, "per_stage", save=True)
    run_ranks(int(np.prod(topology)), lambda r: rank_reshard(r, stored, target))
    _verify("REPACK", root, topology, fmt, values)
    table, points, _, _ = axes.schema(topology)
    loaded = ArrayFabric(table, points)
    readable = _storage(root, topology, fmt, "per_stage")
    before = file_bytes(root)
    run_ranks(
        int(np.prod(topology)), lambda r: gio.rank_load(r, loaded.view(r), readable)
    )
    loaded.assert_values(values)
    assert file_bytes(root) == before
    return loaded


def _accumulate(sums, fabric, weight):
    for (rank, tid), data in _buffers(fabric):
        local = data.view(np.int64).astype(np.float64) * weight
        sums[rank, tid] = sums.get((rank, tid), 0) + local


def _write_mean(fabric, sums, denominator):
    for (rank, tid), data in _buffers(fabric):
        data[:] = (sums[rank, tid] / denominator).astype(np.int64).view(np.uint8)


def _buffers(fabric):
    return [
        ((rank, tid), data)
        for rank, tensors in fabric.buffers.items()
        for tid, data in tensors.items()
    ]


def _log_experts(phase, topology, fabric):
    for tid in (axes.SPECS[4][0], axes.SPECS[5][0]):
        owners = [r for r, t in fabric.points.items() if t[tid].size]
        sample = {r: fabric.buffers[r][tid].view(np.int64).tolist() for r in owners[:2]}
        print(
            f"[CRC+ARANGE][AXES][{phase}] grid={topology} tensor={tid} "
            f"owners={owners} local={sample}",
            flush=True,
        )


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
@pytest.mark.parametrize(
    "weights,shift", [((1, 1, 1), 4), ((1, 2, 3), 5.5)], ids=["mean", "weighted"]
)
@pytest.mark.parametrize("label,src,mid,dst", CASES, ids=[c[0] for c in CASES])
def test_axes_checkpoint_merge_crc(tmp_path, fmt, weights, shift, label, src, mid, dst):
    other = "distcp" if fmt == "safetensors" else "safetensors"
    _, _, base, _ = axes.schema(src)
    expected = {
        tid: (v.astype(np.float64) + shift).astype(np.int64) for tid, v in base.items()
    }
    print(
        f"[CRC+ARANGE][AXES][START] {label} grids={src}->{mid}->{dst} axes=PP,DP,EP,TP "
        f"weights={weights} expected=A+{shift}",
        flush=True,
    )
    sums, originals = {}, {}
    mid_table, mid_points, _, _ = axes.schema(mid)
    for epoch, add in enumerate(SHIFTS):
        values = _with_shift(base, add)
        root = tmp_path / f"in{epoch}"
        stored = _save(root, src, fmt if epoch % 2 == 0 else other, values)
        originals[root] = file_bytes(root)
        loaded = _repack_and_load(
            tmp_path / f"mid{epoch}",
            stored,
            mid,
            other if epoch % 2 == 0 else fmt,
            values,
        )
        if epoch == 0:
            _log_experts("LOADED_ON_MERGE_GRID", mid, loaded)
        _accumulate(sums, loaded, weights[epoch])
    merged = ArrayFabric(mid_table, mid_points)
    _write_mean(merged, sums, sum(weights))
    merged.assert_values(expected)
    out = tmp_path / "merged"
    view = _storage(out, mid, fmt, "rank_multi", save=True)
    run_ranks(int(np.prod(mid)), lambda r: gio.rank_save(r, merged.view(r), view))
    _verify("SAVE_MERGED", out, mid, fmt, expected)
    final_root = tmp_path / "final"
    target = _storage(final_root, dst, other, "single", save=True)
    readable = _storage(out, mid, fmt, "rank_multi")
    run_ranks(int(np.prod(dst)), lambda r: rank_reshard(r, readable, target))
    _verify("CONSOLIDATE", final_root, dst, other, expected, files=1)
    final_table, final_points, _, _ = axes.schema(dst)
    final = ArrayFabric(final_table, final_points)
    final_view = _storage(final_root, dst, other, "single")
    run_ranks(int(np.prod(dst)), lambda r: gio.rank_load(r, final.view(r), final_view))
    final.assert_values(expected)
    _log_experts("FINAL", dst, final)
    for root, contents in originals.items():
        assert file_bytes(root) == contents, root.name
    print(f"[CRC+ARANGE][AXES][CASE][PASS] {label} weights={weights}", flush=True)


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_axes_swapped_expert_payloads_fail_crc_and_values(tmp_path, fmt):
    topology = (2, 1, 2, 2)
    _, _, values, _ = axes.schema(topology)
    stored = _save(tmp_path, topology, fmt, values)
    regions = list(payload_regions(tmp_path, fmt))
    first = next(r for r in regions if ".experts.0." in r[0])
    second = next(
        r
        for r in regions
        if ".experts.2." in r[0] and len(r[3]) == len(first[3]) and r[4] != first[4]
    )
    assert first[3] != second[3]
    for region, data in ((first, second[3]), (second, first[3])):
        with region[4].open("r+b") as file:
            file.seek(region[5])
            file.write(data)
    with pytest.raises(AssertionError):
        _verify("CORRUPT", tmp_path, topology, fmt, values)
    table, points, _, _ = axes.schema((1, 1, 1, 1))
    loaded = ArrayFabric(table, points)
    assert type(gio.rank_load(0, loaded.view(0), stored)) is gb.Ok
    with pytest.raises(AssertionError):
        loaded.assert_values(values)
    print(
        f"[CRC+ARANGE][AXES][CORRUPTION][PASS] swapped {first[0]}<->{second[0]} "
        f"crc=detected values=detected",
        flush=True,
    )


def test_merge_cases_cross_ep_with_other_axes():
    for label, *grids in CASES:
        assert all(len(g) == 4 for g in grids)
        assert any(g[2] > 1 for g in grids), label
    assert any(src[0] > 1 and src[2] > 1 for _, src, _, _ in CASES)

"""Tiny TP/DP/EP/PP checkpoint tests using only public core APIs.

Rank axes are (PP, DP, EP, TP). Dense weights replicate over DP and EP;
expert e lives on EP coordinate e % EP and may shard over TP. PP assigns
whole tensors to stages. Optimizer state also shards over DP. These tests
move weights and optimizer state; they do not run training or MoE routing.
"""

import itertools
import zlib

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_sim.coord import ThreadCoordinator
from tests.support.checkpoint import (
    corrupt_distcp_payload,
    file_bytes,
    payload_file_bytes,
    repack_via_compute,
)
from tests.support.oracle import ArrayFabric
from tests.support.schemas import payload_regions, run_ranks
from tests.support.storage import FileStorage

CASES = [
    ("TP", (1, 1, 1, 3), (1, 1, 1, 2)),
    ("DP", (1, 3, 1, 1), (1, 2, 1, 1)),
    ("EP", (1, 1, 3, 1), (1, 1, 2, 1)),
    ("PP", (3, 1, 1, 1), (2, 1, 1, 1)),
    ("TP_DP", (1, 2, 1, 3), (1, 3, 1, 2)),
    ("TP_EP", (1, 1, 2, 3), (1, 1, 3, 2)),
    ("DP_EP", (1, 2, 3, 1), (1, 3, 2, 1)),
    ("PP_TP", (2, 1, 1, 3), (3, 1, 1, 2)),
    ("PP_DP", (2, 3, 1, 1), (3, 2, 1, 1)),
    ("PP_EP", (2, 1, 3, 1), (3, 1, 2, 1)),
    ("DP_EP_TP", (1, 2, 2, 2), (1, 2, 2, 3)),
    ("PP_DP_TP", (2, 2, 1, 2), (2, 3, 1, 2)),
    ("PP_DP_EP", (2, 2, 2, 1), (3, 1, 3, 1)),
    ("PP_EP_TP", (2, 1, 2, 2), (3, 1, 2, 2)),
    ("PP_DP_EP_TP", (2, 2, 2, 2), (2, 1, 3, 2)),
]


def test_parallel_mode_matrix_covers_every_nonempty_axis_combination():
    expected = {
        axes for axes in itertools.product((False, True), repeat=4) if any(axes)
    }
    actual = {tuple(size > 1 for size in source) for _, source, _ in CASES}
    assert len(CASES) == len(actual) == 15
    assert actual == expected


SPECS = (
    [
        ("layers.0.column", (3, 5), 0, None, "column"),
        ("layers.1.row", (3, 5), 1, None, "row"),
        ("layers.2.norm", (5,), 2, None, "replicate"),
        ("layers.3.optimizer", (3, 5), 3, None, "optimizer"),
    ]
    + [
        (
            f"layers.{expert % 4}.experts.{expert}.weight",
            (2, 3),
            expert % 4,
            expert,
            "row",
        )
        for expert in range(5)
    ]
    + [
        ("layers.0.tiny", (1,), 0, None, "column"),
        ("layers.1.empty", (0, 3), 1, None, "column"),
    ]
)


def _placements(kind, shape, transpose):
    axis = (int(kind == "row") + int(transpose)) % len(shape)
    return [
        gb.Replicate(),
        gb.Shard(0) if kind == "optimizer" else gb.Replicate(),
        gb.Replicate(),
        gb.Replicate() if kind == "replicate" else gb.Shard(axis),
    ]


def _mesh(topology, stage, expert, placements, order, storage):
    world = int(np.prod(topology))
    base = (
        gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
        )
        if storage
        else gb.ArrangedMesh(gt.WorldSize(world))
    )
    slices = [gb.Slice(stage, stage + 1), None, None, None]
    if expert is not None:
        owner = expert % topology[2]
        slices[2] = gb.Slice(owner, owner + 1)
    for axis in (1, 2, 3):
        if storage and slices[axis] is None and type(placements[axis]) is gb.Replicate:
            owner = order % topology[axis]
            slices[axis] = gb.Slice(owner, owner + 1)
    return gb.SlicedMesh(gb.ReshapedMesh(base, list(topology)), slices)


def _dense_points(shape, topology, rank, stage, expert, placements, order, storage):
    coords = tuple(int(i) for i in np.unravel_index(rank, topology))
    if coords[0] != stage or expert is not None and coords[2] != expert % topology[2]:
        return np.array([], dtype=np.int64)
    points = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    for axis in (1, 2, 3):
        fixed = axis == 2 and expert is not None
        size, coord = (1, 0) if fixed else (topology[axis], coords[axis])
        placement = placements[axis]
        if type(placement) is gb.Replicate and storage and coord != order % size:
            return np.array([], dtype=np.int64)
        if type(placement) is gb.Shard:
            points = _dense_chunk(points, placement.axis, size, coord)
    return points.ravel()


def _dense_chunk(points, axis, size, coord):
    chunk = (points.shape[axis] + size - 1) // size
    selection = [slice(None)] * points.ndim
    selection[axis] = slice(coord * chunk, (coord + 1) * chunk)
    return points[tuple(selection)]


def _schema(topology, *, storage=False, transpose=False):
    table, values, stages = [], {}, []
    points = {rank: {} for rank in range(int(np.prod(topology)))}
    for order, (tid, shape, layer, expert, kind) in enumerate(SPECS):
        naive_shape = (int(np.prod(shape)),) if kind == "optimizer" else shape
        placements = _placements(kind, naive_shape, transpose)
        stage = layer % topology[0]
        mesh = _mesh(topology, stage, expert, placements, order, storage)
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta(list(naive_shape), "I64", 8), placements, mesh
        )
        generic = gb.ReshapedDTensor(dt, list(shape)) if kind == "optimizer" else dt
        table.append(gb.NamedDTensor(tid, generic))
        values[tid] = (
            np.arange(int(np.prod(shape)), dtype=np.int64) + order * 1000
        ).reshape(shape)
        stages.append(stage)
        _rank_points(
            points,
            tid,
            naive_shape,
            topology,
            stage,
            expert,
            placements,
            order,
            storage,
        )
    return gb.DTensorTable(table), points, values, stages


def _rank_points(
    points, tid, shape, topology, stage, expert, placements, order, storage
):
    for rank in points:
        points[rank][tid] = _dense_points(
            shape, topology, rank, stage, expert, placements, order, storage
        )


def _storage(root, topology, fmt, kind, *, save=False, transpose=False):
    table, points, _, stages = _schema(topology, storage=True, transpose=transpose)
    world = int(np.prod(topology))
    flat_points = {
        (tid, rank): indices
        for rank, tensors in points.items()
        for tid, indices in tensors.items()
    }
    return FileStorage(
        root,
        table,
        flat_points,
        world,
        fmt,
        kind,
        stages,
        ThreadCoordinator(world) if save else None,
    ).view


def _verify(root, topology, fmt, *, transpose=False):
    _, points, values, _ = _schema(topology, storage=True, transpose=transpose)
    seen = {tid: np.zeros(value.size, dtype=np.int64) for tid, value in values.items()}
    for tid, slot, _, data, _, _ in payload_regions(root, fmt):
        indices = np.arange(values[tid].size) if slot is None else points[slot][tid]
        expected = values[tid].ravel()[indices].copy().tobytes()
        assert zlib.crc32(data) == zlib.crc32(expected), (tid, slot)
        assert data == expected, (tid, slot)
        seen[tid][indices] += 1
    for counts in seen.values():
        np.testing.assert_array_equal(counts, np.ones_like(counts))


def _save(root, topology, fmt, kind, *, transpose=False):
    table, points, values, _ = _schema(topology, transpose=transpose)
    source = ArrayFabric(table, points, values)
    view = _storage(root, topology, fmt, kind, save=True, transpose=transpose)
    assert (
        type(source.view(0).check_constraints())
        is type(view.check_constraints())
        is gb.Ok
    )
    run_ranks(
        int(np.prod(topology)),
        lambda rank: gio.rank_save(rank, source.view(rank), view),
    )
    _verify(root, topology, fmt, transpose=transpose)
    source.assert_values(values)
    return source, _storage(root, topology, fmt, kind, transpose=transpose)


def _load(view, topology, values, *, transpose=False):
    table, points, _, _ = _schema(topology, transpose=transpose)
    target = ArrayFabric(table, points)
    run_ranks(
        int(np.prod(topology)),
        lambda rank: gio.rank_load(rank, target.view(rank), view),
    )
    target.assert_values(values)
    return target


@pytest.mark.parametrize("label,src_p,dst_p", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("kind", ["rank_multi", "per_stage"])
@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_parallel_mode_checkpoint_repartition_matches_serial(
    tmp_path, label, src_p, dst_p, kind, fmt
):
    print(
        f"[CRC+ARANGE][PARALLEL_MODES][START] mode={label} axes=PP,DP,EP,TP source={src_p} target={dst_p} tensors={len(SPECS)} layout={kind} format={fmt}",
        flush=True,
    )
    source, stored = _save(tmp_path / "source", src_p, fmt, kind)
    before = file_bytes(tmp_path / "source")
    table, points, values, _ = _schema(dst_p, transpose=True)
    loaded = _load(stored, dst_p, values, transpose=True)
    _log_expert_placement("SOURCE", src_p, source)
    _log_expert_placement("LOADED", dst_p, loaded)
    other = "distcp" if fmt == "safetensors" else "safetensors"
    output = _storage(
        tmp_path / "repartitioned", dst_p, other, "single", save=True, transpose=True
    )
    reshaped = repack_via_compute(
        int(np.prod(dst_p)), stored, output, table, points, values
    )
    reshaped.assert_values(values)
    _verify(tmp_path / "repartitioned", dst_p, other, transpose=True)
    readable = _storage(
        tmp_path / "repartitioned", dst_p, other, "single", transpose=True
    )
    _load(readable, dst_p, values, transpose=True)
    _assert_serial_files(tmp_path, readable, other)
    assert file_bytes(tmp_path / "source") == before
    source.assert_values(values)
    expert = SPECS[4][0]
    print(
        f"[CRC+ARANGE][PARALLEL_MODES][PASS] mode={label} checks=all_rank_bytes,crc,coverage_once,elastic_load,two_phase_repack,serial_file_bytes,source_unchanged expert={expert} expected={values[expert].ravel().tolist()}",
        flush=True,
    )


def _assert_serial_files(root, source, fmt):
    before = file_bytes(root / "repartitioned")
    serial = (1, 1, 1, 1)
    _save(root / "serial", serial, fmt, "single")
    target = _storage(root / "canonical", serial, fmt, "single", save=True)
    table, points, serial_values, _ = _schema(serial)
    repack_via_compute(1, source, target, table, points, serial_values)
    assert payload_file_bytes(root / "canonical") == payload_file_bytes(root / "serial")
    assert file_bytes(root / "repartitioned") == before


def _log_expert_placement(phase, topology, fabric):
    tid = SPECS[6][0]
    owners = [rank for rank, tensors in fabric.points.items() if tensors[tid].size]
    sample = {
        rank: {
            "global_indices": fabric.points[rank][tid].tolist(),
            "local_values": fabric.buffers[rank][tid].view(np.int64).tolist(),
        }
        for rank in owners[:4]
    }
    print(
        f"[CRC+ARANGE][PARALLEL_MODES][{phase}] tensor={tid} "
        f"pp={2 % topology[0]} ep={2 % topology[2]} owners={owners} sample={sample}",
        flush=True,
    )


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_swapped_expert_payloads_are_detected(tmp_path, fmt):
    topology = (2, 2, 2, 2)
    _, stored = _save(tmp_path, topology, fmt, "rank_multi")
    if fmt == "safetensors":
        regions = list(payload_regions(tmp_path, fmt))
        first = next(row for row in regions if ".experts.0." in row[0])
        second = next(
            row
            for row in regions
            if ".experts.1." in row[0] and len(row[3]) == len(first[3])
        )
        assert first[4] != second[4] and first[3] != second[3]
        _replace_payload(first, second[3])
        _replace_payload(second, first[3])
        fault = f"swap_expert_payloads first={first[0]} second={second[0]}"
    else:
        # Real torch DCP has no per-region offsets; shift one expert tensor.
        corrupt_distcp_payload(tmp_path, ".experts.0.")
        fault = "expert_tensor_shifted"
    with pytest.raises(AssertionError):
        _verify(tmp_path, topology, fmt)
    table, points, values, _ = _schema((1, 1, 1, 1))
    loaded = ArrayFabric(table, points)
    assert type(gio.rank_load(0, loaded.view(0), stored)) is gb.Ok
    with pytest.raises(AssertionError):
        loaded.assert_values(values)
    print(
        f"[CRC+ARANGE][PARALLEL_MODES][EXPECTED_MISMATCH] fault={fault} crc=detected loaded_values=detected",
        flush=True,
    )


def _replace_payload(region, data):
    with region[4].open("r+b") as file:
        file.seek(region[5])
        assert file.write(data) == len(data)

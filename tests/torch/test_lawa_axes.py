"""LAWA on real torch ranks across PP/DP/EP/TP, including topology changes.

Each test spawns real CPU/gloo processes. The world is a (PP, DP, EP, TP)
grid. Every tensor is a native torch.DTensor on the sub-mesh that owns it:
its PP stage, and for experts its EP owner. Ranks outside that sub-mesh
hold an empty local shard, which is how PyTorch represents non-members.

Flow per case:
1. Save k checkpoints from the source grid (rank_save, per-stage files).
2. Load them on the target grid (a different topology), average with
   naive sum and Welford through get_d_tensor / reset_d_tensor.
3. Save the mean to a third layout (one file), reload, and compare every
   rank's local values, then decode the file and check each element once.
Expected values come from a dense numpy mean of the arange checkpoints,
independent of gdtensor ownership code.
"""

import multiprocessing as mp
import itertools
import os
import time
import zlib
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

import gdtensor
import gdtensor.builder as gb
import gdtensor.io as gio
from gdtensor_sim.coord import ProcessCoordinator
from tests.support import axes
from tests.support.oracle import TorchAccessor
from tests.support.schemas import payload_regions
from tests.support.storage import FileStorage

K_VALUES = (1, 2, 3)
EPOCHS = 3

# (label, source grid, target grid); worlds stay at most 16 real processes.
LAWA_CASES = [
    ("PP", (2, 1, 1, 1), (2, 1, 1, 1)),
    ("EP", (1, 1, 2, 1), (1, 1, 2, 1)),
    ("TP_to_DP", (1, 1, 1, 4), (1, 2, 1, 2)),
    ("PP_TP", (2, 1, 1, 2), (2, 1, 1, 2)),
    ("PP_EP", (2, 1, 2, 1), (2, 1, 2, 1)),
    ("EP_TP", (1, 1, 2, 2), (1, 1, 2, 2)),
    ("PP_DP_TP_reshape", (2, 2, 1, 2), (2, 1, 1, 4)),
    ("PP_DP_EP_TP", (2, 2, 2, 2), (2, 2, 2, 2)),
    ("PP_EP_TP_to_PP_DP_EP", (2, 1, 2, 2), (2, 2, 2, 1)),
]


def epoch_values(epoch):
    """Float64 arange checkpoints; epoch e is base * (e + 1) + e^2 - 3e."""
    values = {}
    for order, (tid, shape, _, _, _) in enumerate(axes.SPECS):
        values[tid] = axes.tensor_values(
            order,
            shape,
            np.float64,
            scale=(epoch + 1) / 8,
            shift=epoch * epoch - 3 * epoch,
        )
    return values


def expected_mean(k):
    return {
        tid: np.mean([epoch_values(e)[tid] for e in range(k)], axis=0)
        for tid in epoch_values(0)
    }


def _ok(result):
    assert type(result) is gb.Ok, result
    return result.value


def _storage(root, topology, kind, coord):
    table, points, _, stages = axes.schema(topology, storage=True, dtype="F64")
    return FileStorage(
        root,
        table,
        axes.flat_points(points),
        int(np.prod(topology)),
        "safetensors",
        kind,
        stages,
        coord,
    ).view


_MESHES = {}


def _torch_mesh(torch, topology, stage, expert):
    """One DeviceMesh per (grid, stage, expert owner), built once per process.

    Building a DeviceMesh is a collective over the default group, so every
    rank must build the same meshes in the same order; caching keeps the
    number of collectives fixed instead of growing with every view.
    """
    from torch.distributed.device_mesh import DeviceMesh

    owner = None if expert is None else axes.expert_owner(expert, topology)
    key = (tuple(topology), stage, owner)
    if key not in _MESHES:
        ranks = np.arange(int(np.prod(topology))).reshape(topology)
        sub = ranks[stage : stage + 1]
        if owner is not None:
            sub = sub[:, :, owner : owner + 1]
        _MESHES[key] = DeviceMesh(
            "cpu", torch.tensor(sub.copy()), mesh_dim_names=axes.AXES
        )
    return _MESHES[key]


def _torch_placements(places):
    from torch.distributed.tensor import Replicate, Shard

    return [Shard(p.axis) if type(p) is gb.Shard else Replicate() for p in places]


def compute_view(torch, topology, rank, values):
    """One rank's compute view; every tensor is a native torch.DTensor."""
    from torch.distributed.tensor import DTensor

    nds, points = [], {}
    for order, (tid, shape, layer, expert, kind) in enumerate(axes.SPECS):
        flat = axes.naive_shape(shape, kind)
        places = axes.placements(kind, flat)
        stage = axes.stage_of(layer, topology)
        mesh = _torch_mesh(torch, topology, stage, expert)
        indices = axes.dense_points(
            flat, topology, rank, stage, expert, places, order, False
        )
        dense = torch.from_numpy(np.ascontiguousarray(values[tid]).reshape(-1))
        local = dense[torch.from_numpy(indices)].reshape(
            _local_shape(flat, topology, rank, stage, expert, places)
        )
        dt = DTensor.from_local(
            local.contiguous(),
            mesh,
            _torch_placements(places),
            shape=torch.Size(int(x) for x in flat),
            stride=tuple(int(np.prod(flat[i + 1 :])) for i in range(len(flat))),
        )
        gd = gdtensor.make_gd_tensor(dt)
        generic = gb.ReshapedDTensor(gd, list(shape)) if kind == "optimizer" else gd
        nds.append(gb.NamedDTensor(tid, generic))
        points[tid] = indices
    view = gb.ComputeDTensorsView(
        [s[0] for s in axes.SPECS], gb.DTensorTable(nds), None
    )
    view.data_accessor = TorchAccessor(view, points, torch)
    return view


def _local_shape(flat, topology, rank, stage, expert, places):
    coords = axes.coords_of(rank, topology)
    if coords[0] != stage or (
        expert is not None and coords[2] != axes.expert_owner(expert, topology)
    ):
        return (0,)
    shape = list(flat)
    for axis in (1, 2, 3):
        if type(places[axis]) is gb.Shard:
            size = 1 if axis == 2 and expert is not None else topology[axis]
            coord = 0 if axis == 2 and expert is not None else coords[axis]
            dim = places[axis].axis
            chunk = (shape[dim] + size - 1) // size
            shape[dim] = max(0, min(chunk, shape[dim] - coord * chunk))
    return tuple(shape)


def _nan_view(torch, topology, rank):
    nan = {tid: np.full(shape, np.nan) for tid, shape, *_ in axes.SPECS}
    return compute_view(torch, topology, rank, nan)


def _accumulate(accumulator, current, step, method):
    for tid in gdtensor.get_tensor_ids(current):
        source = gdtensor.get_d_tensor(_ok(gdtensor.get_gd_tensor(current, tid)))
        target = _ok(gdtensor.get_gd_tensor(accumulator, tid))
        previous = gdtensor.get_d_tensor(target)
        merged = (
            previous + source
            if method == "naive"
            else previous + (source - previous) / step
        )
        gdtensor.reset_d_tensor(target, merged)


def _divide(view, divisor):
    for tid in gdtensor.get_tensor_ids(view):
        target = _ok(gdtensor.get_gd_tensor(view, tid))
        gdtensor.reset_d_tensor(target, gdtensor.get_d_tensor(target) / divisor)


def _local_check(view, rank, expected, topology):
    for order, (tid, shape, layer, expert, kind) in enumerate(axes.SPECS):
        dt = gdtensor.get_d_tensor(_ok(gdtensor.get_gd_tensor(view, tid)))
        local = dt.to_local().contiguous().reshape(-1).numpy()
        indices = view.data_accessor.points[tid]
        want = expected[tid].reshape(-1)[indices]
        np.testing.assert_allclose(
            local,
            want,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"rank={rank} tensor={tid} grid={topology}",
        )


def _say(stream, rank, message):
    if rank == 0:
        print(f"[lawa-axes rank=0] {message}", file=stream, flush=True)


def _average_once(torch, rank, root, label, src, dst, coords, stream, k, method):
    zeros = {tid: np.zeros(shape) for tid, shape, *_ in axes.SPECS}
    acc = compute_view(torch, dst, rank, zeros)
    for epoch in range(k):
        current = _nan_view(torch, dst, rank)
        source = _storage(root / f"in{epoch}", src, "per_stage", None)
        _ok(gio.rank_load(rank, current, source))
        _accumulate(acc, current, epoch + 1, method)
    if method == "naive":
        _divide(acc, k)
    out = root / f"{method}{k}"
    _ok(gio.rank_save(rank, acc, _storage(out, dst, "single", coords["dst"])))
    back = _nan_view(torch, dst, rank)
    _ok(gio.rank_load(rank, back, _storage(out, dst, "single", None)))
    _local_check(back, rank, expected_mean(k), dst)
    gd = _ok(gdtensor.get_gd_tensor(back, axes.SPECS[4][0]))
    sample = gdtensor.get_d_tensor(gd).to_local().reshape(-1).tolist()
    _say(
        stream,
        rank,
        f"{label} {src}->{dst} method={method} k={k} reload PASS expert0_local={sample}",
    )


def run_rank(torch, rank, root, label, src, dst, coords, stream, epochs, k_values):
    for epoch in range(epochs):
        view = compute_view(torch, src, rank, epoch_values(epoch))
        _ok(
            gio.rank_save(
                rank,
                view,
                _storage(root / f"in{epoch}", src, "per_stage", coords["src"]),
            )
        )
    _say(stream, rank, f"{label} saved {epochs} checkpoints on source grid {src}")
    for k in k_values:
        for method in ("naive", "welford"):
            _average_once(torch, rank, root, label, src, dst, coords, stream, k, method)


def _worker(rank, init_file, root, label, src, dst, coords, epochs, k_values):
    import torch
    import torch.distributed as dist

    stream = os.fdopen(os.dup(1), "w", buffering=1)
    log = open(Path(root) / f"rank{rank}.log", "wb", buffering=0)
    os.dup2(log.fileno(), 2)
    torch.set_num_threads(1)
    world = int(np.prod(src))
    _init = getattr(dist, "init_process_group")
    _destroy = getattr(dist, "destroy_process_group")
    _init(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )
    try:
        run_rank(
            torch,
            rank,
            Path(root),
            label,
            src,
            dst,
            coords,
            stream,
            epochs,
            k_values,
        )
    finally:
        _destroy()
        log.close()
        stream.close()


def check_files(root, dst, k, method):
    """Decode the averaged single-file checkpoint; each element exactly once."""
    _, points, _, _ = axes.schema(dst, storage=True, dtype="F64")
    expected = expected_mean(k)
    seen = {tid: np.zeros(v.size, dtype=np.int64) for tid, v in expected.items()}
    crcs = {}
    for tid, slot, _, data, _, _ in payload_regions(
        root / f"{method}{k}", "safetensors"
    ):
        indices = np.arange(expected[tid].size) if slot is None else points[slot][tid]
        actual = np.frombuffer(data, dtype=np.float64)
        np.testing.assert_allclose(
            actual, expected[tid].reshape(-1)[indices], rtol=1e-12, atol=1e-12
        )
        seen[tid][indices] += 1
        crcs[tid] = zlib.crc32(data)
    for tid, counts in seen.items():
        np.testing.assert_array_equal(counts, np.ones_like(counts), err_msg=tid)
    return crcs


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
@pytest.mark.parametrize("label,src,dst", LAWA_CASES, ids=[c[0] for c in LAWA_CASES])
def test_lawa_four_axes_real_ranks(tmp_path, label, src, dst):
    _run_case(tmp_path, label, src, dst, EPOCHS, K_VALUES, 240)


@pytest.mark.parametrize(
    "label,src,dst",
    [("EP_TP", (1, 1, 2, 2), (1, 1, 2, 2))],
    ids=["EP_TP_k10"],
)
def test_lawa_k10_four_axes_real_ranks(tmp_path, label, src, dst):
    # N5: 10 checkpoints on a four-axis (EP x TP) grid.
    _run_case(tmp_path, label, src, dst, 10, (1, 10), 300)


def _run_case(tmp_path, label, src, dst, epochs, k_values, join_seconds):
    pytest.importorskip("torch")
    world = int(np.prod(src))
    assert world == int(np.prod(dst)), "one torch job serves both grids"
    ctx = mp.get_context("spawn")
    coords = {
        "src": ProcessCoordinator(world, ctx, timeout=60),
        "dst": ProcessCoordinator(world, ctx, timeout=60),
    }
    print(
        f"[lawa-axes] START {label} source={src} target={dst} world={world} "
        f"tensors={len(axes.SPECS)} epochs={epochs} k={k_values}",
        flush=True,
    )
    procs = [
        ctx.Process(
            target=_worker,
            args=(
                r,
                str(tmp_path / "pg"),
                str(tmp_path),
                label,
                src,
                dst,
                coords,
                epochs,
                k_values,
            ),
        )
        for r in range(world)
    ]
    deadline = time.monotonic() + join_seconds
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(max(0, deadline - time.monotonic()))
        logs = {
            p.name: p.read_text(errors="replace")[-6000:]
            for p in tmp_path.glob("rank*.log")
        }
        assert all(p.exitcode == 0 for p in procs), logs
        for k, m in itertools.product(k_values, ("naive", "welford")):
            check_files(tmp_path, dst, k, m)
        crcs = check_files(tmp_path, dst, k_values[-1], "naive")
        print(
            f"[lawa-axes] PASS {label} files decoded, every element once; "
            f"crc(expert0)={crcs[axes.SPECS[4][0]]:08x}",
            flush=True,
        )
    finally:
        _stop(procs)


def _stop(procs):
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(5)


def test_lawa_cases_cover_pp_ep_and_topology_changes():
    labels = {label for label, src, dst in LAWA_CASES}
    used = set()
    for _, src, dst in LAWA_CASES:
        used |= {axes.AXES[i] for i, n in enumerate(src) if n > 1}
    assert {"pp", "dp", "ep", "tp"} <= used
    assert any(src != dst for _, src, dst in LAWA_CASES)
    assert any(src[0] > 1 and src[2] > 1 for _, src, _ in LAWA_CASES), labels

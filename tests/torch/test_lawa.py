"""LAWA through public IO and bridge functions on real torch ranks."""

import multiprocessing as mp
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

import gdtensor
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_sim.coord import ProcessCoordinator
from tests.support.oracle import TorchAccessor
from tests.support.storage import FileStorage


def _ok(result):
    assert type(result) is gb.Ok, result
    return result.value


def _checkpoint_values(torch, tensor_count):
    base = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) - 5,
        "other": torch.arange(6, dtype=torch.float64).reshape(2, 3) / 7,
        "bias": torch.tensor([-3.5], dtype=torch.float32),
    }
    for i in range(tensor_count - len(base)):
        shape = (i % 5 + 1, i % 7 + 1) if i % 3 else (i % 11 + 1,)
        dtype = torch.float64 if i % 2 else torch.float32
        base[f"layer{i:03d}.weight"] = (
            torch.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) / 13 + i * 17
        )
    return [
        {tid: value * (i + 1) + (i * i - 3 * i) for tid, value in base.items()}
        for i in range(5)
    ]


def _check_saved_average(torch, root, fmt, values, world):
    for k in (1, 2, 3, 5):
        expected = {
            tid: torch.stack([v[tid] for v in values[:k]]).mean(0) for tid in values[0]
        }
        for method in ("naive", "welford"):
            _check_output(torch, root / f"{method}{k}", fmt, expected, world)
            print(f"[lawa] decoded files verified method={method} k={k}", flush=True)


def _check_output(torch, path, fmt, expected, world):
    from safetensors.numpy import load_file

    if fmt == "safetensors":
        decoded = load_file(str(path / "model.safetensors"))
        assert set(decoded) == set(expected)
        for tid, value in expected.items():
            torch.testing.assert_close(torch.from_numpy(decoded[tid].copy()), value)
        return
    from tests.support.schemas import payload_regions

    seen = set()
    for tid, _slot, _info, data, _p, _lo in payload_regions(path, fmt):
        assert tid not in seen, tid
        seen.add(tid)
        value = expected[tid]
        dtype = np.float64 if value.dtype == torch.float64 else np.float32
        actual = np.frombuffer(data, dtype=dtype).reshape(tuple(value.shape))
        torch.testing.assert_close(torch.from_numpy(actual.copy()), value)
    assert seen == set(expected)


def _storage(root, values, fmt, coord, world):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta(
                        list(value.shape),
                        "F64" if value.element_size() == 8 else "F32",
                        value.element_size(),
                    ),
                    [gb.Replicate(), gb.Shard(0)],
                    mesh,
                ),
            )
            for tid, value in values.items()
        ]
    )
    points = {}
    for tid, value in values.items():
        indices = np.arange(value.numel()).reshape(tuple(value.shape))
        rows = (value.shape[0] + world - 1) // world
        for rank in range(world):
            points[tid, rank] = indices[rank * rows : (rank + 1) * rows].ravel()
    return FileStorage(root, table, points, world, fmt, coord=coord).view


def _view(torch, mesh, values, rank):
    from torch.distributed.tensor import DTensor, Replicate, Shard

    nds, points = [], {}
    coords = np.unravel_index(
        mesh.mesh.flatten().tolist().index(rank), tuple(mesh.shape)
    )
    for tid, value in values.items():
        if value.ndim == 2:
            placements = [Shard(1)] if mesh.ndim == 1 else [Shard(0), Shard(1)]
        else:
            placements = [Replicate()] * mesh.ndim
        local, indices = _local_shard(value, mesh, coords, placements, Shard)
        dt = DTensor.from_local(
            local.contiguous(),
            mesh,
            placements,
            shape=value.shape,
            stride=value.stride(),
        )
        gd = gdtensor.make_gd_tensor(dt)
        nds.append(gb.NamedDTensor(tid, gd))
        points[tid] = indices.ravel()
    view = gb.ComputeDTensorsView(list(values), gb.DTensorTable(nds), None)
    view.data_accessor = TorchAccessor(view, points, torch)
    return view


def _local_shard(value, mesh, coords, placements, shard_type):
    indices = np.arange(value.numel()).reshape(tuple(value.shape))
    local = value.clone()
    for size, coord, placement in zip(mesh.shape, coords, placements):
        if isinstance(placement, shard_type):
            axis = placement.dim
            chunk = (local.shape[axis] + size - 1) // size
            index = [slice(None)] * value.ndim
            index[axis] = slice(coord * chunk, (coord + 1) * chunk)
            indices = indices[tuple(index)]
            local = local[tuple(index)]
    return local, indices


def _progress(stream, rank, message):
    if rank == 0:
        print(f"[lawa rank=0] {message}", file=stream, flush=True)


def _dump_values(stream, rank, label, values):
    if rank != 0:
        return
    for tid in list(values)[:3]:
        tensor = values[tid]
        print(
            f"[lawa rank=0] {label} tensor={tid} shape={list(tensor.shape)} values={tensor.tolist()}",
            file=stream,
            flush=True,
        )


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


def _average_checkpoints(
    torch, mesh, root, fmt, coord, world, rank, values, method, k, progress
):
    base = values[0]
    accumulator = _view(
        torch, mesh, {tid: torch.zeros_like(value) for tid, value in base.items()}, rank
    )
    for index, tensors in enumerate(values[:k]):
        current = _view(
            torch,
            mesh,
            {
                tid: torch.full_like(value, float("nan"))
                for tid, value in tensors.items()
            },
            rank,
        )
        _ok(
            gio.rank_load(
                rank,
                current,
                _storage(root / f"input{index}", tensors, fmt, None, world),
            )
        )
        _accumulate(accumulator, current, index + 1, method)
        _progress(
            progress, rank, f"method={method} k={k} merged_checkpoint={index + 1}/{k}"
        )
    if method == "naive":
        _divide(accumulator, k)
    output = root / f"{method}{k}"
    _ok(gio.rank_save(rank, accumulator, _storage(output, base, fmt, coord, world)))
    reloaded = _view(
        torch,
        mesh,
        {tid: torch.full_like(value, float("nan")) for tid, value in base.items()},
        rank,
    )
    _ok(gio.rank_load(rank, reloaded, _storage(output, base, fmt, None, world)))
    _verify_local_mean(torch, reloaded, values[:k], rank, progress, method, k)


def _divide(view, divisor):
    for tid in gdtensor.get_tensor_ids(view):
        target = _ok(gdtensor.get_gd_tensor(view, tid))
        gdtensor.reset_d_tensor(target, gdtensor.get_d_tensor(target) / divisor)


def _verify_local_mean(torch, view, checkpoints, rank, progress, method, k):
    expected_values = {}
    for tid in checkpoints[0]:
        tensor = gdtensor.get_d_tensor(_ok(gdtensor.get_gd_tensor(view, tid)))
        expected = torch.stack([checkpoint[tid] for checkpoint in checkpoints]).mean(0)
        indices = view.data_accessor.points[tid]
        torch.testing.assert_close(
            tensor.to_local().contiguous().reshape(-1), expected.reshape(-1)[indices]
        )
        expected_values[tid] = expected
        if tid in list(checkpoints[0])[:3]:
            _progress(
                progress,
                rank,
                f"method={method} k={k} tensor={tid} actual_local={tensor.to_local().reshape(-1).tolist()} expected_local={expected.reshape(-1)[indices].tolist()}",
            )
    _dump_values(
        progress, rank, f"method={method} k={k} expected_mean", expected_values
    )
    _progress(progress, rank, f"method={method} k={k} saved_and_reloaded=PASS")


def _check_reset(torch, mesh, base, rank, ranks, mesh_shape, names):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate

    view = _view(torch, mesh, base, rank)
    target = _ok(gdtensor.get_gd_tensor(view, "weight"))
    original = gdtensor.get_d_tensor(target)
    replicated = DTensor.from_local(
        base["weight"].clone(),
        mesh,
        [Replicate()] * mesh.ndim,
        shape=base["weight"].shape,
        stride=base["weight"].stride(),
    )
    for bad in (original.double(), replicated):
        with pytest.raises(Exception, match="dtype differs|placements differ"):
            gdtensor.reset_d_tensor(target, bad)
        assert gdtensor.get_d_tensor(target) is original
    different = DeviceMesh(
        "cpu", ranks.flip(0).reshape(mesh_shape), mesh_dim_names=names
    )
    bad = gdtensor.get_d_tensor(
        _ok(gdtensor.get_gd_tensor(_view(torch, different, base, rank), "weight"))
    )
    with pytest.raises(Exception, match="mesh.*differs"):
        gdtensor.reset_d_tensor(target, bad)


def _run_worker(
    torch, rank, root, fmt, dim, reverse, coord, world, tensor_count, progress
):
    from torch.distributed.device_mesh import DeviceMesh

    mesh_shape = (2, world // 2) if dim == "dp_tp" else (world,)
    names = ("dp", "tp") if dim == "dp_tp" else (dim,)
    ranks = torch.arange(world - 1, -1, -1) if reverse else torch.arange(world)
    mesh = DeviceMesh("cpu", ranks.reshape(mesh_shape), mesh_dim_names=names)
    values = _checkpoint_values(torch, tensor_count)
    _progress(
        progress,
        rank,
        f"initialized world={world} tensors={tensor_count} mesh={mesh_shape}",
    )
    for index, tensors in enumerate(values):
        _dump_values(progress, rank, f"checkpoint={index + 1}", tensors)
        view = _view(torch, mesh, tensors, rank)
        _ok(
            gio.rank_save(
                rank, view, _storage(root / f"input{index}", tensors, fmt, coord, world)
            )
        )
        _progress(progress, rank, f"checkpoint={index + 1} saved")
    for k in (1, 2, 3, 5):
        for method in ("naive", "welford"):
            _average_checkpoints(
                torch, mesh, root, fmt, coord, world, rank, values, method, k, progress
            )
    _check_reset(torch, mesh, values[0], rank, ranks, mesh_shape, names)
    _progress(progress, rank, "dtype/placement/mesh rejection=PASS")


def _worker(rank, init_file, root, fmt, dim, reverse, coord, world, tensor_count):
    import resource

    import torch
    import torch.distributed as dist

    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    progress = os.fdopen(os.dup(1), "w", buffering=1)
    log = open(Path(root) / f"rank{rank}.log", "wb", buffering=0)
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        torch.set_num_threads(1)
        _distributed_worker(
            torch,
            dist,
            rank,
            init_file,
            Path(root),
            fmt,
            dim,
            reverse,
            coord,
            world,
            tensor_count,
            progress,
        )
    finally:
        log.close()
        progress.close()


def _distributed_worker(
    torch,
    dist,
    rank,
    init_file,
    root,
    fmt,
    dim,
    reverse,
    coord,
    world,
    tensor_count,
    progress,
):
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    try:
        _run_worker(
            torch, rank, root, fmt, dim, reverse, coord, world, tensor_count, progress
        )
    finally:
        dist.destroy_process_group()


def _stop_workers(processes):
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
@pytest.mark.parametrize(
    "fmt,dim,reverse,world,tensor_count",
    [
        ("safetensors", "dp", False, 2, 3),
        ("distcp", "tp", False, 2, 3),
        ("safetensors", "dp", True, 2, 32),
        ("distcp", "dp_tp", False, 4, 32),
        ("safetensors", "dp_tp", True, 8, 64),
        ("distcp", "dp_tp", False, 16, 16),
        ("distcp", "dp_tp", False, 32, 16),
    ],
)
def test_lawa_real_ranks(tmp_path, fmt, dim, reverse, world, tensor_count):
    torch = pytest.importorskip("torch")
    ctx = mp.get_context("spawn")
    coord = ProcessCoordinator(world, ctx, timeout=30)
    procs = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                str(tmp_path / "pg"),
                str(tmp_path),
                fmt,
                dim,
                reverse,
                coord,
                world,
                tensor_count,
            ),
        )
        for rank in range(world)
    ]
    print(
        f"[lawa] START format={fmt} mesh={dim} reverse={reverse} world={world} tensors={tensor_count}",
        flush=True,
    )
    print(f"[lawa] checkpoint directory={tmp_path}", flush=True)
    # Hosted GitHub runners are slower and more contended than the dedicated
    # digest host; 32 spawn processes plus per-rank gloo checkpointing need a
    # larger join budget there. A deadlocked rank still fails once the deadline
    # passes, so this only removes a flaky false timeout, not the assertion.
    budget = int(os.environ.get("GDTENSOR_LAWA_JOIN_SECONDS", "180"))
    deadline = time.monotonic() + budget
    try:
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(max(0, deadline - time.monotonic()))
        logs = {
            p.name: p.read_text(errors="replace")[-8000:]
            for p in tmp_path.glob("rank*.log")
        }
        exits = {rank: proc.exitcode for rank, proc in enumerate(procs)}
        assert all(code == 0 for code in exits.values()), {"exits": exits, "logs": logs}
        _check_saved_average(
            torch, tmp_path, fmt, _checkpoint_values(torch, tensor_count), world
        )
    finally:
        _stop_workers(procs)

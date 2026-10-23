"""Exercise ownership and byte offsets through public IO, across small worlds."""

import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.oracle import ArrayFabric
from tests.support.schemas import run_ranks, storage_view, tensor_schema, verify_crc

CONFIGS = [p for p in itertools.product(range(1, 9), repeat=3) if int(np.prod(p)) <= 8]
LAYOUTS = [
    ("single", "hf"),
    ("hf", "hf"),
    ("per_rank", "mirror"),
    ("rank_multi", "mirror"),
    ("per_stage", "hf"),
    ("single", "mirror"),
    ("per_rank", "hf"),
]


@pytest.mark.parametrize("p", CONFIGS)
@pytest.mark.parametrize("kind,mode", LAYOUTS)
@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_io_world_layout_matrix(tmp_path, p, kind, mode, fmt):
    world = int(np.prod(p))
    ct, points, values = tensor_schema(p)
    source = ArrayFabric(ct, points, values)
    saved = storage_view(tmp_path, ct, p, kind, mode, fmt, save=True)
    run_ranks(world, lambda rank: gio.rank_save(rank, source.view(rank), saved))
    verify_crc(tmp_path, fmt, ct, p, mode, values)
    target_p = (p[0], 1, p[1] * p[2])
    target, target_points, _ = tensor_schema(target_p, column=True)
    read_view = storage_view(tmp_path, ct, p, kind, mode, fmt)
    loaded = ArrayFabric(target, target_points)
    run_ranks(world, lambda rank: gio.rank_load(rank, loaded.view(rank), read_view))
    loaded.assert_values(values)
    source.assert_values(values)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_save_load_roundtrip_is_idempotent(tmp_path, world):
    p = (1, 1, world)
    ct, points, values = tensor_schema(p)
    fabric = ArrayFabric(ct, points, values)
    for i in range(2):
        root = tmp_path / f"round{i}"
        saved = storage_view(root, ct, p, "single", "hf", "safetensors", save=True)
        run_ranks(world, lambda rank: gio.rank_save(rank, fabric.view(rank), saved))
        reread = ArrayFabric(ct, points)
        read_view = storage_view(root, ct, p, "single", "hf", "safetensors")
        run_ranks(world, lambda rank: gio.rank_load(rank, reread.view(rank), read_view))
        reread.assert_values(values)


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_storage_fixture_roundtrip_rewrites_same_bytes(tmp_path, fmt):
    p = (1, 2, 2)
    ct, points, values = tensor_schema(p)
    fabric = ArrayFabric(ct, points, values)
    sv = storage_view(tmp_path, ct, p, "single", "hf", fmt, save=True)
    run_ranks(4, lambda rank: gio.rank_save(rank, fabric.view(rank), sv))
    reread = ArrayFabric(ct, points)
    run_ranks(
        4,
        lambda rank: gio.rank_load(
            rank, reread.view(rank), storage_view(tmp_path, ct, p, "single", "hf", fmt)
        ),
    )
    rewritten = storage_view(tmp_path, ct, p, "single", "hf", fmt, save=True)
    run_ranks(4, lambda rank: gio.rank_save(rank, reread.view(rank), rewritten))
    verify_crc(tmp_path, fmt, ct, p, "hf", values)


@pytest.mark.parametrize("reverse", [False, True])
def test_rank_start_order_does_not_change_checkpoint(tmp_path, reverse):
    p = (2, 2, 2)
    ct, points, values = tensor_schema(p)
    source = ArrayFabric(ct, points, values)
    ranks = list(reversed(range(8))) if reverse else list(range(8))
    outputs = []
    for directory in ("first", "second"):
        root = tmp_path / directory
        view = storage_view(root, ct, p, "hf", "hf", "safetensors", save=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda rank: gio.rank_save(rank, source.view(rank), view), ranks
                )
            )
        assert all(type(r) is gb.Ok for r in results), results
        outputs.append({f.name: f.read_bytes() for f in root.iterdir()})
    assert outputs[0] == outputs[1]


def test_empty_tensor_moves_no_bytes(tmp_path):
    p = (1, 1, 2)
    ct, points, values = tensor_schema(p)
    empty = ct.named_d_tensors[-1]
    ct = gb.DTensorTable([empty])
    tid = empty.global_tensor_id
    points = {rank: {tid: arrays[tid]} for rank, arrays in points.items()}
    values = {tid: values[tid]}
    source = ArrayFabric(ct, points, values)
    saved = storage_view(tmp_path, ct, p, "single", "hf", "safetensors", save=True)
    run_ranks(2, lambda rank: gio.rank_save(rank, source.view(rank), saved))
    dest = ArrayFabric(ct, points)
    run_ranks(
        2,
        lambda rank: gio.rank_load(
            rank,
            dest.view(rank),
            storage_view(tmp_path, ct, p, "single", "hf", "safetensors"),
        ),
    )
    dest.assert_values(values)

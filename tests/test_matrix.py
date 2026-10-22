"""Exercise ownership and byte offsets through public IO, across small worlds."""

import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tensor_oracle import ArrayFabric
from test_data_correctness import _run, _schema, _storage, _verify_crc

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
    ct, points, values = _schema(p)
    source = ArrayFabric(ct, points, values)
    saved = _storage(tmp_path, ct, p, kind, mode, fmt, save=True)
    _run(world, lambda rank: gio.rank_save(rank, source.view(rank), saved))
    _verify_crc(tmp_path, fmt, ct, p, mode, values)
    target_p = (p[0], 1, p[1] * p[2])
    target, target_points, _ = _schema(target_p, column=True)
    read_view = _storage(tmp_path, ct, p, kind, mode, fmt)
    loaded = ArrayFabric(target, target_points)
    _run(world, lambda rank: gio.rank_load(rank, loaded.view(rank), read_view))
    loaded.assert_values(values)
    converted = ArrayFabric(target, target_points)
    _run(
        world,
        lambda rank: gio.rank_reshard(
            rank, source.view(rank, allow_remote_reads=True), converted.view(rank)
        ),
    )
    converted.assert_values(values)
    source.assert_values(values)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_compute_reshard_is_idempotent(world):
    p = (1, 1, world)
    ct, points, values = _schema(p)
    fabric = ArrayFabric(ct, points, values)
    for _ in range(2):
        _run(
            world,
            lambda rank: gio.rank_reshard(rank, fabric.view(rank), fabric.view(rank)),
        )
        fabric.assert_values(values)


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_storage_fixture_reshard_to_same_view(tmp_path, fmt):
    p = (1, 2, 2)
    ct, points, values = _schema(p)
    fabric = ArrayFabric(ct, points, values)
    sv = _storage(tmp_path, ct, p, "single", "hf", fmt, save=True)
    _run(4, lambda rank: gio.rank_save(rank, fabric.view(rank), sv))
    _run(4, lambda rank: gio.rank_reshard(rank, sv, sv))
    _verify_crc(tmp_path, fmt, ct, p, "hf", values)


@pytest.mark.parametrize("reverse", [False, True])
def test_rank_start_order_does_not_change_checkpoint(tmp_path, reverse):
    p = (2, 2, 2)
    ct, points, values = _schema(p)
    source = ArrayFabric(ct, points, values)
    ranks = list(reversed(range(8))) if reverse else list(range(8))
    outputs = []
    for directory in ("first", "second"):
        root = tmp_path / directory
        view = _storage(root, ct, p, "hf", "hf", "safetensors", save=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda rank: gio.rank_save(rank, source.view(rank), view), ranks
                )
            )
        assert all(type(r) is gt.Ok for r in results), results
        outputs.append({f.name: f.read_bytes() for f in root.iterdir()})
    assert outputs[0] == outputs[1]


def test_empty_tensor_moves_no_bytes(tmp_path):
    p = (1, 1, 2)
    ct, points, values = _schema(p)
    empty = ct.named_d_tensors[-1]
    ct = gb.DTensorTable([empty])
    tid = empty.global_tensor_id
    points = {rank: {tid: arrays[tid]} for rank, arrays in points.items()}
    values = {tid: values[tid]}
    source = ArrayFabric(ct, points, values)
    dest = ArrayFabric(ct, points)
    _run(
        2,
        lambda rank: gio.rank_reshard(
            rank, source.view(rank, allow_remote_reads=True), dest.view(rank)
        ),
    )
    dest.assert_values(values)

"""Independent dense-slice oracle for random shapes and rank permutations."""

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.oracle import ArrayFabric, slice_points
from tests.support.schemas import run_ranks, storage_view, verify_crc


def _random_table(seed):
    rng = np.random.default_rng(seed)
    world = int(rng.choice([1, 2, 3, 4, 6, 8]))
    p = (1, 1, world)
    ranks = rng.permutation(world).tolist()
    mesh = gb.ReshapedMesh(gb.LiteralMesh(ranks, gt.WorldSize(world)), list(p))
    points = {rank: {} for rank in range(world)}
    table, values = [], {}
    for order in range(5):
        shape = tuple(int(x) for x in rng.integers(1, 10, size=int(rng.integers(1, 4))))
        axis = int(rng.integers(len(shape)))
        placements = [gt.Replicate(), gt.Replicate(), gt.Shard(axis)]
        tid = f"tensor{order}"
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta(list(shape), "I64", 8),
            placements,
            gb.SlicedMesh(mesh, [gb.Slice(0, 1), None, None]),
        )
        table.append(gb.NamedDTensor(tid, dt))
        values[tid] = (
            np.arange(int(np.prod(shape)), dtype=np.int64) + order * 10000
        ).reshape(shape)
        for logical, rank in enumerate(ranks):
            points[rank][tid] = slice_points(shape, p, placements, 0, logical)
    return p, gb.DTensorTable(table), points, values


@pytest.mark.parametrize("seed", range(32))
@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_random_shapes_and_permuted_ranks(tmp_path, seed, fmt):
    p, table, points, values = _random_table(seed)
    source = ArrayFabric(table, points, values)
    storage = storage_view(tmp_path, table, p, "single", "hf", fmt, save=True)
    run_ranks(p[2], lambda rank: gio.rank_save(rank, source.view(rank), storage))
    verify_crc(tmp_path, fmt, table, p, "hf", values)
    target = ArrayFabric(table, points)
    readable = storage_view(tmp_path, table, p, "single", "hf", fmt)
    run_ranks(p[2], lambda rank: gio.rank_load(rank, target.view(rank), readable))
    target.assert_values(values)
    source.assert_values(values)

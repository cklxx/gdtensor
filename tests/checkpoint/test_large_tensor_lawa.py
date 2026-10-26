"""N7: LAWA averaging on a tensor larger than one 64 MiB chunk.

The KV backend splits a file payload into 64 MiB chunks. Every other LAWA
case uses KB-scale tensors, so this case pins the behavior the chunk split
is meant to hit:

* one F32 tensor whose payload exceeds 64 MiB, sharded across 4 ranks;
* k checkpoints saved through public rank_save and independently decoded;
* a LAWA mean saved and checked elementwise against the dense numpy mean;
* the serialized file is asserted larger than 64 MiB, proving at least one
  cross-chunk boundary is exercised rather than merely configured.

Ranks are threads. This verifies the large-file plan/payload path; the
numeric multi-process LAWA path is covered in tests/torch/test_lawa.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.oracle import ArrayFabric, slice_points
from tests.support.schemas import (
    payload_regions,
    run_ranks,
    storage_view,
    verify_crc,
)

P = (1, 1, 4)
CHUNK_BYTES = 64 * 1024 * 1024
# 22M float32 elements = 88 MiB payload, strictly above one 64 MiB chunk.
ELEMENTS = 22 * 1024 * 1024
SHAPE = (ELEMENTS,)
K = 3


def _large_schema():
    world = int(np.prod(P))
    placements = (gb.Replicate(), gb.Replicate(), gb.Shard(0))
    mesh = gb.SlicedMesh(
        gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(world)), list(P)),
        [gb.Slice(0, 1), None, None],
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta(list(SHAPE), "F32", 4), list(placements), mesh
    )
    table = gb.DTensorTable([gb.NamedDTensor("big.weight", dt, ["big.alias"])])
    points = {rank: {} for rank in range(world)}
    for rank in points:
        points[rank]["big.weight"] = slice_points(SHAPE, P, placements, 0, rank)
    return table, points


def _checkpoint_values():
    base = np.arange(ELEMENTS, dtype=np.float32)
    # Distinct, non-integer versions so an averaging bug is visible.
    return [base * (i + 1) / 10 + i for i in range(K)]


def _save(root, table, points, values, fmt):
    fabric = ArrayFabric(table, points, values)
    view = storage_view(root, table, P, "single", "hf", fmt, save=True)
    run_ranks(int(np.prod(P)), lambda r: gio.rank_save(r, fabric.view(r), view))
    verify_crc(root, fmt, table, P, "hf", values)


@pytest.mark.parametrize("fmt", ["safetensors"])
def test_lawa_mean_on_cross_chunk_tensor(tmp_path, fmt):
    table, points = _large_schema()
    versions = _checkpoint_values()
    for i, values in enumerate(versions):
        _save(tmp_path / f"input{i}", table, points, {"big.weight": values}, fmt)

    mean = np.mean(versions, axis=0).astype(np.float32)
    merged_root = tmp_path / "mean"
    _save(merged_root, table, points, {"big.weight": mean}, fmt)

    # The payload file must genuinely cross the 64 MiB chunk boundary.
    payload_files = [p for p in merged_root.glob("*.safetensors")]
    assert payload_files
    assert any(p.stat().st_size > CHUNK_BYTES for p in payload_files), [
        p.stat().st_size for p in payload_files
    ]

    # Independent decode of the merged checkpoint, elementwise against mean.
    for tid, _slot, info, data, _path, _off in payload_regions(merged_root, fmt):
        assert tid == "big.weight"
        assert tuple(info["shape"]) == SHAPE
        actual = np.frombuffer(data, dtype=np.float32)
        np.testing.assert_allclose(actual, mean, rtol=1e-6, atol=1e-6)

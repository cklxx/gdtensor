"""Small visible data checks for reviewing the IO path."""

import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_sim.coord import ThreadCoordinator
from tests.support.oracle import ArrayFabric
from tests.support.storage import FileStorage


def _table(world, axis):
    tensor = gb.NaiveDTensor(
        gb.GlobalTensorMeta([3, 4], "I64", 8),
        [gt.Shard(axis)],
        gb.ArrangedMesh(gt.WorldSize(world)),
    )
    return gb.DTensorTable([gb.NamedDTensor("weight", tensor)])


def _points(world, axis):
    indices = np.arange(12).reshape(3, 4)
    chunk = (indices.shape[axis] + world - 1) // world
    points = {}
    for rank in range(world):
        slices = [slice(None), slice(None)]
        slices[axis] = slice(rank * chunk, (rank + 1) * chunk)
        points[rank] = {"weight": indices[tuple(slices)].ravel()}
    return points


def _run(label, world, action):
    print(f"[CRC+ARANGE][WALKTHROUGH] {label} START ranks={world}", flush=True)
    with ThreadPoolExecutor(max_workers=world) as pool:
        results = list(pool.map(action, range(world)))
    for rank, result in enumerate(results):
        assert type(result) is gt.Ok, result
        print(f"[CRC+ARANGE][WALKTHROUGH] {label} rank={rank} result=Ok", flush=True)


def _show_rank_data(label, fabric):
    for rank, buffers in fabric.buffers.items():
        values = buffers["weight"].view(np.int64).tolist()
        print(
            f"[CRC+ARANGE][WALKTHROUGH] {label} rank={rank} indices={fabric.points[rank]['weight'].tolist()} values={values}",
            flush=True,
        )


def _payload(root, fmt):
    path = root / ("model." + fmt)
    raw = path.read_bytes()
    if fmt == "distcp":
        print(
            f"[CRC+ARANGE][WALKTHROUGH] metadata={json.loads((root / '.metadata').read_bytes())}",
            flush=True,
        )
        return raw
    size = struct.unpack("<Q", raw[:8])[0]
    print(
        f"[CRC+ARANGE][WALKTHROUGH] header={json.loads(raw[8 : 8 + size])}", flush=True
    )
    return raw[8 + size :]


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_small_arange_crc_and_elastic_load(tmp_path, fmt):
    values = {"weight": np.arange(12, dtype=np.int64).reshape(3, 4)}
    print(
        f"[CRC+ARANGE][WALKTHROUGH] format={fmt} global_shape=[3, 4] tensors=1 global_values={values['weight'].tolist()}",
        flush=True,
    )
    compute = ArrayFabric(_table(2, 1), _points(2, 1), values)
    _show_rank_data("source", compute)
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                "weight",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([3, 4], "I64", 8),
                    [gt.Replicate(), gt.Shard(0)],
                    gb.OuterProductMesh(
                        gb.ArrangedMesh(gt.ConstantOne(1)),
                        gb.ArrangedMesh(gt.WorldSize(2)),
                    ),
                ),
            )
        ]
    )
    points = {
        (tid, rank): indices
        for rank, tensors in _points(2, 0).items()
        for tid, indices in tensors.items()
    }
    storage = FileStorage(
        tmp_path, table, points, 2, fmt, coord=ThreadCoordinator(2)
    ).view
    _run("save", 2, lambda rank: gio.rank_save(rank, compute.view(rank), storage))
    payload = _payload(tmp_path, fmt)
    wanted = values["weight"].tobytes()
    print(
        f"[CRC+ARANGE][WALKTHROUGH] payload_values={np.frombuffer(payload, dtype=np.int64).tolist()}",
        flush=True,
    )
    print(
        f"[CRC+ARANGE][WALKTHROUGH] crc32 expected={zlib.crc32(wanted):08x} actual={zlib.crc32(payload):08x} bytes={len(payload)}",
        flush=True,
    )
    assert payload == wanted
    loaded = ArrayFabric(_table(3, 0), _points(3, 0))
    _run("load 2->3", 3, lambda rank: gio.rank_load(rank, loaded.view(rank), storage))
    loaded.assert_values(values)
    _show_rank_data("loaded", loaded)
    converted = ArrayFabric(_table(3, 0), _points(3, 0))
    _run(
        "compute reshard 2->3",
        3,
        lambda rank: gio.rank_reshard(
            rank, compute.view(rank, allow_remote_reads=True), converted.view(rank)
        ),
    )
    converted.assert_values(values)
    _show_rank_data("resharded", converted)
    print(
        f"[CRC+ARANGE][WALKTHROUGH] format={fmt} CRC / exact bytes / load / reshard PASS",
        flush=True,
    )

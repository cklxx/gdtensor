"""Shared checkpoint schemas and independent dense/file oracles."""

import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import safetensors

import gdtensor.builder as gb
import gdtensor.tag as gt
from gdtensor_sim.coord import ThreadCoordinator
from tests.support.oracle import slice_points
from tests.support.storage import FileStorage

SRC = (2, 2, 2)
DST = (2, 1, 3)
LAYOUTS = [
    ("single", "hf"),
    ("hf", "hf"),
    ("per_rank", "mirror"),
    ("rank_multi", "mirror"),
    ("per_stage", "hf"),
]


def run_ranks(world, fn):
    with ThreadPoolExecutor(max_workers=world) as ex:
        results = list(ex.map(fn, range(world)))
    assert all(type(r) is gb.Ok for r in results), results


def tensor_schema(p, column=False, tensor_count=8):
    specs: list[tuple[str, tuple[int, ...], str, tuple[gt.Box, ...], bool]] = [
        (
            "matrix",
            (19, 37),
            "<i8",
            (gb.Replicate(), gb.Replicate(), gb.Shard(1 if column else 0)),
            False,
        ),
        (
            "cube",
            (3, 17, 11),
            "<f8",
            (gb.Replicate(), gb.Shard(1), gb.Shard(2 if column else 0)),
            False,
        ),
        ("nested", (5, 3), "<i8", (gb.Replicate(), gb.Shard(0), gb.Shard(0)), False),
        ("flat", (3, 97), "<f4", (gb.Replicate(), gb.Shard(0), gb.Shard(0)), True),
        ("half", (5, 7), "<f2", (gb.Replicate(), gb.Replicate(), gb.Shard(1)), False),
        ("tiny", (1,), "<i8", (gb.Replicate(), gb.Replicate(), gb.Shard(0)), False),
        ("singleton", (1,), "<f8", (gb.Replicate(),) * 3, False),
        ("empty", (0, 5), "<i8", (gb.Replicate(),) * 3, False),
    ]
    for i in range(tensor_count - len(specs)):
        _, shape, dtype, placements, flat = specs[i % 4]
        specs.append((f"layer{i:03d}.weight", shape, dtype, placements, flat))
    world = int(np.prod(p))
    table, values = [], {}
    points = {rank: {} for rank in range(world)}
    for order, (tid, shape, dtype, placements, flat) in enumerate(specs):
        dtype = np.dtype(dtype)
        stage = order % p[0]
        underlying_shape = (int(np.prod(shape)),) if flat else shape
        mesh = gb.SlicedMesh(
            gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(world)), p),
            [gb.Slice(stage, stage + 1), None, None],
        )
        name = {"<i8": "I64", "<f8": "F64", "<f4": "F32", "<f2": "F16"}[dtype.str]
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta(list(underlying_shape), name, dtype.itemsize),
            list(placements),
            mesh,
        )
        if flat:
            dt = gb.ReshapedDTensor(dt, list(shape))
        table.append(gb.NamedDTensor(tid, dt, [tid + ".alias"]))
        # Values stay unique across elements and tensor ids, including offsets >256.
        seed = 1000 if dtype.itemsize == 2 else order * 10000
        values[tid] = (np.arange(int(np.prod(shape)), dtype=dtype) + seed).reshape(
            shape
        )
        for rank in points:
            points[rank][tid] = slice_points(
                underlying_shape, p, placements, stage, rank
            )
    return gb.DTensorTable(table), points, values


def storage_view(root, ct, p, kind, mode, fmt, *, save=False):
    world = int(np.prod(p))
    table = []
    for order, nd in enumerate(ct.named_d_tensors):
        dt = (
            nd.d_tensor.underlying_d_tensor
            if type(nd.d_tensor) is gb.ReshapedDTensor
            else nd.d_tensor
        )
        slices = _storage_slices(dt, p, order)
        mesh = gb.SlicedMesh(
            gb.ReshapedMesh(
                gb.SlotOuterProductMesh(
                    gb.ArrangedMesh(gt.ConstantOne(1)),
                    gb.ArrangedMesh(gt.WorldSize(world)),
                ),
                p,
            ),
            slices,
        )
        placements = [
            gb.Shard(0) if mode == "hf" and type(pl) is gb.Shard else pl
            for pl in dt.placements
        ]
        stored = gb.NaiveDTensor(dt.global_tensor_meta, placements, mesh)
        if type(nd.d_tensor) is gb.ReshapedDTensor:
            stored = gb.ReshapedDTensor(stored, nd.d_tensor.shape)
        table.append(gb.NamedDTensor(nd.global_tensor_id, stored, nd.alias))
    return FileStorage(
        root,
        gb.DTensorTable(table),
        storage_points(ct, p, mode),
        world,
        fmt,
        kind,
        [i % p[0] for i in range(len(table))],
        ThreadCoordinator(world) if save else None,
    ).view


def _storage_slices(tensor, parallel, order):
    slices = [gb.Slice(order % parallel[0], order % parallel[0] + 1), None, None]
    for dim in (1, 2):
        if type(tensor.placements[dim]) is gb.Replicate:
            slices[dim] = gb.Slice(order % parallel[dim], order % parallel[dim] + 1)
    return slices


def storage_points(ct, p, mode):
    expected = {}
    for order, nd in enumerate(ct.named_d_tensors):
        dt = nd.d_tensor
        if type(dt) is gb.ReshapedDTensor:
            dt = dt.underlying_d_tensor
        placements = [
            gb.Shard(0) if mode == "hf" and type(pl) is gb.Shard else pl
            for pl in dt.placements
        ]
        for rank in range(int(np.prod(p))):
            expected[nd.global_tensor_id, rank] = slice_points(
                dt.global_tensor_meta.global_shape,
                p,
                placements,
                order % p[0],
                rank,
                storage=True,
                order=order,
            )
    return expected


def payload_regions(root, fmt):
    if fmt == "safetensors":
        files = sorted(root.glob("*.safetensors"))
        assert files
        for path in files:
            yield from _safetensors_regions(path)
    else:
        doc = json.loads((root / ".metadata").read_bytes())
        assert doc["format"] == "gdtensor-distcp"
        for chunk in doc["chunks"]:
            path = root / chunk["file"]
            lo, size = chunk["offset"], chunk["size"]
            data = path.read_bytes()[lo : lo + size]
            assert len(data) == size
            yield chunk["tensor_id"], chunk["slot"], chunk, data, path, lo
    assert not list(root.glob("*.tmp"))


def _safetensors_regions(path):
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    decoded = dict(safetensors.deserialize(raw))
    cursor = 0
    for key, info in header.items():
        if key == "__metadata__":
            continue
        lo, hi = info["data_offsets"]
        assert lo == cursor and lo <= hi
        cursor = hi
        data = raw[8 + header_len + lo : 8 + header_len + hi]
        assert len(data) == hi - lo
        assert bytes(decoded[key]["data"]) == data
        tid, sep, slot = key.partition("/slot")
        yield tid, int(slot) if sep else None, info, data, path, 8 + header_len + lo
    assert len(raw) == 8 + header_len + cursor


def verify_crc(root, fmt, ct, p, mode, values):
    print(
        f"[CRC+ARANGE][VERIFY][START] directory={root.name} format={fmt} ranks={int(np.prod(p))} tensors={len(values)}",
        flush=True,
    )
    points = storage_points(ct, p, mode)
    seen = {tid: np.zeros(value.size, dtype=np.int64) for tid, value in values.items()}
    for tid, slot, info, data, _, _ in payload_regions(root, fmt):
        value = values[tid]
        indices = np.arange(value.size) if slot is None else points[tid, slot]
        want = value.reshape(-1)[indices].copy().tobytes()
        assert len(data) == len(want)
        assert zlib.crc32(data) == zlib.crc32(want), (tid, slot)
        assert data == want, (tid, slot)
        seen[tid][indices] += 1
        if fmt != "safetensors":
            continue
        assert int(np.prod(info["shape"])) == len(indices)
        if slot is None:
            assert info["shape"] == list(value.shape)
        assert (
            info["dtype"]
            == gb.global_tensor_meta_of(
                next(nd for nd in ct.named_d_tensors if nd.global_tensor_id == tid)
            ).dtype
        )
    for counts in seen.values():
        np.testing.assert_array_equal(counts, np.ones_like(counts))
    print(
        "[CRC+ARANGE][VERIFY][PASS] checks=payload_crc,exact_bytes,every_element_stored_once",
        flush=True,
    )

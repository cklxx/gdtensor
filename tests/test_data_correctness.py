"""Public IO verified with dense arange arrays and independent file decoding."""

from __future__ import annotations

import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import safetensors

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_sim.coord import ThreadCoordinator
from storage_fixture import FileStorage
from tensor_oracle import ArrayFabric, slice_points


SRC = (2, 2, 2)
DST = (2, 1, 3)
LAYOUTS = [
    ("single", "hf"),
    ("hf", "hf"),
    ("per_rank", "mirror"),
    ("rank_multi", "mirror"),
    ("per_stage", "hf"),
]


def _run(world, fn):
    with ThreadPoolExecutor(max_workers=world) as ex:
        results = list(ex.map(fn, range(world)))
    assert all(type(r) is gt.Ok for r in results), results


def _schema(p, column=False, tensor_count=8):
    specs: list[tuple[str, tuple[int, ...], str, tuple[gt.Box, ...], bool]] = [
        (
            "matrix",
            (19, 37),
            "<i8",
            (gt.Replicate(), gt.Replicate(), gt.Shard(1 if column else 0)),
            False,
        ),
        (
            "cube",
            (3, 17, 11),
            "<f8",
            (gt.Replicate(), gt.Shard(1), gt.Shard(2 if column else 0)),
            False,
        ),
        ("nested", (5, 3), "<i8", (gt.Replicate(), gt.Shard(0), gt.Shard(0)), False),
        ("flat", (3, 97), "<f4", (gt.Replicate(), gt.Shard(0), gt.Shard(0)), True),
        ("half", (5, 7), "<f2", (gt.Replicate(), gt.Replicate(), gt.Shard(1)), False),
        ("tiny", (1,), "<i8", (gt.Replicate(), gt.Replicate(), gt.Shard(0)), False),
        ("singleton", (1,), "<f8", (gt.Replicate(),) * 3, False),
        ("empty", (0, 5), "<i8", (gt.Replicate(),) * 3, False),
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


def _storage(root, ct, p, kind, mode, fmt, *, save=False):
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
                gb.OuterProductMesh(
                    gb.ArrangedMesh(gt.ConstantOne(1)),
                    gb.ArrangedMesh(gt.WorldSize(world)),
                ),
                p,
            ),
            slices,
        )
        placements = [
            gt.Shard(0) if mode == "hf" and type(pl) is gt.Shard else pl
            for pl in dt.placements
        ]
        stored = gb.NaiveDTensor(dt.global_tensor_meta, placements, mesh)
        if type(nd.d_tensor) is gb.ReshapedDTensor:
            stored = gb.ReshapedDTensor(stored, nd.d_tensor.shape)
        table.append(gb.NamedDTensor(nd.global_tensor_id, stored, nd.alias))
    return FileStorage(
        root,
        gb.DTensorTable(table),
        _storage_points(ct, p, mode),
        world,
        fmt,
        kind,
        [i % p[0] for i in range(len(table))],
        ThreadCoordinator(world) if save else None,
    ).view


def _storage_slices(tensor, parallel, order):
    slices = [gb.Slice(order % parallel[0], order % parallel[0] + 1), None, None]
    for dim in (1, 2):
        if type(tensor.placements[dim]) is gt.Replicate:
            slices[dim] = gb.Slice(order % parallel[dim], order % parallel[dim] + 1)
    return slices


def _storage_points(ct, p, mode):
    expected = {}
    for order, nd in enumerate(ct.named_d_tensors):
        dt = nd.d_tensor
        if type(dt) is gb.ReshapedDTensor:
            dt = dt.underlying_d_tensor
        placements = [
            gt.Shard(0) if mode == "hf" and type(pl) is gt.Shard else pl
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


def _regions(root, fmt):
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


def _verify_crc(root, fmt, ct, p, mode, values):
    points = _storage_points(ct, p, mode)
    seen = {tid: np.zeros(value.size, dtype=np.int64) for tid, value in values.items()}
    for tid, slot, info, data, _, _ in _regions(root, fmt):
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


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
@pytest.mark.parametrize("kind,mode", LAYOUTS)
def test_io_arange_crc_elastic_and_reshard(tmp_path, fmt, kind, mode):
    ct, points, values = _schema(SRC)
    target, target_points, _ = _schema(DST, column=True)
    source = ArrayFabric(ct, points, values)
    root = tmp_path / "source"
    sv = _storage(root, ct, SRC, kind, mode, fmt, save=True)
    _run(int(np.prod(SRC)), lambda r: gio.rank_save(r, source.view(r), sv))
    _verify_crc(root, fmt, ct, SRC, mode, values)

    loaded = ArrayFabric(target, target_points)
    read_view = _storage(root, ct, SRC, kind, mode, fmt)
    _run(int(np.prod(DST)), lambda r: gio.rank_load(r, loaded.view(r), read_view))
    loaded.assert_values(values)
    source.assert_values(values)

    for branch in ("compute", "storage"):
        destination = ArrayFabric(target, target_points)
        src = lambda r: (
            source.view(r, allow_remote_reads=True)
            if branch == "compute"
            else read_view
        )
        _run(
            int(np.prod(DST)),
            lambda r: gio.rank_reshard(r, src(r), destination.view(r)),
        )
        destination.assert_values(values)

    # Exercise compute->storage and storage->storage, then load both results.
    for branch in ("compute", "storage"):
        out = tmp_path / branch
        output_fmt = "distcp" if fmt == "safetensors" else "safetensors"
        dst = _storage(out, target, DST, "per_rank", "mirror", output_fmt, save=True)
        src = lambda r: (
            loaded.view(r, allow_remote_reads=True)
            if branch == "compute"
            else read_view
        )
        _run(int(np.prod(DST)), lambda r: gio.rank_reshard(r, src(r), dst))
        _verify_crc(out, output_fmt, target, DST, "mirror", values)
        back = ArrayFabric(ct, points)
        output = _storage(out, target, DST, "per_rank", "mirror", output_fmt)
        _run(int(np.prod(SRC)), lambda r: gio.rank_load(r, back.view(r), output))
        back.assert_values(values)


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_io_crc_detects_payload_corruption(tmp_path, fmt):
    ct, points, values = _schema(SRC)
    source = ArrayFabric(ct, points, values)
    sv = _storage(tmp_path, ct, SRC, "single", "hf", fmt, save=True)
    _run(int(np.prod(SRC)), lambda r: gio.rank_save(r, source.view(r), sv))
    _verify_crc(tmp_path, fmt, ct, SRC, "hf", values)
    tid, slot, info, data, path, start = next(_regions(tmp_path, fmt))
    with path.open("r+b") as f:
        f.seek(start)
        f.write(bytes([data[0] ^ 1]))
    with pytest.raises(AssertionError):
        _verify_crc(tmp_path, fmt, ct, SRC, "hf", values)
    # Raw IO does not promise checksum validation. The independent oracle must fail.
    loaded = ArrayFabric(ct, points)
    read_view = _storage(tmp_path, ct, SRC, "single", "hf", fmt)
    _run(int(np.prod(SRC)), lambda r: gio.rank_load(r, loaded.view(r), read_view))
    with pytest.raises(AssertionError):
        loaded.assert_values(values)


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
@pytest.mark.parametrize(
    "src_p,dst_p,tensor_count,kind,mode",
    [
        ((1, 1, 3), (1, 1, 2), 16, "single", "hf"),
        ((2, 2, 4), (2, 2, 3), 32, "rank_multi", "mirror"),
        ((2, 4, 4), (2, 2, 4), 64, "per_rank", "mirror"),
    ],
)
def test_many_ranks_many_tensors(tmp_path, fmt, src_p, dst_p, tensor_count, kind, mode):
    ct, points, values = _schema(src_p, tensor_count=tensor_count)
    target, target_points, _ = _schema(dst_p, column=True, tensor_count=tensor_count)
    source = ArrayFabric(ct, points, values)
    saved = _storage(tmp_path, ct, src_p, kind, mode, fmt, save=True)
    _run(int(np.prod(src_p)), lambda r: gio.rank_save(r, source.view(r), saved))
    _verify_crc(tmp_path, fmt, ct, src_p, mode, values)
    loaded = ArrayFabric(target, target_points)
    read_view = _storage(tmp_path, ct, src_p, kind, mode, fmt)
    _run(int(np.prod(dst_p)), lambda r: gio.rank_load(r, loaded.view(r), read_view))
    loaded.assert_values(values)
    resharded = ArrayFabric(target, target_points)
    _run(
        int(np.prod(dst_p)),
        lambda r: gio.rank_reshard(
            r, source.view(r, allow_remote_reads=True), resharded.view(r)
        ),
    )
    resharded.assert_values(values)
    source.assert_values(values)

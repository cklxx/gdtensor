"""Data correctness: CRC checks and arrange-tensor round trips.

CRC:
* file backend: every safetensors tensor payload matches zlib.crc32 of the
  expected pattern bytes; headers are excluded.

Arrange tensor:
* A tensor filled with its global element index (mod 256) is sharded, saved,
  loaded under a different parallelism, and resharded. Every rank's local
  buffer must equal the owned slice of the global arrange vector.
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np
import pytest

from conftest import assert_all_ok, verify_fabric
from gdtensor.coord import ThreadCoordinator
from gdtensor.dtensor import (
    DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor, Replicate, Shard,
    global_tensor_meta_of, owned_intervals,
)
from gdtensor.formats import make_storage_view
from gdtensor.iobackend import MiB
from gdtensor.layout import build_payload_index
from gdtensor.models import (
    ModelConfig, ParallelConfig, compute_mesh, compute_table, fill_pattern, layout_builder,
    pattern_bytes, storage_table, tensor_seed,
)
from gdtensor.runner import _run_threads, compute_view, load_threads, reshard_threads, save_threads
from gdtensor.views import ComputeFabric

M = ModelConfig(layers=2, hidden=64, ffn=128, vocab=256)
P = ParallelConfig(tp=2, dp=2, pp=2)


# ------------------------------------------------------------------ CRC helpers


def _safetensors_tensor_crcs(root):
    """{tensor_key: crc32(payload)} for every tensor in every safetensors file."""
    out = {}
    for name in sorted(f for f in os.listdir(root) if f.endswith(".safetensors")):
        with open(os.path.join(root, name), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            import json
            header = json.loads(f.read(n))
            data_start = 8 + n
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                lo, hi = info["data_offsets"]
                f.seek(data_start + lo)
                out[f"{name}:{key}"] = zlib.crc32(f.read(hi - lo))
    return out


def _expected_crc(tensor_id, itemsize, numel):
    return zlib.crc32(pattern_bytes(tensor_id, itemsize, np.arange(numel, dtype=np.int64)).tobytes())


# ------------------------------------------------------------------ file backend CRC


def _expected_file_payload_crc(st, lay, path):
    """CRC of the concatenated slot payloads stored in ``path``.

    Each slot holds one rank's owned elements, in ascending global order.
    """
    idx = build_payload_index(st, lay)
    fp = idx.files[path]
    buf = bytearray()
    for slot in fp.slots:
        nd = next(x for x in st.named_d_tensors if x.global_tensor_id == slot.pos.tensor_id)
        buf += pattern_bytes(slot.pos.tensor_id, slot.itemsize,
                             np.concatenate([np.arange(a, b, dtype=np.int64)
                                             for a, b in slot.intervals])).tobytes()
    return zlib.crc32(bytes(buf))


def _safetensors_payload_crc(path):
    """CRC of the data region (header excluded) of one safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n)
        return zlib.crc32(f.read())


@pytest.mark.parametrize("kind", ["single", "hf", "per_rank", "per_stage"])
def test_file_backend_payload_crc_matches_pattern(tmp_path, kind):
    root = str(tmp_path / kind)
    os.makedirs(root)
    ct = compute_table(M, P)
    mode = "hf" if kind in ("single", "hf", "per_stage") else "mirror"
    st = storage_table(ct, P, mode)
    lay = layout_builder(kind, st, P, "safetensors", 2)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    res, _ = save_threads(ct, st, lay, root, fab, P.world)
    assert_all_ok(res)
    paths = [f for f in os.listdir(root) if f.endswith(".safetensors")]
    assert paths
    for name in paths:
        got = _safetensors_payload_crc(os.path.join(root, name))
        want = _expected_file_payload_crc(st, lay, name)
        assert got == want, name


def test_file_backend_crc_detects_corruption(tmp_path):
    root = str(tmp_path / "corrupt")
    os.makedirs(root)
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("single", st, P, "safetensors")
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, P.world)[0])
    path = os.path.join(root, "model.safetensors")
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.seek(size // 2)
        f.write(b"\xff" * 8)  # flip bytes inside a payload
    assert _safetensors_payload_crc(path) != _expected_file_payload_crc(st, lay, "model.safetensors")


# ------------------------------------------------------------------ arrange tensor


def _arrange_tensor(tid: str, shape, dtype="U8", itemsize=1):
    """A tensor whose byte at global element i is (i + seed) & 0xFF.

    pattern_bytes already does (i*131 + b*29 + seed) & 0xFF; for an arrange
    tensor we override fill with the raw index so shard boundaries are easy
    to verify.
    """
    return NamedDTensor(tid, NaiveDTensor(GlobalTensorMeta(shape, dtype, itemsize),
                                          (Replicate(), Replicate(), Shard(0)),
                                          compute_mesh(P, 0)))


def _arrange_fill(nd, iv):
    e = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in iv])
    return ((e + tensor_seed(nd.global_tensor_id)) & 0xFF).astype(np.uint8).ravel()


def test_arrange_tensor_save_load_elastic(tmp_path):
    """Save arrange tensors under TP2 DP2 PP2, load under TP4 DP1 PP2.

    Each loaded rank's local buffer must equal the global arrange vector at
    its owned intervals.
    """
    tids = ["arr0", "arr1", "arr2"]
    shapes = [(1024,), (256, 16), (64, 64)]
    nds = [_arrange_tensor(t, s) for t, s in zip(tids, shapes)]
    ct = DTensorTable(nds)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("single", st, P, "safetensors")
    root = str(tmp_path / "arrange")
    os.makedirs(root)
    fab = ComputeFabric(ct).allocate(fill=_arrange_fill)
    assert_all_ok(save_threads(ct, st, lay, root, fab, P.world)[0])

    p2 = ParallelConfig(tp=4, dp=1, pp=2)
    ct2 = DTensorTable([NamedDTensor(t, NaiveDTensor(GlobalTensorMeta(s, "U8", 1),
                                                    (Replicate(), Replicate(), Shard(0)),
                                                    compute_mesh(p2, 0)))
                        for t, s in zip(tids, shapes)])
    fab2 = ComputeFabric(ct2).allocate()
    assert_all_ok(load_threads(ct2, st, lay, root, fab2, p2.world)[0])
    for r in range(p2.world):
        for nd in ct2.named_d_tensors:
            buf = fab2.buffers.get(r, {}).get(nd.global_tensor_id)
            if buf is None:
                continue
            iv = owned_intervals(nd, "compute", r)
            want = _arrange_fill(nd, iv)
            assert np.array_equal(buf, want), f"rank {r} {nd.global_tensor_id} arrange mismatch"


def test_arrange_tensor_reshard_compute_to_compute(tmp_path):
    tids = ["r0", "r1"]
    shapes = [(512,), (128, 128)]
    p_src = ParallelConfig(tp=2, dp=2, pp=1)
    p_dst = ParallelConfig(tp=4, dp=1, pp=1)
    ct_src = DTensorTable([NamedDTensor(t, NaiveDTensor(GlobalTensorMeta(s, "U8", 1),
                                                       (Replicate(), Shard(0), Shard(0)),
                                                       compute_mesh(p_src, 0)))
                           for t, s in zip(tids, shapes)])
    ct_dst = DTensorTable([NamedDTensor(t, NaiveDTensor(GlobalTensorMeta(s, "U8", 1),
                                                       (Replicate(), Replicate(), Shard(0)),
                                                       compute_mesh(p_dst, 0)))
                           for t, s in zip(tids, shapes)])
    fab_src = ComputeFabric(ct_src).allocate(fill=_arrange_fill)
    fab_dst = ComputeFabric(ct_dst).allocate()
    assert_all_ok(reshard_threads(p_dst.world,
                                  lambda r: compute_view(ct_src, fab_src, r),
                                  lambda r: compute_view(ct_dst, fab_dst, r)))
    for r in range(p_dst.world):
        for nd in ct_dst.named_d_tensors:
            buf = fab_dst.buffers.get(r, {}).get(nd.global_tensor_id)
            if buf is None:
                continue
            iv = owned_intervals(nd, "compute", r)
            assert np.array_equal(buf, _arrange_fill(nd, iv)), f"rank {r} {nd.global_tensor_id}"


def test_arrange_tensor_reshard_storage_to_storage(tmp_path):
    tids = ["s0", "s1"]
    shapes = [(768,), (48, 48)]
    ct = DTensorTable([_arrange_tensor(t, s) for t, s in zip(tids, shapes)])
    src_st = storage_table(ct, P, "hf")
    src_lay = layout_builder("single", src_st, P, "safetensors")
    dst_st = storage_table(ct, P, "mirror")
    dst_lay = layout_builder("per_rank", dst_st, P, "safetensors")
    src_root = str(tmp_path / "src")
    dst_root = str(tmp_path / "dst")
    os.makedirs(src_root); os.makedirs(dst_root)
    fab = ComputeFabric(ct).allocate(fill=_arrange_fill)
    assert_all_ok(save_threads(ct, src_st, src_lay, src_root, fab, P.world)[0])

    sv_src = make_storage_view(src_st.tensor_ids, src_st, src_lay, src_root)
    sv_dst = make_storage_view(dst_st.tensor_ids, dst_st, dst_lay, dst_root,
                               coord=ThreadCoordinator(P.world))
    assert_all_ok(reshard_threads(P.world, lambda r: sv_src, lambda r: sv_dst))

    # Load the destination storage back into compute and verify arrange.
    fab2 = ComputeFabric(ct).allocate()
    assert_all_ok(load_threads(ct, dst_st, dst_lay, dst_root, fab2, P.world)[0])
    for r in range(P.world):
        for nd in ct.named_d_tensors:
            buf = fab2.buffers.get(r, {}).get(nd.global_tensor_id)
            if buf is None:
                continue
            iv = owned_intervals(nd, "compute", r)
            assert np.array_equal(buf, _arrange_fill(nd, iv)), f"rank {r} {nd.global_tensor_id}"

"""Shared helpers. Verification paths are independent of the write path:
safetensors files are decoded by the official ``safetensors`` library and a
hand-written header parser; distcp files through ``.metadata``."""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
import pytest
import safetensors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdtensor.dtensor import global_tensor_meta_of, iv_normalize, iv_total, named_of, owned_intervals  # noqa: E402
from gdtensor.models import fill_pattern, full_tensor_bytes, interval_elems, pattern_bytes  # noqa: E402


def raw_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def verify_safetensors_dir(root, storage_table):
    """Every byte of every tensor is present exactly once and correct."""
    seen = {nd.global_tensor_id: [] for nd in storage_table.named_d_tensors}
    files = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
    assert files, "no safetensors file published"
    assert not [f for f in os.listdir(root) if f.endswith(".tmp")], "temp files left behind"
    order = {nd.global_tensor_id: i for i, nd in enumerate(storage_table.named_d_tensors)}
    for f in files:
        decoded = dict(safetensors.deserialize(open(os.path.join(root, f), "rb").read()))
        header, _ = raw_header(os.path.join(root, f))
        entries = [(k, v) for k, v in header.items() if k != "__metadata__"]
        # header order == DTensorTable order, and offsets follow it
        keys = [k.split("/slot")[0] for k, _ in entries]
        assert keys == sorted(keys, key=lambda k: order[k]), f"{f}: header order != table order"
        offs = [v["data_offsets"] for _, v in entries]
        assert offs == sorted(offs) and all(a[1] == b[0] for a, b in zip(offs, offs[1:])), f"{f}: payload not contiguous"
        for key, info in decoded.items():
            tid, _, slot = key.partition("/slot")
            nd = named_of(storage_table, tid)
            isz = global_tensor_meta_of(nd).itemsize
            if slot:
                iv = owned_intervals(nd, "storage", int(slot))
            else:
                iv = np.array([[0, global_tensor_meta_of(nd).numel]], dtype=np.int64)
                assert list(info["shape"]) == list(global_tensor_meta_of(nd).global_shape)
            want = pattern_bytes(tid, isz, interval_elems(iv)).tobytes()
            assert bytes(info["data"]) == want, f"{f}:{key} payload mismatch"
            seen[tid].append(iv)
    idx_path = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        assert set(idx["weight_map"]) == set(seen)
        assert all(idx["weight_map"][k] in files for k in idx["weight_map"])
        assert idx["metadata"]["total_size"] == sum(global_tensor_meta_of(nd).nbytes for nd in storage_table.named_d_tensors)
    for tid, ivs in seen.items():
        nd = named_of(storage_table, tid)
        numel = global_tensor_meta_of(nd).numel
        allv = np.concatenate(ivs) if ivs else np.zeros((0, 2), dtype=np.int64)
        assert iv_total(allv) == numel, f"{tid}: stored {iv_total(allv)} of {numel} elements (dup or gap)"
        assert iv_total(iv_normalize(allv)) == numel
    return files


def verify_distcp_dir(root, storage_table):
    meta = json.load(open(os.path.join(root, ".metadata")))
    seen = {}
    for c in meta["chunks"]:
        nd = named_of(storage_table, c["tensor_id"])
        isz = global_tensor_meta_of(nd).itemsize
        iv = owned_intervals(nd, "storage", c["slot"])
        with open(os.path.join(root, c["file"]), "rb") as f:
            f.seek(c["offset"])
            data = f.read(c["size"])
        assert data == pattern_bytes(c["tensor_id"], isz, interval_elems(iv)).tobytes()
        seen.setdefault(c["tensor_id"], []).append(iv)
    for nd in storage_table.named_d_tensors:
        ivs = seen.get(nd.global_tensor_id, [])
        assert iv_total(np.concatenate(ivs)) == global_tensor_meta_of(nd).numel


def verify_fabric(fabric, table):
    for r, bufs in fabric.buffers.items():
        for tid, buf in bufs.items():
            nd = named_of(table, tid)
            want = fill_pattern(nd, fabric.intervals(r, tid))
            assert np.array_equal(buf, want), f"rank {r} {tid} mismatch"


def assert_all_ok(results):
    bad = {r: x.error for r, x in results.items() if not x.ok}
    assert not bad, bad


@pytest.fixture
def root(tmp_path):
    return str(tmp_path)

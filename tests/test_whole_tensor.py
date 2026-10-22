"""Whole-tensor detection: strided slots must NOT be declared whole.

Removing the contiguity term in ``_whole_tensor_files`` misclassifies a
Shard(1) (strided) layout as whole, which then emits one header entry whose
data_offsets cover bytes that are not contiguous in global order.
"""
from __future__ import annotations

import json
import struct

import numpy as np

from gdtensor._dtensor import DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor, Replicate, Shard
from gdtensor._formats import _whole_tensor_files, safetensors_header
from gdtensor._layout import build_layout, build_payload_index
from gdtensor._mesh import Slice
from gdtensor._models import storage_mesh


def _strided_table(world=2):
    meta = GlobalTensorMeta((2, 4), "U8", 1)
    sm = storage_mesh(world, (1, 1, world), (Slice(0, 1), Slice(0, 1), None))
    return DTensorTable([NamedDTensor("w", NaiveDTensor(meta, (Replicate(), Replicate(), Shard(1)), sm))])


def test_strided_slots_are_not_whole():
    st = _strided_table()
    lay = build_layout(st, 2, "safetensors", lambda t, w, nd: "f.safetensors")
    idx = build_payload_index(st, lay)
    whole = _whole_tensor_files(st, idx)
    assert list(whole.values()) == [False]
    raw = safetensors_header(st, idx, "f.safetensors")
    n = struct.unpack("<Q", raw[:8])[0]
    keys = set(json.loads(raw[8:8 + n]))
    assert "w" not in keys
    assert {"w/slot0", "w/slot1"} <= keys


def test_contiguous_slots_are_whole():
    # Shard(0): slots partition the flat tensor, adjacency holds
    meta = GlobalTensorMeta((8,), "U8", 1)
    sm = storage_mesh(2, (1, 1, 2), (Slice(0, 1), Slice(0, 1), None))
    st = DTensorTable([NamedDTensor("w", NaiveDTensor(meta, (Replicate(), Replicate(), Shard(0)), sm))])
    lay = build_layout(st, 2, "safetensors", lambda t, w, nd: "f.safetensors")
    idx = build_payload_index(st, lay)
    assert list(_whole_tensor_files(st, idx).values()) == [True]

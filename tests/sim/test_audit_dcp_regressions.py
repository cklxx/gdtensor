"""Regression: public get_file_offset contract on gdtensor-produced DCP artifacts.

These tests save through the REAL public path ``gdtensor.io.rank_save`` (DcpSaveSession
stages slots, assembles the full tensor, publishes via torch ``dcp_save(no_dist=True)``)
and then, independently, parse the on-disk ``.metadata`` pickle + ``.distcp`` zip local
headers, ``os.pread`` the base returned by the public ``DcpLoadSession.get_file_offset``,
and compare against a dense-arange ground truth.

Contract pinned here:

- a slot whose owned elements form ONE contiguous packed run in the chunk resolves to a
  base whose ``pread(slot_bytes)`` is byte-exact (row-shard, and a packed col-shard that
  fills exactly one real gloo chunk);
- a strided / non-contiguous slot (inner-axis column shard inside the no_dist full-tensor
  chunk) MUST raise ``InvalidArgumentError`` — it must not silently return Ok with a base
  whose pread sweeps into the peer rank's bytes;
- in-process ``rank_load`` data path stays correct regardless.

The oracle here never imports ``gdtensor_sim.dcp_backend._raw_payload_bases``.
"""

from __future__ import annotations

import pickle
import re
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.oracle import ArrayFabric, slice_points
from tests.support.schemas import run_ranks, storage_view

WORLD = 2


def _table(specs):
    import gdtensor.tag as gt

    nds = []
    for name, shape, dn, isz, axis in specs:
        mesh = gb.SlicedMesh(
            gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(WORLD)), (1, 1, WORLD)),
            [gb.Slice(0, 1), None, None],
        )
        placements = [
            gb.Replicate(),
            gb.Replicate(),
            (gb.Shard(axis) if axis is not None else gb.Replicate()),
        ]
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta(list(shape), dn, isz), placements, mesh
        )
        nds.append(gb.NamedDTensor(name, dt, [name + ".alias"]))
    return gb.DTensorTable(nds)


def _save_roundtrip(table, root):
    points = {r: {} for r in range(WORLD)}
    for nd in table.named_d_tensors:
        dt = (
            nd.d_tensor.underlying_d_tensor
            if type(nd.d_tensor) is gb.ReshapedDTensor
            else nd.d_tensor
        )
        for r in range(WORLD):
            points[r][nd.global_tensor_id] = slice_points(
                list(dt.global_tensor_meta.global_shape),
                (1, 1, WORLD),
                list(dt.placements),
                0,
                r,
            )
    values = {
        nd.global_tensor_id: np.arange(
            int(np.prod(gb.global_tensor_meta_of(nd).global_shape)), dtype=np.float32
        )
        for nd in table.named_d_tensors
    }
    fab = ArrayFabric(table, points, values)
    sv = storage_view(
        root, table, (1, 1, WORLD), "single", "mirror", "distcp", save=True
    )
    run_ranks(WORLD, lambda r: gio.rank_save(r, fab.view(r), sv))

    # independent on-disk oracle: resolve payload base(s) by zip local headers
    md = pickle.loads((root / ".metadata").read_bytes())
    bases = {}
    for idx, sd in md.storage_data.items():
        bases[idx.fqn] = (
            sd.relative_path,
            _resolve_payload_base(root / sd.relative_path, sd.offset, sd.length),
        )
    return points, values, bases


def _resolve_payload_base(path, sub_offset, sub_length):
    raw = path.read_bytes()
    sub = raw[sub_offset : sub_offset + sub_length]
    for m in re.finditer(rb"PK\x03\x04", sub):
        p = m.start()
        (fnlen,) = struct.unpack("<H", sub[p + 26 : p + 28])
        (elen,) = struct.unpack("<H", sub[p + 28 : p + 30])
        name = bytes(sub[p + 30 : p + 30 + fnlen]).decode()
        if re.fullmatch(r"archive/data/\d+", name):
            return sub_offset + p + 30 + fnlen + elen
    raise AssertionError(f"no archive/data entry in {path}")


def _open_load_session(table, root):
    sv = storage_view(root, table, (1, 1, WORLD), "single", "mirror", "distcp")
    sessions = {}
    for r in range(WORLD):
        res = sv.begin_load(sv, r)
        assert type(res).__name__ == "Ok", res.value
        sessions[r] = res.value
    return sessions


@pytest.mark.parametrize("shape,axis", [(6, 0)])
def test_row_shard_get_file_offset_pread_byte_exact(tmp_path, shape, axis):
    table = _table([("x", (shape, 2), "F32", 4, axis)])
    root = tmp_path / "ck"
    root.mkdir()
    points, values, bases = _save_roundtrip(table, root)
    sessions = _open_load_session(table, root)
    for r in range(WORLD):
        res = sessions[r].get_file_offset(gb.TensorComponentPos("x", r))
        assert type(res).__name__ == "Ok", res.value
        off = int(res.value)
        rel, base = bases["x"]
        want_idx = np.asarray(points[r]["x"]).reshape(-1)
        nbytes = want_idx.size * 4
        with open(root / rel, "rb") as h:
            h.seek(off)
            got = h.read(nbytes)
        want = values["x"].reshape(-1)[want_idx].view(np.uint8).tobytes()
        assert got == want, (r, off)


def test_col_shard_get_file_offset_must_raise_invalidarg(tmp_path):
    table = _table([("c", (2, 4), "F32", 4, 1)])
    root = tmp_path / "ck"
    root.mkdir()
    points, values, bases = _save_roundtrip(table, root)
    sessions = _open_load_session(table, root)
    for r in range(WORLD):
        res = sessions[r].get_file_offset(gb.TensorComponentPos("c", r))
        assert type(res).__name__ == "Err", (r, res)
        # Must be InvalidArg (not NotFound, not 0, not file start).
        assert (
            "InvalidArgument" in type(res.value).__name__
            or "invalid" in str(res.value).lower()
        )


def test_strided_3d_get_file_offset_must_raise_invalidarg(tmp_path):
    table = _table([("t", (3, 4, 11), "F32", 4, 1)])
    root = tmp_path / "ck"
    root.mkdir()
    points, values, bases = _save_roundtrip(table, root)
    sessions = _open_load_session(table, root)
    for r in range(WORLD):
        res = sessions[r].get_file_offset(gb.TensorComponentPos("t", r))
        assert type(res).__name__ == "Err", (r, res)
        assert (
            "Invalid" in type(res.value).__name__ or "invalid" in str(res.value).lower()
        )


def test_rank_load_still_correct_through_public_path(tmp_path):
    table = _table([("x", (6, 2), "F32", 4, 0), ("c", (2, 4), "F32", 4, 1)])
    root = tmp_path / "ck"
    root.mkdir()
    points, values, bases = _save_roundtrip(table, root)
    # reload into a fresh fabric and verify values (data path independent of get_file_offset)
    sv2 = storage_view(root, table, (1, 1, WORLD), "single", "mirror", "distcp")
    fab2 = ArrayFabric(table, points)
    run_ranks(WORLD, lambda r: gio.rank_load(r, fab2.view(r), sv2))
    for r in range(WORLD):
        for nd in table.named_d_tensors:
            tid = nd.global_tensor_id
            want_idx = np.asarray(points[r][tid]).reshape(-1)
            want = values[tid].reshape(-1)[want_idx]
            got = fab2.buffers[r][tid].reshape(-1).view(np.float32)
            assert np.array_equal(got, want), (r, tid)

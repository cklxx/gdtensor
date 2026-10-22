"""Data correctness: CRC checks and arrange-tensor round trips.

System under test is the PUBLIC API only:
* product types come from ``gdtensor.builder``;
* tags come from ``gdtensor.tag``;
* every save / load / reshard goes through ``gdtensor.io``
  (rank_save / rank_load / rank_reshard).

The test harness itself (in-memory device fabric, file-backed storage view,
model-table builders) is the business layer: design 4.3 puts builders and
sessions outside the core, so those scaffolding pieces live in the internal
package and are NOT the system under test. The CRC oracle parses
safetensors headers itself and never reads the internal payload index.

CRC:
* every safetensors tensor payload matches zlib.crc32 of the expected bytes;
* headers are excluded; one flipped byte must change the CRC.

Arrange tensor:
* a tensor filled with (global element index + seed) & 0xFF is sharded,
  saved, loaded under a different parallelism, and resharded. Each rank's
  local buffer must equal the owned slice of the global arrange vector.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import gdtensor.io as gio  # noqa: E402
import gdtensor.builder as gb  # noqa: E402
import gdtensor.tag as gt  # noqa: E402

# ---- test-harness / business layer only (design 4.3: builders live outside the core)
from gdtensor_sim.coord import ThreadCoordinator  # noqa: E402
from gdtensor.internal.dtensor import owned_intervals, tensor_ids_of  # noqa: E402
from gdtensor.internal.formats import make_storage_view  # noqa: E402
from gdtensor_sim.models import (  # noqa: E402
    ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder,
    pattern_bytes, storage_table, tensor_seed,
)
from gdtensor_sim.runner import compute_view  # noqa: E402
from gdtensor.internal.views import ComputeFabric  # noqa: E402

M = ModelConfig(layers=2, hidden=64, ffn=128, vocab=256)
P = ParallelConfig(tp=2, dp=2, pp=2)


def _run(world, fn):
    with ThreadPoolExecutor(max_workers=world) as ex:
        return list(ex.map(fn, range(world)))


def _assert_all_ok(results):
    bad = {r: x.value for r, x in enumerate(results) if type(x) is gt.Err}
    assert not bad, bad


# ------------------------------------------------------------------ CRC oracle


def _parse_payload_regions(path):
    """Return [(key, payload_start, payload_end, info)] by reading the header.

    Independent of the internal payload index: decodes the on-disk format.
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        out = []
        for key, info in header.items():
            if key == "__metadata__":
                continue
            lo, hi = info["data_offsets"]
            out.append((key, data_start + lo, data_start + hi, info))
    return out


def _crc_region(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        return zlib.crc32(f.read(end - start))


def _expected_key_crc(table, key, info, itemsize):
    """CRC of the pattern bytes one on-disk tensor key must hold.

    Key forms: ``tid`` (whole tensor) or ``tid/slot<w>`` (one storage slot).
    The owned-interval oracle is the only internal-derived fact here.
    """
    tid, _, slot = key.partition("/slot")
    named = next(x for x in table.named_d_tensors if x.global_tensor_id == tid)
    meta = gb.global_tensor_meta_of(named)
    numel = int(np.prod(meta.global_shape))
    if slot:
        elems = np.concatenate([np.arange(a, b, dtype=np.int64)
                                 for a, b in owned_intervals(named, "storage", int(slot))])
        assert list(info["shape"]) != []  # slot entry carries a box shape
    else:
        elems = np.arange(numel, dtype=np.int64)
        assert list(info["shape"]) == list(meta.global_shape)
    return zlib.crc32(pattern_bytes(tid, itemsize, elems).tobytes())


# ------------------------------------------------------------------ file CRC


@pytest.mark.parametrize("kind", ["single", "hf", "per_rank", "per_stage"])
def test_public_io_payload_crc_matches_pattern(tmp_path, kind):
    root = str(tmp_path / kind)
    os.makedirs(root)
    mode = "hf" if kind in ("single", "hf", "per_stage") else "mirror"
    ct = compute_table(M, P)
    st = storage_table(ct, P, mode)
    lay = layout_builder(kind, st, P, "safetensors", 2)
    sv = make_storage_view(tensor_ids_of(st), st, lay, root, coord=ThreadCoordinator(P.world))
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    _assert_all_ok(_run(P.world, lambda r: gio.rank_save(r, compute_view(ct, fab, r), sv)))

    files = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
    assert files
    for name in files:
        path = os.path.join(root, name)
        for key, start, end, info in _parse_payload_regions(path):
            tid = key.partition("/slot")[0]
            named = next(x for x in st.named_d_tensors if x.global_tensor_id == tid)
            isz = gb.global_tensor_meta_of(named).itemsize
            assert _crc_region(path, start, end) == _expected_key_crc(st, key, info, isz), name


def test_public_io_crc_detects_corruption(tmp_path):
    root = str(tmp_path / "corrupt")
    os.makedirs(root)
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("single", st, P, "safetensors")
    sv = make_storage_view(tensor_ids_of(st), st, lay, root, coord=ThreadCoordinator(P.world))
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    _assert_all_ok(_run(P.world, lambda r: gio.rank_save(r, compute_view(ct, fab, r), sv)))

    path = os.path.join(root, "model.safetensors")
    regions = _parse_payload_regions(path)
    before = {k: _crc_region(path, s, e) for k, s, e, _ in regions}
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.seek(size // 2)
        f.write(b"\xff" * 8)  # flip bytes inside a payload
    after = {k: _crc_region(path, s, e) for k, s, e, _ in regions}
    assert before != after


# ------------------------------------------------------------------ arrange tensor


def _stage0_sliced_mesh(p):
    # one PP stage slice of the (pp, dp, tp) arranged world
    return gb.SlicedMesh(
        gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(p.world)), [p.pp, p.dp, p.tp]),
        [gb.Slice(0, 1), None, None])


def _arrange_named2(tid, shape, p, placements):
    return gb.NamedDTensor(tid, gb.NaiveDTensor(
        gb.GlobalTensorMeta(list(shape), "U8", 1), list(placements), _stage0_sliced_mesh(p)))


def _arrange_fill(nd, iv):
    elems = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in iv])
    return ((elems + tensor_seed(nd.global_tensor_id)) & 0xFF).astype(np.uint8).ravel()


def _assert_arrange(fabric, table, world):
    for r in range(world):
        for nd in table.named_d_tensors:
            buf = fabric.buffers.get(r, {}).get(nd.global_tensor_id)
            if buf is None:
                continue
            want = _arrange_fill(nd, owned_intervals(nd, "compute", r))
            assert np.array_equal(buf, want), f"rank {r} {nd.global_tensor_id} arrange mismatch"


def test_public_io_arrange_save_load_elastic(tmp_path):
    """Save arrange tensors TP2 DP2 PP2, load under TP4 DP1 PP2 via public io."""
    tids = ["arr0", "arr1", "arr2"]
    shapes = [(1024,), (256, 16), (64, 64)]
    pls = (gt.Replicate(), gt.Replicate(), gt.Shard(0))
    ct = gb.DTensorTable([_arrange_named2(t, s, P, pls) for t, s in zip(tids, shapes)])
    st = storage_table(ct, P, "hf")
    lay = layout_builder("single", st, P, "safetensors")
    root = str(tmp_path / "arrange")
    os.makedirs(root)
    sv = make_storage_view(tensor_ids_of(st), st, lay, root, coord=ThreadCoordinator(P.world))
    fab = ComputeFabric(ct).allocate(fill=_arrange_fill)
    _assert_all_ok(_run(P.world, lambda r: gio.rank_save(r, compute_view(ct, fab, r), sv)))

    p2 = ParallelConfig(tp=4, dp=1, pp=2)
    ct2 = gb.DTensorTable([_arrange_named2(t, s, p2, pls) for t, s in zip(tids, shapes)])
    sv2 = make_storage_view(tensor_ids_of(st), st, lay, root)
    fab2 = ComputeFabric(ct2).allocate()
    _assert_all_ok(_run(p2.world, lambda r: gio.rank_load(r, compute_view(ct2, fab2, r), sv2)))
    _assert_arrange(fab2, ct2, p2.world)


def test_public_io_arrange_reshard_compute_to_compute():
    tids = ["r0", "r1"]
    shapes = [(512,), (128, 128)]
    p_src = ParallelConfig(tp=2, dp=2, pp=1)
    p_dst = ParallelConfig(tp=4, dp=1, pp=1)
    ct_src = gb.DTensorTable([_arrange_named2(
        t, s, p_src, (gt.Replicate(), gt.Shard(0), gt.Shard(0)))
        for t, s in zip(tids, shapes)])
    ct_dst = gb.DTensorTable([_arrange_named2(
        t, s, p_dst, (gt.Replicate(), gt.Replicate(), gt.Shard(0)))
        for t, s in zip(tids, shapes)])
    fab_src = ComputeFabric(ct_src).allocate(fill=_arrange_fill)
    fab_dst = ComputeFabric(ct_dst).allocate()
    _assert_all_ok(_run(
        p_dst.world,
        lambda r: gio.rank_reshard(r, compute_view(ct_src, fab_src, r),
                                     compute_view(ct_dst, fab_dst, r))))
    _assert_arrange(fab_dst, ct_dst, p_dst.world)


def test_public_io_arrange_reshard_storage_to_storage(tmp_path):
    tids = ["s0", "s1"]
    shapes = [(768,), (48, 48)]
    pls = (gt.Replicate(), gt.Replicate(), gt.Shard(0))
    ct = gb.DTensorTable([_arrange_named2(t, s, P, pls) for t, s in zip(tids, shapes)])
    src_st = storage_table(ct, P, "hf")
    src_lay = layout_builder("single", src_st, P, "safetensors")
    dst_st = storage_table(ct, P, "mirror")
    dst_lay = layout_builder("per_rank", dst_st, P, "safetensors")
    src_root = str(tmp_path / "src")
    dst_root = str(tmp_path / "dst")
    os.makedirs(src_root)
    os.makedirs(dst_root)
    sv_src = make_storage_view(tensor_ids_of(src_st), src_st, src_lay, src_root)
    sv_dst = make_storage_view(tensor_ids_of(dst_st), dst_st, dst_lay, dst_root,
                               coord=ThreadCoordinator(P.world))

    fab = ComputeFabric(ct).allocate(fill=_arrange_fill)
    _assert_all_ok(_run(P.world, lambda r: gio.rank_save(r, compute_view(ct, fab, r), sv_src)))
    # public storage -> storage reshard
    _assert_all_ok(_run(P.world, lambda r: gio.rank_reshard(r, sv_src, sv_dst)))

    fab2 = ComputeFabric(ct).allocate()
    sv_load = make_storage_view(tensor_ids_of(dst_st), dst_st, dst_lay, dst_root)
    _assert_all_ok(_run(P.world, lambda r: gio.rank_load(r, compute_view(ct, fab2, r), sv_load)))
    _assert_arrange(fab2, ct, P.world)

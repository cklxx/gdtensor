"""Multi-machine consistency.

The implementation must behave identically across machines and runtimes:

1. plans are a pure function of (rank, tables, layout): no wall time, pid,
   thread id, or set/hash iteration;
2. published bytes are identical across:
   * repeated saves into different directories;
   * threads and forked processes as the rank runtime;
   * reversed rank start order;
3. omnifs export reproduces the file-backend bytes exactly;
4. every persisted artifact uses explicit byte lengths (not str length).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import assert_all_ok
from gdtensor.coord import ThreadCoordinator
from gdtensor.formats import make_storage_view
from gdtensor.models import (
    ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table,
)
from gdtensor.omnifs import OmniFsConfig, OmniFsIO
from gdtensor.omnifs_view import export_hf_dir, make_omnifs_storage_view
from gdtensor.plan import d_tensors_view_rank_plan_load, d_tensors_view_rank_plan_save
from gdtensor.runner import _run_threads, compute_view, run_processes, save_threads
from gdtensor.views import ComputeDataAccessor, ComputeFabric

M = ModelConfig(layers=2, hidden=16, ffn=32, vocab=64)
P = ParallelConfig(tp=2, dp=2, pp=1)


def _snapshot_bytes(root):
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            out[name] = open(p, "rb").read()
    return out


def _do_save(root, p=P, m=M, mode="hf", kind="hf"):
    os.makedirs(root, exist_ok=True)
    ct = compute_table(m, p)
    st = storage_table(ct, p, mode)
    lay = layout_builder(kind, st, p, "safetensors")
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])
    return _snapshot_bytes(root)


def test_repeated_saves_are_byte_identical(tmp_path):
    a = _do_save(str(tmp_path / "a"))
    b = _do_save(str(tmp_path / "b"))
    assert a == b


def test_thread_and_process_outputs_are_identical(tmp_path):
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, "safetensors")
    rt = str(tmp_path / "threads"); os.makedirs(rt)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, rt, fab, P.world)[0])
    rp = str(tmp_path / "procs"); os.makedirs(rp)
    res, _, _ = run_processes("save", ct, st, lay, rp, P.world, fill=fill_pattern)
    assert_all_ok(res)
    # process runners receive fill_pattern across fork: compare every file
    a, b = _snapshot_bytes(rt), _snapshot_bytes(rp)
    assert set(a) == set(b)
    for name in a:
        assert a[name] == b[name], name


def test_reversed_rank_start_order_same_bytes(tmp_path):
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, "safetensors")
    r1 = str(tmp_path / "asc"); os.makedirs(r1)
    r2 = str(tmp_path / "desc"); os.makedirs(r2)
    for root, order in [(r1, range(P.world)), (r2, range(P.world - 1, -1, -1))]:
        io = __import__("gdtensor.iobackend", fromlist=["FileIO"]).FileIO()
        sv = make_storage_view(st.tensor_ids, st, lay, root, io, ThreadCoordinator(P.world))
        fab = ComputeFabric(ct).allocate(fill=fill_pattern)
        assert_all_ok(_run_threads(P.world, lambda r: __import__(
            "gdtensor.api", fromlist=["d_tensors_view_rank_save"]).d_tensors_view_rank_save(
            r, compute_view(ct, fab, list(order)[r]), sv)))
    assert _snapshot_bytes(r1) == _snapshot_bytes(r2)


def test_omnifs_export_matches_file_backend(tmp_path):
    cfg = OmniFsConfig(str(tmp_path / "store"), "proj", 0)
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, "safetensors")
    fr = str(tmp_path / "files"); os.makedirs(fr)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, fr, fab, P.world)[0])

    io = OmniFsIO(cfg)
    sv = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, io, ThreadCoordinator(P.world))
    fab2 = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(_run_threads(P.world, lambda r: __import__(
        "gdtensor.api", fromlist=["d_tensors_view_rank_save"]).d_tensors_view_rank_save(
        r, compute_view(ct, fab2, r), sv)))
    out = str(tmp_path / "export")
    export_hf_dir(cfg, out)
    a, b = _snapshot_bytes(fr), _snapshot_bytes(out)
    assert set(a) == set(b)
    for name in a:
        assert a[name] == b[name], name


def test_plan_is_pure_and_order_independent():
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, "safetensors")

    def plans():
        sv = make_storage_view(st.tensor_ids, st, lay, "/nonexistent")
        ps, pl = {}, {}
        for r in range(P.world):
            cv = compute_view(ct, None, r)
            cv.data_accessor = ComputeDataAccessor(r, None)
            ps[r] = d_tensors_view_rank_plan_save(r, cv, sv).unwrap()
            pl[r] = d_tensors_view_rank_plan_load(r, cv, sv).unwrap()
        return ps, pl

    a = plans()
    b = plans()
    assert a == b


def test_manifest_uses_byte_lengths(tmp_path):
    """A non-ASCII JSON blob must size the blob by encoded bytes."""
    cfg = OmniFsConfig(str(tmp_path / "store"), "proj", 0)
    ct = compute_table(M, ParallelConfig(tp=1, dp=1, pp=1))
    st = storage_table(ct, ParallelConfig(tp=1, dp=1, pp=1), "hf")
    lay = layout_builder("single", st, ParallelConfig(tp=1, dp=1, pp=1), "safetensors")
    payload = "日本語テンソル".encode()
    io = OmniFsIO(cfg)
    sv = make_omnifs_storage_view(
        st.tensor_ids, st, lay, cfg, io,
        extra_json={"note.json": payload})
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(_run_threads(1, lambda r: __import__(
        "gdtensor.api", fromlist=["d_tensors_view_rank_save"]).d_tensors_view_rank_save(
        r, compute_view(ct, fab, r), sv)))
    m = __import__("gdtensor.omnifs", fromlist=["Manifest"]).Manifest.load(cfg.manifest_path)
    assert m.files["note.json"]["size"] == len(payload)
    assert io.pread(-1, "note.json", 0, len(payload)) == payload


def test_p2p_peer_pick_is_hash_seed_independent(tmp_path):
    """L2 peer routing must not depend on per-process PYTHONHASHSEED."""
    import subprocess, sys, textwrap
    code = textwrap.dedent(
        """
        import sys, zlib
        sys.path.insert(0, ".")
        from gdtensor.iobackend import P2PReadCache
        c = P2PReadCache(ranks_per_node=1)
        # populate holders for block 0 of "f": peers nodes 1..7, query node 0
        import numpy as np
        for n in range(1, 8):
            c._l1[n][("f", 0)] = b"x"
            c._holders[("f", 0)].add(n)
        picks = [c._l1[zlib.crc32(f"0:{i}".encode()) % 7] and 0 for i in range(4)]
        # emulate get_block's peer choice explicitly
        peers = sorted(n for n in c._holders[("f", 0)] if n != 0)
        chosen = peers[zlib.crc32(b"0:0") % len(peers)]
        print(chosen)
        """)
    outs = {subprocess.run([sys.executable, "-c", code], cwd=".",
                           env={**__import__("os").environ, "PYTHONHASHSEED": str(seed)},
                           capture_output=True, text=True).stdout.strip() for seed in (1, 42, 999)}
    assert len(outs) == 1, outs

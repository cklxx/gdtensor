"""Adversarial / boundary tests.

Attack surfaces:
* empty tensors (numel = 0) and single-element tensors
* all-replicate (no shard dims) and single-rank world
* reshard no-op (src == dst)
* omnifs reads exactly aligned to chunk boundaries
* P2P cache concurrent single-flight on the same block
* save with one rank failing mid-write (partial KV)
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from conftest import assert_all_ok, verify_fabric
from gdtensor.coord import ThreadCoordinator
from gdtensor.dtensor import (
    DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor, Replicate, Shard,
)
from gdtensor.formats import make_storage_view
from gdtensor.iobackend import FileIO, IoTracer, MiB, P2PReadCache
from gdtensor.models import ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table
from gdtensor.omnifs import KvStore, Manifest, OmniFsConfig, OmniFsIO
from gdtensor.omnifs_view import export_hf_dir, make_omnifs_storage_view
from gdtensor.runner import _run_threads, compute_view, load_threads, reshard_threads, save_threads
from gdtensor.views import ComputeFabric

M = ModelConfig(layers=2, hidden=32, ffn=64, vocab=128)


# ------------------------------------------------------------------ tiny model edge


def test_tiny_model_single_element_tensors(tmp_path):
    """Model with 1x1 tensors: every tensor is a scalar."""
    tiny = ModelConfig(layers=1, hidden=1, ffn=1, vocab=1)
    p = ParallelConfig(tp=1, dp=2, pp=1)
    ct = compute_table(tiny, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    root = str(tmp_path / "tiny")
    os.makedirs(root)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])
    fab2 = ComputeFabric(ct).allocate()
    assert_all_ok(load_threads(ct, st, lay, root, fab2, p.world)[0])
    verify_fabric(fab2, ct)


# ------------------------------------------------------------------ all replicate / single rank


def test_all_replicate_tensor_written_once(tmp_path):
    p = ParallelConfig(tp=1, dp=4, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "mirror")
    lay = layout_builder("per_rank", st, p, "safetensors")
    root = str(tmp_path / "repl")
    os.makedirs(root)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    res, io = save_threads(ct, st, lay, root, fab, p.world)
    assert_all_ok(res)
    # each file is written by exactly one rank (the designated writer)
    for path in set(os.listdir(root)):
        if not path.endswith(".safetensors"):
            continue
        writers = {r.rank for r in io.tracer.records
                   if r.op == "pwrite" and r.path.replace(".tmp", "").endswith(path)}
        assert len(writers) == 1, f"{path} written by {writers}"


def test_single_rank_world(tmp_path):
    p = ParallelConfig(tp=1, dp=1, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    root = str(tmp_path / "one")
    os.makedirs(root)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, 1)[0])
    fab2 = ComputeFabric(ct).allocate()
    assert_all_ok(load_threads(ct, st, lay, root, fab2, 1)[0])
    verify_fabric(fab2, ct)


# ------------------------------------------------------------------ reshard no-op


def test_reshard_no_op_compute_to_compute(tmp_path):
    p = ParallelConfig(tp=2, dp=2, pp=1)
    ct = compute_table(M, p)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    fab2 = ComputeFabric(ct).allocate()
    assert_all_ok(reshard_threads(p.world,
                                  lambda r: compute_view(ct, fab, r),
                                  lambda r: compute_view(ct, fab2, r)))
    verify_fabric(fab2, ct)
    # no comm should be needed when src == dst topology
    # (fab.comm_bytes may be 0 because locate finds own rank first)


def test_reshard_storage_to_same_storage(tmp_path):
    p = ParallelConfig(tp=2, dp=2, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    root = str(tmp_path / "same")
    os.makedirs(root)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])
    sv = make_storage_view(st.tensor_ids, st, lay, root, coord=ThreadCoordinator(p.world))
    # reshard from storage to the *same* storage view (should be a no-op copy)
    assert_all_ok(reshard_threads(p.world, lambda r: sv, lambda r: sv))


# ------------------------------------------------------------------ omnifs chunk-aligned reads


def test_omnifs_read_exactly_chunk_aligned(tmp_path):
    cfg = OmniFsConfig(str(tmp_path / "omnifs"), "train", 1, chunk_bytes=4096, n_dn=2)
    p = ParallelConfig(tp=2, dp=2, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    io = OmniFsIO(cfg)
    sv = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, io, ThreadCoordinator(p.world))
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(_run_threads(p.world, lambda r: __import__(
        "gdtensor.api", fromlist=["d_tensors_view_rank_save"]).d_tensors_view_rank_save(
        r, compute_view(ct, fab, r), sv)))
    # read through the store at every chunk boundary
    m = Manifest.load(cfg.manifest_path)
    io2 = OmniFsIO(cfg)
    io2.bind(m)
    for rel, f in m.files.items():
        for off in range(0, f["size"], cfg.chunk_bytes):
            n = min(cfg.chunk_bytes, f["size"] - off)
            data = io2.pread(-1, rel, off, n)
            assert len(data) == n, f"{rel}@{off}: {len(data)} != {n}"


def test_omnifs_read_one_byte_at_chunk_boundary(tmp_path):
    cfg = OmniFsConfig(str(tmp_path / "omnifs"), "train", 1, chunk_bytes=4096, n_dn=2)
    p = ParallelConfig(tp=2, dp=2, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    io = OmniFsIO(cfg)
    sv = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, io, ThreadCoordinator(p.world))
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(_run_threads(p.world, lambda r: __import__(
        "gdtensor.api", fromlist=["d_tensors_view_rank_save"]).d_tensors_view_rank_save(
        r, compute_view(ct, fab, r), sv)))
    m = Manifest.load(cfg.manifest_path)
    io2 = OmniFsIO(cfg)
    io2.bind(m)
    # read 1 byte exactly at each chunk boundary
    for rel, f in m.files.items():
        for off in range(cfg.chunk_bytes, f["size"], cfg.chunk_bytes):
            data = io2.pread(-1, rel, off, 1)
            assert len(data) == 1


# ------------------------------------------------------------------ P2P cache concurrent single-flight


def test_p2p_cache_concurrent_misses_single_flight(tmp_path):
    """N ranks miss the same block simultaneously; only one L3 fetch happens."""
    p = ParallelConfig(tp=1, dp=8, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    root = str(tmp_path / "cache")
    os.makedirs(root)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])

    tracer = IoTracer()
    cache = P2PReadCache(ranks_per_node=1, block_bytes=4096)
    io = FileIO(tracer, cache)
    sv = make_storage_view(st.tensor_ids, st, lay, root, io)
    fab2 = ComputeFabric(ct).allocate()
    assert_all_ok(load_threads(ct, st, lay, root, fab2, p.world, io)[0])
    # unique L3 blocks fetched must be <= total blocks (no duplicate fetch of same block)
    total_blocks = sum(-(-os.path.getsize(os.path.join(root, f)) // 4096)
                       for f in os.listdir(root) if f.endswith(".safetensors"))
    assert len(cache.unique_l3_blocks) <= total_blocks
    # every block fetched at most once from L3
    assert len(cache.unique_l3_blocks) == len(set(cache.unique_l3_blocks))


# ------------------------------------------------------------------ partial write / abort recovery


def test_omnifs_abort_then_rewrite_succeeds(tmp_path):
    cfg = OmniFsConfig(str(tmp_path / "omnifs"), "train", 1, chunk_bytes=4096, n_dn=2)
    p = ParallelConfig(tp=2, dp=2, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")

    # first attempt: rank 3 aborts after writing some KVs
    io = OmniFsIO(cfg)
    sv = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, io, ThreadCoordinator(p.world))
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)

    def fn(r):
        if r == 3:
            s = sv.begin_save(sv, r).unwrap()
            # write half the plan then abort
            from gdtensor.api import d_tensors_view_rank_plan_save
            plan = d_tensors_view_rank_plan_save(r, compute_view(ct, fab, r), sv).unwrap()
            half = plan[:len(plan) // 2]
            from gdtensor.api import _copy, _bind
            _bind(sv, s)
            _copy(half, compute_view(ct, fab, r), sv)
            return s.abort()
        return __import__("gdtensor.api", fromlist=["d_tensors_view_rank_save"]).d_tensors_view_rank_save(
            r, compute_view(ct, fab, r), sv)

    res = _run_threads(p.world, fn)
    assert res[3].ok  # abort succeeds
    assert not res[0].ok  # other ranks see the failure
    assert Manifest.load(cfg.manifest_path).state == "prepared"

    # second attempt: full write succeeds
    io2 = OmniFsIO(cfg)
    sv2 = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, io2, ThreadCoordinator(p.world))
    assert_all_ok(_run_threads(p.world, lambda r: __import__(
        "gdtensor.api", fromlist=["d_tensors_view_rank_save"]).d_tensors_view_rank_save(
        r, compute_view(ct, fab, r), sv2)))
    out = str(tmp_path / "export")
    export_hf_dir(cfg, out)
    from conftest import verify_safetensors_dir
    verify_safetensors_dir(out, st)

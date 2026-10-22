"""Adversarial / boundary tests.

Attack surfaces:
* empty tensors (numel = 0) and single-element tensors
* all-replicate (no shard dims) and single-rank world
* reshard no-op (src == dst)
* save with one rank aborting mid-write, then a full rewrite
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from conftest import assert_all_ok, verify_fabric
from gdtensor.internal.dtensor import tensor_ids_of as _ids  # noqa: E402
from gdtensor.internal.api import d_tensors_view_rank_save
from gdtensor.internal.coord import ThreadCoordinator
from gdtensor.internal.dtensor import (
    DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor, Replicate, Shard,
)
from gdtensor.internal.formats import make_storage_view
from gdtensor.internal.iobackend import FileIO, IoTracer, MiB
from gdtensor.internal.models import ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table
from gdtensor.internal.runner import _run_threads, compute_view, load_threads, reshard_threads, save_threads
from gdtensor.internal.views import ComputeFabric

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
    sv = make_storage_view(_ids(st), st, lay, root, coord=ThreadCoordinator(p.world))
    # reshard from storage to the *same* storage view (should be a no-op copy)
    assert_all_ok(reshard_threads(p.world, lambda r: sv, lambda r: sv))


# ------------------------------------------------------------------ partial write / abort recovery


def test_abort_then_rewrite_succeeds(tmp_path):
    """One rank aborts after a partial write; a full rewrite then publishes."""
    p = ParallelConfig(tp=2, dp=2, pp=1)
    ct = compute_table(M, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    root = str(tmp_path / "ck")
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    sv = make_storage_view(_ids(st), st, lay, root, coord=ThreadCoordinator(p.world))

    def fn(r):
        if r == 3:
            from gdtensor.internal.api import _bind, _copy
            from gdtensor.internal.plan import d_tensors_view_rank_plan_save
            s = sv.begin_save(sv, r).unwrap()
            plan = d_tensors_view_rank_plan_save(r, compute_view(ct, fab, r), sv).unwrap()
            _bind(sv, s)
            _copy(plan[:len(plan) // 2], compute_view(ct, fab, r), sv)
            return s.abort()
        return d_tensors_view_rank_save(r, compute_view(ct, fab, r), sv)

    res = _run_threads(p.world, fn)
    assert res[3].ok  # abort succeeds
    assert not res[0].ok  # other ranks see the failure
    assert not any(f.endswith(".safetensors") for f in os.listdir(root))  # nothing published

    sv2 = make_storage_view(_ids(st), st, lay, root, coord=ThreadCoordinator(p.world))
    assert_all_ok(_run_threads(p.world, lambda r: d_tensors_view_rank_save(r, compute_view(ct, fab, r), sv2)))
    from conftest import verify_safetensors_dir
    verify_safetensors_dir(root, st)

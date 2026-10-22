"""Numbers from "HF 文件格式与并行下的读写行为" (Qwen3-235B-A22B, layer 0, TP=4).

Plan-level checks only (no data, no session): the plan API is pure. Write
pieces here are logical ranges; a 64 MiB object-store part size is used only
to count writers per part.
"""
from __future__ import annotations

import collections
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import assert_all_ok, verify_safetensors_dir
from gdtensor.internal.dtensor import tensor_ids_of as _ids  # noqa: E402
from gdtensor.internal.dtensor import DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor
from gdtensor_sim.models import (
    ModelConfig, ParallelConfig, compute_mesh, compute_table, fill_pattern, layout_builder,
    storage_table,
)
from gdtensor.internal.api import d_tensors_view_rank_plan_save
from gdtensor.internal.ranges import RangeFromStorageView
from gdtensor.internal.dtensor import Replicate, Shard
from gdtensor.internal.iobackend import FileIO
from gdtensor_sim.coord import ThreadCoordinator
from gdtensor.internal.formats import make_storage_view
from gdtensor_sim.runner import compute_view, reshard_threads
from gdtensor.internal.views import ComputeDataAccessor, ComputeFabric

MiB = 1 << 20
KiB = 1 << 10
CHUNK_BYTES = 64 * MiB  # object-store part size used for the writer count


def _qwen3_layer0_table(tp: int):
    """q_proj [8192,4096] dim0, o_proj [4096,8192] dim1, q_norm [128] rep, BF16."""
    p = ParallelConfig(tp=tp, dp=1, pp=1)
    nds = [
        NamedDTensor("q_proj", NaiveDTensor(GlobalTensorMeta((8192, 4096), "BF16", 2),
                                            (Replicate(), Replicate(), Shard(0)), compute_mesh(p, 0))),
        NamedDTensor("o_proj", NaiveDTensor(GlobalTensorMeta((4096, 8192), "BF16", 2),
                                            (Replicate(), Replicate(), Shard(1)), compute_mesh(p, 0))),
        NamedDTensor("q_norm", NaiveDTensor(GlobalTensorMeta((128,), "BF16", 2),
                                            (Replicate(), Replicate(), Replicate()), compute_mesh(p, 0))),
    ]
    return p, DTensorTable(nds)


def test_qwen3_tp4_write_pattern():
    p, ct = _qwen3_layer0_table(4)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    by_tensor = {tid: [] for tid in _ids(ct)}
    for r in range(p.world):
        cv = compute_view(ct, None, r)
        cv.data_accessor = ComputeDataAccessor(r, None)
        plan = d_tensors_view_rank_plan_save(r, cv, make_storage_view(_ids(st), st, lay, "/nonexistent")).unwrap()
        for _, d in plan:
            by_tensor[_tid_of(plan, d)].append(d)
    q = by_tensor["q_proj"]
    assert len(q) == 4 and all(x.size == 16 * MiB for x in q)
    # o_proj: each rank writes one 4 KiB piece per row, 4096 rows.
    o = by_tensor["o_proj"]
    assert len(o) == 4 * 4096 and all(x.size == 4 * KiB for x in o)
    # q_norm: 256 B tensor, exactly one writer.
    n = by_tensor["q_norm"]
    assert len(n) == 1 and n[0].size == 256


def _tid_of(plan, dst: RangeFromStorageView) -> str:
    for s, d in plan:
        if d is dst:
            return s.global_tensor_id
    raise KeyError


def test_q_proj_and_o_proj_offsets_unaligned():
    """File offset = 8 + N + data_offsets. The JSON header N pushes every
    tensor start off the 1 MiB boundary (doc: P, Q are not aligned)."""
    from gdtensor.internal.formats import safetensors_header
    from gdtensor.internal.layout import build_payload_index
    p, ct = _qwen3_layer0_table(4)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    data_start = len(safetensors_header(st, build_payload_index(st, lay), "model.safetensors"))
    assert data_start % MiB != 0
    sv = make_storage_view(_ids(st), st, lay, "/nonexistent")
    seen = []
    for r in range(p.world):
        cv = compute_view(ct, None, r)
        cv.data_accessor = ComputeDataAccessor(r, None)
        plan = d_tensors_view_rank_plan_save(r, cv, sv).unwrap()
        seen += [(s.global_tensor_id, d.contiguous_payload_offset, d.size) for s, d in plan]
    for tid, off, size in seen:
        if size >= 4 * KiB:
            assert (data_start + off) % MiB != 0


def test_o_proj_one_chunk_many_writers():
    """One 64 MiB chunk carries o_proj rows from every TP rank (chunk 58 idea)."""
    p, ct = _qwen3_layer0_table(4)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    sv = make_storage_view(_ids(st), st, lay, "/nonexistent")
    per_chunk = collections.defaultdict(set)
    for r in range(p.world):
        cv = compute_view(ct, None, r)
        cv.data_accessor = ComputeDataAccessor(r, None)
        for s, d in d_tensors_view_rank_plan_save(r, cv, sv).unwrap():
            if s.global_tensor_id == "o_proj":
                c0 = d.contiguous_payload_offset // CHUNK_BYTES
                per_chunk[c0].add(r)
    assert max(len(v) for v in per_chunk.values()) == 4


# ------------------------------------------------------------------ aggregate switch


def test_aggregate_switch_moves_dim1_writes_to_one_rank(tmp_path):
    """aggregate_tp is a business layout choice (design 4.3): each TP-dim1
    tensor is stored by one TP rank. The thin executor (B.1.1) does not batch
    pieces, so the switch changes who writes, not the piece count."""
    m = ModelConfig(layers=2, hidden=256, ffn=512, vocab=1024)
    p = ParallelConfig(tp=4, dp=2, pp=1)
    ct = compute_table(m, p)
    dim1 = [nd.global_tensor_id for nd in ct.named_d_tensors
            if isinstance(nd.d_tensor, NaiveDTensor) and type(nd.d_tensor.placements[2]) is Shard and nd.d_tensor.placements[2].value == 1]
    assert dim1
    writers = {}
    for agg in (False, True):
        d = str(tmp_path / ("agg" if agg else "raw"))
        st = storage_table(ct, p, "hf", aggregate_tp=agg)
        lay = layout_builder("single", st, p, "safetensors")
        fab = ComputeFabric(ct).allocate(fill=fill_pattern)
        sv = make_storage_view(_ids(st), st, lay, d, FileIO(), ThreadCoordinator(p.world))
        assert_all_ok(reshard_threads(p.world, lambda r: compute_view(ct, fab, r), lambda r: sv))
        verify_safetensors_dir(d, st)
        from gdtensor.internal.plan import d_tensors_view_rank_plan_reshard
        per_tensor = collections.defaultdict(set)
        for r in range(p.world):
            plan = d_tensors_view_rank_plan_reshard(r, compute_view(ct, fab, r), sv).unwrap()
            for src, _ in plan:
                if src.global_tensor_id in dim1:
                    per_tensor[src.global_tensor_id].add(r)
        writers[agg] = max(len(v) for v in per_tensor.values())
    assert writers[False] == p.tp
    assert writers[True] == 1


def test_zero_flat_shard_size(tmp_path):
    """ZeRO m/v: flattened and split over the DP x TP group (doc: 128 B case)."""
    p, ct = _qwen3_layer0_table(4)
    # q_norm m state: FP32, 128 elems = 512 B; split over TP=4 -> 128 B each.
    from gdtensor.internal.dtensor import ReshapedDTensor
    flat = GlobalTensorMeta((128,), "F32", 4)
    m = NamedDTensor("q_norm.exp_avg", ReshapedDTensor(
        NaiveDTensor(flat, (Replicate(), Replicate(), Shard(0)), compute_mesh(p, 0)), (128,)))
    table = DTensorTable(ct.named_d_tensors + [m])
    st = storage_table(table, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    sv = make_storage_view(_ids(st), st, lay, "/nonexistent")
    sizes = []
    for r in range(p.world):
        cv = compute_view(table, None, r)
        cv.data_accessor = ComputeDataAccessor(r, None)
        for s, d in d_tensors_view_rank_plan_save(r, cv, sv).unwrap():
            if s.global_tensor_id == "q_norm.exp_avg":
                sizes.append(d.size)
    assert sorted(sizes) == [128, 128, 128, 128]

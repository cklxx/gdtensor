"""Use case matrix of DTensorsView design section 7, plus invariants (section 6)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from conftest import assert_all_ok, raw_header, verify_distcp_dir, verify_fabric, verify_safetensors_dir
from gdtensor.internal.dtensor import nbytes_of, tensor_ids_of as _ids  # noqa: E402
from gdtensor.internal.dtensor import (
    DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor, Replicate, Shard, global_tensor_meta_of,
)
from gdtensor.internal.formats import make_storage_view
from gdtensor.internal.layout import TensorComponentPos, build_layout, build_payload_index, file_path_of
from gdtensor.internal.mesh import ArrangedMesh, OuterProductMesh, Slice, SlicedMesh
from gdtensor.tag import ConstantOne, TensorCount, WorldSize
from gdtensor.internal.models import (
    ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table,
)
from gdtensor.internal.plan import (
    check_compute_d_tensors_access_disjoint, check_compute_d_tensors_access_full_covering,
    check_physical_layout, check_storage_d_tensors_access_disjoint, check_storage_d_tensors_access_full_covering,
    check_storage_mesh_disjoint, d_tensors_view_rank_plan_load, d_tensors_view_rank_plan_reshard,
    d_tensors_view_rank_plan_save,
)
from gdtensor.internal.runner import compute_view, load_threads, reshard_threads, run_processes, save_threads
from gdtensor.internal.views import ComputeFabric

M = ModelConfig(layers=4, hidden=32, ffn=64, vocab=128)
MOE = ModelConfig(layers=2, hidden=32, ffn=48, vocab=128, experts=4)
P = ParallelConfig(tp=2, dp=2, pp=2)


def roundtrip(root, m, p, mode, kind, fmt, hf_shards=2, load_p=None):
    ct = compute_table(m, p)
    st = storage_table(ct, p, mode)
    lay = layout_builder(kind, st, p, fmt, hf_shards)
    assert check_physical_layout(st, lay).ok, check_physical_layout(st, lay)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    res, io = save_threads(ct, st, lay, root, fab, p.world)
    assert_all_ok(res)
    (verify_safetensors_dir if fmt == "safetensors" else verify_distcp_dir)(root, st)
    lp = load_p or p
    ct2 = compute_table(m, lp)
    fab2 = ComputeFabric(ct2).allocate()
    res2, io2 = load_threads(ct2, st, lay, root, fab2, lp.world)
    assert_all_ok(res2)
    verify_fabric(fab2, ct2)
    return ct, st, lay, io


# ------------------------------------------------------------------ matrix


@pytest.mark.parametrize("fmt", ["distcp", "safetensors"])
def test_per_rank_file(root, fmt):
    _, st, lay, _ = roundtrip(root, M, P, "mirror", "per_rank", fmt)
    leaders = {file_path_of(lay, pos).unwrap(): pos.rank for pos in lay.tensor_entries}
    assert all(f"__{r}_0." in path for path, r in leaders.items())


def test_multi_rank_one_safetensors(root):
    ct, st, lay, io = roundtrip(root, M, P, "hf", "single", "safetensors")
    writers = {r.rank for r in io.tracer.records if r.op == "pwrite" and r.path.endswith("model.safetensors.tmp")}
    assert writers == set(range(P.world))  # every rank pwrites into the one shared file
    header, _ = raw_header(os.path.join(root, "model.safetensors"))
    assert len([k for k in header if "/slot" in k]) == 0  # HF case: one entry per tensor


def test_hf_sharded_files_and_moe(root):
    roundtrip(root, MOE, P, "hf", "hf", "safetensors", hf_shards=3)
    assert os.path.exists(os.path.join(root, "model.safetensors.index.json"))


def test_one_rank_many_files(root):
    _, st, lay, _ = roundtrip(root, M, P, "mirror", "rank_multi", "safetensors")
    paths_of_rank0 = {file_path_of(lay, pos).unwrap() for pos in lay.tensor_entries if pos.rank == 0}
    assert len(paths_of_rank0) >= 2
    assert len(set(lay.file_paths.values())) == len(lay.file_paths)  # one list, one path


@pytest.mark.parametrize("fmt", ["distcp", "safetensors"])
def test_per_pp_stage(root, fmt):
    _, st, lay, _ = roundtrip(root, M, P, "hf", "per_stage", fmt)
    assert sorted(set(lay.file_paths.values())) == [f"stage0.{fmt}", f"stage1.{fmt}"]


def test_tensor_across_files(root):
    """per_rank + Shard: one tensor spans P.tp files; each list row slice is
    that tensor's single-tensor mesh cells."""
    _, st, lay, _ = roundtrip(root, M, P, "mirror", "per_rank", "safetensors")
    tid = "model.layers.0.self_attn.q_proj.weight"
    files = {file_path_of(lay, pos).unwrap() for pos in lay.tensor_entries if pos.tensor_id == tid}
    assert len(files) == P.tp


def test_replicated_written_once(root):
    """DP=4 replicas: payload bytes written == file payload size (no dup)."""
    ct, st, lay, io = roundtrip(root, M, ParallelConfig(tp=2, dp=4, pp=1), "hf", "single", "safetensors")
    total_payload = sum(nbytes_of(global_tensor_meta_of(x)) for x in st.named_d_tensors)
    _, data_start = raw_header(os.path.join(root, "model.safetensors"))
    payload_written = sum(r.size for r in io.tracer.records if r.op == "pwrite" and r.offset >= data_start)
    assert payload_written == total_payload
    writers = {r.rank for r in io.tracer.records if r.op == "pwrite" and r.offset >= data_start}
    assert len(writers) == 8  # writes rotate across DP replicas


def test_load_with_other_parallelism(root):
    """Save with TP2 DP2 PP2, load with TP4 DP1 PP2 (elastic restore)."""
    roundtrip(root, M, P, "hf", "hf", "safetensors", load_p=ParallelConfig(tp=4, dp=1, pp=2))


# ------------------------------------------------------------------ empty slots


def _empty_slot_fixture():
    W = 3
    base = OuterProductMesh(ArrangedMesh(ConstantOne(1)), ArrangedMesh(WorldSize(W)))
    A = NamedDTensor("A", NaiveDTensor(GlobalTensorMeta((200,), "U8", 1), (Replicate(), Shard(0)),
                                       SlicedMesh(base, (None, Slice(0, 3, 2)))))  # columns 0, 2; column 1 empty
    B = NamedDTensor("B", NaiveDTensor(GlobalTensorMeta((200,), "U8", 1), (Replicate(), Replicate()),
                                       SlicedMesh(base, (None, Slice(0, 1)))))
    st = DTensorTable([A, B])
    cA = NamedDTensor("A", NaiveDTensor(GlobalTensorMeta((200,), "U8", 1), (Shard(0),), ArrangedMesh(WorldSize(W))))
    cB = NamedDTensor("B", NaiveDTensor(GlobalTensorMeta((200,), "U8", 1), (Replicate(),), ArrangedMesh(WorldSize(W))))
    ct = DTensorTable([cA, cB])
    lay = build_layout(st, W, "safetensors", lambda t, w, nd: "f.safetensors")
    return ct, st, lay


def test_empty_slot_offsets(root):
    """Design 5.3 example: A0=100, A1=empty, A2=100, B0=200."""
    ct, st, lay = _empty_slot_fixture()
    idx = build_payload_index(st, lay)
    cpo = {pos: s.contiguous_payload_offset for pos, (_, s) in idx.slots.items()}
    assert cpo == {TensorComponentPos("A", 0): 0, TensorComponentPos("A", 2): 100, TensorComponentPos("B", 0): 200}
    assert TensorComponentPos("A", 1) not in lay.tensor_entries  # empty slot: no key
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    sv = make_storage_view(_ids(st), st, lay, root)
    session = sv.begin_save(sv, 0).unwrap()
    data_start = raw_header(os.path.join(root, "f.safetensors.tmp"))[1]
    got = {k: session.get_file_offset(k).unwrap() for k in cpo}
    assert got == {k: data_start + v for k, v in cpo.items()}
    session.abort().unwrap()
    res, _ = save_threads(ct, st, lay, root, fab, 3)
    assert_all_ok(res)
    verify_safetensors_dir(root, st)


# ------------------------------------------------------------------ reshard


def test_reshard_storage_to_storage_hf_to_dcp(tmp_path):
    src, dst = str(tmp_path / "hf"), str(tmp_path / "dcp")
    os.makedirs(src); os.makedirs(dst)
    ct, st, lay, _ = roundtrip(src, M, P, "hf", "hf", "safetensors")
    st2 = storage_table(ct, P, "mirror")
    lay2 = layout_builder("per_rank", st2, P, "distcp")
    from gdtensor.internal.coord import ThreadCoordinator
    sv_src = make_storage_view(_ids(st), st, lay, src)
    sv_dst = make_storage_view(_ids(st2), st2, lay2, dst, coord=ThreadCoordinator(P.world))
    assert_all_ok(reshard_threads(P.world, lambda r: sv_src, lambda r: sv_dst))
    verify_distcp_dir(dst, st2)


def test_reshard_storage_to_compute(root):
    ct, st, lay, _ = roundtrip(root, M, P, "hf", "hf", "safetensors")
    p2 = ParallelConfig(tp=1, dp=8, pp=1)
    ct2 = compute_table(M, p2)
    fab = ComputeFabric(ct2).allocate()
    sv = make_storage_view(_ids(st), st, lay, root)
    assert_all_ok(reshard_threads(p2.world, lambda r: sv, lambda r: compute_view(ct2, fab, r)))
    verify_fabric(fab, ct2)


def test_reshard_compute_to_compute_no_files(root):
    ct = compute_table(M, P)
    p2 = ParallelConfig(tp=4, dp=2, pp=1)
    ct2 = compute_table(M, p2)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    fab2 = ComputeFabric(ct2).allocate()
    res = reshard_threads(p2.world, lambda r: compute_view(ct, fab, r), lambda r: compute_view(ct2, fab2, r))
    assert_all_ok(res)
    verify_fabric(fab2, ct2)
    assert fab.comm_bytes > 0 and os.listdir(root) == []


def test_reshard_assert_meta_mismatch():
    ct = compute_table(M, P)
    other = compute_table(ModelConfig(layers=4, hidden=32, ffn=64, vocab=256), P)
    fab, fab2 = ComputeFabric(ct).allocate(), ComputeFabric(other).allocate()
    res = d_tensors_view_rank_plan_reshard(0, compute_view(ct, fab, 0), compute_view(other, fab2, 0))
    assert type(res).__name__ == 'Err' and "global_tensor_meta differs" in res.value.msg


# ------------------------------------------------------------------ mixed format


def test_mixed_format_needs_two_layouts(tmp_path):
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    for fmt in ("safetensors", "distcp"):
        d = str(tmp_path / fmt)
        os.makedirs(d)
        lay = layout_builder("per_stage", st, P, fmt)
        assert lay.container_format == fmt
        fab = ComputeFabric(ct).allocate(fill=fill_pattern)
        res, _ = save_threads(ct, st, lay, d, fab, P.world)
        assert_all_ok(res)
        (verify_safetensors_dir if fmt == "safetensors" else verify_distcp_dir)(d, st)


# ------------------------------------------------------------------ invariants / checks


@pytest.mark.parametrize("kind", ["hf", "single", "per_rank", "rank_multi", "per_stage"])
def test_plan_union_disjoint_and_covering(kind):
    ct = compute_table(MOE, P)
    st = storage_table(ct, P, "hf" if kind in ("hf", "single", "per_stage") else "mirror")
    lay = layout_builder(kind, st, P, "safetensors")
    fab = ComputeFabric(ct).allocate()
    sv = make_storage_view(_ids(st), st, lay, "/nonexistent")
    paths = sorted(set(lay.file_paths.values()))
    union_save, union_load = [], []
    for r in range(P.world):
        cv = compute_view(ct, fab, r)
        ps = d_tensors_view_rank_plan_save(r, cv, sv).unwrap()
        pl = d_tensors_view_rank_plan_load(r, cv, sv).unwrap()
        assert type(check_compute_d_tensors_access_disjoint(ct, pl)).__name__ == "Ok"
        union_save += ps
        union_load += pl
    # Shared-file invariant: all ranks' storage writes are disjoint and cover every file.
    assert type(check_storage_d_tensors_access_disjoint(st, lay, paths, union_save)).__name__ == 'Ok'
    assert type(check_storage_d_tensors_access_full_covering(st, lay, paths, union_save)).__name__ == 'Ok'
    assert type(check_compute_d_tensors_access_full_covering(ct, union_save)).__name__ == 'Ok'


def test_check_detects_overlap():
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("single", st, P, "safetensors")
    fab = ComputeFabric(ct).allocate()
    sv = make_storage_view(_ids(st), st, lay, "/nonexistent")
    plan = d_tensors_view_rank_plan_save(0, compute_view(ct, fab, 0), sv).unwrap()
    res = check_storage_d_tensors_access_disjoint(st, lay, ["model.safetensors"], plan + plan[:1])
    assert not res.ok and "overlap" in res.error
    res = check_storage_d_tensors_access_full_covering(st, lay, ["model.safetensors"], plan)
    assert not res.ok


def test_storage_mesh_disjoint_check():
    base = OuterProductMesh(ArrangedMesh(TensorCount(2)), ArrangedMesh(WorldSize(4)))
    a = SlicedMesh(base, (Slice(0, 1), Slice(0, 2)))
    b = SlicedMesh(base, (Slice(0, 1), Slice(2, 4)))
    c = SlicedMesh(base, (Slice(0, 2), Slice(1, 3)))
    assert check_storage_mesh_disjoint(a, b).ok
    assert not check_storage_mesh_disjoint(a, c).ok


def test_layout_is_deterministic_and_serializable():
    ct = compute_table(MOE, P)
    st = storage_table(ct, P, "hf")
    a = layout_builder("hf", st, P, "safetensors", 3)
    b = layout_builder("hf", storage_table(compute_table(MOE, P), P, "hf"), P, "safetensors", 3)
    # section 6: PhysicalLayoutTable is deterministic configuration data.
    assert a == b and hash(a) == hash(b)
    from gdtensor.internal.formats import safetensors_header
    ia, ib = build_payload_index(st, a), build_payload_index(st, b)
    for path in ia.files:
        assert safetensors_header(st, ia, path) == safetensors_header(st, ib, path)


def test_shared_header_has_no_rank_info():
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("single", st, P, "safetensors")
    from gdtensor.internal.formats import safetensors_header
    h = safetensors_header(st, build_payload_index(st, lay), "model.safetensors")
    assert b"slot" not in h and b"rank" not in h


# ------------------------------------------------------------------ session lifecycle


def test_abort_on_one_rank_discards_shared_file(root):
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("single", st, P, "safetensors")
    from gdtensor.internal.coord import ThreadCoordinator
    from gdtensor.internal.runner import _run_threads
    sv = make_storage_view(_ids(st), st, lay, root, coord=ThreadCoordinator(P.world))

    def rank_fn(r):
        s = sv.begin_save(sv, r).unwrap()
        return s.abort() if r == 3 else s.commit()

    res = _run_threads(P.world, rank_fn)
    assert res[3].ok and not res[0].ok
    assert os.listdir(root) == []


def test_load_session_states(root):
    ct, st, lay, _ = roundtrip(root, M, ParallelConfig(tp=2, dp=1, pp=1), "hf", "single", "safetensors")
    sv = make_storage_view(_ids(st), st, lay, root)
    s = sv.begin_load(sv, 0).unwrap()
    assert s.finish().ok
    assert not s.finish().ok and not s.get_file_offset(TensorComponentPos("lm_head.weight", 0)).ok


def test_distcp_without_metadata_fails(root):
    ct, st, lay, _ = roundtrip(root, M, P, "mirror", "per_rank", "distcp")
    os.remove(os.path.join(root, ".metadata"))
    sv = make_storage_view(_ids(st), st, lay, root)
    res = sv.begin_load(sv, 0)
    assert not res.ok and ".metadata is missing" in res.error


# ------------------------------------------------------------------ real processes


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_multiprocess_concurrent_pwrite(root, fmt):
    ct = compute_table(MOE, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, fmt, 2)
    res, recs, _ = run_processes("save", ct, st, lay, root, P.world, fill=fill_pattern)
    assert_all_ok(res)
    (verify_safetensors_dir if fmt == "safetensors" else verify_distcp_dir)(root, st)
    res2, _, bufs = run_processes("load", ct, st, lay, root, P.world)
    assert_all_ok(res2)
    fab = ComputeFabric(ct)
    fab.buffers = bufs
    verify_fabric(fab, ct)

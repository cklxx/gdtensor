"""Explicit coverage for B.1.3 check APIs at two granularities.

Slot level  : check_physical_layout, check_storage_mesh_disjoint
Range level : check_compute_* / check_storage_* on save / load / reshard plans

Positive tests use business-built layouts. Negative tests hand-mutate one
invariant at a time.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest

from conftest import assert_all_ok
from gdtensor._api import d_tensors_view_rank_plan_reshard
from gdtensor._layout import (
    DisjointStorageMeshList, PhysicalLayoutTable, TensorComponentPos, build_payload_index,
    leader_tensor_component_pos_of,
)
from gdtensor._mesh import ArrangedMesh, storage_mesh_cells
from gdtensor._models import ModelConfig, ParallelConfig, compute_table, layout_builder, storage_table

M = ModelConfig(layers=4, hidden=32, ffn=64, vocab=128)
MOE = ModelConfig(layers=2, hidden=32, ffn=48, vocab=128, experts=4)
P = ParallelConfig(tp=2, dp=2, pp=2)
from gdtensor._plan import (
    check_compute_d_tensors_access_disjoint, check_compute_d_tensors_access_full_covering,
    check_physical_layout, check_storage_d_tensors_access_disjoint, check_storage_d_tensors_access_full_covering,
    check_storage_mesh_disjoint, d_tensors_view_rank_plan_load, d_tensors_view_rank_plan_save,
)
from gdtensor._ranges import RangeFromComputeView, RangeFromStorageView
from gdtensor._formats import make_storage_view
from gdtensor._runner import compute_view
from gdtensor._views import ComputeFabric


def _fixture(m=M, p=P, kind="hf", mode="hf"):
    ct = compute_table(m, p)
    st = storage_table(ct, p, mode)
    lay = layout_builder(kind, st, p, "safetensors")
    fab = ComputeFabric(ct).allocate()
    sv = make_storage_view(st.tensor_ids, st, lay, "/nonexistent")
    return ct, st, lay, fab, sv


def _rank_plans(ct, sv, world):
    ps, pl = {}, {}
    for r in range(world):
        cv = compute_view(ct, None, r)
        from gdtensor._views import ComputeDataAccessor
        cv.data_accessor = ComputeDataAccessor(r, None)
        ps[r] = d_tensors_view_rank_plan_save(r, cv, sv).unwrap()
        pl[r] = d_tensors_view_rank_plan_load(r, cv, sv).unwrap()
    return ps, pl


# ================================================================== slot level


def test_valid_layouts_pass_physical_layout_check():
    ct = compute_table(MOE, P)
    for kind, mode in [("hf", "hf"), ("single", "hf"), ("per_stage", "hf"),
                       ("per_rank", "mirror"), ("rank_multi", "mirror")]:
        st = storage_table(ct, P, mode)
        lay = layout_builder(kind, st, P, "safetensors")
        assert check_physical_layout(st, lay).ok, (kind, check_physical_layout(st, lay).error)


def test_real_lists_are_pairwise_disjoint():
    ct = compute_table(MOE, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, "safetensors", 3)
    lists = {leader_tensor_component_pos_of(l): l for l in lay.tensor_entries.values()}
    uniq = list(lists.values())
    assert len(uniq) == 3
    for i, a in enumerate(uniq):
        for j, b in enumerate(uniq):
            if i != j:
                assert check_storage_mesh_disjoint(a.disjoint_storage_mesh_list[0],
                                                    b.disjoint_storage_mesh_list[0]).ok or True
        # check against every mesh of the other list
    for i, a in enumerate(uniq):
        for j, b in enumerate(uniq):
            if i == j:
                continue
            for ma in a.disjoint_storage_mesh_list:
                for mb in b.disjoint_storage_mesh_list:
                    assert check_storage_mesh_disjoint(ma, mb).ok


def test_layout_rejects_duplicate_file_path():
    _, st, lay, _, _ = _fixture(kind="per_rank", mode="mirror")
    bad = copy.deepcopy(lay)
    bad.file_paths = {k: "same.safetensors" for k in bad.file_paths}
    res = check_physical_layout(st, bad)
    assert not res.ok and "share one file path" in res.error


def test_layout_rejects_leader_without_path():
    _, st, lay, _, _ = _fixture(kind="per_rank", mode="mirror")
    bad = copy.deepcopy(lay)
    some_leader = next(iter(bad.file_paths))
    del bad.file_paths[some_leader]
    res = check_physical_layout(st, bad)
    assert not res.ok and "has no file path" in res.error


def test_layout_rejects_path_for_non_leader():
    _, st, lay, _, _ = _fixture(kind="hf")
    bad = copy.deepcopy(lay)
    bad.file_paths[TensorComponentPos("not-a-leader", 99)] = "x.safetensors"
    res = check_physical_layout(st, bad)
    assert not res.ok and "exactly the list leaders" in res.error


def test_layout_rejects_pos_outside_its_list():
    _, st, lay, _, _ = _fixture(kind="per_rank", mode="mirror")
    bad = copy.deepcopy(lay)
    # Re-point one (tensor, rank) at a list that does not cover it.
    pos = next(iter(bad.tensor_entries))
    other = next(l for p, l in bad.tensor_entries.items() if l is not bad.tensor_entries[pos])
    bad.tensor_entries[pos] = other
    res = check_physical_layout(st, bad)
    assert not res.ok and "not covered by its own list" in res.error


def test_layout_rejects_row_label_mismatch():
    _, st, lay, _, _ = _fixture(kind="per_rank", mode="mirror")
    bad = copy.deepcopy(lay)
    pos = next(iter(bad.tensor_entries))
    lst = bad.tensor_entries[pos]
    bad.tensor_entries[pos] = DisjointStorageMeshList(lst.disjoint_storage_mesh_list, tuple(reversed(lst.tensor_ids)))
    res = check_physical_layout(st, bad)
    assert not res.ok and "rows differ" in res.error


def test_storage_mesh_disjoint_rejects_overlap():
    from gdtensor._mesh import OuterProductMesh, Slice, SlicedMesh
    base = OuterProductMesh(ArrangedMesh(2, "tensor_count"), ArrangedMesh(4, "world_size"))
    a = SlicedMesh(base, (Slice(0, 1), Slice(0, 2)))
    c = SlicedMesh(base, (Slice(0, 2), Slice(1, 3)))
    res = check_storage_mesh_disjoint(a, c)
    assert not res.ok and "overlap" in res.error


def test_storage_mesh_disjoint_accepts_touching_boundary():
    from gdtensor._mesh import OuterProductMesh, Slice, SlicedMesh
    base = OuterProductMesh(ArrangedMesh(1, "tensor_count"), ArrangedMesh(4, "world_size"))
    a = SlicedMesh(base, (None, Slice(0, 2)))
    b = SlicedMesh(base, (None, Slice(2, 4)))
    assert check_storage_mesh_disjoint(a, b).ok


def test_empty_slot_layout_passes():
    # 3-column world where one tensor uses columns 0,2 and another column 0.
    from gdtensor._dtensor import DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor, Replicate, Shard
    from gdtensor._mesh import ArrangedMesh, OuterProductMesh, Slice, SlicedMesh
    W = 3
    base = OuterProductMesh(ArrangedMesh(1, "constant_one"), ArrangedMesh(W, "world_size"))
    a = NamedDTensor("A", NaiveDTensor(GlobalTensorMeta((200,), "U8", 1),
                                       (Replicate(), Shard(0)),
                                       SlicedMesh(base, (None, Slice(0, 3, 2)))))
    b = NamedDTensor("B", NaiveDTensor(GlobalTensorMeta((200,), "U8", 1),
                                       (Replicate(), Replicate()),
                                       SlicedMesh(base, (None, Slice(0, 1)))))
    st = DTensorTable([a, b])
    from gdtensor._layout import build_layout
    lay = build_layout(st, W, "safetensors", lambda t, w, nd: "f.safetensors")
    assert check_physical_layout(st, lay).ok
    assert TensorComponentPos("A", 1) not in lay.tensor_entries  # empty slot: no entry


# ================================================================== range level


def test_per_rank_save_src_is_disjoint():
    ct, st, lay, fab, sv = _fixture(MOE, P, "hf")
    ps, _ = _rank_plans(ct, sv, P.world)
    for r, plan in ps.items():
        assert check_compute_d_tensors_access_disjoint(ct, plan).ok, r


def test_save_union_storage_and_compute_checks():
    ct, st, lay, fab, sv = _fixture(MOE, P, "hf")
    ps, _ = _rank_plans(ct, sv, P.world)
    union = [x for plan in ps.values() for x in plan]
    paths = sorted(set(lay.file_paths.values()))
    assert check_storage_d_tensors_access_disjoint(st, lay, paths, union).ok
    assert check_storage_d_tensors_access_full_covering(st, lay, paths, union).ok
    assert check_compute_d_tensors_access_disjoint(ct, union).ok
    assert check_compute_d_tensors_access_full_covering(ct, union).ok


def test_compute_range_overlap_is_detected():
    ct, st, lay, fab, sv = _fixture(M, P, "single")
    plan = ps0 = d_tensors_view_rank_plan_save(0, _cv(ct, 0), sv).unwrap()
    # Duplicate one compute source pair: two readers of the same bytes.
    bad = plan + [plan[0]]
    res = check_compute_d_tensors_access_disjoint(ct, bad)
    assert not res.ok and "overlap" in res.error


def test_storage_range_crossing_slot_boundary_is_detected():
    ct, st, lay, _, sv = _fixture(M, P, "single")
    index = sv.data_accessor.index
    fp = index.files["model.safetensors"]
    s0, s1 = fp.slots[0], fp.slots[1]
    # Start one byte before the end of slot 0, read two bytes into slot 1.
    rng = RangeFromStorageView("model.safetensors", s1.contiguous_payload_offset - 1, 2)
    plan = [(RangeFromComputeView("x", 0, 2), rng)]
    res = check_storage_d_tensors_access_disjoint(st, lay, ["model.safetensors"], plan)
    assert not res.ok and "crosses a slot boundary" in res.error


def test_storage_range_on_unknown_file_is_detected():
    ct, st, lay, _, _ = _fixture(M, P, "single")
    plan = [(RangeFromComputeView("x", 0, 4), RangeFromStorageView("ghost.safetensors", 0, 4))]
    res = check_storage_d_tensors_access_disjoint(st, lay, ["model.safetensors"], plan)
    assert not res.ok and "unknown file" in res.error


def test_compute_range_out_of_bounds_is_detected():
    ct, st, lay, _, sv = _fixture(M, P, "single")
    plan = d_tensors_view_rank_plan_save(0, _cv(ct, 0), sv).unwrap()
    s0, d0 = plan[0]
    # Move the compute source past the tensor end: full_covering must fail.
    bad = [(RangeFromComputeView(s0.global_tensor_id, 1 << 40, s0.size), d0)]
    res = check_compute_d_tensors_access_full_covering(ct, bad)
    assert not res.ok and "covered" in res.error


def test_storage_range_gap_is_detected():
    ct, st, lay, _, sv = _fixture(M, P, "single")
    plan = d_tensors_view_rank_plan_save(0, _cv(ct, 0), sv).unwrap()
    # Drop the first pair: the file no longer starts at offset 0.
    res = check_storage_d_tensors_access_full_covering(st, lay, ["model.safetensors"], plan[1:])
    assert not res.ok and "covered" in res.error


def test_load_semantics_per_rank_disjoint_union_covering():
    ct, st, lay, fab, sv = _fixture(M, P, "hf")
    ps, pl = _rank_plans(ct, sv, P.world)
    # Every rank's load destination is internally disjoint.
    for r, plan in pl.items():
        assert check_compute_d_tensors_access_disjoint(ct, plan).ok
    # Across ranks the union still covers every tensor (replicated ranks overlap
    # on compute: disjointness is a per-rank property, not a union property).
    union = [x for plan in pl.values() for x in plan]
    assert check_compute_d_tensors_access_full_covering(ct, union).ok
    replicated_overlap = not check_compute_d_tensors_access_disjoint(ct, union).ok
    assert replicated_overlap  # documents that load-union overlap is expected
    # Replicated tensors are read from storage by every replica rank: storage
    # source ranges legitimately overlap across ranks.
    norm = "model.layers.0.input_layernorm.weight"
    src = [s for plan in pl.values() for s, _ in plan if not hasattr(s, "global_tensor_id")]
    # every replica rank reads the same storage region at least once
    paths = sorted({s.file_path for s in src})
    assert check_storage_d_tensors_access_disjoint(st, lay, paths, src).ok is False


def test_reshard_storage_to_storage_passes_checks():
    ct = compute_table(MOE, P)
    src_st = storage_table(ct, P, "hf")
    src_lay = layout_builder("hf", src_st, P, "safetensors", 2)
    dst_st = storage_table(ct, P, "mirror")
    dst_lay = layout_builder("per_rank", dst_st, P, "safetensors")
    ssv = make_storage_view(src_st.tensor_ids, src_st, src_lay, "/nonexistent")
    dsv = make_storage_view(dst_st.tensor_ids, dst_st, dst_lay, "/nonexistent")
    # Destination side only: pair each storage dst with a zero-size compute src
    # so the check filters in exactly the dst storage ranges.
    union_dst = []
    for r in range(P.world):
        plan = d_tensors_view_rank_plan_reshard(r, ssv, dsv).unwrap()
        union_dst += [(RangeFromComputeView("__ignored__", 0, 0), d) for _, d in plan]
    paths = sorted(set(dst_lay.file_paths.values()))
    assert check_storage_d_tensors_access_disjoint(dst_st, dst_lay, paths, union_dst).ok
    assert check_storage_d_tensors_access_full_covering(dst_st, dst_lay, paths, union_dst).ok


def test_reshard_compute_to_compute_per_rank_disjoint_and_covering():
    ct = compute_table(M, P)
    p2 = ParallelConfig(tp=4, dp=2, pp=1)
    ct2 = compute_table(M, p2)
    fab = ComputeFabric(ct).allocate()
    fab2 = ComputeFabric(ct2).allocate()
    union_dst = []
    for r in range(p2.world):
        plan = d_tensors_view_rank_plan_reshard(r, compute_view(ct, fab, r), compute_view(ct2, fab2, r)).unwrap()
        # Destination side only: zero-size storage src keeps the src compute
        # ranges out of the check.
        dst = [(RangeFromStorageView("__ignored__", 0, 0), d) for _, d in plan]
        assert check_compute_d_tensors_access_disjoint(ct2, dst).ok
        union_dst += dst
    assert check_compute_d_tensors_access_full_covering(ct2, union_dst).ok


def _cv(table, rank):
    from gdtensor._views import ComputeDataAccessor, ComputeDTensorsView
    cv = ComputeDTensorsView(table.tensor_ids, table, ComputeDataAccessor(rank, None))
    cv.data_accessor = ComputeDataAccessor(rank, None)
    return cv


def _sv(storage_view, rank):
    return storage_view

def test_two_lists_mapping_to_one_path_is_rejected():
    """check_physical_layout rejects two list leaders sharing one path."""
    ct = compute_table(M, P)
    st = storage_table(ct, P, "mirror")
    lay = layout_builder("per_rank", st, P, "safetensors")
    leaders = sorted(lay.file_paths)
    assert len(leaders) >= 2
    victim_paths = dict(lay.file_paths)
    victim_paths[leaders[1]] = victim_paths[leaders[0]]
    bad = PhysicalLayoutTable(lay.container_format, dict(lay.tensor_entries), victim_paths)
    from gdtensor._plan import check_physical_layout
    assert not check_physical_layout(st, bad).ok
    from gdtensor._layout import build_payload_index
    with pytest.raises(Exception):
        build_payload_index(st, bad)

"""Runtime plan invariants observed only through the public IO API.

The executors validate one rank's plan before they open any session:
* save: the designated writer reads exactly the elements it owns;
* load: every rank fills every element it owns, per rank, not over a union;
* storage access stays inside the payload and never overlaps inside a rank.

Upstream planning already rejects every malformation reachable through public
builders, so these tests pin the observable contracts (exact per-rank bytes,
no session on a rejected plan, every replica gets the full tensor) and guard
against regressions in either layer.
"""

import numpy as np

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.memory import Memory, compute_view, storage_view


def _two_rank_shard():
    """2-rank compute shard over 8 U8 elements; two storage columns."""
    src, saved = Memory(range(8)), Memory([0xA5] * 8)
    events = []
    cv, sv = compute_view(src), storage_view(saved, events)
    ct = cv.d_tensor_table.named_d_tensors[0].d_tensor
    ct.set_mesh(gb.ArrangedMesh(gt.WorldSize(2)))
    ct.set_placements([gb.Shard(0)])
    st = sv.d_tensor_table.named_d_tensors[0].d_tensor
    st.set_mesh(
        gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
    )
    st.set_placements([gb.Replicate(), gb.Shard(0)])
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
            )
        ],
        ["w"],
    )
    sv.physical_layout.set_tensor_entries(
        {gb.TensorComponentPos("w", rank): group for rank in (0, 1)}
    )
    assert type(cv.check_constraints()) is type(sv.check_constraints()) is gb.Ok
    return cv, sv, saved, events


def test_each_rank_saves_and_reloads_exactly_its_owned_shard():
    cv, sv, saved, events = _two_rank_shard()
    for rank in range(2):
        assert type(gio.rank_save(rank, cv, sv)) is gb.Ok
    np.testing.assert_array_equal(saved.data, np.arange(8, dtype=np.uint8))
    for rank in range(2):
        out = Memory([0xA5] * 8)
        loaded = gb.ComputeDTensorsView(cv.tensor_ids, cv.d_tensor_table, out)
        assert type(gio.rank_load(rank, loaded, sv)) is gb.Ok
        want = np.full(8, 0xA5, dtype=np.uint8)
        want[4 * rank : 4 * (rank + 1)] = np.arange(
            4 * rank, 4 * (rank + 1), dtype=np.uint8
        )
        np.testing.assert_array_equal(out.data, want)


def test_missing_storage_column_is_rejected_before_any_session():
    # Only column 0 has a physical slot. Rank 0 persists its half; rank 1
    # must be rejected and must never open a save session.
    cv = compute_view(Memory(range(8)))
    tensor = cv.d_tensor_table.named_d_tensors[0].d_tensor
    tensor.set_mesh(gb.ArrangedMesh(gt.WorldSize(2)))
    tensor.set_placements([gb.Shard(0)])
    events = []
    sv = storage_view(Memory([0] * 8), events)
    dt = sv.d_tensor_table.named_d_tensors[0].d_tensor
    dt.set_mesh(
        gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
    )
    dt.set_placements([gb.Replicate(), gb.Shard(0)])
    base = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList(
        [gb.SlicedMesh(base, [None, gb.Slice(0, 1)])],
        ["w"],
    )
    sv.physical_layout.tensor_entries = {gb.TensorComponentPos("w", 0): group}
    first = gio.rank_save(0, cv, sv)
    assert type(first) is gb.Ok, first
    events.clear()
    result = gio.rank_save(1, cv, sv)
    assert type(result) is gb.Err
    assert not events, "rank 1 opened a session for an uncovered plan"


def test_rank_load_without_its_column_is_rejected_before_any_session():
    # Same partial layout: only column 0 exists. Rank 1 must not load a
    # tensor shard that has no source, at planning or via the per-rank guard.
    cv = compute_view(Memory([0xA5] * 8))
    tensor = cv.d_tensor_table.named_d_tensors[0].d_tensor
    tensor.set_mesh(gb.ArrangedMesh(gt.WorldSize(2)))
    tensor.set_placements([gb.Shard(0)])
    events = []
    sv = storage_view(Memory(range(8)), events)
    dt = sv.d_tensor_table.named_d_tensors[0].d_tensor
    dt.set_mesh(
        gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(2))
        )
    )
    dt.set_placements([gb.Replicate(), gb.Shard(0)])
    base = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(2))
    )
    group = gb.DisjointStorageMeshList(
        [gb.SlicedMesh(base, [None, gb.Slice(0, 1)])],
        ["w"],
    )
    sv.physical_layout.tensor_entries = {gb.TensorComponentPos("w", 0): group}
    result = gio.rank_load(1, cv, sv)
    assert type(result) is gb.Err
    assert not events, "rank 1 opened a load session with no source shard"


def test_every_replica_independently_loads_the_full_tensor():
    # Two fully replicated compute ranks. A union-based covering check would
    # pass if the two ranks together read the tensor once; the per-rank rule
    # requires each replica to receive all 8 bytes on its own.
    sv = storage_view(Memory(range(8)), [])
    cv_mesh = gb.ArrangedMesh(gt.WorldSize(2))
    for rank in range(2):
        out = Memory([0xA5] * 8)
        cv = gb.ComputeDTensorsView(
            ["w"],
            gb.DTensorTable(
                [
                    gb.NamedDTensor(
                        "w",
                        gb.NaiveDTensor(
                            gb.GlobalTensorMeta([8], "U8", 1),
                            [gb.Replicate()],
                            cv_mesh,
                        ),
                    )
                ]
            ),
            out,
        )
        result = gio.rank_load(rank, cv, sv)
        assert type(result) is gb.Ok, result
        np.testing.assert_array_equal(out.data, np.arange(8, dtype=np.uint8))

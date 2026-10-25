"""Mixed chain: each segment opens its own source session (design line 386).

Two disjoint tensor-id segments share the same underlying storage backend (same
begin factory) but MUST get two distinct session objects. Segment A completes
its bytes; segment B's read fails with the exact error, A's bytes are already
delivered, and both sessions finish exactly once.
"""

import numpy as np

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.memory import Memory


class _RecSession:
    def __init__(self, accessor, events, read_error=None):
        self.accessor = accessor
        self.events = events
        self.read_error = read_error
        self.init_count = 0
        self.finish_count = 0

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        self.init_count += 1
        return gb.Ok(self.accessor)

    def finish(self):
        self.finish_count += 1
        return gb.Ok(None)


def _storage_view(accessor, tid, world):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([world], "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh
    )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(world))
            )
        ],
        [tid],
    )
    entries = {gb.TensorComponentPos(tid, rank): group for rank in range(world)}
    layout = gb.PhysicalLayoutTable(
        "mem", entries, {gb.TensorComponentPos(tid, 0): "mem"}
    )
    return gb.StorageDTensorsView(
        [tid],
        gb.DTensorTable([gb.NamedDTensor(tid, dt, [tid + ".a"])]),
        accessor,
        layout,
        lambda v, r: gb.Ok(None),
        lambda v, r: gb.Ok(None),
    )


def _compute_view(accessor, tid, world):
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([world], "U8", 1),
        [gb.Replicate()],
        gb.ArrangedMesh(gt.WorldSize(world)),
    )
    return gb.ComputeDTensorsView(
        [tid], gb.DTensorTable([gb.NamedDTensor(tid, dt, [tid + ".a"])]), accessor
    )


def test_mixed_two_segments_each_own_session_bytes_and_finish():
    world = 2
    src_a = Memory(list(np.arange(world, dtype=np.uint8)))
    src_b = Memory(list(np.arange(world, dtype=np.uint8) + 100))
    dst_a = Memory([0] * world)
    dst_b = Memory([0] * world)
    sessions = []

    def factory(accessor):
        def begin(v, rank):
            s = _RecSession(accessor, [])
            sessions.append(s)
            return gb.Ok(s)

        return begin

    sv_a = _storage_view(src_a, "a", world)
    sv_a.begin_load = factory(src_a)
    sv_b = _storage_view(src_b, "b", world)
    sv_b.begin_load = factory(src_b)
    cv_a = _compute_view(dst_a, "a", world)
    cv_b = _compute_view(dst_b, "b", world)

    seg_a = gb.DirectDTensorsReadChain(src=sv_a, dst=cv_a)
    seg_b = gb.DirectDTensorsReadChain(src=sv_b, dst=cv_b)
    chain = gb.MixedDTensorsReadChain(
        disjoint_tensor_ids_and_read_chain_list=[
            gb.DisjointTensorIdsAndReadChain(tensor_ids=["a"], read_chain=seg_a),
            gb.DisjointTensorIdsAndReadChain(tensor_ids=["b"], read_chain=seg_b),
        ]
    )
    res = gio.chain_load(0, chain)
    assert type(res).__name__ == "Ok", res
    assert len(sessions) == 2, len(sessions)
    assert sessions[0].finish_count == 1 and sessions[1].finish_count == 1
    assert sessions[0].init_count == 1 and sessions[1].init_count == 1
    np.testing.assert_array_equal(dst_a.data, np.arange(world, dtype=np.uint8))
    np.testing.assert_array_equal(dst_b.data, np.arange(world, dtype=np.uint8) + 100)


def test_mixed_segment_b_fails_a_bytes_kept_both_finish():
    world = 2
    src_a = Memory(list(np.arange(world, dtype=np.uint8)))
    boom = gb.GDTensorIOError(operation="read", msg="boom-b")
    src_b = Memory(list(np.arange(world, dtype=np.uint8)), error=boom)
    dst_a = Memory([0] * world)
    dst_b = Memory([0] * world)
    sessions = []

    def factory(accessor):
        def begin(v, rank):
            s = _RecSession(accessor, [])
            sessions.append(s)
            return gb.Ok(s)

        return begin

    sv_a = _storage_view(src_a, "a", world)
    sv_a.begin_load = factory(src_a)
    sv_b = _storage_view(src_b, "b", world)
    sv_b.begin_load = factory(src_b)
    cv_a = _compute_view(dst_a, "a", world)
    cv_b = _compute_view(dst_b, "b", world)

    seg_a = gb.DirectDTensorsReadChain(src=sv_a, dst=cv_a)
    seg_b = gb.DirectDTensorsReadChain(src=sv_b, dst=cv_b)
    chain = gb.MixedDTensorsReadChain(
        disjoint_tensor_ids_and_read_chain_list=[
            gb.DisjointTensorIdsAndReadChain(tensor_ids=["a"], read_chain=seg_a),
            gb.DisjointTensorIdsAndReadChain(tensor_ids=["b"], read_chain=seg_b),
        ]
    )
    res = gio.chain_load(0, chain)
    assert type(res).__name__ == "Err", res
    assert res.value.msg == "boom-b", res
    np.testing.assert_array_equal(dst_a.data, np.arange(world, dtype=np.uint8))
    assert len(sessions) == 2, len(sessions)
    assert sessions[0].finish_count == 1 and sessions[1].finish_count == 1

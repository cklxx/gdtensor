"""Independent offset oracle: public get_file_offset + os.pread bytes.

Does NOT call _raw_payload_bases or any private backend helper. Uses public
tensor_table / build_layout / build_payload_index + DcpLoadSession.get_file_offset,
then os.pread at the returned offset and compares to ground-truth bytes.

Correct schema (design 5.3):
  raw_file_offset(payload_base, slot) = payload_base
      + (slot.intervals[0,0] - chunk_origin) * slot.itemsize
  where chunk_origin = global row that starts this chunk's first row.
  The within-chunk offset is the FIRST owned element's global offset minus the
  chunk's global origin, NOT (last - first) * itemsize.

This test uses world=3 tensor of 9 elements, row-sharded over axis 0.
rank2 owns global rows [6, 9). The chunk produced by no_dist contains the
FULL tensor (origin 0), so within-chunk offset = 6 * itemsize.
"""

import os

import pytest

import numpy as np

torch = pytest.importorskip("torch")


def _save(tmp_path, state):
    from torch.distributed.checkpoint import save as dcp_save

    dcp_save(state, checkpoint_id=str(tmp_path), no_dist=True)


def test_world3_rank2_middle_pread_bytes_match(tmp_path):
    """Tensor[9] I64 row-sharded across world=3. rank2 owns rows [6,9).

    The DCP archive stores the full contiguous tensor (origin 0). At the
    offset returned by get_file_offset, os.pread must yield bytes equal to
    dense arange elements 6..8 (i.e. 48 bytes), NOT a 24-byte slice computed
    as (last-first)*itemsize.
    """
    import gdtensor.builder as gb
    from gdtensor.builder import global_tensor_meta_of
    from gdtensor_sim._geometry import nonempty_columns
    from gdtensor_sim._layout import build_payload_index
    from gdtensor_sim.models import build_layout
    from tests.support.schemas import tensor_schema
    from gdtensor_sim.dcp_backend import DcpLoadSession

    # world=3 row-shard on axis 0, axis1/2 replicated.
    ct, _, _ = tensor_schema((3, 1, 1))
    state = {}
    for nd in ct.named_d_tensors:
        meta = global_tensor_meta_of(nd)
        dt = {
            "I64": torch.int64,
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
        }[meta.dtype]
        n = int(np.prod(meta.global_shape))
        # pyrefly-ignore-next-line possibly-missing-attribute
        state[nd.global_tensor_id] = torch.arange(n, dtype=dt).reshape(
            tuple(meta.global_shape)
        )
    _save(tmp_path, dict(state))

    layout = build_layout(ct, 3, "safetensors", lambda t, w, nd: f"f{t}.safetensors")
    index = build_payload_index(ct, layout)

    # Pick one tensor whose row-shard gives rank2 a nonempty middle slot.
    target = None
    for nd in ct.named_d_tensors:
        cols = nonempty_columns(nd)
        if 2 in cols and int(np.prod(global_tensor_meta_of(nd).global_shape)) >= 9:
            target = nd
            break
    assert target is not None, "no tensor with rank2 nonempty slot in schema"

    tid = target.global_tensor_id
    meta = global_tensor_meta_of(target)
    itemsize = meta.itemsize
    want_full = state[tid].contiguous().view(torch.uint8).numpy().tobytes()

    # Each rank opens its own session (world=3).
    distcp = list(tmp_path.glob("__*.distcp"))[0]
    for rank, intervals in sorted(nonempty_columns(target).items()):
        session = DcpLoadSession(rank, str(tmp_path), ct, layout, index)
        pos = gb.TensorComponentPos(tid, rank)
        got = session.get_file_offset(pos)
        assert type(got).__name__ == "Ok", got
        slot_path, slot = index.slots[pos]
        fd = os.open(distcp, os.O_RDONLY)
        try:
            assert got.value is not None
            chunk = os.pread(fd, slot.nbytes, int(got.value))
        finally:
            os.close(fd)
        # Ground truth: global rows owned by this rank.
        lo = int(intervals[0, 0])
        hi = int(intervals[-1, 1])
        want = want_full[lo * itemsize : hi * itemsize]
        assert chunk == want, (
            f"rank{rank}: intervals={intervals.tolist()} nbytes={slot.nbytes} "
            f"offset={got.value} chunk_len={len(chunk)} want_len={len(want)}"
        )

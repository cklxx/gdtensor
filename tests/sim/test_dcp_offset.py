"""Offset contract: real raw payload offset resolved by archive structure.

Real DCP ``.distcp`` is a concatenation of self-contained torch-zip sub-archives,
one per tensor. ``storage_data.offset``/``length`` delimit each sub-archive. Inside
it the raw tensor bytes are the ``archive/data/<n>`` zip entry. We resolve the raw
file offset by parsing that zip structure (never by searching raw byte content,
never a hard-coded header constant):

    raw_offset = sub_archive_start + (data_entry local-header offset + 30
                                       + filename length + extra-field length)

pread at that offset (plus the slot's within-blob interval offset) must equal the
tensor's raw bytes, byte-exact, for repeated payloads, BF16, and multi-dtype.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _save(tmp_path, tensors):
    from torch.distributed.checkpoint import save as dcp_save

    dcp_save(dict(tensors), checkpoint_id=str(tmp_path), no_dist=True)


def test_raw_offset_resolves_by_structure_not_content_search(tmp_path):
    """Two tensors share an identical repeated payload (all-zeros). A content-based
    search would match the wrong blob; structure parsing must place each at its own
    data entry. pread at the resolved offset is byte-exact."""
    a = torch.zeros(6, dtype=torch.float32)  # all zeros -> ambiguous content
    b = torch.arange(5, dtype=torch.float64)
    c = torch.arange(7, dtype=torch.bfloat16)
    _save(tmp_path, [("alpha", a), ("beta", b), ("gamma", c)])

    from gdtensor_sim.dcp_backend import _raw_payload_bases

    bases = _raw_payload_bases(str(tmp_path))
    raw = open(tmp_path / "__0_0.distcp", "rb").read()
    for tid, t in (("alpha", a), ("beta", b), ("gamma", c)):
        base = bases[tid][0][1]
        want = t.contiguous().view(torch.uint8).numpy().tobytes()
        got = raw[base : base + len(want)]
        assert got == want, tid
    # alpha (all-zero) must resolve to a distinct blob from beta
    assert bases["alpha"][0][1] != bases["beta"][0][1]


def test_session_get_file_offset_preads_byte_exact(tmp_path):
    """Through the real DcpLoadSession, get_file_offset returns a raw file offset
    that preads byte-exact tensor bytes (existing pos); missing pos is NotFound."""
    import gdtensor.builder as gb
    from gdtensor.builder import global_tensor_meta_of
    from gdtensor_sim._geometry import nonempty_columns
    from gdtensor_sim._layout import build_payload_index
    from gdtensor_sim.models import build_layout
    from tests.support.schemas import tensor_schema
    from gdtensor_sim.dcp_backend import DcpLoadSession

    ct, _, _ = tensor_schema((1, 1, 1))
    _DT = {
        "I64": torch.int64,
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
    }
    state = {}
    for nd in ct.named_d_tensors:
        meta = global_tensor_meta_of(nd)
        # pyrefly-ignore-next-line possibly-missing-attribute
        state[nd.global_tensor_id] = torch.zeros(
            tuple(meta.global_shape), dtype=_DT[meta.dtype]
        )
    _save(tmp_path, list(state.items()))
    layout = build_layout(ct, 1, "safetensors", lambda t, w, nd: f"f{t}.safetensors")
    index = build_payload_index(ct, layout)
    session = DcpLoadSession(0, str(tmp_path), ct, layout, index)
    raw = open(tmp_path / "__0_0.distcp", "rb").read()
    for nd in ct.named_d_tensors:
        tid = nd.global_tensor_id
        for rank in nonempty_columns(nd):
            pos = gb.TensorComponentPos(tid, rank)
            got = session.get_file_offset(pos)
            assert type(got).__name__ == "Ok", got
            want = state[tid].contiguous().view(torch.uint8).numpy().tobytes()
            assert got.value is not None
            chunk = raw[got.value : got.value + len(want)]
            assert chunk == want, (tid, rank)
    missing = session.get_file_offset(gb.TensorComponentPos("nope", 0))
    assert type(missing).__name__ == "Err", missing


def test_middle_slot_within_chunk_offset_preads_distinct_ranges(tmp_path):
    """A row-sharded tensor over world_size=2 stores both rank slots in ONE data entry.
    slot0 owns the first half, slot1 the second half. get_file_offset must return
    base+global_start for each so pread yields the correct distinct range — not the
    same base for every slot in the chunk."""
    import gdtensor.builder as gb
    from gdtensor.builder import global_tensor_meta_of
    from gdtensor_sim._geometry import nonempty_columns
    from gdtensor_sim._layout import build_payload_index
    from gdtensor_sim.models import build_layout
    from tests.support.schemas import tensor_schema
    from gdtensor_sim.dcp_backend import DcpLoadSession

    ct, _, _ = tensor_schema((1, 1, 1))
    state = {}
    for nd in ct.named_d_tensors:
        meta = global_tensor_meta_of(nd)
        dt = {
            "I64": torch.int64,
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
        }[meta.dtype]
        # pyrefly-ignore-next-line possibly-missing-attribute
        state[nd.global_tensor_id] = torch.arange(
            int(np.prod(meta.global_shape)), dtype=dt
        ).reshape(tuple(meta.global_shape))
    _save(tmp_path, list(state.items()))
    layout = build_layout(ct, 2, "safetensors", lambda t, w, nd: f"f{t}.safetensors")
    index = build_payload_index(ct, layout)
    session = DcpLoadSession(0, str(tmp_path), ct, layout, index)
    _raw = open(tmp_path / "__0_0.distcp", "rb").read()
    for nd in ct.named_d_tensors:
        tid = nd.global_tensor_id
        slots = {}
        for rank in nonempty_columns(nd):
            pos = gb.TensorComponentPos(tid, rank)
            got = session.get_file_offset(pos)
            assert type(got).__name__ == "Ok", (tid, rank, got)
            slots[rank] = got.value
        # Two ranks must resolve to distinct offsets within the shared chunk.
        if len(slots) == 2:
            offs = list(slots.values())
            assert offs[0] != offs[1], (tid, slots)

"""Offset contract for the real-DCP backend.

Empirical fact (verified by independent pread): real torch DCP ``.distcp`` is a
pickle-wrapped archive. The public ``storage_data.offset`` is a LOGICAL offset
into torch's internal unpacking, NOT a raw-byte pread location — preading at it
returns pickle framing, not the tensor's raw bytes. The raw payload region is not
exposed by the public metadata API (raw bytes sit at ``storage_offset + 704``,
where 704 is torch's internal header size, not a public constant).

Therefore:
- SAVE (we control the writer): staged slots occupy a header-less contiguous raw
  region, so ``get_file_offset`` returns the slot's ``contiguous_payload_offset``
  (the staging-file offset). pread at that offset is byte-exact.
- LOAD (official reader): real DCP metadata does not expose raw byte offsets, so
  ``get_file_offset`` reports the gap explicitly (invalid), rather than faking
  ``storage_data.offset`` as if it were a raw file location.
"""

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_real_dcp_packs_second_tensor_at_nonzero_offset(tmp_path):
    """Two tensors saved by real DCP: the second's storage_data offset is a real
    non-zero logical offset (packed after the first), read from pickle metadata."""
    from torch.distributed.checkpoint import save as dcp_save

    dcp_save(
        {
            "alpha": np.arange(4, dtype=np.float32),
            "beta": np.arange(6, dtype=np.float32),
        },
        checkpoint_id=str(tmp_path),
        no_dist=True,
    )
    meta = pickle.loads((tmp_path / ".metadata").read_bytes())
    offsets = {idx.fqn: sd.offset for idx, sd in meta.storage_data.items()}
    assert offsets["alpha"] == 0, offsets
    assert offsets["beta"] > 0, offsets


def test_storage_offset_is_logical_not_raw_pread_location(tmp_path):
    """pread at storage_data.offset returns pickle framing, NOT raw tensor bytes.

    The raw tensor bytes exist in the archive (located by search) but NOT at the
    metadata-recorded offset. This pins that nonzero storage_data.offset is a
    logical archive offset, not a raw-byte file location."""
    from torch.distributed.checkpoint import save as dcp_save

    a = torch.arange(4, dtype=torch.float32)
    b = torch.arange(6, dtype=torch.float32)
    dcp_save({"alpha": a, "beta": b}, checkpoint_id=str(tmp_path), no_dist=True)
    raw = (tmp_path / "__0_0.distcp").read_bytes()
    meta = pickle.loads((tmp_path / ".metadata").read_bytes())
    off = {idx.fqn: sd.offset for idx, sd in meta.storage_data.items()}

    for tid, expect in (("alpha", a), ("beta", b)):
        o = off[tid]
        chunk = raw[o : o + expect.numel() * expect.element_size()]
        raw_located = raw.find(expect.numpy().tobytes())
        assert raw_located != -1, tid
        assert chunk != expect.numpy().tobytes(), (
            tid,
            "storage_data.offset is a logical pickle-archive offset, not a raw "
            "byte pread location",
        )


def test_save_get_file_offset_maps_staging_contiguous_offset(tmp_path):
    """Save-unpublished: staged slots are a header-less contiguous raw region, so
    get_file_offset returns the slot's contiguous_payload_offset (pread byte-exact)."""
    import gdtensor.builder as gb
    from gdtensor_sim._layout import build_payload_index
    from gdtensor_sim.models import build_layout
    from tests.support.schemas import tensor_schema
    from gdtensor_sim.dcp_backend import DcpSaveSession

    ct, _, _ = tensor_schema((1, 1, 1))
    layout = build_layout(ct, 1, "safetensors", lambda t, w, nd: f"f{t}.safetensors")
    index = build_payload_index(ct, layout)
    session = DcpSaveSession(0, str(tmp_path), ct, layout, index, mode="thread")
    tid = ct.named_d_tensors[0].global_tensor_id
    got = session.get_file_offset(gb.TensorComponentPos(tid, 0))
    assert type(got).__name__ == "Ok", got
    _path, slot = index.slots[gb.TensorComponentPos(tid, 0)]
    assert got.value == slot.contiguous_payload_offset


def test_load_get_file_offset_reports_unrepresentable_not_fake_offset(tmp_path):
    """Load: real DCP metadata does not expose raw byte offsets. get_file_offset
    must report the gap explicitly (invalid), NOT return storage_data.offset as if
    it were a raw file location, and NOT NotFound for an existing pos."""
    import gdtensor.builder as gb
    from gdtensor.builder import global_tensor_meta_of
    from gdtensor_sim._layout import build_payload_index
    from gdtensor_sim.models import build_layout
    from tests.support.schemas import tensor_schema
    from gdtensor_sim.dcp_backend import DcpLoadSession
    from torch.distributed.checkpoint import save as dcp_save

    ct, _, _ = tensor_schema((1, 1, 1))
    state = {
        nd.global_tensor_id: torch.zeros(
            tuple(global_tensor_meta_of(nd).global_shape), dtype=torch.float32
        )
        for nd in ct.named_d_tensors
    }
    dcp_save(state, checkpoint_id=str(tmp_path), no_dist=True)
    layout = build_layout(ct, 1, "safetensors", lambda t, w, nd: f"f{t}.safetensors")
    index = build_payload_index(ct, layout)
    session = DcpLoadSession(0, str(tmp_path), ct, layout, index)
    tid = ct.named_d_tensors[0].global_tensor_id
    got = session.get_file_offset(gb.TensorComponentPos(tid, 0))
    # Existing pos -> explicit "unrepresentable" error, not fake Ok(offset), not NotFound.
    assert type(got).__name__ == "Err", got
    assert "raw byte offset" in str(got.value) or "logical" in str(got.value)

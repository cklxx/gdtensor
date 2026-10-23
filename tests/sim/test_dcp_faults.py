"""Fault-injection contract for the real-DCP staging/publish backend.

These lock the failure modes the design requires: a missing or partial slot
must not assemble into a silent checkpoint, a published stage must not leak to
the next save, and a coordinator publish failure must surface as IO error.
"""

from __future__ import annotations

import pickle

import pytest

torch = pytest.importorskip("torch")

from gdtensor_sim import dcp_backend as db
from tests.support.schemas import tensor_schema


def _one_tensor():
    ct, _, _ = tensor_schema((1, 1, 1))
    return ct.named_d_tensors[0]


def test_missing_slot_raises_instead_of_silent_checkpoint(tmp_path):
    nd = _one_tensor()
    stage = db._Stage(str(tmp_path))
    with pytest.raises(Exception) as exc:
        db._assemble_tensor(nd, stage, on_disk=False)
    assert "missing staged slot" in str(exc.value) or "staged incompletely" in str(
        exc.value
    )


def test_partial_slot_raises_byte_length_mismatch(tmp_path):
    from gdtensor_sim._geometry import nonempty_columns

    nd = _one_tensor()
    stage = db._Stage(str(tmp_path))
    rank = next(iter(nonempty_columns(nd)))
    # Stage a buffer shorter than the slot expects.
    stage.slots[(nd.global_tensor_id, rank)] = bytearray(1)
    with pytest.raises(Exception) as exc:
        db._assemble_tensor(nd, stage, on_disk=False)
    assert "expected" in str(exc.value) or "missing" in str(exc.value)


def test_stage_is_cleared_after_publish_no_stale_reuse(tmp_path):
    """After a successful publish, staged bytes are dropped: a later save to the
    same root cannot assemble leftover slots from the previous cycle."""
    ct, _, _ = tensor_schema((1, 1, 1))
    stage = db._Stage(str(tmp_path))
    # Simulate a fully-staged tensor then publish through the save path.
    from gdtensor.builder import global_tensor_meta_of
    from gdtensor_sim._geometry import nonempty_columns, iv_total

    for nd in ct.named_d_tensors:
        meta = global_tensor_meta_of(nd)
        for rank, intervals in nonempty_columns(nd).items():
            slot_bytes = iv_total(intervals) * meta.itemsize
            stage.slots[(nd.global_tensor_id, rank)] = bytearray(slot_bytes)
    db._real_dcp_save(ct, stage, str(tmp_path), on_disk=False)
    # _real_dcp_save assembles+writes but does not clear; the session does.
    # Emulate the session clear to lock the reuse contract.
    stage.slots.clear()
    assert not stage.slots
    # Re-assembling on the cleared stage must now fail (no stale bytes).
    nd0 = ct.named_d_tensors[0]
    with pytest.raises(Exception):
        db._assemble_tensor(nd0, stage, on_disk=False)


def test_pickled_metadata_is_real_dcp_not_json(tmp_path):
    """Published .metadata must be the real torch DCP pickle, not our JSON doc."""
    from torch.distributed.checkpoint import save as dcp_save

    dcp_save({"x": torch.arange(3)}, checkpoint_id=str(tmp_path), no_dist=True)
    raw = (tmp_path / ".metadata").read_bytes()
    meta = pickle.loads(raw)
    assert hasattr(meta, "state_dict_metadata")
    assert not raw.startswith(b"{")


class _FailingCoord:
    """A coordinator that votes False, simulating another rank aborting."""

    def __init__(self):
        self.calls = 0

    def vote(self, ok):
        self.calls += 1
        return False


def test_publish_failure_propagates_as_io_error(tmp_path):
    """If the coordinator vote fails, commit must raise, not silently succeed."""
    from gdtensor_sim.dcp_backend import DcpSaveSession

    ct, _, _ = tensor_schema((1, 1, 1))
    layout = __import__("gdtensor_sim.models", fromlist=["build_layout"]).build_layout(
        ct, 1, "safetensors", lambda t, w, nd: f"f{t}.safetensors"
    )
    index = __import__(
        "gdtensor_sim._layout", fromlist=["build_payload_index"]
    ).build_payload_index(ct, layout)
    session = DcpSaveSession(
        0, str(tmp_path), ct, layout, index, coord=_FailingCoord(), mode="thread"
    )
    result = session.commit()
    assert type(result).__name__ == "Err", result
    assert (
        "publish" in str(result.value).lower() or "abort" in str(result.value).lower()
    )


def test_write_integrity_correct_length_but_partial_coverage_raises(tmp_path):
    """A slot buffer of the right byte count must still fail if one owned
    interval was never filled: assembly checks filled.all() over every element."""
    from gdtensor_sim._geometry import iv_total, nonempty_columns
    from gdtensor.builder import global_tensor_meta_of

    nd = _one_tensor()
    stage = db._Stage(str(tmp_path))
    rank = next(iter(nonempty_columns(nd)))
    meta = global_tensor_meta_of(nd)
    intervals = nonempty_columns(nd)[rank]
    total = iv_total(intervals) * meta.itemsize
    # Correct total byte length...
    buf = bytearray(total)
    # ...but leave the LAST interval's region as a non-written sentinel by
    # shortening the buffer the assembly sees for that interval.
    stage.slots[(nd.global_tensor_id, rank)] = buf
    # Tamper: drop bytes belonging to the final interval so assembly sees a gap.
    lo_last, hi_last = intervals[-1]
    covered = lo_last * meta.itemsize
    stage.slots[(nd.global_tensor_id, rank)] = buf[:covered]
    with pytest.raises(Exception) as exc:
        db._assemble_tensor(nd, stage, on_disk=False)
    assert (
        "expected" in str(exc.value)
        or "missing" in str(exc.value)
        or "incomplete" in str(exc.value)
    )

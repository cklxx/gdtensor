"""Hermetic contract tests for the real-DCP backend dtype codec.

These lock the byte-exact dtype contract of gdtensor_sim.dcp_backend: staged
slot bytes are the raw element bytes of the tensor dtype, and BF16 is never
widened through float32. No mesh / compute fabric is needed -- only the raw
buffer -> torch tensor reconstruction that the staging publisher relies on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gdtensor_sim import dcp_backend as db


class _Meta:
    def __init__(self, shape, dtype, itemsize):
        self.global_shape = list(shape)
        self.dtype = dtype
        self.itemsize = itemsize


def test_bf16_slot_bytes_reconstruct_without_float32_widening():
    shape = (2, 3)
    values = [1.0, 2.0, -3.0, 0.5, 100.0, -0.25]
    expected = torch.tensor(values, dtype=torch.bfloat16).reshape(shape)
    raw = bytes(expected.reshape(-1).view(torch.uint8).numpy())
    assert len(raw) == expected.numel() * 2

    out = db._torch_from_raw(raw, _Meta(shape, "BF16", 2))

    assert out.dtype == torch.bfloat16
    assert out.shape == torch.Size(shape)
    assert torch.equal(out, expected)
    # The 2-byte pattern must survive byte-exact; no float32 round trip.
    assert bytes(out.reshape(-1).view(torch.uint8).numpy()) == raw


def test_bf16_roundtrip_via_real_dcp_preserves_bytes():
    from torch.distributed.checkpoint import load as dcp_load, save as dcp_save

    import tempfile

    shape = (4, 2)
    expected = torch.randn(shape, dtype=torch.bfloat16)
    raw_before = bytes(expected.reshape(-1).view(torch.uint8).numpy())

    with tempfile.TemporaryDirectory() as root:
        dcp_save({"x": expected.clone()}, checkpoint_id=root, no_dist=True)
        out = {"x": torch.empty(shape, dtype=torch.bfloat16)}
        dcp_load(out, checkpoint_id=root, no_dist=True)

    assert out["x"].dtype == torch.bfloat16
    assert torch.equal(out["x"], expected)
    assert bytes(out["x"].reshape(-1).view(torch.uint8).numpy()) == raw_before


def test_empty_tensor_does_not_raise_on_view():
    out = db._torch_from_raw(b"", _Meta((0, 5), "I64", 8))
    assert out.dtype == torch.int64
    assert out.shape == torch.Size((0, 5))


def test_real_dtypes_reconstruct():
    cases = [
        ((6,), "F32", 4, torch.float32, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ((2, 3), "F16", 2, torch.float16, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ((4,), "I64", 8, torch.int64, [1, 2, 3, 4]),
    ]
    for shape, name, itemsize, dt, values in cases:
        t = torch.tensor(values, dtype=dt).reshape(shape)
        raw = bytes(t.reshape(-1).view(torch.uint8).numpy())
        out = db._torch_from_raw(raw, _Meta(shape, name, itemsize))
        assert out.dtype == dt
        assert torch.equal(out, t)

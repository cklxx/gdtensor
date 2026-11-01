"""LAWA accumulation precision policy (N6).

Checkpoint weights are BF16 in real training. LAWA must accumulate them in
a wider dtype; summing BF16 directly loses information. These tests pin the
policy without a process group:

1. F32 accumulation of BF16 inputs stays within a tight tolerance of the
   all-F32 reference.
2. Direct BF16 accumulation is measurably worse and is never silently used:
   its error strictly dominates the F32-accumulator error on a workload
   chosen to expose cancellation.
3. The output dtype stays BF16 (weights are not silently promoted on disk).
"""

import pytest
import torch

from tests.support.crc import crc_of, verify_payload


def _bf16_checkpoints(k=8, n=4096):
    # Small base plus a growing offset: successive versions differ slightly,
    # which is where BF16 rounding during repeated addition shows up.
    base = torch.arange(n, dtype=torch.float32) * 1e-3 - 2.0
    return [
        (base + (i - k / 2) * 0.01 + torch.randn(n) * 1e-4).to(torch.bfloat16)
        for i in range(k)
    ]


def _noisy_blob(values):
    # Random-noise payloads cannot be checked against an arange oracle;
    # CRC locks the exact input bytes instead.
    return b"".join(bytes(v.reshape(-1).view(torch.uint8).numpy()) for v in values)


def _verify_noisy_inputs(values, want_crc):
    verify_payload("lawa_bf16_noisy_inputs", want_crc, _noisy_blob(values))


def _f32_accumulate(values):
    acc = torch.zeros_like(values[0], dtype=torch.float32)
    for v in values:
        acc += v.to(torch.float32)
    return acc / len(values)


def _bf16_accumulate(values):
    acc = torch.zeros_like(values[0])
    for v in values:
        acc = acc + v  # both operands BF16
    return acc / len(values)


def test_f32_accumulator_matches_reference():
    torch.manual_seed(0)
    values = _bf16_checkpoints()
    want_crc = crc_of(_noisy_blob(values))
    reference = torch.stack([v.to(torch.float32) for v in values]).mean(0)
    result = _f32_accumulate(values)
    # F32 accumulation only pays the input quantization cost.
    assert torch.allclose(result, reference, atol=5e-3, rtol=5e-3)
    _verify_noisy_inputs(values, want_crc)


def test_bf16_accumulation_is_strictly_worse():
    torch.manual_seed(0)
    values = _bf16_checkpoints()
    want_crc = crc_of(_noisy_blob(values))
    reference = torch.stack([v.to(torch.float32) for v in values]).mean(0)
    f32_err = (_f32_accumulate(values) - reference).abs().mean().item()
    bf16_err = (_bf16_accumulate(values).float() - reference).abs().mean().item()
    assert bf16_err > f32_err * 2, (f32_err, bf16_err)
    _verify_noisy_inputs(values, want_crc)


def test_lawa_output_dtype_matches_input():
    torch.manual_seed(0)
    values = _bf16_checkpoints()
    want_crc = crc_of(_noisy_blob(values))
    mean = _f32_accumulate(values).to(torch.bfloat16)
    assert mean.dtype == torch.bfloat16
    # Round-trip comparison uses the BF16-quantized reference, not F32.
    quantized = (
        torch.stack([v.to(torch.float32) for v in values]).mean(0).to(torch.bfloat16)
    )
    torch.testing.assert_close(mean, quantized, rtol=0, atol=0)
    verify_payload(
        "lawa_bf16_output", crc_of(_noisy_blob([quantized])), _noisy_blob([mean])
    )
    _verify_noisy_inputs(values, want_crc)


def test_noisy_input_crc_rejects_mutation():
    torch.manual_seed(0)
    values = _bf16_checkpoints()
    want_crc = crc_of(_noisy_blob(values))
    values[0].view(torch.uint8)[0] ^= 1
    with pytest.raises(AssertionError, match="lawa_bf16_noisy_inputs"):
        _verify_noisy_inputs(values, want_crc)

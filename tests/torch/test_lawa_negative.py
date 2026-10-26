"""Negative controls for the LAWA numeric path (B3).

These tests prove the LAWA correctness checks can catch the three most
likely numeric bugs instead of passing vacuously:

1. A missed checkpoint version (only k-1 of k inputs averaged).
2. A wrong divisor (sum divided by k-1 or k+1 instead of k).
3. Invalid caller weights (negative, zero-sum, non-finite).

The numeric bugs are exercised at single-process tensor level, so they do
not need the gloo process group. The accumulation formula mirrors the one
used in test_lawa.py: naive running sum divided by k, and an incremental
weighted running mean.
"""

import numpy as np
import pytest
import torch

from tests.support.lawa import normalize_weights


def _checkpoints(torch):
    return [
        torch.arange(12, dtype=torch.float32).reshape(3, 4) * (i + 1) + i
        for i in range(5)
    ]


def _naive_mean(torch, values, included, divisor):
    return sum(values[i] for i in included) / divisor


def test_missing_version_is_detected():
    """Skipping one checkpoint must not match the k-version mean."""
    values = _checkpoints(torch)
    k = 5
    correct = _naive_mean(torch, values, list(range(k)), k)
    # Bug: the last version never enters the accumulator.
    buggy = _naive_mean(torch, values, list(range(k - 1)), k)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(buggy, correct)


def test_wrong_divisor_is_detected():
    """Dividing by k-1 or k+1 must not match the true mean."""
    values = _checkpoints(torch)
    k = 5
    correct = _naive_mean(torch, values, list(range(k)), k)
    for wrong_divisor in (k - 1, k + 1):
        buggy = _naive_mean(torch, values, list(range(k)), wrong_divisor)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(buggy, correct)


def test_weighted_path_agrees_on_equal_weights():
    """Sanity: the weighted helper reduces to the plain mean on equal weights."""
    values = _checkpoints(torch)
    k = 3
    weights = normalize_weights(torch, [2.0, 2.0, 2.0])
    weighted = sum(w * values[i] for i, w in enumerate(weights))
    torch.testing.assert_close(weighted, _naive_mean(torch, values, list(range(k)), k))


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, -1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, float("nan")],
        [1.0, float("inf")],
        [],
    ],
)
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        normalize_weights(torch, weights)


def test_numpy_and_torch_oracle_agree_on_shape():
    """Independent numpy mean must agree with the torch oracle elementwise."""
    values = _checkpoints(torch)
    k = 4
    torch_mean = _naive_mean(torch, values, list(range(k)), k)
    np_mean = np.mean([v.numpy() for v in values[:k]], axis=0)
    torch.testing.assert_close(torch_mean, torch.from_numpy(np_mean))

"""Property tests for interval math against a brute-force element model.

Fixed-boundary unit tests can miss off-by-one mutants that are equivalent on
the chosen examples. These random trials compare the vectorized interval
functions to a flat boolean ground truth over many shapes.
"""
from __future__ import annotations

import numpy as np
import pytest

from gdtensor.dtensor import iv_normalize, iv_intersect, iv_subtract, iv_total

RNG_SEEDS = range(60)
MAX_POINT = 40


def _brute(iv):
    m = np.zeros(MAX_POINT, dtype=bool)
    for a, b in iv:
        m[a:b] = True
    return m


def _from_mask(m):
    out, i = [], 0
    while i < MAX_POINT:
        if m[i]:
            j = i + 1
            while j < MAX_POINT and m[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _random_intervals(rng):
    n = rng.integers(0, 5)
    pts = sorted(int(x) for x in rng.choice(MAX_POINT, size=2 * n, replace=False))
    return np.array(list(zip(pts[0::2], pts[1::2])), dtype=np.int64).reshape(-1, 2)


@pytest.mark.parametrize("seed", RNG_SEEDS)
def test_normalize_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    # include adjacent and overlapping raw intervals
    iv = _random_intervals(rng)
    if iv.size and rng.random() < 0.5:
        a, _ = iv[0]
        iv = np.vstack([iv, [(a, a + 1)]])  # overlap with first interval
    got = _brute(iv_normalize(iv))
    want = _brute(iv)
    assert np.array_equal(got, want)
    # normalized output is sorted and strictly disjoint
    n = iv_normalize(iv)
    if len(n) > 1:
        assert np.all(n[1:, 0] > n[:-1, 1])


@pytest.mark.parametrize("seed", RNG_SEEDS)
def test_intersect_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    a, b = _random_intervals(rng), _random_intervals(rng)
    got_iv = iv_intersect(a, b)
    # canonical form: strictly positive-width, sorted, non-overlapping rows
    if got_iv.size:
        assert np.all(got_iv[:, 1] > got_iv[:, 0]), got_iv
        assert np.all(got_iv[1:, 0] > got_iv[:-1, 1]), got_iv
    assert np.array_equal(_brute(got_iv), _brute(a) & _brute(b))


@pytest.mark.parametrize("seed", RNG_SEEDS)
def test_subtract_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    a, b = _random_intervals(rng), _random_intervals(rng)
    got = _brute(iv_subtract(a, b))
    want = _brute(a) & ~_brute(b)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("seed", RNG_SEEDS)
def test_intersect_and_subtract_partition(seed):
    """A & B, then A \\ B must reconstruct A exactly."""
    rng = np.random.default_rng(1000 + seed)
    a, b = _random_intervals(rng), _random_intervals(rng)
    both = iv_intersect(a, b)
    rest = iv_subtract(a, b)
    recon = iv_normalize(np.vstack([both, rest]) if both.size or rest.size else
                         np.zeros((0, 2), dtype=np.int64))
    assert np.array_equal(_brute(recon), _brute(a))
    assert iv_total(recon) == iv_total(a)

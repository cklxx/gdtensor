"""Direct unit tests for interval math and plan generation.

These attack the invariants that the higher-level tests only cover indirectly.
"""
from __future__ import annotations

import numpy as np
import pytest

from gdtensor.internal.dtensor import iv_intersect, iv_normalize, iv_prefix, iv_subtract
from gdtensor.internal.plan import _pair
from gdtensor.internal.ranges import RangeFromComputeView, RangeFromStorageView
from gdtensor.internal.result import Fail


def _iv(lst):
    return np.array(lst, dtype=np.int64).reshape(-1, 2)


# ------------------------------------------------------------------ iv_normalize


def test_iv_normalize_merges_overlapping():
    a = _iv([(0, 5), (3, 8), (10, 12)])
    got = iv_normalize(a)
    assert got.tolist() == [[0, 8], [10, 12]]


def test_iv_normalize_merges_touching():
    a = _iv([(0, 5), (5, 10)])
    got = iv_normalize(a)
    assert got.tolist() == [[0, 10]]


def test_iv_normalize_sorts():
    a = _iv([(10, 12), (0, 5)])
    got = iv_normalize(a)
    assert got.tolist() == [[0, 5], [10, 12]]


def test_iv_normalize_drops_empty():
    a = _iv([(0, 0), (5, 10)])
    got = iv_normalize(a)
    assert got.tolist() == [[5, 10]]


def test_iv_normalize_empty():
    assert iv_normalize(np.zeros((0, 2), dtype=np.int64)).shape == (0, 2)


# ------------------------------------------------------------------ iv_intersect


def test_iv_intersect_basic():
    a = _iv([(0, 10), (20, 30)])
    b = _iv([(5, 25)])
    got = iv_intersect(a, b)
    assert got.tolist() == [[5, 10], [20, 25]]


def test_iv_intersect_no_overlap():
    a = _iv([(0, 5)])
    b = _iv([(10, 15)])
    assert iv_intersect(a, b).shape == (0, 2)


def test_iv_intersect_touching_is_empty():
    a = _iv([(0, 5)])
    b = _iv([(5, 10)])
    assert iv_intersect(a, b).shape == (0, 2)


def test_iv_intersect_multiple_b():
    a = _iv([(0, 100)])
    b = _iv([(0, 10), (20, 30), (40, 50)])
    got = iv_intersect(a, b)
    assert got.tolist() == [[0, 10], [20, 30], [40, 50]]


# ------------------------------------------------------------------ iv_subtract


def test_iv_subtract_basic():
    a = _iv([(0, 10)])
    b = _iv([(3, 7)])
    got = iv_subtract(a, b)
    assert got.tolist() == [[0, 3], [7, 10]]


def test_iv_subtract_b_covers_a():
    a = _iv([(0, 10)])
    b = _iv([(0, 10)])
    assert iv_subtract(a, b).shape == (0, 2)


def test_iv_subtract_b_starts_before_a():
    a = _iv([(5, 10)])
    b = _iv([(0, 7)])
    got = iv_subtract(a, b)
    assert got.tolist() == [[7, 10]]


def test_iv_subtract_b_ends_after_a():
    a = _iv([(0, 10)])
    b = _iv([(5, 15)])
    got = iv_subtract(a, b)
    assert got.tolist() == [[0, 5]]


def test_iv_subtract_multiple_a_b():
    a = _iv([(0, 10), (20, 30)])
    b = _iv([(5, 25)])
    got = iv_subtract(a, b)
    assert got.tolist() == [[0, 5], [25, 30]]


def test_iv_subtract_b_with_gap():
    a = _iv([(0, 30)])
    b = _iv([(5, 10), (20, 25)])
    got = iv_subtract(a, b)
    assert got.tolist() == [[0, 5], [10, 20], [25, 30]]


def test_iv_subtract_empty_a():
    a = np.zeros((0, 2), dtype=np.int64)
    b = _iv([(0, 10)])
    assert iv_subtract(a, b).shape == (0, 2)


def test_iv_subtract_empty_b():
    a = _iv([(0, 10)])
    b = np.zeros((0, 2), dtype=np.int64)
    assert iv_subtract(a, b).tolist() == [[0, 10]]


def test_iv_subtract_b_overlaps_two_a_intervals():
    """The j-pointer must not skip the second a-interval."""
    a = _iv([(0, 10), (15, 25)])
    b = _iv([(5, 20)])
    got = iv_subtract(a, b)
    assert got.tolist() == [[0, 5], [20, 25]]


def test_iv_subtract_b_splits_each_a():
    a = _iv([(0, 10), (20, 30)])
    b = _iv([(2, 4), (22, 24)])
    got = iv_subtract(a, b)
    assert got.tolist() == [[0, 2], [4, 10], [20, 22], [24, 30]]


# ------------------------------------------------------------------ _pair


def _crange(tid, off, size):
    return RangeFromComputeView(tid, off, size)


def _srange(path, off, size):
    return RangeFromStorageView(path, off, size)


def test_pair_exact_match():
    dst = [(_iv([(0, 100)]), ("compute", "t", 1))]
    src = [(_iv([(0, 100)]), ("storage", "f", 0, 1))]
    plan = _pair(dst, src)
    assert len(plan) == 1
    s, d = plan[0]
    assert s.contiguous_payload_offset == 0 and s.size == 100
    assert d.global_tensor_offset == 0 and d.size == 100


def test_pair_src_split_into_two():
    dst = [(_iv([(0, 100)]), ("compute", "t", 1))]
    src = [(_iv([(0, 40)]), ("storage", "f", 0, 1)),
           (_iv([(40, 100)]), ("storage", "f", 40, 1))]
    plan = _pair(dst, src)
    assert len(plan) == 2
    assert plan[0][1].size == 40
    assert plan[1][1].global_tensor_offset == 40


def test_pair_dst_split_by_src_gap():
    dst = [(_iv([(0, 100)]), ("compute", "t", 1))]
    src = [(_iv([(0, 40)]), ("storage", "f", 0, 1)),
           (_iv([(60, 100)]), ("storage", "f", 60, 1))]
    # dst [40,60) has no src -> must raise (full coverage invariant)
    with pytest.raises(Fail, match="no source covers"):
        _pair(dst, src)


def test_pair_multiple_dst_intervals():
    dst = [(_iv([(0, 50), (80, 130)]), ("compute", "t", 1))]
    src = [(_iv([(0, 130)]), ("storage", "f", 0, 1))]
    plan = _pair(dst, src)
    assert len(plan) == 2
    assert plan[0][1].global_tensor_offset == 0 and plan[0][1].size == 50
    assert plan[1][1].global_tensor_offset == 80 and plan[1][1].size == 50


def test_pair_src_larger_than_dst():
    dst = [(_iv([(20, 80)]), ("compute", "t", 1))]
    src = [(_iv([(0, 100)]), ("storage", "f", 0, 1))]
    plan = _pair(dst, src)
    assert len(plan) == 1
    assert plan[0][0].contiguous_payload_offset == 20
    assert plan[0][0].size == 60
    assert plan[0][1].global_tensor_offset == 20


def test_pair_empty_plan_when_no_overlap():
    dst = [(_iv([(0, 10)]), ("compute", "t", 1))]
    src = [(_iv([(20, 30)]), ("storage", "f", 20, 1))]
    with pytest.raises(Fail, match="no source covers"):
        _pair(dst, src)


# ------------------------------------------------------------------ mutants: touching-only cases


def test_iv_intersect_only_touching_is_empty():
    a = _iv([(0, 5)])
    b = _iv([(5, 10)])
    assert iv_intersect(a, b).size == 0
    # intersect of two touching-only multi-interval lists must stay empty
    a2 = _iv([(0, 5), (10, 15)])
    b2 = _iv([(5, 10), (15, 20)])
    assert iv_intersect(a2, b2).size == 0


def test_iv_prefix_last_element_of_interval():
    # points at the final element: looser end check would accept the point
    # just past the interval
    a = _iv([(10, 20)])
    assert iv_prefix(a, np.array([19])).tolist() == [9]
    with pytest.raises(Exception):
        iv_prefix(a, np.array([20]))

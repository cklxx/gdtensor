"""Setup optimizations must not change any plan, file mapping or leader.

The golden records were produced by the implementation that still hashed a
shared mesh list once per layout entry and filled the storage slot grid in
Python loops. Every case must still give the same record: the same save
plans, the same load plans on the saving topology and on a different one
(reshard), the same file for every layout entry, and the same leader cell
and constraint result for every storage mesh list.
"""

import json
from pathlib import Path

import pytest

from tests.support import setup_plans

GOLDEN = json.loads(
    (Path(setup_plans.__file__).parent / "setup_plan_golden.json").read_text()
)


@pytest.fixture(scope="module")
def records():
    return setup_plans.collect()


def test_cases_match_golden_set(records):
    assert sorted(records) == sorted(GOLDEN)
    assert sum(key.startswith("layout/") for key in GOLDEN) == 60
    assert sum(key.startswith("mesh/") for key in GOLDEN) == 80


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_setup_record_is_unchanged(records, key):
    assert records[key] == GOLDEN[key]


def test_golden_layout_cases_exercise_real_plans():
    layouts = [value for key, value in GOLDEN.items() if key.startswith("layout/")]
    assert all(value["check"] == "Ok" for value in layouts)
    assert all(value["save"][0] > 0 for value in layouts)
    assert all(value["load"][0] > 0 for value in layouts)
    assert all(value["reshard"][0] > 0 for value in layouts)

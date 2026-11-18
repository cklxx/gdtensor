"""Stable plan snapshots for propagation-only optimizations."""

import hashlib
import json
import sys
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

import gdtensor.io as gio
from tests.support.cache_shapes import _build


def _case(
    name, layers, nodes=3, k=2, steps=None, width=None, sizes=None, replicate=None
):
    return pytest.param(
        name,
        dict(
            seed=0,
            k=k,
            nodes=nodes,
            layers=layers,
            steps=steps or [3] * len(layers),
            width=width or [2] * len(layers),
            sizes=sizes or [17, 9],
            replicate=replicate or [False, True],
            coordinator="thread",
            bandwidth=0.0,
            repeat=1,
        ),
        id=name,
    )


CASES = [
    _case("single-rank-one-layer", [[0]], nodes=1, k=1),
    _case("one-layer-subset", [[0, 2]]),
    _case("two-layer-fanout", [[0], [0, 2]]),
    _case(
        "two-layer-uneven",
        [[0, 1], [0, 1, 2, 3]],
        nodes=4,
        steps=[2, 4],
        width=[2, 1],
        sizes=[64, 33, 7],
        replicate=[False, True, False],
    ),
    _case(
        "two-layer-replicated",
        [[0, 2], [1, 2]],
        steps=[4, 2],
        width=[1, 3],
        sizes=[31, 16, 5],
        replicate=[True, True, False],
    ),
]

# Digests are computed from the unoptimized planner and must not be regenerated
# as part of an optimization change.
EXPECTED = {
    "single-rank-one-layer": "047bb4686a07a1186ad936afdaaa041959a7883d9559c2c8faaeb41bf21e2585",
    "one-layer-subset": "a5ad7e7c239b663ac40dcd38cd3a8e4ecd54ca3b61a6c0c01b35bce287d9991d",
    "two-layer-fanout": "fdc0fb270ed5b9edd9650f52004be2ea35a335a108c03f807d88bd763bbb22a5",
    "two-layer-uneven": "d15fe20f471f14f4261c8cdcc3589819feaea05c9549b86227f1cc3db0336f54",
    "two-layer-replicated": "465550191ee63b673797b973ca4b9b5368aab77f934f3e103cb411752b014f5e",
}


def _normalize(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            "fields": {
                f.name: _normalize(getattr(value, f.name)) for f in fields(value)
            },
        }
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "items": _normalize(value.tolist())}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            "dict": sorted(
                [(_normalize(key), _normalize(item)) for key, item in value.items()],
                key=repr,
            )
        }
    if isinstance(value, tuple):
        return {"tuple": [_normalize(item) for item in value]}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((_normalize(item) for item in value), key=repr)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(type(value).__name__)


def _plan_digest(chain) -> str:
    api = sys.modules[gio.chain_load.__module__]
    plan = api._slots(chain).plan
    propagation = sys.modules[f"{api.__package__}.propagation"]
    snapshot = {
        "plan": _normalize(plan),
        "copy_plans": {
            str(rank): _normalize(propagation.rank_copy_plans(plan, rank))
            for rank in plan.ranks
        },
    }
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.parametrize("name,case", CASES)
def test_plan_snapshot(name, case):
    assert _plan_digest(_build(case)["chain"]) == EXPECTED[name]


@pytest.mark.parametrize(
    "pieces,lo,hi",
    [
        pytest.param(
            [
                ("b", [[2, 4], [6, 8]], 200),
                ("a", [[0, 2], [4, 6]], 100),
            ],
            1,
            8,
            id="disjoint-unsorted-noncontiguous",
        ),
        pytest.param(
            [("first", [[4, 10]], 400), ("second", [[0, 6]], 0)],
            2,
            9,
            id="overlap-preserves-source-priority",
        ),
        pytest.param(
            [("first", [[0, 8]], 0), ("second", [[0, 8]], 800)],
            1,
            7,
            id="duplicate-uses-first-source",
        ),
        pytest.param([("a", [[0, 2], [5, 8]], 100)], 1, 7, id="gap"),
        pytest.param([], 1, 7, id="empty"),
    ],
)
def test_storage_piece_pairing_matches_generic(pieces, lo, hi):
    api = sys.modules[gio.chain_load.__module__]
    propagation = sys.modules[f"{api.__package__}.propagation"]
    plan = sys.modules[f"{api.__package__}.plan"]
    chain = _build(CASES[0].values[1])["chain"]
    side = plan._Side(chain.src)
    tid = chain.dst.tensor_ids[0]
    isz = propagation.global_tensor_meta_of(
        propagation.named_of(side.table, tid)
    ).itemsize
    side._storage_by_tid[tid] = [
        (np.array(iv, dtype=np.int64), path, cpo, isz) for path, iv, cpo in pieces
    ]
    seg = propagation.Segment(0, 0, tid, lo * isz, (hi - lo) * isz, (tid, 0, 0))
    iv = np.array([[lo, hi]], dtype=np.int64)
    candidates = side.src_candidates(tid, 0)

    try:
        pairs = plan._pair([(iv, ("compute", tid, isz))], candidates)
    except Exception as expected:
        with pytest.raises(type(expected)) as actual:
            propagation._storage_pieces(side, seg)
        assert str(actual.value) == str(expected)
    else:
        expected = sorted(
            ((src, dst.global_tensor_offset - seg.off) for src, dst in pairs),
            key=lambda piece: piece[1],
        )
        assert propagation._storage_pieces(side, seg) == expected

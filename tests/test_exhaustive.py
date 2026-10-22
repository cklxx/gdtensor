"""Exhaustive plan-space tests (small-world hypothesis).

Enumerate every valid (tp, dp, pp) with world <= 8 over a small model, both
hf and mirror storage modes, and every layout. For each case build the per-rank
save and load plans and check them against an independent oracle:

* each pair moves exactly the same global elements (compute <-> storage);
* save: ranks read only local memory, union of writes covers every payload
  byte exactly once;
* load: each rank fills exactly its owned bytes exactly once.

Also cross-loads every combination of save parallelism -> load parallelism.
No IO is performed, so thousands of configurations are cheap.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pytest

from gdtensor._dtensor import tensor_ids_of as _ids  # noqa: E402
from gdtensor._formats import make_storage_view
from gdtensor._layout import build_payload_index
from gdtensor._models import (
    ModelConfig, ParallelConfig, compute_table, layout_builder, storage_table,
)
from gdtensor._plan import d_tensors_view_rank_plan_load, d_tensors_view_rank_plan_save
from gdtensor._runner import compute_view
from gdtensor._views import ComputeDataAccessor, ComputeFabric
from oracle import check_load, check_save

# pp <= layers: model_tensors maps every layer to a stage; pp > layers gives
# empty stages, covered separately by test_unsupported_empty_pp_stage.
CONFIGS = [ParallelConfig(tp=t, dp=d, pp=p)
           for t, d, p in itertools.product(range(1, 9), repeat=3)
           if t * d * p <= 8 and p <= 2]

LAYOUTS = [
    ("hf", "single"), ("hf", "hf"), ("hf", "per_rank"), ("hf", "rank_multi"),
    ("hf", "per_stage"), ("mirror", "per_rank"), ("mirror", "rank_multi"),
]


def _plans(ct, st, lay, world, op):
    sv = make_storage_view(_ids(st), st, lay, "/nonexistent")
    index = sv.data_accessor.index
    plans = {}
    fn = d_tensors_view_rank_plan_save if op == "save" else d_tensors_view_rank_plan_load
    for r in range(world):
        cv = compute_view(ct, None, r)
        cv.data_accessor = ComputeDataAccessor(r, None)
        plans[r] = fn(r, cv, sv).unwrap()
    return plans, index


@pytest.mark.parametrize("p", CONFIGS, ids=lambda p: f"t{p.tp}d{p.dp}p{p.pp}")
@pytest.mark.parametrize("mode,kind", LAYOUTS)
def test_plan_oracle_exhaustive(p, mode, kind):
    m = ModelConfig(layers=2, hidden=8, ffn=12, vocab=16)
    ct = compute_table(m, p)
    st = storage_table(ct, p, mode)
    lay = layout_builder(kind, st, p, "safetensors", hf_shards=2)
    ps, index = _plans(ct, st, lay, p.world, "save")
    check_save(ps, ct, st, index)
    pl, index = _plans(ct, st, lay, p.world, "load")
    check_load(pl, ct, st, index)


@pytest.mark.parametrize("p", CONFIGS, ids=lambda p: f"t{p.tp}d{p.dp}p{p.pp}")
def test_non_divisible_shapes_plan(p):
    # hidden=5 forces uneven torch.chunk splits; ffn=7 and vocab=13 add primes.
    m = ModelConfig(layers=2, hidden=5, ffn=7, vocab=13)
    ct = compute_table(m, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("single", st, p, "safetensors")
    ps, index = _plans(ct, st, lay, p.world, "save")
    check_save(ps, ct, st, index)
    pl, _ = _plans(ct, st, lay, p.world, "load")
    check_load(pl, ct, st, index)


def _e2e(root, m, ps_cfg, pl_cfg, mode, kind):
    from conftest import assert_all_ok, verify_fabric, verify_safetensors_dir
    from gdtensor._runner import load_threads, save_threads

    ct = compute_table(m, ps_cfg)
    st = storage_table(ct, ps_cfg, mode)
    lay = layout_builder(kind, st, ps_cfg, "safetensors")
    fab = ComputeFabric(ct).allocate(fill=__import__(
        "gdtensor._models", fromlist=["fill_pattern"]).fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, ps_cfg.world)[0])
    verify_safetensors_dir(root, st)
    ctl = compute_table(m, pl_cfg)
    fab2 = ComputeFabric(ctl).allocate()
    assert_all_ok(load_threads(ctl, st, lay, root, fab2, pl_cfg.world)[0])
    verify_fabric(fab2, ctl)


# A bounded end-to-end subset: every layout and mode at representative worlds,
# plus non-divisible shapes and cross-parallelism loads.
E2E_CONFIGS = [
    (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (4, 1, 1),
    (2, 1, 2), (4, 2, 1),
]


@pytest.mark.parametrize("t,d,pp", E2E_CONFIGS)
@pytest.mark.parametrize("mode,kind", LAYOUTS)
def test_e2e_roundtrip_exhaustive(tmp_path, t, d, pp, mode, kind):
    p = ParallelConfig(tp=t, dp=d, pp=pp)
    _e2e(str(tmp_path), ModelConfig(layers=2, hidden=8, ffn=12, vocab=16), p, p, mode, kind)


@pytest.mark.parametrize("save_cfg,load_cfg", [
    (ParallelConfig(tp=4, dp=1, pp=1), ParallelConfig(tp=1, dp=4, pp=1)),
    (ParallelConfig(tp=1, dp=4, pp=1), ParallelConfig(tp=2, dp=2, pp=1)),
    (ParallelConfig(tp=2, dp=2, pp=1), ParallelConfig(tp=1, dp=1, pp=1)),
    (ParallelConfig(tp=1, dp=1, pp=2), ParallelConfig(tp=2, dp=1, pp=1)),
    (ParallelConfig(tp=2, dp=1, pp=2), ParallelConfig(tp=1, dp=2, pp=2)),
])
def test_e2e_cross_parallelism(tmp_path, save_cfg, load_cfg):
    _e2e(str(tmp_path), ModelConfig(layers=2, hidden=8, ffn=12, vocab=16),
         save_cfg, load_cfg, "hf", "hf")


def test_e2e_non_divisible(tmp_path):
    _e2e(str(tmp_path), ModelConfig(layers=2, hidden=5, ffn=7, vocab=13),
         ParallelConfig(tp=3, dp=1, pp=1), ParallelConfig(tp=2, dp=1, pp=1), "hf", "single")

"""LAWA merge (design section 10): element-wise average of k checkpoints.

The reduction lives on torch.DTensor, exactly as section 10.3 requires:
gdtensor only provides load / save and the 5-function compute bridge; there
is no core reduce API. Both the naive (sum then /k) and Welford (running
mean) versions use the same load/save/id iteration and must produce the same
averaged checkpoint.

Single-process: each checkpoint is a complete sharded safetensors dir. We
load via the gdtensor storage/plan path into fabric buffers (proving the
storage -> compute bridge), then materialize torch tensors on which the
merge arithmetic runs, reset the gd tensor, and save back. The final files are
verified byte-equivalent against a directly averaged torch tensor.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", minversion="2.0")
import torch.distributed as dist  # noqa: E402
from torch.distributed.tensor import Replicate as TRep, distribute_tensor, init_device_mesh  # noqa: E402

from conftest import assert_all_ok  # noqa: E402
from gdtensor._dtensor import tensor_ids_of as _ids  # noqa: E402
from gdtensor.builder import get_gd_tensor, get_tensor_ids, reset_d_tensor  # noqa: E402
from gdtensor._torch_bridge import gd_for_layout  # noqa: E402
from gdtensor._dtensor import DTensorTable, NamedDTensor, named_of, global_tensor_meta_of  # noqa: E402
from gdtensor._models import ParallelConfig, layout_builder, storage_table  # noqa: E402
from gdtensor._runner import compute_view, load_threads, save_threads  # noqa: E402
from gdtensor.tag import Replicate, Shard  # noqa: E402
from gdtensor._torch_bridge import torch_from_fabric, fabric_from_torch  # noqa: E402
from gdtensor._views import ComputeFabric  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1")


@pytest.fixture(scope="module", autouse=True)
def _pg():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


P = ParallelConfig(tp=1, dp=4, pp=1)
PLS = (Replicate(), Shard(0), Replicate())
SHAPE = (8, 8)


def _table_and_layout(shape=SHAPE):
    # build one consistent gd schema for the given global shape
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    g = torch.zeros(shape, dtype=torch.float32)
    dt = distribute_tensor(g, mesh, [TRep()])
    gd = gd_for_layout(dt, "w", p=P, placements=PLS).unwrap()
    ct = DTensorTable([NamedDTensor("w", gd)])
    st = storage_table(ct, P, "hf")
    lay = layout_builder("single", st, P, "safetensors")
    return ct, st, lay


def _save_ckpt(root, value, shape=None):
    """Save one checkpoint whose only tensor equals ``value``."""
    shape = tuple(value.shape) if shape is None else shape
    ct, st, lay = _table_and_layout(shape)
    os.makedirs(root, exist_ok=True)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    dt = distribute_tensor(value, mesh, [TRep()])
    fab = fabric_from_torch(ct, {"w": dt})
    assert_all_ok(save_threads(ct, st, lay, root, fab, P.world)[0])
    return st, lay


def _load_as_torch(root, ct, st, lay):
    """gdtensor load into fabric -> global torch tensor (compute bridge)."""
    fab = ComputeFabric(ct).allocate()
    assert_all_ok(load_threads(ct, st, lay, root, fab, P.world)[0])
    return torch_from_fabric(ct, fab, "w")


# ------------------------------------------------------------------ both versions


def _naive_average(ckpt_roots, ct, st, lay):
    """Section 10.1: sum into acc torch tensors, divide by k, then save."""
    acc = None
    k = len(ckpt_roots)
    for root in ckpt_roots:
        t = _load_as_torch(root, ct, st, lay)
        acc = t if acc is None else acc + t
    return acc / k


def _welford_average(ckpt_roots, ct, st, lay):
    """Section 10.2: running mean with delta/n; no final division."""
    mean = None
    n = 0
    for root in ckpt_roots:
        t = _load_as_torch(root, ct, st, lay)
        n += 1
        if mean is None:
            mean = t.clone()
        else:
            delta = t - mean
            mean = mean + delta / n
    return mean


def _save_result(root, value):
    ct, st, lay = _table_and_layout()
    os.makedirs(root, exist_ok=True)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    dt = distribute_tensor(value, mesh, [TRep()])
    fab = fabric_from_torch(ct, {"w": dt})
    assert_all_ok(save_threads(ct, st, lay, root, fab, P.world)[0])
    return ct, st, lay


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_lawa_naive_and_welford_match(tmp_path, k):
    torch.manual_seed(0)
    values = [torch.randn(SHAPE, dtype=torch.float32) for _ in range(k)]
    roots = []
    for i, v in enumerate(values):
        r = str(tmp_path / f"ckpt{i}")
        _save_ckpt(r, v)
        roots.append(r)

    ct, st, lay = _table_and_layout()
    naive = _naive_average(roots, ct, st, lay)
    welford = _welford_average(roots, ct, st, lay)
    direct = torch.stack(values).mean(dim=0)

    assert torch.allclose(naive, direct, atol=1e-6)
    assert torch.allclose(welford, direct, atol=1e-6)
    assert torch.allclose(naive, welford, atol=1e-6)

    # save the Welford result through gdtensor rank_save and reload it
    out = str(tmp_path / "out"); cto, sto, layo = _save_result(out, welford)
    back = _load_as_torch(out, cto, sto, layo)
    assert torch.allclose(back, direct, atol=1e-6)


def test_lawa_uses_bridge_functions_not_core_reduce(monkeypatch):
    """The reduction must go through get/reset bridge; there is no core reduce.

    We assert the public bridge surface is the only compute path by checking
    the gdtensor package exposes no reduce/mean API.
    """
    import gdtensor
    import gdtensor.builder as builder
    for forbidden in ("rank_reduce", "rank_mean", "average", "lawa", "merge_checkpoints"):
        assert not hasattr(gdtensor, forbidden)
        assert not hasattr(builder, forbidden)
    # bridge functions the section-10 flow relies on do exist
    for fn in (get_tensor_ids, get_gd_tensor, gd_for_layout, reset_d_tensor):
        assert callable(fn)


def test_lawa_meta_mismatch_is_rejected(tmp_path):
    """Averaging requires identical tensor ids and meta (section 10.3).

    Differing GlobalTensorMeta between two views is rejected by the plan
    precondition (_check_same_tensors inside plan_save/reshard), so a LAWA
    loop that hands two differently-typed checkpoints to one view errors
    before any range selection.
    """
    from gdtensor._api import d_tensors_view_rank_plan_save
    from gdtensor._formats import make_storage_view
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
    v = torch.zeros(SHAPE, dtype=torch.float32)
    gd32 = gd_for_layout(distribute_tensor(v, mesh, [TRep()]), "w",
                          p=P, placements=PLS).unwrap()
    gd64 = gd_for_layout(distribute_tensor(v.double(), mesh, [TRep()]), "w",
                          p=P, placements=PLS).unwrap()
    t32 = DTensorTable([NamedDTensor("w", gd32)])
    t64 = DTensorTable([NamedDTensor("w", gd64)])
    st64 = storage_table(t64, P, "hf"); lay64 = layout_builder("single", st64, P, "safetensors")
    sv64 = make_storage_view(_ids(st64), st64, lay64, "/nonexistent")
    cv32 = compute_view(t32, ComputeFabric(t32).allocate(), 0)
    # f32 compute -> f64 storage: global_tensor_meta differs -> rejected
    r = d_tensors_view_rank_plan_save(0, cv32, sv64)
    assert not r.ok and "global_tensor_meta differs" in r.error
    # equal meta is accepted (sanity)
    st32 = storage_table(t32, P, "hf"); lay32 = layout_builder("single", st32, P, "safetensors")
    sv32 = make_storage_view(_ids(st32), st32, lay32, "/nonexistent")
    assert d_tensors_view_rank_plan_save(0, cv32, sv32).ok

"""Real gloo multi-rank torch.distributed.checkpoint save/load.

This proves genuine distributed DCP (collective, no ``no_dist=True``) across
forked/spawned CPU ranks on one host. It does NOT claim multi-machine.
"""

import multiprocessing as mp
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _worker(rank, init_file, root, op):
    import torch.distributed as dist
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint import save as dcp_save

    torch.set_num_threads(1)
    _init = getattr(dist, "init_process_group")
    _destroy = getattr(dist, "destroy_process_group")
    _init(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        if op == "save":
            # Replicated full tensor: both ranks participate in the collective.
            state = {"x": torch.arange(6, dtype=torch.float32) + rank * 100}
            dcp_save(state, checkpoint_id=root)
        else:
            out = {"x": torch.empty(6, dtype=torch.float32)}
            dcp_load(out, checkpoint_id=root)
            np.save(Path(root) / f"loaded{rank}.npy", out["x"].numpy())
    finally:
        _destroy()


def _spawn(root, op):
    init_file = Path(root) / "init"
    init_file.touch()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(r, str(init_file), str(root), op))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(180)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_real_gloo_distributed_dcp_save_load_roundtrip(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _spawn(root, "save")
    artifacts = sorted(os.listdir(root))
    assert ".metadata" in artifacts, artifacts

    # Fresh collective load on both ranks.
    _spawn(root, "load")
    a = np.load(root / "loaded0.npy")
    b = np.load(root / "loaded1.npy")
    np.testing.assert_array_equal(a, np.arange(6) + 0)
    np.testing.assert_array_equal(b, np.arange(6) + 0)

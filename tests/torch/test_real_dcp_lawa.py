"""F2: LAWA over real torch.distributed.checkpoint files.

All other LAWA tests average either safetensors or the simulated distcp
container. Here the inputs and the averaged output are genuine DCP
checkpoints written by a collective gloo 2-rank job:

1. Each rank saves its local Shard(0) of a 6x2 tensor for k versions through
   the real torch.distributed.checkpoint.save collective.
2. Each rank loads its own shard of every version, computes the running mean
   (the caller-side LAWA operation), and saves the mean shard back through
   DCP.
3. The parent independently loads the merged checkpoint (no_dist) and checks
   it elementwise against a dense numpy mean of the k global versions.
"""

import multiprocessing as mp
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

GLOBAL_ROWS = 6
COLS = 2
ROWS_PER_RANK = GLOBAL_ROWS // 2
K = 3


def _global_version(index):
    base = torch.arange(GLOBAL_ROWS * COLS, dtype=torch.float32).reshape(
        GLOBAL_ROWS, COLS
    )
    return base * (index + 1) + index * index


def _worker(rank, init_file, root):
    import torch.distributed as dist
    import torch.distributed.tensor as dtt
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint import save as dcp_save
    from torch.distributed.device_mesh import DeviceMesh

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
        mesh = DeviceMesh("cpu", [0, 1])
        for index in range(K):
            global_t = _global_version(index)
            local = global_t[rank * ROWS_PER_RANK : (rank + 1) * ROWS_PER_RANK].clone()
            sharded = dtt.DTensor.from_local(
                local,
                mesh,
                [dtt.Shard(0)],
                shape=torch.Size((GLOBAL_ROWS, COLS)),
                stride=torch.Size((COLS, 1)),
            )
            dcp_save({"x": sharded}, checkpoint_id=str(Path(root) / f"in{index}"))

        # Caller-side LAWA: accumulate this rank's local shard across versions.
        # The load target must be a sharded DTensor on the same mesh so DCP
        # plans this rank's chunk; a plain 3x2 tensor raises a global/local
        # size mismatch.
        accumulator = torch.zeros(ROWS_PER_RANK, COLS, dtype=torch.float32)
        for index in range(K):
            empty = torch.empty(ROWS_PER_RANK, COLS, dtype=torch.float32)
            sharded_buf = dtt.DTensor.from_local(
                empty,
                mesh,
                [dtt.Shard(0)],
                shape=torch.Size((GLOBAL_ROWS, COLS)),
                stride=torch.Size((COLS, 1)),
            )
            dcp_load({"x": sharded_buf}, checkpoint_id=str(Path(root) / f"in{index}"))
            accumulator += sharded_buf.to_local()
        mean_local = accumulator / K

        mean_shard = dtt.DTensor.from_local(
            mean_local.contiguous(),
            mesh,
            [dtt.Shard(0)],
            shape=torch.Size((GLOBAL_ROWS, COLS)),
            stride=torch.Size((COLS, 1)),
        )
        dcp_save({"x": mean_shard}, checkpoint_id=str(Path(root) / "mean"))
    finally:
        _destroy()


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_lawa_over_real_dcp_collective(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    init_file = root / "init"
    init_file.touch()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(r, str(init_file), str(root)))
        for r in range(2)
    ]
    end = time.monotonic() + 240
    for p in procs:
        p.start()
    for p in procs:
        p.join(max(0, end - time.monotonic()))
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    # Independent dense oracle and a fresh no_dist read of the real DCP output.
    expected = torch.stack([_global_version(i) for i in range(K)]).mean(0)
    from torch.distributed.checkpoint import load as dcp_load

    merged = {"x": torch.empty(GLOBAL_ROWS, COLS, dtype=torch.float32)}
    dcp_load(merged, checkpoint_id=str(root / "mean"), no_dist=True)
    np.testing.assert_allclose(
        merged["x"].numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
    )

    # The inputs are real distinct checkpoints; their mean must differ from any
    # single version, proving the average actually consumed all k files.
    for index in range(K):
        assert not torch.allclose(merged["x"], _global_version(index))

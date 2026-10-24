"""Real gloo 2-rank sharded DCP via torch DTensor.from_local(Shard(0)).

Each rank holds the local row shard of a global 6x2 tensor and wraps it in a
DTensor on a 1D DeviceMesh over the two ranks. Saving that state dict through the
DCP collective must produce multiple storage chunks for the fqn (one per rank),
with global row offsets 0 and 3. Load reads back the real per-rank global slice.
"""

import multiprocessing as mp
import os
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

GLOBAL_ROWS = 6
COLS = 2
ROWS_PER_RANK = GLOBAL_ROWS // 2


def _worker(rank, init_file, root, op):
    import torch.distributed as dist
    import torch.distributed.tensor as dt
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
        global_t = torch.arange(GLOBAL_ROWS * COLS, dtype=torch.float32).reshape(
            GLOBAL_ROWS, COLS
        )
        local = global_t[rank * ROWS_PER_RANK : (rank + 1) * ROWS_PER_RANK].clone()
        mesh = DeviceMesh("cpu", [0, 1])
        sharded = dt.DTensor.from_local(
            local,
            mesh,
            [dt.Shard(0)],
            shape=torch.Size((GLOBAL_ROWS, COLS)),
            stride=torch.Size((COLS, 1)),
        )
        if op == "save":
            dcp_save({"x": sharded}, checkpoint_id=root)
        else:
            out = {"x": torch.empty(GLOBAL_ROWS, COLS, dtype=torch.float32)}
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
def test_real_dtensor_sharded_dcp_multi_chunk_metadata(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _spawn(root, "save")

    meta = pickle.loads((root / ".metadata").read_bytes())
    entries = sorted(
        (idx.fqn, sd.relative_path, sd.offset, sd.length)
        for idx, sd in meta.storage_data.items()
    )
    # Real sharded DTensor: more than one payload chunk for the fqn.
    assert len(entries) >= 2, entries
    offsets = [e[2] for e in entries]
    assert offsets[0] == 0, offsets
    # second chunk sits after the first payload blob (offset deltas are the chunk sizes).

    _spawn(root, "load")
    global_t = torch.arange(GLOBAL_ROWS * COLS, dtype=torch.float32).reshape(
        GLOBAL_ROWS, COLS
    )
    for rank in range(2):
        got = np.load(root / f"loaded{rank}.npy")
        want = global_t.numpy()
        np.testing.assert_array_equal(got, want)

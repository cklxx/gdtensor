"""Real gloo 2-rank col-sharded DCP via DTensor.from_local(Shard(1)).

Global matrix 2x4; rank 0 owns columns [0,2), rank 1 owns [2,4). Each local shard
is 2x2. The packed chunk stores each shard's columns contiguously, so the slot's
owned row intervals map to local byte offsets inside the chunk without a global-end
boundary Error. Public dcp_load reads each rank's shard bytes back correctly.

Rendezvous hygiene: every process-group lifetime uses a fresh FileStore path in a
temporary directory outside the checkpoint, and init is retried with backoff so a
spawned peer that is briefly not listening (high-load CI) cannot fail the test.
Workers are reaped on success, failure, or timeout.
"""

import multiprocessing as mp
import os
import pickle
import time
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

torch = pytest.importorskip("torch")

GLOBAL_ROWS = 2
GLOBAL_COLS = 4
COLS_PER_RANK = GLOBAL_COLS // 2
INIT_RETRIES = 5
INIT_BACKOFF_SECONDS = 2


def _init_process_group(dist, init_file, rank):
    _init = getattr(dist, "init_process_group")
    last = None
    for attempt in range(INIT_RETRIES):
        try:
            _init(
                "gloo",
                init_method="file://" + init_file,
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=120),
            )
            return
        except RuntimeError as exc:
            last = exc
            time.sleep(INIT_BACKOFF_SECONDS if attempt < INIT_RETRIES - 1 else 0)
    assert last is not None
    raise last


def _worker(rank, init_file, root, op):
    import torch.distributed as dist
    import torch.distributed.tensor as dt
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint import save as dcp_save
    from torch.distributed.device_mesh import DeviceMesh

    torch.set_num_threads(1)
    _destroy = getattr(dist, "destroy_process_group")
    _init_process_group(dist, init_file, rank)
    try:
        global_t = torch.arange(GLOBAL_ROWS * GLOBAL_COLS, dtype=torch.float32).reshape(
            GLOBAL_ROWS, GLOBAL_COLS
        )
        local = global_t[:, rank * COLS_PER_RANK : (rank + 1) * COLS_PER_RANK].clone()
        mesh = DeviceMesh("cpu", [0, 1])
        sharded = dt.DTensor.from_local(
            local,
            mesh,
            [dt.Shard(1)],
            shape=torch.Size((GLOBAL_ROWS, GLOBAL_COLS)),
            stride=torch.Size((GLOBAL_COLS, 1)),
        )
        if op == "save":
            dcp_save({"x": sharded}, checkpoint_id=root)
        else:
            out = {"x": torch.empty(GLOBAL_ROWS, GLOBAL_COLS, dtype=torch.float32)}
            dcp_load(out, checkpoint_id=root)
            np.save(Path(root) / f"loaded{rank}.npy", out["x"].numpy())
    finally:
        _destroy()


def _wait_workers(procs, op, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while any(p.is_alive() for p in procs):
        codes = [p.exitcode for p in procs]
        assert all(code in (None, 0) for code in codes), f"{op}: worker failed: {codes}"
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"{op}: worker timeout: {codes}"
        for p in procs:
            p.join(min(0.05, remaining))
    codes = [p.exitcode for p in procs]
    assert all(code == 0 for code in codes), f"{op}: worker failed: {codes}"


def _stop_workers(procs):
    started = [p for p in procs if p.pid is not None]
    for p in started:
        if p.is_alive():
            p.terminate()
    for p in started:
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(5)
        assert not p.is_alive(), f"worker {p.pid} survived cleanup"


def _run_workers(procs, op, timeout_seconds):
    try:
        for p in procs:
            p.start()
        _wait_workers(procs, op, timeout_seconds)
    finally:
        _stop_workers(procs)


def _spawn(root, op, timeout_seconds=180):
    # FileStore requires a fresh path for every process-group lifetime.
    # Keep rendezvous state outside checkpoint data; reap workers before removal.
    with TemporaryDirectory(prefix="dcp-rendezvous-", dir=Path(root).parent) as tmp:
        init_file = Path(tmp) / "init"
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_worker, args=(r, str(init_file), str(root), op))
            for r in range(2)
        ]
        _run_workers(procs, op, timeout_seconds)


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_real_dtensor_col_sharded_dcp_roundtrip(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _spawn(root, "save")

    meta = pickle.loads((root / ".metadata").read_bytes())
    entries = sorted(
        (idx.fqn, sd.relative_path, sd.offset, sd.length)
        for idx, sd in meta.storage_data.items()
    )
    assert len(entries) >= 2, entries

    _spawn(root, "load")
    global_t = torch.arange(GLOBAL_ROWS * GLOBAL_COLS, dtype=torch.float32).reshape(
        GLOBAL_ROWS, GLOBAL_COLS
    )
    for rank in range(2):
        got = np.load(root / f"loaded{rank}.npy")
        # public load returns the full global tensor; each rank's shard columns.
        want = global_t.numpy()
        np.testing.assert_array_equal(got, want)

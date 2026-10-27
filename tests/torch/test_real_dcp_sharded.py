"""Real gloo 2-rank sharded DCP via torch DTensor.from_local(Shard(0)).

Each rank holds the local row shard of a global 6x2 tensor and wraps it in a
DTensor on a 1D DeviceMesh over the two ranks. Saving that state dict through the
DCP collective must produce multiple storage chunks for the fqn (one per rank),
with global row offsets 0 and 3. Load reads back the real per-rank global slice.
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

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


def _record_rendezvous(rank, init_file, root, op):
    if rank == 0:
        path = Path(init_file)
        assert not path.exists(), "reused rendezvous state"
        path.write_bytes(b"stale FileStore state")
        (Path(root) / f"{op}.txt").write_text(init_file)


def _failing_worker(rank, init_file, root, op):
    if rank == 1:
        raise RuntimeError("injected worker failure")
    time.sleep(60)


def _hanging_worker(rank, init_file, root, op):
    time.sleep(60)


def test_spawn_uses_fresh_rendezvous_and_removes_stale_state(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_worker", _record_rendezvous)
    root = tmp_path / "ckpt"
    root.mkdir()
    for op in ("save", "load", "load_again"):
        _spawn(root, op)
    paths = [(root / f"{op}.txt").read_text() for op in ("save", "load", "load_again")]
    assert len(set(paths)) == 3
    assert all(not Path(path).parent.exists() for path in paths)


@pytest.mark.parametrize("fault", ["failure", "timeout"])
def test_spawn_reaps_workers_on_failure_or_timeout(tmp_path, monkeypatch, fault):
    target = _failing_worker if fault == "failure" else _hanging_worker
    monkeypatch.setattr(sys.modules[__name__], "_worker", target)
    before = {p.pid for p in mp.active_children()}
    message = "worker failed" if fault == "failure" else "worker timeout"
    with pytest.raises(AssertionError, match=message):
        _spawn(tmp_path, fault, timeout_seconds=30 if fault == "failure" else 3)
    assert {p.pid for p in mp.active_children()} <= before
    assert not list(tmp_path.parent.glob("dcp-rendezvous-*"))


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

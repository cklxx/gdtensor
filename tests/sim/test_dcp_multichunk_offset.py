"""Offset contract against a REAL gloo 2-rank sharded DTensor checkpoint.

The checkpoint is produced by ``dt.DTensor.from_local(local, mesh, [Shard(0)])``
saved through the DCP collective: each rank's rows live in a separate
``__<rank>_0.distcp`` file, and ``.metadata.storage_data`` has one chunk per rank
for the fqn (global row offsets 0 and 3, chunk size 3x2).

Two storage layouts are pinned through the public ``DcpLoadSession.get_file_offset``:

- a slot whose owned byte range sits inside ONE chunk resolves to that chunk's raw
  base and preads byte-exact oracle bytes from the correct file;
- a slot that would need a base from a DIFFERENT chunk file cannot be expressed as a
  single int offset — the actual ``(file, offset, length)`` metadata is recorded and
  the pread impossibility shown (a single base cannot name which file).
"""

import multiprocessing as mp
import os
import pickle
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

GLOBAL_ROWS = 6
COLS = 2
ROWS_PER_RANK = GLOBAL_ROWS // 2


def _save_worker(rank, init_file, root):
    import torch.distributed as dist
    import torch.distributed.tensor as dt
    from torch.distributed.checkpoint import save as dcp_save
    from torch.distributed.device_mesh import DeviceMesh

    torch.set_num_threads(1)
    # pyrefly-ignore-next-line possibly-missing-attribute
    getattr(dist, "init_process_group")(
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
        dcp_save({"x": sharded}, checkpoint_id=root)
    finally:
        # pyrefly-ignore-next-line possibly-missing-attribute
        getattr(dist, "destroy_process_group")()


def _spawn_save(root):
    init_file = Path(root) / "init"
    init_file.touch()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_save_worker, args=(r, str(init_file), str(root)))
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
def test_real_multichunk_offset_chunk_cover_and_crossfile_counterexample(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _spawn_save(root)

    meta = pickle.loads((root / ".metadata").read_bytes())
    chunks = [
        (idx.fqn, sd.relative_path, sd.offset, sd.length)
        for idx, sd in meta.storage_data.items()
    ]
    assert len(chunks) >= 2, chunks
    files = sorted(set(c[1] for c in chunks))
    assert len(files) >= 2, files  # chunks live in separate __<rank>_0.distcp files

    # Resolve each chunk's raw payload base by archive structure.
    from gdtensor_sim.dcp_backend import _raw_payload_bases

    bases = _raw_payload_bases(str(root))
    x_chunks = bases["x"]
    assert len(x_chunks) == 2, x_chunks

    global_t = torch.arange(GLOBAL_ROWS * COLS, dtype=torch.float32).reshape(
        GLOBAL_ROWS, COLS
    )
    oracle = [
        global_t[r * ROWS_PER_RANK : (r + 1) * ROWS_PER_RANK]
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
        for r in range(2)
    ]

    # Layout 1: a slot exactly covering one chunk preads byte-exact from ITS file.
    for r, (rel, base, nbytes) in enumerate(x_chunks):
        raw = open(root / rel, "rb").read()
        got = raw[base : base + len(oracle[r])]
        assert got == oracle[r], (r, rel, base)
        assert nbytes == len(oracle[r]), (r, nbytes, len(oracle[r]))

    # Cross-file counterexample: rank1's chunk lives in a DIFFERENT file. A single
    # int offset returned by get_file_offset cannot name which file, so the caller
    # cannot pread rank1's bytes using rank0's file. Record the actual metadata.
    r0_rel, r0_base, _ = x_chunks[0]
    r1_rel, r1_base, _ = x_chunks[1]
    assert r0_rel != r1_rel, (r0_rel, r1_rel)
    raw0 = open(root / r0_rel, "rb").read()
    # If a caller used the rank0 file + rank1-style offset, it does NOT yield rank1
    # oracle bytes (the single-base contract is file-agnostic and thus wrong across files).
    wrong = raw0[r1_base : r1_base + len(oracle[1])]
    assert wrong != oracle[1], "cross-file pread must not match rank1 oracle"

"""Real gloo 2-rank sharded DCP: public DcpLoadSession.get_file_offset + pread.

Saves a real DTensor Shard(0) global (6,2) f32 via two gloo ranks, then in the
main process opens a public DcpLoadSession for each rank and verifies that
os.pread at the returned offset yields exactly that rank's global row slice.

Also verifies world=1 single-rank checkpoint that writes the tensor across
two storage chunks returns an Err (single base offset cannot span files),
not a NotFound / silent success.
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
ITEMSIZE = 4  # float32

# Col-shard fixture: global (2,4), world=2, Shard(1).
C_ROWS, C_COLS = 2, 4
C_COLS_PER_RANK = C_COLS // 2


def _save_col_worker(rank, init_file, root):
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
        global_t = torch.arange(C_ROWS * C_COLS, dtype=torch.float32).reshape(
            C_ROWS, C_COLS
        )
        local = global_t[
            :, rank * C_COLS_PER_RANK : (rank + 1) * C_COLS_PER_RANK
        ].clone()
        mesh = DeviceMesh("cpu", [0, 1])
        sharded = dt.DTensor.from_local(
            local,
            mesh,
            [dt.Shard(1)],
            shape=torch.Size((C_ROWS, C_COLS)),
            stride=torch.Size((C_COLS, 1)),
        )
        dcp_save({"c": sharded}, checkpoint_id=root)
    finally:
        # pyrefly-ignore-next-line possibly-missing-attribute
        getattr(dist, "destroy_process_group")()


def _spawn_col(root):
    init_file = Path(root) / "init_col"
    init_file.touch()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_save_col_worker, args=(r, str(init_file), str(root)))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(180)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]


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


def _build_gdtensor_table_world2():
    import gdtensor.builder as gb
    import gdtensor.tag as gt

    world = 2
    placements = [gb.Replicate(), gb.Shard(0)]
    mesh = gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(world)), [1, 2])
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([GLOBAL_ROWS, COLS], "F32", ITEMSIZE),
        placements,
        mesh,
    )
    nd = gb.NamedDTensor("x", dt, ["x.alias"])
    # DisjointStorageMeshList needs tensor_ids labels ordered like the table.
    return gb.DTensorTable([nd])


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_real_sharded_public_session_pread_bytes_match(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    _spawn_save(root)

    import gdtensor.builder as gb
    from gdtensor_sim._layout import build_payload_index
    from gdtensor_sim.models import build_layout
    from gdtensor_sim.dcp_backend import DcpLoadSession

    table = _build_gdtensor_table_world2()
    layout = build_layout(table, 2, "safetensors", lambda t, w, nd: f"__{w}_0.distcp")
    index = build_payload_index(table, layout)

    want = torch.arange(GLOBAL_ROWS * COLS, dtype=torch.float32).reshape(
        GLOBAL_ROWS, COLS
    )
    want_bytes = want.contiguous().view(torch.uint8).numpy().tobytes()

    # Distcp files produced by the 2-rank gloo save.
    distcp_files = sorted(root.glob("__*.distcp"))
    assert distcp_files, list(root.iterdir())

    for rank in range(2):
        session = DcpLoadSession(rank, str(root), table, layout, index)
        pos = gb.TensorComponentPos("x", rank)
        got = session.get_file_offset(pos)
        assert type(got).__name__ == "Ok", got
        _, slot = index.slots[pos]
        path = gb.file_path_of(layout, pos)
        assert type(path).__name__ == "Ok", path
        distcp_path = root / path.value
        lo = rank * ROWS_PER_RANK
        hi = (rank + 1) * ROWS_PER_RANK
        want_chunk = want_bytes[lo * COLS * ITEMSIZE : hi * COLS * ITEMSIZE]
        fd = os.open(distcp_path, os.O_RDONLY)
        try:
            assert got.value is not None
            chunk = os.pread(fd, slot.nbytes, int(got.value))
        finally:
            os.close(fd)
        assert chunk == want_chunk, (
            f"rank{rank}: pread {len(chunk)}B at offset {got.value} "
            f"want {len(want_chunk)}B"
        )
        session.finish()


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_real_col_sharded_public_session_pread_bytes_match(tmp_path):
    """Shard(1) global (2,4): rank1 owns columns [2,4) of both rows.

    The rank1 chunk is a packed 2D local region [row0 cols2,3][row1 cols2,3] =
    16 contiguous bytes. get_file_offset must point at chunk start and pread
    16B must equal dense[:, rank*2:rank*2+2].tobytes().
    """
    root = tmp_path / "ckpt_col"
    root.mkdir()
    _spawn_col(root)

    import gdtensor.builder as gb
    import gdtensor.tag as gt
    from gdtensor_sim._layout import build_payload_index
    from gdtensor_sim.models import build_layout
    from gdtensor_sim.dcp_backend import DcpLoadSession

    placements = [gb.Replicate(), gb.Shard(1)]
    mesh = gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(2)), [1, 2])
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([C_ROWS, C_COLS], "F32", ITEMSIZE),
        placements,
        mesh,
    )
    table = gb.DTensorTable([gb.NamedDTensor("c", dt, ["c.alias"])])
    layout = build_layout(table, 2, "safetensors", lambda t, w, nd: f"__{w}_0.distcp")
    index = build_payload_index(table, layout)

    want = torch.arange(C_ROWS * C_COLS, dtype=torch.float32).reshape(C_ROWS, C_COLS)
    _want_bytes = want.contiguous().view(torch.uint8).numpy().tobytes()

    for rank in range(2):
        session = DcpLoadSession(rank, str(root), table, layout, index)
        pos = gb.TensorComponentPos("c", rank)
        got = session.get_file_offset(pos)
        assert type(got).__name__ == "Ok", got
        _, slot = index.slots[pos]
        path = gb.file_path_of(layout, pos)
        assert type(path).__name__ == "Ok", path
        fd = os.open(root / path.value, os.O_RDONLY)
        try:
            assert got.value is not None
            chunk = os.pread(fd, slot.nbytes, int(got.value))
        finally:
            os.close(fd)
        want_chunk = want[
            :, rank * C_COLS_PER_RANK : (rank + 1) * C_COLS_PER_RANK
        ].contiguous()
        want_bytes_chunk = want_chunk.view(torch.uint8).numpy().tobytes()
        assert chunk == want_bytes_chunk, (
            f"rank{rank}: pread {len(chunk)}B at offset {got.value} "
            f"want {len(want_bytes_chunk)}B; intervals={slot.intervals.tolist()}"
        )
        session.finish()


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_world1_slot_spanning_two_chunks_returns_err(tmp_path):
    """Real 2-rank gloo save produces two chunks for fqn 'x'.

    gdtensor layout is built for world=1 so the single slot owns the FULL
    tensor and points at one distcp path. The actual archive has two chunks;
    the matched chunk's nbytes is half, so slot.nbytes > nbytes and
    get_file_offset must return an Err (not NotFound, not a wrong offset).
    public rank_load via torch reader still returns the full dense tensor.
    """
    root = tmp_path / "ckpt_multi"
    root.mkdir()
    _spawn_save(root)

    import gdtensor.builder as gb
    import gdtensor.tag as gt
    from gdtensor_sim._layout import build_payload_index
    from gdtensor_sim.models import build_layout
    from gdtensor_sim.dcp_backend import DcpLoadSession

    # world=1: single slot owns the whole tensor.
    placements = [gb.Replicate(), gb.Replicate()]
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([GLOBAL_ROWS, COLS], "F32", ITEMSIZE),
        placements,
        mesh,
    )
    table = gb.DTensorTable([gb.NamedDTensor("x", dt, ["x.alias"])])
    layout = build_layout(table, 1, "safetensors", lambda t, w, nd: "__0_0.distcp")
    index = build_payload_index(table, layout)

    # Inspect actual metadata chunks.

    meta = pickle.loads((root / ".metadata").read_bytes())
    chunk_entries = [
        (idx.fqn, sd.relative_path, sd.offset, sd.length)
        for idx, sd in meta.storage_data.items()
        if idx.fqn == "x"
    ]
    assert len(chunk_entries) == 2, chunk_entries

    session = DcpLoadSession(0, str(root), table, layout, index)
    pos = gb.TensorComponentPos("x", 0)
    got = session.get_file_offset(pos)
    assert type(got).__name__ == "Err", got
    msg = str(got.value)
    assert "single base offset" in msg or "not contained" in msg, msg
    # Must NOT be NotFound.
    assert "not_found" not in type(got.value).__name__.lower(), got
    session.finish()

    # Public gdtensor gio.rank_load (not torch dcp_load) reads full dense tensor.
    import gdtensor.io as gio
    from tests.support.memory import Memory

    want_full = torch.arange(GLOBAL_ROWS * COLS, dtype=torch.float32).reshape(
        GLOBAL_ROWS, COLS
    )
    dst_np = np.zeros((GLOBAL_ROWS, COLS), dtype=np.float32)
    dst_mem = Memory(list(dst_np.tobytes()))
    storage_view = gb.StorageDTensorsView(
        ["x"],
        table,
        None,
        layout,
        lambda view, rank: gb.Ok(None),
        lambda view, rank: gb.Ok(DcpLoadSession(rank, str(root), table, layout, index)),
    )
    compute_dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([GLOBAL_ROWS, COLS], "F32", ITEMSIZE),
        [gb.Replicate()],
        gb.ArrangedMesh(gt.WorldSize(1)),
    )
    compute_table = gb.DTensorTable([gb.NamedDTensor("x", compute_dt, ["x.alias"])])
    compute_view = gb.ComputeDTensorsView(["x"], compute_table, dst_mem)
    res = gio.rank_load(0, compute_view, storage_view)
    assert type(res).__name__ == "Ok", res
    np.testing.assert_array_equal(
        dst_mem.data.view(np.float32).reshape(GLOBAL_ROWS, COLS), want_full.numpy()
    )

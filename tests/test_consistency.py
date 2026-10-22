"""Multi-machine consistency.

The implementation must behave identically across machines and runtimes:

1. plans are a pure function of (rank, tables, layout): no wall time, pid,
   thread id, or set/hash iteration;
2. published bytes are identical across:
   * repeated saves into different directories;
   * threads and forked processes as the rank runtime;
   * reversed rank start order;
3. plans and published bytes do not depend on PYTHONHASHSEED (each
   process on each machine has its own hash salt).
"""
from __future__ import annotations

import os


from conftest import assert_all_ok
from gdtensor.internal.dtensor import tensor_ids_of as _ids  # noqa: E402
from gdtensor.internal.coord import ThreadCoordinator
from gdtensor.internal.formats import make_storage_view
from gdtensor.internal.models import (
    ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table,
)
from gdtensor.internal.plan import d_tensors_view_rank_plan_load, d_tensors_view_rank_plan_save
from gdtensor.internal.runner import _run_threads, compute_view, run_processes, save_threads
from gdtensor.internal.views import ComputeDataAccessor, ComputeFabric

M = ModelConfig(layers=2, hidden=16, ffn=32, vocab=64)
P = ParallelConfig(tp=2, dp=2, pp=1)


def _snapshot_bytes(root):
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            out[name] = open(p, "rb").read()
    return out


def _do_save(root, p=P, m=M, mode="hf", kind="hf"):
    os.makedirs(root, exist_ok=True)
    ct = compute_table(m, p)
    st = storage_table(ct, p, mode)
    lay = layout_builder(kind, st, p, "safetensors")
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, root, fab, p.world)[0])
    return _snapshot_bytes(root)


def test_repeated_saves_are_byte_identical(tmp_path):
    a = _do_save(str(tmp_path / "a"))
    b = _do_save(str(tmp_path / "b"))
    assert a == b


def test_thread_and_process_outputs_are_identical(tmp_path):
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, "safetensors")
    rt = str(tmp_path / "threads"); os.makedirs(rt)
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)
    assert_all_ok(save_threads(ct, st, lay, rt, fab, P.world)[0])
    rp = str(tmp_path / "procs"); os.makedirs(rp)
    res, _, _ = run_processes("save", ct, st, lay, rp, P.world, fill=fill_pattern)
    assert_all_ok(res)
    # process runners receive fill_pattern across fork: compare every file
    a, b = _snapshot_bytes(rt), _snapshot_bytes(rp)
    assert set(a) == set(b)
    for name in a:
        assert a[name] == b[name], name


def test_reversed_rank_start_order_same_bytes(tmp_path):
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, "safetensors")
    r1 = str(tmp_path / "asc"); os.makedirs(r1)
    r2 = str(tmp_path / "desc"); os.makedirs(r2)
    for root, order in [(r1, range(P.world)), (r2, range(P.world - 1, -1, -1))]:
        io = __import__("gdtensor.internal.iobackend", fromlist=["FileIO"]).FileIO()
        sv = make_storage_view(_ids(st), st, lay, root, io, ThreadCoordinator(P.world))
        fab = ComputeFabric(ct).allocate(fill=fill_pattern)
        assert_all_ok(_run_threads(P.world, lambda r: __import__(
            "gdtensor.internal.api", fromlist=["d_tensors_view_rank_save"]).d_tensors_view_rank_save(
            r, compute_view(ct, fab, list(order)[r]), sv)))
    assert _snapshot_bytes(r1) == _snapshot_bytes(r2)


def test_plan_is_pure_and_order_independent():
    ct = compute_table(M, P)
    st = storage_table(ct, P, "hf")
    lay = layout_builder("hf", st, P, "safetensors")

    def plans():
        sv = make_storage_view(_ids(st), st, lay, "/nonexistent")
        ps, pl = {}, {}
        for r in range(P.world):
            cv = compute_view(ct, None, r)
            cv.data_accessor = ComputeDataAccessor(r, None)
            ps[r] = d_tensors_view_rank_plan_save(r, cv, sv).unwrap()
            pl[r] = d_tensors_view_rank_plan_load(r, cv, sv).unwrap()
        return ps, pl

    a = plans()
    b = plans()
    assert a == b


def test_plan_and_bytes_are_hash_seed_independent(tmp_path):
    """Save plans and published bytes must match across PYTHONHASHSEED values."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import hashlib, os, sys
        sys.path.insert(0, "tests"); sys.path.insert(0, ".")
        from gdtensor.internal.formats import make_storage_view
        from gdtensor.internal.models import ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table
        from gdtensor.internal.plan import d_tensors_view_rank_plan_save
        from gdtensor.internal.runner import compute_view, save_threads
        from gdtensor.internal.views import ComputeFabric
        m = ModelConfig(layers=2, hidden=16, ffn=32, vocab=64)
        p = ParallelConfig(tp=2, dp=2, pp=1)
        ct = compute_table(m, p); st = storage_table(ct, p, "hf")
        from gdtensor.internal.dtensor import tensor_ids_of as st_named_ids
        st_named_ids = st_named_ids(st)
        lay = layout_builder("hf", st, p, "safetensors")
        root = sys.argv[1]
        fab = ComputeFabric(ct).allocate(fill=fill_pattern)
        res, _ = save_threads(ct, st, lay, root, fab, p.world)
        assert all(type(r).__name__ == 'Ok' for r in res.values())
        sv = make_storage_view(st_named_ids, st, lay, root)
        h = hashlib.sha256()
        for r in range(p.world):
            h.update(repr(d_tensors_view_rank_plan_save(r, compute_view(ct, fab, r), sv).value).encode())
        for name in sorted(os.listdir(root)):
            h.update(name.encode()); h.update(open(os.path.join(root, name), "rb").read())
        print(h.hexdigest())
        """)
    outs = set()
    for seed in (1, 42, 999):
        root = str(tmp_path / f"s{seed}")
        out = subprocess.run([sys.executable, "-c", code, root], cwd=os.path.dirname(os.path.dirname(__file__)),
                             env={**os.environ, "PYTHONHASHSEED": str(seed)}, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        outs.add(out.stdout.strip())
    assert len(outs) == 1, outs

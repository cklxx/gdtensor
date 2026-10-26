"""Basic IO isolation: public rank_save / rank_load / Direct chain_load must keep
working even when the cache global state is fault-injected to fail.

Fault injection (external boundary): we obtain the already-loaded runtime executor
module by reflecting the public entry point's ``__module__`` (``sys.modules``), then
replace its cache globals (`_slots`, `_view_slots`, `_table_lock`, `_NodeState`,
`_SlotTable`) so ANY use raises. We never mutate production private data / counters;
a Direct read chain must bypass every cache global. No static internal import.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt

G = 8
WORLD = 2
TID = "w"


def _executor_module():
    """Reflect the runtime module that implements the public chain entry.

    This is the fault-injection boundary: we do not statically import the internal
    package; we look up the module object the public entry point already lives in.
    """
    return sys.modules[gio.chain_load.__module__]


# ------------------------------------------------------------------ accessor/session


class Buf:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.uint8).copy()
        self.lock = threading.Lock()

    @staticmethod
    def _off(rng):
        if hasattr(rng, "global_tensor_offset"):
            return rng.global_tensor_offset
        return rng.contiguous_payload_offset

    def read_to_buffer(self, rng, cb):
        with self.lock:
            seg = self.data[self._off(rng) : self._off(rng) + rng.size]
            return cb(seg.ctypes.data, seg.nbytes)

    def write_from_buffer(self, rng, cb):
        with self.lock:
            return self._write(rng, cb)

    def _write(self, rng, cb):
        off = self._off(rng)
        data = np.empty(rng.size, dtype=np.uint8)
        res = cb(data.ctypes.data, data.nbytes)
        if type(res) is gb.Ok:
            self.data[off : off + rng.size] = data
        return res


class Session:
    def __init__(self, accessor):
        self.accessor = accessor

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        return gb.Ok(self.accessor)

    def finish(self):
        return gb.Ok(None)

    def commit(self):
        return gb.Ok(None)

    def abort(self):
        return gb.Ok(None)


# ------------------------------------------------------------------ view builders


def _storage_view(buf):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(WORLD))
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([G], "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh
    )
    grp = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)),
                gb.ArrangedMesh(gt.WorldSize(WORLD)),
            )
        ],
        [TID],
    )
    leader = gb.TensorComponentPos(TID, 0)
    entries = {gb.TensorComponentPos(TID, r): grp for r in range(WORLD)}
    layout = gb.PhysicalLayoutTable("safetensors", entries, {leader: "memory"})
    table = gb.DTensorTable([gb.NamedDTensor(TID, dt)])

    def session_factory(v, r):
        return gb.Ok(Session(buf))

    return gb.StorageDTensorsView(
        [TID], table, buf, layout, session_factory, session_factory
    )


def _compute_view(buf):
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([G], "U8", 1),
        [gb.Shard(0)],
        gb.ArrangedMesh(gt.WorldSize(WORLD)),
    )
    return gb.ComputeDTensorsView(
        [TID], gb.DTensorTable([gb.NamedDTensor(TID, dt)]), buf
    )


def _owned(rank):
    """Independent oracle: Shard(0) over WORLD ranks."""
    per = G // WORLD
    return rank * per, (rank + 1) * per


# ----------------------------------------------------- fault injection fixtures


class _Boom:
    def __getattr__(self, name):
        raise RuntimeError(f"cache global accessed under fault: {name}")

    def __getitem__(self, k):
        raise RuntimeError("cache _view_slots accessed under fault")

    def __setitem__(self, k, v):
        raise RuntimeError("cache _view_slots written under fault")

    def __len__(self):
        raise RuntimeError("cache _view_slots len under fault")


@pytest.fixture
def cache_fault(monkeypatch):
    api = _executor_module()

    def _raise(*a, **k):
        raise RuntimeError("cache _slots() used under fault")

    monkeypatch.setattr(api, "_slots", _raise)
    monkeypatch.setattr(api, "_view_slots", _Boom())
    monkeypatch.setattr(api, "_table_lock", _Boom())
    boom_node = type(
        "_BoomNode",
        (),
        {
            "__init__": lambda self, *a, **k: (_ for _ in ()).throw(
                RuntimeError("cache state constructed under fault")
            )
        },
    )
    monkeypatch.setattr(api, "_NodeState", boom_node)
    monkeypatch.setattr(api, "_SlotTable", boom_node)
    yield


# ------------------------------------------------------------------- tests


def test_direct_chain_load_ok_under_cache_fault(cache_fault):
    src = Buf(np.arange(G, dtype=np.uint8))
    dsts = [Buf(np.zeros(G, np.uint8)) for _ in range(WORLD)]
    for rank in range(WORLD):
        chain = gb.DirectDTensorsReadChain(
            src=_storage_view(src), dst=_compute_view(dsts[rank])
        )
        res = gio.chain_load(rank, chain)
        assert type(res) is gb.Ok, res
        lo, hi = _owned(rank)
        want = np.arange(lo, hi, dtype=np.uint8)
        assert bytes(dsts[rank].data[lo:hi]) == bytes(want)


def test_direct_chain_never_touches_cache_globals():
    api = _executor_module()
    before = len(api._view_slots)
    src = Buf(np.arange(G, dtype=np.uint8))
    for rank in range(WORLD):
        dst = Buf(np.zeros(G, np.uint8))
        chain = gb.DirectDTensorsReadChain(
            src=_storage_view(src), dst=_compute_view(dst)
        )
        res = gio.chain_load(rank, chain)
        assert type(res) is gb.Ok, res
    assert len(api._view_slots) == before, "direct chain created a cache table"


def test_rank_save_load_ok_under_cache_fault(cache_fault):
    stored = Buf(np.zeros(G, np.uint8))
    for rank in range(WORLD):
        src_buf = Buf(np.arange(G, dtype=np.uint8))
        res = gio.rank_save(rank, _compute_view(src_buf), _storage_view(stored))
        assert type(res) is gb.Ok, res
    for rank in range(WORLD):
        dst = Buf(np.zeros(G, np.uint8))
        res = gio.rank_load(rank, _compute_view(dst), _storage_view(stored))
        assert type(res) is gb.Ok, res
        lo, hi = _owned(rank)
        want = np.arange(lo, hi, dtype=np.uint8)
        assert bytes(dst.data[lo:hi]) == bytes(want)


def test_concurrent_ranks_independent_under_fault(cache_fault):
    src = Buf(np.arange(G, dtype=np.uint8))
    dsts = [Buf(np.zeros(G, np.uint8)) for _ in range(WORLD)]
    errs = []

    def worker(rank):
        try:
            chain = gb.DirectDTensorsReadChain(
                src=_storage_view(src), dst=_compute_view(dsts[rank])
            )
            res = gio.chain_load(rank, chain)
            assert type(res) is gb.Ok, res
        except BaseException as e:  # noqa: BLE001
            errs.append((rank, e))

    ts = [threading.Thread(target=worker, args=(r,), daemon=True) for r in range(WORLD)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10.0)
    assert not any(t.is_alive() for t in ts), "deadlock"
    assert not errs, errs
    for rank in range(WORLD):
        lo, hi = _owned(rank)
        want = np.arange(lo, hi, dtype=np.uint8)
        assert bytes(dsts[rank].data[lo:hi]) == bytes(want)


def test_direct_path_does_not_import_torch():
    # Physical check in a FRESH interpreter: cannot rely on sys.modules order when
    # torch is already imported by other suites.
    code = (
        "import numpy as np, gdtensor.builder as gb, gdtensor.io as gio,"
        " gdtensor.tag as gt, sys\n"
        "G,W,TID=8,2,'w'\n"
        "class Buf:\n"
        "  def __init__(s,d): s.data=np.array(d,dtype=np.uint8).copy()\n"
        "  def _off(s,r): return getattr(r,'global_tensor_offset',"
        "getattr(r,'contiguous_payload_offset',0))\n"
        "  def read_to_buffer(s,r,cb):\n"
        "    o=s._off(r); return cb(s.data[o:o+r.size].ctypes.data,r.size)\n"
        "  def write_from_buffer(s,r,cb):\n"
        "    d=np.empty(r.size,dtype=np.uint8); res=cb(d.ctypes.data,r.size)\n"
        "    if type(res) is gb.Ok:\n"
        "      o=s._off(r); s.data[o:o+r.size]=d\n"
        "    return res\n"
        "class S:\n"
        "  def get_file_offset(s,p): return gb.Ok(0)\n"
        "  def init_data_accessor(s): return gb.Ok(s.b)\n"
        "  def finish(s): return gb.Ok(None)\n"
        "  def commit(s): return gb.Ok(None)\n"
        "  def abort(s): return gb.Ok(None)\n"
        "mesh=gb.SlotOuterProductMesh(gb.ArrangedMesh(gt.ConstantOne(1)),"
        "gb.ArrangedMesh(gt.WorldSize(W)))\n"
        "dt=gb.NaiveDTensor(gb.GlobalTensorMeta([G],'U8',1),[gb.Replicate(),gb.Shard(0)],mesh)\n"
        "g=gb.DisjointStorageMeshList([gb.SlotOuterProductMesh("
        "gb.ArrangedMesh(gt.TensorCount(1)),gb.ArrangedMesh(gt.WorldSize(W)))],[TID])\n"
        "ent={gb.TensorComponentPos(TID,r):g for r in range(W)}\n"
        "lay=gb.PhysicalLayoutTable('safetensors',ent,{gb.TensorComponentPos(TID,0):'m'})\n"
        "tbl=gb.DTensorTable([gb.NamedDTensor(TID,dt)])\n"
        "src=Buf(np.arange(G,dtype=np.uint8)); dst=Buf(np.zeros(G,np.uint8))\n"
        "def sf(v,r):\n"
        "  s=S(); s.b=src; return gb.Ok(s)\n"
        "sv=gb.StorageDTensorsView([TID],tbl,src,lay,sf,sf)\n"
        "cdt=gb.NaiveDTensor(gb.GlobalTensorMeta([G],'U8',1),[gb.Shard(0)],"
        "gb.ArrangedMesh(gt.WorldSize(W)))\n"
        "cv=gb.ComputeDTensorsView([TID],gb.DTensorTable([gb.NamedDTensor(TID,cdt)]),dst)\n"
        "ch=gb.DirectDTensorsReadChain(src=sv,dst=cv)\n"
        "assert type(gio.chain_load(0,ch)) is gb.Ok\n"
        "assert bytes(dst.data[0:4])==bytes(np.arange(4,dtype=np.uint8))\n"
        "assert 'torch' not in sys.modules\n"
    )
    # Deterministic clean env: only what a fresh interpreter needs. Do not inherit
    # pytest's process env (which may already have torch imported in the parent);
    # the point is a physical, order-independent check.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

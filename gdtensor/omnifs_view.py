"""StorageDTensorsView on the OmniFS backend (snapshot protocol).

Same core types and executors as the file backend; only the session differs:

begin_save(rank)
    Every rank derives the same manifest (pure function of table + layout;
    this stands for "rank0 syncs the write plan"). rank0 checks the snapshot
    is not committed, drops leftovers of a failed attempt (rewritable), and
    writes the manifest in state "prepared". Barrier. File leaders write the
    safetensors header as KVs (the header is known from the plan; no backfill).
commit()
    Each rank fsyncs its KVs and layout entries. Vote 1 = "all writers done".
    rank0 checks full coverage, writes JSON blobs (index.json), and moves the
    manifest to "committed" (the only publish point). Vote 2 = "rank0 commit
    ok". On any failure every rank sees the snapshot as not visible.
abort()
    Vote "fail"; the snapshot stays invisible; rank0 drops its KVs.
begin_load(rank)
    Requires a committed manifest. Headers are read through the host read
    path (L1/L2/L3), so shared headers are P2P-amortized.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .dtensor import DTensorTable
from .formats import HF_INDEX_NAME, SafeTensorsLoadSession, _whole_tensor_files, safetensors_header
from .dtensor import global_tensor_meta_of
from .layout import PayloadIndex, PhysicalLayoutTable, TensorComponentPos, build_payload_index
from .omnifs import Manifest, OmniFsConfig, OmniFsIO, _atomic_write, build_manifest
from .result import ResultError, catch


def snapshot_files(table: DTensorTable, index: PayloadIndex) -> Dict[str, bytes]:
    return {path: safetensors_header(table, index, path) for path in index.files}


def hf_index_json(table: DTensorTable, index: PayloadIndex) -> Optional[bytes]:
    if len(index.files) < 2:
        return None
    whole = _whole_tensor_files(table, index)
    if not all(whole.values()):
        return None
    wm = {table.named_d_tensors[t].global_tensor_id: path for (path, t) in sorted(whole, key=lambda k: k[1])}
    total = sum(global_tensor_meta_of(nd).nbytes for nd in table.named_d_tensors)
    return json.dumps({"metadata": {"total_size": total}, "weight_map": wm}, indent=2).encode()


def plan_manifest(cfg: OmniFsConfig, table: DTensorTable, index: PayloadIndex, extra_json: Dict[str, bytes]):
    headers = snapshot_files(table, index)
    sizes = {p: len(h) + index.files[p].payload_size for p, h in headers.items()}
    js = dict(extra_json)
    idx = hf_index_json(table, index)
    if idx is not None:
        js[HF_INDEX_NAME] = idx
    return build_manifest(cfg, sizes, js), headers


class OmniFsSaveSession:
    def __init__(self, rank: int, cfg: OmniFsConfig, table: DTensorTable, layout: PhysicalLayoutTable,
                 index: PayloadIndex, io: OmniFsIO, coord=None, extra_json: Optional[Dict[str, bytes]] = None) -> None:
        if layout.container_format != "safetensors":
            raise ResultError("OmniFS stores HF safetensors only")
        self.rank, self.cfg, self.table, self.layout, self.index, self.io, self.coord = rank, cfg, table, layout, index, io, coord
        self.state = "open"
        self.manifest, self.headers = plan_manifest(cfg, table, index, extra_json or {})
        self.coordinator = min(fp.leader.rank for fp in index.files.values())
        err = None
        if rank == self.coordinator:
            try:
                if os.path.exists(cfg.manifest_path):
                    old = Manifest.load(cfg.manifest_path)
                    if old.state == "committed":
                        raise ResultError(f"snapshot {cfg.snap} is already committed")
                io.kv.drop()  # leftovers of a failed attempt: rewritable
                _atomic_write(cfg.manifest_path, self.manifest.to_bytes())
            except Exception as exc:  # noqa: BLE001
                err = exc
        ok = self._vote(err is None)  # barrier: manifest prepared before any write
        if not ok:
            self.state = "aborted"
            raise ResultError(f"create snapshot failed: {err}" if err else "create snapshot failed on rank0")
        io.bind(self.manifest)
        for path, hdr in self.headers.items():
            if index.files[path].leader.rank == rank:
                io.pwrite(rank, path, 0, hdr)

    def _vote(self, ok: bool) -> bool:
        return ok if self.coord is None else self.coord.vote(ok)

    def abspath(self, rel: str) -> str:
        return rel

    @catch
    def get_file_offset(self, pos: TensorComponentPos) -> int:
        if self.state != "open":
            raise ResultError(f"get_file_offset on {self.state} session")
        hit = self.index.slots.get(TensorComponentPos(*pos))
        if hit is None:
            raise ResultError(f"{tuple(pos)} has no physical file")
        path, slot = hit
        return len(self.headers[path]) + slot.contiguous_payload_offset

    @catch
    def commit(self) -> None:
        if self.state != "open":
            raise ResultError(f"commit on {self.state} session")
        self.io.kv.flush(self.rank)
        if not self._vote(True):
            self.state = "aborted"
            raise ResultError("another rank aborted; snapshot not visible")
        err = None
        if self.rank == self.coordinator:
            try:
                for rel, content in self.manifest.json_blobs.items():
                    self.io.pwrite(self.rank, rel, 0, content.encode())  # JSON blob: whole overwrite
                self.io.kv.flush(self.rank)
                self.io.kv.check_full_coverage(self.manifest)
                self.manifest.state = "committed"
                _atomic_write(self.cfg.manifest_path, self.manifest.to_bytes())  # publish point
            except Exception as exc:  # noqa: BLE001
                err = exc
        ok = self._vote(err is None)
        if not ok:
            self.state = "aborted"
            raise ResultError(f"commit failed on rank0: {err}" if err else "commit failed on rank0")
        self.state = "committed"

    @catch
    def abort(self) -> None:
        if self.state in ("committed", "aborted"):
            raise ResultError(f"abort on {self.state} session")
        self.io.kv.flush(self.rank)
        self._vote(False)
        self.state = "aborted"
        if self.rank == self.coordinator:
            self.io.kv.drop()  # manifest stays "prepared": invisible, rewritable


class OmniFsLoadSession(SafeTensorsLoadSession):
    def __init__(self, rank: int, cfg: OmniFsConfig, table, layout, index, io: OmniFsIO,
                 needed_files: Optional[List[str]] = None) -> None:
        if not os.path.exists(cfg.manifest_path):
            raise ResultError(f"snapshot {cfg.snap} does not exist")
        m = Manifest.load(cfg.manifest_path)
        if m.state != "committed":
            raise ResultError(f"snapshot {cfg.snap} is not committed (state {m.state})")
        io.bind(m)
        super().__init__(rank, "", table, layout, index, io, needed_files=needed_files)


def make_omnifs_storage_view(tensor_ids, table: DTensorTable, layout: PhysicalLayoutTable,
                             cfg: OmniFsConfig, io: Optional[OmniFsIO] = None, coord=None,
                             extra_json: Optional[Dict[str, bytes]] = None):
    from .views import StorageDataAccessor, StorageDTensorsView

    io = io or OmniFsIO(cfg)
    index = build_payload_index(table, layout)

    @catch
    def begin_save(view, rank: int):
        return OmniFsSaveSession(rank, cfg, view.d_tensor_table, view.physical_layout, index, io, coord, extra_json)

    @catch
    def begin_load(view, rank: int, needed_files: Optional[List[str]] = None):
        return OmniFsLoadSession(rank, cfg, view.d_tensor_table, view.physical_layout, index, io, needed_files)

    view = StorageDTensorsView(list(tensor_ids), table, StorageDataAccessor(index=index, io=io), layout,
                               begin_save, begin_load)
    view.root = cfg.root
    return view


def export_hf_dir(cfg: OmniFsConfig, out_dir: str, io: Optional[OmniFsIO] = None) -> List[str]:
    """Read every file of a committed snapshot through the store (verification
    and FUSE-style access; no P2P)."""
    io = io or OmniFsIO(cfg)
    m = Manifest.load(cfg.manifest_path)
    if m.state != "committed":
        raise ResultError("snapshot is not committed")
    io.bind(m)
    os.makedirs(out_dir, exist_ok=True)
    for rel, f in m.files.items():
        with open(os.path.join(out_dir, rel), "wb") as fh:
            pos = 0
            while pos < f["size"]:
                n = min(m.chunk_bytes, f["size"] - pos)
                fh.write(io.pread(-1, rel, pos, n))
                pos += n
    return sorted(m.files)

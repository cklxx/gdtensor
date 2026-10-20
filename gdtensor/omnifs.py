"""OmniFS store backend (OmniFS-Store design, scheme 2.1 "EIC-KV ChunkStore").

Mapping chain (all deterministic, nothing per-tensor is stored):
    HF file --manifest--> blob --rule--> chunk (offset // chunk_bytes)
    chunk == KeySet --one pwrite piece--> one variable-length KV
    KV --crc32(key) % n_dn--> DN   (non-affinity placement, read scheme 2)

On-disk layout under ``cfg.root`` (one local dir simulates every service):
    meta/<project>/step-<step>/manifest.json       omnifs-meta (business meta)
    store/<project>/step-<step>/dn<d>/r<rank>.log  EIC DN data (SSD tier)
    store/<project>/step-<step>/journal/r<rank>.jsonl
                                                   KeySet layout entries
                                                   (key, chunk_offset, length)
    storemeta/<project>/step-<step>/sink.json      sink state (storage side)
    <hdfs_root>/<project>/step-<step>/<file>       HDD tier after sink (HF dir)

Rules taken from the design:
* rank0 creates the snapshot manifest (state "prepared"); every rank writes
  only its own ranges straight to the store; after a barrier rank0 checks the
  full layout, writes the JSON blobs, and commits the manifest. The commit is
  the only publish point. An uncommitted snapshot is invisible and rewritable.
* KV keys are unique per range, so a retry rewrites the same key with the same
  bytes (idempotent). A key with different bytes, or two keys that overlap, is
  a conflict ("final consistency, no overwrite").
* Sink does not change the manifest. Reads route by sink.json, never by an SSD
  miss.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .iobackend import CHUNK_BYTES, IoRecord, IoTracer, P2PReadCache
from .result import ResultError


@dataclass(frozen=True)
class OmniFsConfig:
    root: str
    project: str = "proj"
    step: int = 0
    chunk_bytes: int = CHUNK_BYTES
    n_dn: int = 8

    @property
    def snap(self) -> str:
        return os.path.join(self.project, f"step-{self.step}")

    @property
    def meta_dir(self) -> str:
        return os.path.join(self.root, "meta", self.snap)

    @property
    def store_dir(self) -> str:
        return os.path.join(self.root, "store", self.snap)

    @property
    def storemeta_dir(self) -> str:
        return os.path.join(self.root, "storemeta", self.snap)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.meta_dir, "manifest.json")

    @property
    def sink_path(self) -> str:
        return os.path.join(self.storemeta_dir, "sink.json")


def _atomic_write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# =================================================================== manifest


@dataclass
class Manifest:
    project: str
    step: int
    chunk_bytes: int
    state: str  # prepared | committed
    files: Dict[str, dict]  # rel path -> {"blob_id", "size", "kind"}
    json_blobs: Dict[str, str] = field(default_factory=dict)  # rel -> content (written at commit)

    def to_bytes(self) -> bytes:
        return json.dumps(self.__dict__, sort_keys=True, indent=1).encode()

    @staticmethod
    def load(path: str) -> "Manifest":
        with open(path, "rb") as f:
            return Manifest(**json.loads(f.read()))

    def blob(self, rel: str) -> dict:
        try:
            return self.files[rel]
        except KeyError:
            raise ResultError(f"{rel} is not in snapshot {self.project}/step-{self.step}") from None


def build_manifest(cfg: OmniFsConfig, file_sizes: Dict[str, int], json_blobs: Dict[str, bytes]) -> Manifest:
    files = {}
    for i, rel in enumerate(sorted(file_sizes) + sorted(json_blobs)):
        size = file_sizes.get(rel, len(json_blobs.get(rel, b"")))
        files[rel] = {"blob_id": f"blob-{i:05d}", "size": size,
                      "kind": "json" if rel in json_blobs else "safetensors"}
    return Manifest(cfg.project, cfg.step, cfg.chunk_bytes, "prepared", files,
                    {k: v.decode() for k, v in json_blobs.items()})


def split_by_chunk(offset: int, size: int, chunk: int):
    """Yield (chunk_idx, chunk_offset, piece_offset, piece_len)."""
    pos, end = offset, offset + size
    while pos < end:
        idx, off = divmod(pos, chunk)
        n = min(chunk - off, end - pos)
        yield idx, off, pos - offset, n
        pos += n


# =================================================================== KV store


@dataclass(frozen=True)
class KvEntry:
    key: str
    blob_id: str
    chunk: int
    chunk_offset: int
    length: int
    dn: int
    rank: int
    log_offset: int
    crc: int


class KvStore:
    """EIC-KV simulator: DN logs + per-rank KeySet layout journals."""

    def __init__(self, cfg: OmniFsConfig, tracer: Optional[IoTracer] = None) -> None:
        self.cfg = cfg
        self.tracer = tracer or IoTracer()
        self._lock = threading.Lock()
        self._wfd: Dict[Tuple[int, int], int] = {}
        self._wsize: Dict[Tuple[int, int], int] = {}
        self._jfd: Dict[int, int] = {}
        self._rfd: Dict[str, int] = {}
        self._layout: Optional[Dict[Tuple[str, int], List[KvEntry]]] = None

    # ---------------------------------------------------------- write side

    def _log_path(self, dn: int, rank: int) -> str:
        return os.path.join(self.cfg.store_dir, f"dn{dn}", f"r{rank}.log")

    def _journal_path(self, rank: int) -> str:
        return os.path.join(self.cfg.store_dir, "journal", f"r{rank}.jsonl")

    def _writer(self, rank: int, dn: int) -> Tuple[int, int]:
        key = (rank, dn)
        with self._lock:
            if key not in self._wfd:
                path = self._log_path(dn, rank)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
                self._wfd[key] = fd
                self._wsize[key] = os.fstat(fd).st_size
            if rank not in self._jfd:
                jp = self._journal_path(rank)
                os.makedirs(os.path.dirname(jp), exist_ok=True)
                self._jfd[rank] = os.open(jp, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            return self._wfd[key], self._jfd[rank]

    def put(self, rank: int, blob_id: str, chunk: int, chunk_offset: int, data) -> KvEntry:
        """WriteChunk: one pwrite piece -> one new KV in the chunk's KeySet."""
        view = memoryview(data)
        key = f"{blob_id}#{chunk}@{chunk_offset}+{len(view)}"
        dn = zlib.crc32(key.encode()) % self.cfg.n_dn
        fd, jfd = self._writer(rank, dn)
        log_off = self._wsize[(rank, dn)]
        done = 0
        while done < len(view):
            done += os.pwrite(fd, view[done:], log_off + done)
        self._wsize[(rank, dn)] = log_off + len(view)
        ent = KvEntry(key, blob_id, chunk, chunk_offset, len(view), dn, rank, log_off, zlib.crc32(view))
        os.write(jfd, (json.dumps(ent.__dict__, separators=(",", ":")) + "\n").encode())
        self.tracer.add(IoRecord(rank, "kv_put", f"{blob_id}#{chunk}", chunk_offset, len(view),
                                 time.perf_counter(), f"dn{dn}"))
        return ent

    def flush(self, rank: int) -> None:
        """Make this rank's KVs and layout entries durable, then close them."""
        with self._lock:
            keys = [k for k in self._wfd if k[0] == rank]
            fds = [self._wfd.pop(k) for k in keys] + ([self._jfd.pop(rank)] if rank in self._jfd else [])
            for k in keys:
                self._wsize.pop(k, None)
        for fd in fds:
            os.fsync(fd)
            os.close(fd)

    def drop(self) -> None:
        """Remove every KV and layout entry of this snapshot (abort / reclaim)."""
        self.close()
        shutil.rmtree(self.cfg.store_dir, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            for fd in list(self._wfd.values()) + list(self._jfd.values()) + list(self._rfd.values()):
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._wfd.clear(); self._wsize.clear(); self._jfd.clear(); self._rfd.clear()
            self._layout = None

    # ---------------------------------------------------------- layout

    def layout(self, refresh: bool = False) -> Dict[Tuple[str, int], List[KvEntry]]:
        """KeySet layouts: (blob_id, chunk) -> entries sorted by chunk_offset.

        Merges all rank journals; deduplicates retries; rejects conflicts and
        overlaps. Cached (the SDK "layout cache")."""
        with self._lock:
            if self._layout is not None and not refresh:
                return self._layout
        jdir = os.path.join(self.cfg.store_dir, "journal")
        by_key: Dict[str, KvEntry] = {}
        if os.path.isdir(jdir):
            for name in sorted(os.listdir(jdir)):
                with open(os.path.join(jdir, name), "rb") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        e = KvEntry(**json.loads(line))
                        old = by_key.get(e.key)
                        if old is not None and old.crc != e.crc:
                            raise ResultError(f"KV conflict: key {e.key} rewritten with different bytes")
                        by_key.setdefault(e.key, e)  # retry with same bytes: keep first copy
        out: Dict[Tuple[str, int], List[KvEntry]] = defaultdict(list)
        for e in by_key.values():
            out[(e.blob_id, e.chunk)].append(e)
        for k, lst in out.items():
            lst.sort(key=lambda e: e.chunk_offset)
            for a, b in zip(lst, lst[1:]):
                if b.chunk_offset < a.chunk_offset + a.length:
                    raise ResultError(f"KV overlap in chunk {k}: {a.key} / {b.key}")
        with self._lock:
            self._layout = dict(out)
            return self._layout

    def check_full_coverage(self, manifest: Manifest) -> None:
        """rank0 summary check before commit: every blob byte has exactly one KV."""
        lay = self.layout(refresh=True)
        cb = manifest.chunk_bytes
        for rel, f in manifest.files.items():
            n_chunks = -(-f["size"] // cb)
            for c in range(n_chunks):
                want = min(cb, f["size"] - c * cb)
                pos = 0
                for e in lay.get((f["blob_id"], c), []):
                    if e.chunk_offset != pos:
                        raise ResultError(f"{rel}: chunk {c} gap at {pos}")
                    pos += e.length
                if pos != want:
                    raise ResultError(f"{rel}: chunk {c} covers {pos} of {want} bytes")

    # ---------------------------------------------------------- read side

    def _reader(self, dn: int, rank: int) -> int:
        path = self._log_path(dn, rank)
        with self._lock:
            fd = self._rfd.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._rfd[path] = fd
            return fd

    def read_chunk(self, reader_rank: int, blob_id: str, chunk: int, offset: int, length: int) -> bytes:
        """ReadChunk, scheme 2: intersect the layout, read KV pieces across DNs,
        assemble. A gap is an error: only written ranges can be read."""
        entries = self.layout().get((blob_id, chunk), [])
        out = bytearray()
        pos, end = offset, offset + length
        starts = [e.chunk_offset for e in entries]
        import bisect
        k = max(bisect.bisect_right(starts, pos) - 1, 0)
        while pos < end:
            if k >= len(entries) or entries[k].chunk_offset > pos or entries[k].chunk_offset + entries[k].length <= pos:
                raise ResultError(f"{blob_id}#{chunk}: no KV covers offset {pos}")
            e = entries[k]
            lo = pos - e.chunk_offset
            n = min(e.length - lo, end - pos)
            data = os.pread(self._reader(e.dn, e.rank), n, e.log_offset + lo)
            if len(data) != n:
                raise ResultError(f"short KV read {e.key}")
            self.tracer.add(IoRecord(reader_rank, "kv_get", f"{blob_id}#{chunk}", pos, n, time.perf_counter(), f"dn{e.dn}"))
            out += data
            pos += n
            k += 1
        return bytes(out)


# =================================================================== IO facade


class _NoFds:
    def close_rank(self, rank: int, prefix: Optional[str] = None) -> None:
        pass

    def close_path(self, path: str) -> None:
        pass


class OmniFsIO:
    """Same interface as FileIO (pwrite / pread by path), backed by OmniFS.

    Paths are file names inside the snapshot (fixed 2-level dir: project/step).
    Writes: SDK straight to the store, split at chunk boundaries.
    Reads: omnifs-host path L1 -> L2 (P2P) -> L3; L3 routes by sink state.
    """

    def __init__(self, cfg: OmniFsConfig, tracer: Optional[IoTracer] = None,
                 read_cache: Optional[P2PReadCache] = None) -> None:
        self.cfg = cfg
        self.tracer = tracer or IoTracer()
        self.read_cache = read_cache
        self.kv = KvStore(cfg, self.tracer)
        self.fds = _NoFds()
        self.manifest: Optional[Manifest] = None
        self._sink: Optional[dict] = None
        self._sink_checked = False
        self._lock = threading.Lock()

    def bind(self, manifest: Manifest) -> None:
        self.manifest = manifest

    # ---------------------------------------------------------- write

    def pwrite(self, rank: int, path: str, offset: int, data) -> None:
        f = self.manifest.blob(path)
        view = memoryview(data)
        if offset + len(view) > f["size"]:
            raise ResultError(f"{path}: write past blob end ({offset}+{len(view)} > {f['size']})")
        for idx, coff, poff, n in split_by_chunk(offset, len(view), self.cfg.chunk_bytes):
            self.kv.put(rank, f["blob_id"], idx, coff, view[poff:poff + n])
        self.tracer.add(IoRecord(rank, "pwrite", path, offset, len(view), time.perf_counter()))

    # ---------------------------------------------------------- read

    def sink_state(self) -> Optional[dict]:
        with self._lock:
            if not self._sink_checked:
                p = self.cfg.sink_path
                self._sink = json.load(open(p)) if os.path.exists(p) else None
                self._sink_checked = True
            return self._sink

    def _backend_read(self, rank: int, path: str, offset: int, size: int, level: str = "L3") -> bytes:
        f = self.manifest.blob(path)
        size = max(0, min(size, f["size"] - offset))  # a cache block may pass EOF
        if size == 0:
            return b""
        sink = self.sink_state()
        if sink is not None and sink.get("state") == "sunk":
            hp = os.path.join(sink["location"], path)
            with open(hp, "rb") as fh:
                data = os.pread(fh.fileno(), size, offset)
            self.tracer.add(IoRecord(rank, "pread", path, offset, len(data), time.perf_counter(), f"{level}-hdd"))
            return data
        parts = []
        for idx, coff, _, n in split_by_chunk(offset, size, self.cfg.chunk_bytes):
            parts.append(self.kv.read_chunk(rank, f["blob_id"], idx, coff, n))
        data = b"".join(parts)
        self.tracer.add(IoRecord(rank, "pread", path, offset, len(data), time.perf_counter(), f"{level}-ssd"))
        return data

    def pread(self, rank: int, path: str, offset: int, size: int) -> bytes:
        if size == 0:
            return b""
        if self.read_cache is None:
            data = self._backend_read(rank, path, offset, size, "direct")
        else:
            blk = self.read_cache.block
            parts = []
            fetch = lambda off, n: self._backend_read(rank, path, off, n)  # noqa: E731
            for idx in range(offset // blk, (offset + size - 1) // blk + 1):
                b, level = self.read_cache.get_block(rank, path, idx, fetch, self.tracer)
                lo = max(offset - idx * blk, 0)
                hi = min(offset + size - idx * blk, len(b))
                parts.append(b[lo:hi])
                self.tracer.count(f"{level}_bytes", hi - lo)
            data = b"".join(parts)
            self.tracer.count("requested_bytes", size)
        if len(data) != size:
            raise IOError(f"short read {path}@{offset}: {len(data)} of {size} bytes")
        return data


# =================================================================== sink (DMC)


def sink_snapshot(cfg: OmniFsConfig, hdfs_root: str, reclaim: bool = True) -> dict:
    """SSD -> HDD sink of one committed snapshot.

    1. list files and chunks from the manifest (read-only; never modified);
    2. rebuild each chunk from its KeySet layout;
    3. append each chunk to a temp file, then concat the chunks into the blob
       file under <hdfs_root>/<project>/step-<step>/ (HF directory);
    4. verify size and CRC, publish sink.json (storage side), then reclaim the
       SSD KVs and layouts.
    """
    m = Manifest.load(cfg.manifest_path)
    if m.state != "committed":
        raise ResultError("only a committed snapshot can sink")
    kv = KvStore(cfg)
    out_dir = os.path.join(hdfs_root, cfg.snap)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    stats = {"files": 0, "chunks": 0, "bytes": 0}
    crcs = {}
    for rel, f in sorted(m.files.items()):
        n_chunks = -(-f["size"] // cfg.chunk_bytes)
        chunk_tmps = []
        crc = 0
        for c in range(n_chunks):
            n = min(cfg.chunk_bytes, f["size"] - c * cfg.chunk_bytes)
            data = kv.read_chunk(-1, f["blob_id"], c, 0, n)
            crc = zlib.crc32(data, crc)
            tmp = os.path.join(out_dir, f"{rel}.chunk{c}.tmp")
            with open(tmp, "wb") as fh:
                fh.write(data)
            chunk_tmps.append(tmp)
            stats["chunks"] += 1
        final_tmp = os.path.join(out_dir, f"{rel}.concat.tmp")
        with open(final_tmp, "wb") as out:  # local stand-in for HDFS concat
            for tmp in chunk_tmps:
                with open(tmp, "rb") as fh:
                    shutil.copyfileobj(fh, out)
                os.remove(tmp)
            out.flush()
            os.fsync(out.fileno())
        if os.path.getsize(final_tmp) != f["size"]:
            raise ResultError(f"sink {rel}: size {os.path.getsize(final_tmp)} != {f['size']}")
        with open(final_tmp, "rb") as fh:
            if zlib.crc32(fh.read()) != crc:
                raise ResultError(f"sink {rel}: CRC mismatch after concat")
        os.replace(final_tmp, os.path.join(out_dir, rel))
        crcs[rel] = crc
        stats["files"] += 1
        stats["bytes"] += f["size"]
    kv.close()
    state = {"state": "sunk", "location": out_dir, "crc32": crcs}
    _atomic_write(cfg.sink_path, json.dumps(state, indent=1).encode())
    if reclaim:
        shutil.rmtree(cfg.store_dir, ignore_errors=True)  # SSD KV data + layouts
    stats["seconds"] = round(time.perf_counter() - t0, 3)
    stats["location"] = out_dir
    return stats

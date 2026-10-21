"""IO backend: traced pwrite / pread, plus the OmniFS read path.

Read path (OmniFS host): L1 node-local 1 MiB block cache -> L2 peer node
cache (P2P) -> L3 store (real pread). Concurrent misses on one block are
single-flight: one reader goes to L3, the others wait (LazyLoad lock).

Write path: every rank pwrites straight to the store (no host, no cache).
"""
from __future__ import annotations

import os
import threading
import time
import zlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MiB = 1 << 20
CHUNK_BYTES = 64 * MiB  # OmniFS-Store fixed chunk size


@dataclass(frozen=True)
class IoRecord:
    rank: int
    op: str  # "pwrite" | "pread"
    path: str
    offset: int
    size: int
    t: float
    level: str = "L3"  # pread only: L1 / L2 / L3 / direct


class IoTracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: List[IoRecord] = []
        self.counters: Dict[str, int] = defaultdict(int)

    def add(self, rec: IoRecord) -> None:
        with self._lock:
            self.records.append(rec)

    def count(self, key: str, n: int) -> None:
        with self._lock:
            self.counters[key] += n


class FdPool:
    """Per-rank file descriptor cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fds: Dict[Tuple[int, str, str], int] = {}

    def get(self, rank: int, path: str, mode: str) -> int:
        key = (rank, path, mode)
        with self._lock:
            fd = self._fds.get(key)
            if fd is None:
                if mode == "w":
                    # O_CREAT: any rank may write into a shared file before the
                    # file's slot owners have opened it (OmniFS pwrite model).
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
                else:
                    fd = os.open(path, os.O_RDONLY)
                self._fds[key] = fd
            return fd

    def close_rank(self, rank: int, prefix: Optional[str] = None) -> None:
        with self._lock:
            for key in [k for k in self._fds if k[0] == rank and (prefix is None or k[1].startswith(prefix))]:
                os.close(self._fds.pop(key))

    def close_path(self, path: str) -> None:
        with self._lock:
            for key in [k for k in self._fds if k[1] == path]:
                os.close(self._fds.pop(key))


def eic_replica_policy(path: str) -> int:
    """OmniFS-Store 2.1: SafeTensor 3 cache replicas, hot JSON 10."""
    return 10 if path.endswith(".json") else 3


class P2PReadCache:
    """Simulated read cache shared by all ranks of one job.

    * ``max_replicas=None`` (omnifs-host P2P): every node that reads a block
      keeps a copy, so replicas grow with readers.
    * ``max_replicas=f(path)`` (EIC semi-managed memory cache): a block has
      at most f(path) cached copies; other nodes read a peer copy remotely
      (counted as L2) and do not keep it.
    """

    def __init__(self, ranks_per_node: int, block_bytes: int = MiB, l1_capacity_blocks: int = 4096,
                 max_replicas=None) -> None:
        self.max_replicas = max_replicas
        self.ranks_per_node = max(1, ranks_per_node)
        self.block = block_bytes
        self.cap = l1_capacity_blocks
        self._lock = threading.Lock()
        self._l1: Dict[int, "OrderedDict[Tuple[str, int], bytes]"] = defaultdict(OrderedDict)
        self._holders: Dict[Tuple[str, int], set] = defaultdict(set)  # block -> nodes
        self._inflight: Dict[Tuple[str, int], threading.Event] = {}
        self.unique_l3_blocks: set = set()

    def replicas(self, path: str, idx: int) -> int:
        with self._lock:
            return len(self._holders.get((path, idx), ()))

    def node_of(self, rank: int) -> int:
        return rank // self.ranks_per_node

    def _put(self, node: int, key: Tuple[str, int], data: bytes) -> None:
        l1 = self._l1[node]
        l1[key] = data
        l1.move_to_end(key)
        self._holders[key].add(node)
        while len(l1) > self.cap:
            old, _ = l1.popitem(last=False)
            self._holders[old].discard(node)

    def get_block(self, rank: int, path: str, idx: int, fetch, tracer: IoTracer) -> bytes:
        node = self.node_of(rank)
        key = (path, idx)
        while True:
            with self._lock:
                data = self._l1[node].get(key)
                if data is not None:
                    self._l1[node].move_to_end(key)
                    return data, "L1"
                peers = sorted(n for n in self._holders.get(key, ()) if n != node)
                if peers:
                    holders = self._holders[key]
                    # Stable peer pick: Python hash() is salted per process,
                    # which would change L2 routing across machines.
                    pick = zlib.crc32(f"{node}:{idx}".encode()) % len(peers)
                    data = self._l1[peers[pick]][key]
                    if self.max_replicas is None or len(holders) < self.max_replicas(path):
                        self._put(node, key, data)  # diffuse one more replica
                    tracer.count("L2_block_bytes", len(data))
                    return data, "L2"
                ev = self._inflight.get(key)
                if ev is None:
                    ev = threading.Event()
                    self._inflight[key] = ev
                    owner = True
                else:
                    owner = False
            if not owner:
                ev.wait()
                continue  # retry: block is now in some L1 (or the fetch failed)
            try:
                data = fetch(idx * self.block, self.block)
                with self._lock:
                    self._put(node, key, data)
                    self.unique_l3_blocks.add(key)
                tracer.count("L3_block_bytes", len(data))
                return data, "L3"
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
                ev.set()


class FileIO:
    """Traced pwrite / pread against a local directory acting as the store."""

    def __init__(self, tracer: Optional[IoTracer] = None, read_cache: Optional[P2PReadCache] = None) -> None:
        self.tracer = tracer or IoTracer()
        self.read_cache = read_cache
        self.fds = FdPool()

    def pwrite(self, rank: int, path: str, offset: int, data) -> None:
        fd = self.fds.get(rank, path, "w")
        view = memoryview(data)
        done = 0
        while done < len(view):
            done += os.pwrite(fd, view[done:], offset + done)
        self.tracer.add(IoRecord(rank, "pwrite", path, offset, len(view), time.perf_counter()))

    def _pread_direct(self, rank: int, path: str, offset: int, size: int, level: str) -> bytes:
        fd = self.fds.get(rank, path, "r")
        out = bytearray()
        while len(out) < size:
            chunk = os.pread(fd, size - len(out), offset + len(out))
            if not chunk:
                break
            out += chunk
        self.tracer.add(IoRecord(rank, "pread", path, offset, len(out), time.perf_counter(), level))
        return bytes(out)

    def pread(self, rank: int, path: str, offset: int, size: int) -> bytes:
        if size == 0:
            return b""
        if self.read_cache is None:
            data = self._pread_direct(rank, path, offset, size, "direct")
        else:
            blk = self.read_cache.block
            parts = []
            fetch = lambda off, n: self._pread_direct(rank, path, off, n, "L3")  # noqa: E731
            for idx in range(offset // blk, (offset + size - 1) // blk + 1):
                b, level = self.read_cache.get_block(rank, path, idx, fetch, self.tracer)
                lo = max(offset - idx * blk, 0)
                hi = min(offset + size - idx * blk, len(b))
                parts.append(b[lo:hi])
                self.tracer.count(f"{level}_bytes", hi - lo)  # bytes served to the caller
            data = b"".join(parts)
            self.tracer.count("requested_bytes", size)
        if len(data) != size:
            raise IOError(f"short read {path}@{offset}: {len(data)} of {size} bytes")
        return data

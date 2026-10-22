"""Internal file IO: traced pwrite / pread against a local directory.

This module is an implementation detail of the safetensors / distcp sessions
(design section 8: format implementers own file IO). It is not public API.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

MiB = 1 << 20


@dataclass(frozen=True)
class IoRecord:
    rank: int
    op: str  # "pwrite" | "pread"
    path: str
    offset: int
    size: int
    t: float


class IoTracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: List[IoRecord] = []
        self.counters: Dict[str, int] = defaultdict(int)

    def add(self, rec: IoRecord) -> None:
        with self._lock:
            self.records.append(rec)
            self.counters[f"{rec.op}_bytes"] += rec.size


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
                    # O_CREAT: any writer rank of a shared file may open it
                    # first; byte ranges of different ranks are disjoint.
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


class FileIO:
    """Traced pwrite / pread against a local directory acting as the store."""

    def __init__(self, tracer: Optional[IoTracer] = None) -> None:
        self.tracer = tracer or IoTracer()
        self.fds = FdPool()

    def pwrite(self, rank: int, path: str, offset: int, data) -> None:
        fd = self.fds.get(rank, path, "w")
        view = memoryview(data)
        done = 0
        while done < len(view):
            done += os.pwrite(fd, view[done:], offset + done)
        self.tracer.add(IoRecord(rank, "pwrite", path, offset, len(view), time.perf_counter()))

    def pread(self, rank: int, path: str, offset: int, size: int) -> bytes:
        if size == 0:
            return b""
        fd = self.fds.get(rank, path, "r")
        out = bytearray()
        while len(out) < size:
            chunk = os.pread(fd, size - len(out), offset + len(out))
            if not chunk:
                break
            out += chunk
        self.tracer.add(IoRecord(rank, "pread", path, offset, len(out), time.perf_counter()))
        if len(out) != size:
            raise IOError(f"short read {path}@{offset}: {len(out)} of {size} bytes")
        return bytes(out)

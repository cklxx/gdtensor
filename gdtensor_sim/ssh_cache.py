"""Bounded SSH cache data path for explicit two-host integration verification.

SSH provides authentication and transport. This module is a test backend, not a
peer discovery service or a cross-host coordinator. Each request is a bounded
process and every read verifies length and SHA256 before delivering bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import threading

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb


class SSHCache(ga.DTensorDataAccessor):
    def __init__(self, host, root, helper, python_command, timeout=15):
        if not re.fullmatch(r"(?:[a-zA-Z0-9_.-]+@)?[a-zA-Z0-9][a-zA-Z0-9_.-]*", host):
            raise ValueError("invalid SSH destination")
        if timeout <= 0 or not python_command or not helper.startswith("/"):
            raise ValueError("positive timeout and absolute helper required")
        self.host, self.root, self.helper = host, root, helper
        self.python_command, self.timeout = list(python_command), timeout
        self.lock = threading.RLock()
        self.records = {}
        self.tx = self.rx = 0
        self.requests = 0

    def _request(self, operation, key, data=b"", digest=None):
        if len(data) > 64 * (1 << 20):
            raise ValueError("segment size limit")
        request = json.dumps(
            dict(op=operation, key=key, size=len(data), digest=digest)
        ).encode()
        command = shlex.join(self.python_command + [self.helper, self.root])
        out = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                self.host,
                command,
            ],
            input=request + b"\n" + data,
            capture_output=True,
            timeout=self.timeout,
            check=True,
        )
        header, payload = out.stdout.split(b"\n", 1)
        response = json.loads(header)
        if not response.get("ok"):
            raise OSError(response.get("error", "peer cache request failed"))
        self.requests += 1
        self.tx += len(data)
        self.rx += len(payload)
        return response, payload

    def _key(self, rng):
        namespace = hashlib.sha256(
            repr(getattr(rng, "source_identity", None)).encode()
        ).hexdigest()
        return (
            namespace,
            getattr(rng, "cache_node", 0),
            rng.global_tensor_id,
            rng.global_tensor_offset,
            rng.size,
        )

    def _record(self, rng):
        exact = self._key(rng)
        if exact in self.records:
            return exact, self.records[exact]
        namespace, node, tid, off, size = exact
        hits = [
            (key, digest)
            for key, digest in self.records.items()
            if key[:3] == (namespace, node, tid)
            and key[3] <= off
            and off + size <= key[3] + key[4]
        ]
        if len(hits) != 1:
            return None
        return hits[0]

    def has_bytes(self, global_range) -> bool:
        with self.lock:
            return self._record(global_range) is not None

    def write_from_buffer(self, global_range, write_from_buffer):
        buf = np.empty(global_range.size, np.uint8)
        result = write_from_buffer(buf.ctypes.data, global_range.size)
        if type(result) is not gb.Ok:
            return result
        data = buf.tobytes()
        key = self._key(global_range)
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            self._request("write", key, data, digest)
            self.records[key] = digest
        return result

    def read_to_buffer(self, global_range, read_to_buffer):
        with self.lock:
            raw = self._read_range(global_range)
        buf = np.frombuffer(raw, np.uint8)
        return read_to_buffer(buf.ctypes.data, global_range.size)

    def _read_range(self, global_range):
        hit = self._record(global_range)
        if hit is None:
            raise OSError("peer bytes not published")
        key, digest = hit
        _, raw = self._request("read", key, digest=digest)
        if len(raw) != key[4] or hashlib.sha256(raw).hexdigest() != digest:
            raise OSError("peer cache checksum mismatch")
        off = global_range.global_tensor_offset - key[3]
        return raw[off : off + global_range.size]

    def corrupt(self):
        key = next(iter(self.records))
        self._request("corrupt", key)

    def close(self):
        self._request("cleanup", ["cleanup"])

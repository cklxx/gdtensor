"""Bounded SSH cache data path for explicit two-host integration verification.

SSH provides authentication and transport. This module is a test backend, not a
peer discovery service or a cross-host coordinator. Each response has a JSON
header with a declared payload size. Payloads are length- and SHA256-checked
before consumer callbacks.
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

_MAX_PAYLOAD = 64 * (1 << 20)
_MAX_HEADER = 65536


def _expire(process, expired):
    expired.set()
    try:
        process.kill()
    except ProcessLookupError:
        pass


def _send_request(process, request, data):
    try:
        process.stdin.write(request + b"\n")
        for offset in range(0, len(data), 1 << 20):
            process.stdin.write(data[offset : offset + (1 << 20)])
        process.stdin.close()
    except BrokenPipeError:
        pass


def _receive_response(process):
    header = process.stdout.readline(_MAX_HEADER + 1)
    if len(header) > _MAX_HEADER or not header.endswith(b"\n"):
        raise OSError("invalid peer cache response header")
    response = json.loads(header)
    if not response.get("ok"):
        raise OSError(response.get("error", "peer cache request failed"))
    size = response.get("size")
    if type(size) is not int or not 0 <= size <= _MAX_PAYLOAD:
        raise OSError("invalid peer cache response size")
    chunks = []
    remaining = size
    while remaining:
        chunk = process.stdout.read(min(remaining, 1 << 20))
        if not chunk:
            raise OSError("short peer cache response")
        chunks.append(chunk)
        remaining -= len(chunk)
    if process.stdout.read(1):
        raise OSError("peer cache response exceeds declared size")
    return response, chunks


def _raise_expired(expired, timeout):
    if expired.is_set():
        raise subprocess.TimeoutExpired("ssh", timeout) from None


def _close_process(process):
    if process.poll() is None:
        process.kill()
    process.wait()
    process.stdout.close()
    if process.stdin.closed:
        return
    try:
        process.stdin.close()
    except BrokenPipeError:
        pass


def _collect_records(reader, global_ranges):
    records = []
    for rng in global_ranges:
        record = reader(rng)
        if record is None:
            return None
        records.append(record)
    return records


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
        if len(data) > _MAX_PAYLOAD:
            raise ValueError("segment size limit")
        request = json.dumps(
            dict(op=operation, key=key, size=len(data), digest=digest)
        ).encode()
        command = shlex.join(self.python_command + [self.helper, self.root])
        process = subprocess.Popen(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                self.host,
                command,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        expired = threading.Event()
        timer = threading.Timer(self.timeout, lambda: _expire(process, expired))
        timer.start()
        try:
            return self._exchange(process, request, data, expired)
        except BaseException:
            _raise_expired(expired, self.timeout)
            raise
        finally:
            timer.cancel()
            _close_process(process)

    def _exchange(self, process, request, data, expired):
        _send_request(process, request, data)
        response, chunks = _receive_response(process)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, "ssh")
        _raise_expired(expired, self.timeout)
        payload = b"".join(chunks)
        self.requests += 1
        self.tx += len(data)
        self.rx += len(payload)
        return response, payload

    def _key(self, rng):
        identity = getattr(rng, "source_identity", None)
        namespace = getattr(rng, "source_digest", None) or ga._content_digest(identity)
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
        return self.has_bytes_batch([global_range])

    def has_bytes_batch(self, global_ranges) -> bool:
        with self.lock:
            return self._probe_records(global_ranges)

    def _probe_records(self, global_ranges):
        records = _collect_records(self._record, global_ranges)
        if records is None:
            return False
        if not records:
            return True
        response, _ = self._request("probe", [key for key, _digest in records])
        present = response.get("present")
        return (
            type(present) is list
            and len(present) == len(records)
            and all(type(value) is bool and value for value in present)
        )

    def write_from_buffer(self, global_range, write_from_buffer):
        buf = np.empty(global_range.size, np.uint8)
        result = write_from_buffer(buf.ctypes.data, global_range.size)
        if type(result) is not gb.Ok:
            return result
        key = self._key(global_range)
        digest = hashlib.sha256(buf).hexdigest()
        with self.lock:
            self._request("write", key, buf, digest)
            self.records[key] = digest
        return result

    def read_to_buffer(self, global_range, read_to_buffer):
        raw = self.read_bytes_batch([global_range])[0]
        buf = np.frombuffer(raw, np.uint8)
        return read_to_buffer(buf.ctypes.data, global_range.size)

    def read_bytes_batch(self, global_ranges):
        with self.lock:
            return self._read_records(global_ranges)

    def _read_records(self, global_ranges):
        records = _collect_records(self._record, global_ranges)
        if records is None:
            raise OSError("peer bytes not published")
        if not records:
            return []
        keys = [key for key, _digest in records]
        _response, raw = self._request("read_batch", keys)
        expected = sum(key[4] for key in keys)
        if len(raw) != expected:
            raise OSError("peer cache response length mismatch")
        output = []
        start = 0
        for rng, (key, digest) in zip(global_ranges, records):
            end = start + key[4]
            whole = raw[start:end]
            output.append(self._slice_record(rng, key, digest, whole))
            start = end
        return output

    def _slice_record(self, rng, key, digest, whole):
        if hashlib.sha256(whole).hexdigest() != digest:
            raise OSError("peer cache checksum mismatch")
        off = rng.global_tensor_offset - key[3]
        return whole[off : off + rng.size]

    def corrupt(self):
        key = next(iter(self.records))
        self._request("corrupt", key)

    def close(self):
        self.records.clear()

    def cleanup(self):
        with self.lock:
            self._request("cleanup", ["cleanup"])
            self.records.clear()

"""SSH transport fault tests; no external host or service required."""

import ctypes
import hashlib
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import gdtensor.builder as gb
from gdtensor_conn.ssh_cache import SSHCache
from tests.support import p2p_helper


def cache():
    return SSHCache("user@peer", "/tmp/gdtensor-p2p-test", "/tmp/helper.py", ["python"])


def rng(identity="source", off=0, size=4):
    return SimpleNamespace(
        source_identity=identity,
        cache_node=0,
        global_tensor_id="w",
        global_tensor_offset=off,
        size=size,
    )


def fill(address, size):
    ctypes.memmove(address, b"abcd", size)
    return gb.Ok(None)


def frame(payload=b"", **fields):
    header = json.dumps(dict(ok=True, size=len(payload), **fields)).encode()
    return header + b"\n" + payload


class FakeStdin:
    def __init__(self, process):
        self.process = process
        self.data = bytearray()
        self.closed = False

    def write(self, data):
        if self.closed:
            raise BrokenPipeError()
        self.data.extend(data)
        return len(data)

    def close(self):
        if not self.closed:
            self.closed = True
            self.process.respond()


class FakeStdout:
    def __init__(self, process):
        self.process = process
        self.buffer = None

    def _buffer(self):
        self.process.respond()
        if self.process.killed:
            return io.BytesIO()
        if self.buffer is None:
            self.buffer = io.BytesIO(self.process.output)
        return self.buffer

    def readline(self, size):
        return self._buffer().readline(size)

    def read(self, size=-1):
        return self._buffer().read(size)

    def close(self):
        pass


class FakeProcess:
    def __init__(self, handler):
        self.handler = handler
        self.stdin = FakeStdin(self)
        self.stdout = FakeStdout(self)
        self.output = b""
        self.request = None
        self.returncode = None
        self.killed = False
        self.waits = 0
        self.responded = False

    def respond(self):
        if self.responded or self.killed:
            return
        self.responded = True
        header, _, data = bytes(self.stdin.data).partition(b"\n")
        self.request = json.loads(header)
        self.output = self.handler(self.request, data)

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self):
        self.waits += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def poll(self):
        return self.returncode


class ImmediateTimer:
    def __init__(self, _timeout, callback):
        self.callback = callback

    def start(self):
        self.callback()

    def cancel(self):
        pass


def install(monkeypatch, handler):
    processes = []

    def popen(*_args, **_kwargs):
        process = FakeProcess(handler)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    return processes


@pytest.mark.parametrize("failure", ["disconnect", "timeout", "malformed", "denied"])
def test_failed_remote_write_never_publishes(monkeypatch, failure):
    def handler(_request, _data):
        if failure == "disconnect":
            raise subprocess.CalledProcessError(255, "ssh")
        if failure == "malformed":
            return b"invalid"
        return json.dumps(dict(ok=False, size=0, error="denied")).encode() + b"\n"

    processes = install(monkeypatch, handler)
    if failure == "timeout":
        monkeypatch.setattr("gdtensor_conn.ssh_cache.threading.Timer", ImmediateTimer)
    acc = cache()
    with pytest.raises((subprocess.SubprocessError, ValueError, OSError)):
        acc.write_from_buffer(rng(), fill)
    assert not acc.records and not acc.has_bytes(rng())
    if failure == "timeout":
        assert processes[0].killed and processes[0].waits


@pytest.mark.parametrize("payload", [b"abc", b"bad!"])
def test_bad_remote_read_never_calls_consumer(monkeypatch, payload):
    acc = cache()
    acc.records[acc._key(rng())] = hashlib.sha256(b"abcd").hexdigest()
    install(monkeypatch, lambda _request, _data: frame(payload))
    called = []
    with pytest.raises(OSError):
        acc.read_to_buffer(rng(), lambda *a: called.append(a))
    assert not called


def test_duplicate_write_is_idempotent_and_source_scoped(monkeypatch):
    store = {}

    def handler(request, data):
        keys = request["key"] if request["op"] == "read_batch" else [request["key"]]
        if request["op"] == "write":
            store[repr(request["key"])] = data
            return frame()
        if request["op"] == "probe":
            return frame(present=[repr(key) in store for key in request["key"]])
        return frame(b"".join(store[repr(key)] for key in keys))

    install(monkeypatch, handler)
    acc = cache()
    assert type(acc.write_from_buffer(rng(), fill)) is gb.Ok
    assert type(acc.write_from_buffer(rng(), fill)) is gb.Ok
    assert len(acc.records) == len(store) == 1
    assert not acc.has_bytes(rng("other"))
    out = []
    assert (
        type(
            acc.read_to_buffer(
                rng(off=1, size=2),
                lambda p, n: (out.append(ctypes.string_at(p, n)), gb.Ok(None))[1],
            )
        )
        is gb.Ok
    )
    assert out == [b"bc"]


@pytest.mark.parametrize(
    "output",
    [
        json.dumps(dict(ok=True, size=64 * (1 << 20) + 1)).encode() + b"\n",
        json.dumps(dict(ok=True, size=-1)).encode() + b"\n",
        frame(b"abc")[:-1],
        frame(b"") + b"extra",
    ],
    ids=["too-large", "negative", "short", "trailing"],
)
def test_invalid_response_fails_before_consumer(monkeypatch, output):
    acc = cache()
    acc.records[acc._key(rng())] = hashlib.sha256(b"abcd").hexdigest()
    install(monkeypatch, lambda _request, _data: output)
    called = []
    with pytest.raises(OSError):
        acc.read_to_buffer(rng(), lambda *a: called.append(a))
    assert not called


def test_batch_probe_and_read_preserve_order(monkeypatch):
    store = {}

    def handler(request, data):
        if request["op"] == "write":
            store[repr(request["key"])] = data
            return frame()
        if request["op"] == "probe":
            return frame(present=[repr(key) in store for key in request["key"]])
        assert request["op"] == "read_batch"
        return frame(b"".join(store[repr(key)] for key in request["key"]))

    install(monkeypatch, handler)
    acc = cache()
    for off in (0, 4):
        assert type(acc.write_from_buffer(rng(off=off), fill)) is gb.Ok
    requests = [rng(off=5, size=3), rng(off=1, size=2)]
    assert acc.has_bytes_batch(requests)
    assert acc.read_bytes_batch(requests) == [b"bcd", b"bc"]


def test_batch_probe_rejects_non_boolean_presence(monkeypatch):
    acc = cache()
    acc.records[acc._key(rng())] = "digest"
    install(monkeypatch, lambda _request, _data: frame(present=[1]))
    assert not acc.has_bytes(rng())


def test_helper_rejects_oversized_file_before_read(tmp_path, monkeypatch):
    root = tmp_path / "gdtensor-p2p-test"
    root.mkdir()
    key = ["source", 0, "w", 0, 4]
    path = p2p_helper.path_of(root, key)
    with path.open("wb") as f:
        f.truncate(p2p_helper.MAX_PAYLOAD + 1)
    request = json.dumps({"op": "read_batch", "key": [key]}).encode() + b"\n"
    monkeypatch.setattr(sys, "argv", ["p2p_helper.py", str(root)])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(request)))

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("oversized file was opened")

    monkeypatch.setattr(type(path), "open", unexpected_read)
    with pytest.raises(ValueError, match="response size limit"):
        p2p_helper.main()


def test_close_preserves_shared_root_and_cleanup_is_explicit(monkeypatch):
    operations = []
    install(
        monkeypatch, lambda request, _data: operations.append(request["op"]) or frame()
    )
    acc = cache()
    acc.close()
    assert operations == []
    acc.cleanup()
    assert operations == ["cleanup"]


@dataclass
class _ReprIdentity:
    value: int

    def __repr__(self):
        return "same repr"


def test_precomputed_source_digest_reuses_cache_key(monkeypatch):
    import gdtensor.access as ga

    identity = _ReprIdentity(1)
    request = rng(identity)
    request.source_digest = ga._content_digest(identity)

    def unexpected_digest(_identity):
        raise AssertionError("source identity was hashed again")

    monkeypatch.setattr(ga, "_content_digest", unexpected_digest)
    assert cache()._key(request)[0] == request.source_digest


def test_source_keys_use_identity_content_not_repr():
    acc = cache()
    first, second = _ReprIdentity(1), _ReprIdentity(2)
    acc.records[acc._key(rng(first))] = "digest"
    assert acc._record(rng(first)) is not None
    assert acc._record(rng(second)) is None


@pytest.mark.parametrize("host", ["-oProxyCommand=bad", "peer;bad", "user@peer bad"])
def test_invalid_host_is_rejected(host):
    with pytest.raises(ValueError):
        SSHCache(host, "/tmp/gdtensor-p2p-test", "/tmp/helper.py", ["python"])

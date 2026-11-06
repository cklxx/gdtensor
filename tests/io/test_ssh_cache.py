"""SSH transport fault tests; no external host or service required."""

import ctypes
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

import gdtensor.builder as gb
from gdtensor_sim.ssh_cache import SSHCache


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


@pytest.mark.parametrize("failure", ["disconnect", "timeout", "malformed", "denied"])
def test_failed_remote_write_never_publishes(monkeypatch, failure):
    def run(*_a, **_k):
        if failure == "disconnect":
            raise subprocess.CalledProcessError(255, "ssh")
        if failure == "timeout":
            raise subprocess.TimeoutExpired("ssh", 1)
        raw = b"invalid" if failure == "malformed" else b'{"ok":false}\n'
        return SimpleNamespace(stdout=raw)

    monkeypatch.setattr(subprocess, "run", run)
    acc = cache()
    with pytest.raises((subprocess.SubprocessError, ValueError, OSError)):
        acc.write_from_buffer(rng(), fill)
    assert not acc.records and not acc.has_bytes(rng())


@pytest.mark.parametrize("payload", [b"abc", b"bad!"])
def test_bad_remote_read_never_calls_consumer(monkeypatch, payload):
    acc = cache()
    acc.records[acc._key(rng())] = hashlib.sha256(b"abcd").hexdigest()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout=b'{"ok":true}\n' + payload),
    )
    called = []
    with pytest.raises(OSError):
        acc.read_to_buffer(rng(), lambda *a: called.append(a))
    assert not called


def test_duplicate_write_is_idempotent_and_source_scoped(monkeypatch):
    store = {}

    def run(*_a, **kwargs):
        header, payload = kwargs["input"].split(b"\n", 1)
        doc = json.loads(header)
        key = repr(doc["key"])
        if doc["op"] == "write":
            store[key] = payload
            payload = b""
        else:
            payload = store[key]
        return SimpleNamespace(stdout=b'{"ok":true}\n' + payload)

    monkeypatch.setattr(subprocess, "run", run)
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


@pytest.mark.parametrize("host", ["-oProxyCommand=bad", "peer;bad", "user@peer bad"])
def test_invalid_host_is_rejected(host):
    with pytest.raises(ValueError):
        SSHCache(host, "/tmp/gdtensor-p2p-test", "/tmp/helper.py", ["python"])

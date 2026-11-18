#!/usr/bin/env python3
"""Check cache coordination assumptions from independent shared-FS clients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

from gdtensor.internal.api import _MAGIC


def atomic_json(path: Path, value: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(value, separators=(",", ":")).encode())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def publication(writer: int, round_no: int) -> dict:
    payload = (f"{writer}:{round_no}:" + "x" * 65536).encode()
    return {
        "writer": writer,
        "round": round_no,
        "payload": payload.decode(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def check_publication(value: dict, clients: int, rounds: int) -> None:
    payload = value["payload"].encode()
    if not (
        type(value["writer"]) is int
        and -1 <= value["writer"] < clients
        and type(value["round"]) is int
        and -1 <= value["round"] < rounds
        and hashlib.sha256(payload).hexdigest() == value["sha256"]
    ):
        raise ValueError("partial or invalid atomic publication")


def wait_files(paths: list[Path], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for _ in range(math.ceil(timeout * 20)):
        if all(path.is_file() for path in paths):
            return
        time.sleep(0.05)
        if time.monotonic() >= deadline:
            break
    missing = [path.name for path in paths if not path.is_file()]
    raise TimeoutError(f"preflight clients missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="job coordination directory")
    parser.add_argument("--job", required=True, help="fresh preflight run identifier")
    parser.add_argument("--chain", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--client", type=int, required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if not (
        2 <= args.clients <= 128
        and 0 <= args.client < args.clients
        and 1 <= args.rounds <= 1000
        and math.isfinite(args.timeout)
        and 0 < args.timeout <= 600
        and args.job
        and "/" not in args.job
    ):
        parser.error("invalid client, round, timeout, or job value")

    import fcntl

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    work = root / f".preflight-{args.job}"
    work.mkdir(exist_ok=True)
    lock_path = work / "lock"
    state_path = work / "state.json"
    publish_path = work / "published.json"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not state_path.exists():
            atomic_json(state_path, {"holder": None, "count": 0})
        if not publish_path.exists():
            atomic_json(publish_path, publication(-1, -1))
        fcntl.flock(lock, fcntl.LOCK_UN)

    expected = {
        "protocol": _MAGIC,
        "chain": args.chain,
        "namespace": args.namespace,
        "directory": str(root),
        "clients": args.clients,
    }
    ready_paths = [work / f"ready-{i}.json" for i in range(args.clients)]
    atomic_json(ready_paths[args.client], expected)
    wait_files(ready_paths, args.timeout)
    for path in ready_paths:
        if read_json(path) != expected:
            raise ValueError(f"preflight configuration mismatch: {path.name}")

    held_path = work / "held.json"
    probe_paths = [work / f"probe-{i}.json" for i in range(args.clients)]
    if args.client == 0:
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            atomic_json(held_path, {"client": 0})
            wait_files(probe_paths[1:], args.timeout)
            acquired = [
                i
                for i in range(1, args.clients)
                if read_json(probe_paths[i])["acquired"]
            ]
            fcntl.flock(lock, fcntl.LOCK_UN)
        if acquired:
            raise RuntimeError(f"shared flock admitted clients {acquired}")
    else:
        wait_files([held_path], args.timeout)
        with lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                acquired = False
            else:
                acquired = True
                fcntl.flock(lock, fcntl.LOCK_UN)
        atomic_json(probe_paths[args.client], {"acquired": acquired})

    with lock_path.open("a+b") as lock:
        for _ in range(args.rounds):
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = read_json(state_path)
                if state["holder"] is not None:
                    raise RuntimeError("shared flock did not exclude another client")
                state["holder"] = args.client
                atomic_json(state_path, state)
                time.sleep(0.01)
                if read_json(state_path)["holder"] != args.client:
                    raise RuntimeError("shared flock lost ownership")
                state = read_json(state_path)
                state["holder"] = None
                state["count"] += 1
                atomic_json(state_path, state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            for _ in range(1_000_000):
                if stop.is_set():
                    return
                check_publication(read_json(publish_path), args.clients, args.rounds)
                time.sleep(0.001)
            raise RuntimeError("atomic publication reader iteration cap")
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for round_no in range(args.rounds):
            if stop.is_set():
                break
            atomic_json(publish_path, publication(args.client, round_no))
    finally:
        stop.set()
        thread.join(args.timeout)
    if thread.is_alive() or errors:
        raise RuntimeError("atomic replace check failed") from (
            errors[0] if errors else None
        )

    done_paths = [work / f"done-{i}.json" for i in range(args.clients)]
    atomic_json(done_paths[args.client], {"ok": True})
    wait_files(done_paths, args.timeout)
    state = read_json(state_path)
    if state != {"holder": None, "count": args.clients * args.rounds}:
        raise RuntimeError(f"flock transaction count mismatch: {state}")
    print(
        json.dumps(
            dict(
                ok=True,
                hostname=socket.gethostname(),
                pid=os.getpid(),
                **expected,
                flock_transactions=state["count"],
                atomic_publications=args.clients * args.rounds,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

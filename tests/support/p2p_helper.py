"""One bounded, authenticated SSH cache request; standard library only."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

MAX_PAYLOAD = 64 * (1 << 20)


def path_of(root, key):
    name = (
        hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
        + ".bin"
    )
    return root / name


def _write(root, request):
    size = request["size"]
    if size < 0 or size > MAX_PAYLOAD:
        raise ValueError("segment size limit")
    data = sys.stdin.buffer.read(size)
    if len(data) != size or sys.stdin.buffer.read(1):
        raise ValueError("incomplete peer write")
    if hashlib.sha256(data).hexdigest() != request["digest"]:
        raise ValueError("peer write checksum mismatch")
    path = path_of(root, request["key"])
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=root, prefix=path.name + ".", delete=False
        ) as f:
            tmp = Path(f.name)
            assert f.write(data) == size
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return {}


def _read_batch(root, keys):
    paths = [path_of(root, key) for key in keys]
    sizes = [path.stat().st_size for path in paths]
    if (
        any(size < 0 or size > MAX_PAYLOAD for size in sizes)
        or sum(sizes) > MAX_PAYLOAD
    ):
        raise ValueError("response size limit")
    chunks = []
    for path, size in zip(paths, sizes):
        with path.open("rb") as f:
            data = f.read(size + 1)
        if len(data) != size:
            raise ValueError("peer cache file changed during read")
        chunks.append(data)
    return {}, b"".join(chunks)


def main():
    root = Path(sys.argv[1]).resolve()
    if not root.name.startswith("gdtensor-p2p-"):
        raise ValueError("dedicated directory required")
    request = json.loads(sys.stdin.buffer.readline(65537))
    root.mkdir(parents=True, exist_ok=True)
    op = request["op"]
    if op == "write":
        return _write(root, request)
    if op == "probe":
        return {"present": [path_of(root, key).is_file() for key in request["key"]]}
    if op == "read_batch":
        return _read_batch(root, request["key"])
    if op == "corrupt":
        with path_of(root, request["key"]).open("r+b") as f:
            f.write(b"bad")
        return {}
    if op == "cleanup":
        shutil.rmtree(root)
        return {}
    raise ValueError("unknown operation")


if __name__ == "__main__":
    try:
        result = main()
        fields, payload = result if isinstance(result, tuple) else (result, b"")
        header = json.dumps(dict(ok=True, size=len(payload), **fields)).encode()
        sys.stdout.buffer.write(header + b"\n" + payload)
    except Exception as exc:
        sys.stdout.buffer.write(
            json.dumps(dict(ok=False, size=0, error=str(exc))).encode() + b"\n"
        )

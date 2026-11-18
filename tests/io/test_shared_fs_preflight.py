"""Independent-process checks for the shared-filesystem preflight entrypoint."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "cache_fs_preflight.py"


def _client(root, client, chain="chain-a"):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--job",
            "test-run",
            "--chain",
            chain,
            "--namespace",
            "namespace-a",
            "--client",
            str(client),
            "--clients",
            "2",
            "--rounds",
            "3",
            "--timeout",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def test_preflight_two_clients_verify_flock_and_atomic_replace(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: _client(tmp_path, i), range(2)))
    for result in results:
        assert result.returncode == 0, result.stderr
        record = json.loads(result.stdout)
        assert record["ok"] and record["flock_transactions"] == 6
        assert record["atomic_publications"] == 6


def test_preflight_rejects_mismatched_chain(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_client, tmp_path, i, f"chain-{i}") for i in range(2)]
        results = [future.result() for future in futures]
    assert all(result.returncode != 0 for result in results)
    assert all("configuration mismatch" in result.stderr for result in results)

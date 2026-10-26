"""Real-process smoke: run ``python -m gdtensor_sim.sim`` end-to-end as a
subprocess for both checkpoint formats and a layout, with a bounded timeout and
a fresh unique work dir.

Calling ``main()`` in-process proves the function works, but not that the
console entry point ``python -m gdtensor_sim.sim`` wires argv, dispatch and the
report step together. This test exercises the real module entry point.

Regression for P2-2: the reporting step used to raise FileNotFoundError on a
layout-derived PayloadIndex file that torch ``dcp_save(no_dist)`` never wrote.
"""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_cli(root, fmt, layout):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    env["GDTENSOR_TORCH_TESTS"] = "1"
    argv = [
        sys.executable,
        "-m",
        "gdtensor_sim.sim",
        "--root",
        str(root),
        "--layers",
        "1",
        "--hidden",
        "16",
        "--ffn",
        "32",
        "--vocab",
        "64",
        "--tp",
        "2",
        "--dp",
        "1",
        "--pp",
        "1",
        "--format",
        fmt,
        "--layout",
        layout,
        "--report",
        str(root) + "_report",
    ]
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=180)


@pytest.mark.parametrize("fmt", ["distcp", "safetensors"])
@pytest.mark.parametrize("layout", ["single", "per_rank"])
def test_cli_module_entrypoint_real_process(tmp_path, fmt, layout):
    root = tmp_path / f"ck_{fmt}_{layout}"
    proc = _run_cli(root, fmt, layout)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    summary = json.loads(proc.stdout)
    assert summary["load_bytes_verified"] is True, summary

    report = json.loads(
        (tmp_path / f"ck_{fmt}_{layout}_report" / "report.json").read_text()
    )
    assert report["load_bytes_verified"] is True

    # P2-2: distcp stats must come from real on-disk metadata files, and the
    # payload/metadata split must separate the .metadata from payload bytes.
    if fmt == "distcp":
        assert (root / ".metadata").exists()
        payload_files = list(root.rglob("*.distcp"))
        assert payload_files, "no real distcp payload file on disk"
        assert (root / ".metadata").stat().st_size > 0
        assert sum(p.stat().st_size for p in payload_files) > 0

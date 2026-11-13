"""Run real torchrun ranks through the format/cache Cartesian product."""

import os
import signal
import subprocess
import sys
from contextlib import suppress
from itertools import product
from pathlib import Path

import pytest

from tests.support.checkpoint import small_schema
from tests.support.schemas import verify_crc

ROOT = Path(__file__).resolve().parents[2]
FORMATS = ("safetensors", "distcp")
CACHE_MODES = ("direct", "one", "two")
PARALLELS = {2: (1, 1, 2), 4: (1, 2, 2)}


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
@pytest.mark.parametrize("world", [2, 4])
def test_gloo_checkpoint_format_cache_matrix(tmp_path, world):
    pytest.importorskip("torch")
    torchrun = Path(sys.executable).with_name("torchrun")
    assert torchrun.is_file(), torchrun
    root = tmp_path / f"world-{world}"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONOPTIMIZE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), env.get("PYTHONPATH")))
    )
    command = [
        str(torchrun),
        "--standalone",
        "--start-method=spawn",
        f"--nproc-per-node={world}",
        "--module",
        "tests.torch.gloo_checkpoint_matrix",
        "--",
        "--root",
        str(root),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"torchrun matrix timed out\n{stdout}\n{stderr}")
    assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert f"MATRIX PASS world={world} cases=12" in stdout

    table, _points, values = small_schema(PARALLELS[world], tensor_count=8)
    for fmt in FORMATS:
        verify_crc(
            root / f"source-{fmt}", fmt, table, PARALLELS[world], "mirror", values
        )
    for src_fmt, dst_fmt, mode in product(FORMATS, FORMATS, CACHE_MODES):
        target = root / f"{src_fmt}-to-{dst_fmt}-{mode}" / "target"
        verify_crc(target, dst_fmt, table, PARALLELS[world], "mirror", values)

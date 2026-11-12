"""Launch the Gloo coordinator example through two torchrun nodes."""

import os
import signal
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import pytest

from gdtensor_sim.coord import GlooCoordinator


ROOT = Path(__file__).resolve().parents[2]


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_node(torchrun, node, port, env):
    command = [
        str(torchrun),
        "--nnodes=2",
        f"--node-rank={node}",
        "--nproc-per-node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=127.0.0.1:{port}",
        f"--rdzv-id=gdtensor-{os.getpid()}",
        "--module",
        "gdtensor_sim.gloo_torchrun",
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
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return 124, stdout + "\nTIMEOUT after 180 seconds\n", stderr
    return process.returncode, stdout, stderr


def _has_port_conflict(results):
    return any(
        "address already in use" in (result[1] + result[2]).lower()
        for result in results
    )


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_gloo_wait_uses_distributed_barrier(monkeypatch):
    dist = pytest.importorskip("torch.distributed")
    called = []
    monkeypatch.setattr(dist, "barrier", lambda: called.append("called"))
    GlooCoordinator().wait()
    assert called == ["called"]


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_gloo_coordinator_two_torchrun_nodes():
    pytest.importorskip("torch")
    torchrun = Path(sys.executable).with_name("torchrun")
    assert torchrun.is_file(), torchrun
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONOPTIMIZE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), env.get("PYTHONPATH")))
    )
    for _ in range(3):
        port = _free_port()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda node: _run_node(torchrun, node, port, env), range(2)
                )
            )
        if not _has_port_conflict(results):
            break
    failures = {
        node: {"returncode": result[0], "stdout": result[1], "stderr": result[2]}
        for node, result in enumerate(results)
        if result[0]
    }
    assert not failures, failures
    output = "".join(result[1] for result in results)
    assert "PASS backend=gloo device=cpu world_size=4" in output

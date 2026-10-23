"""Run behavioral mutations in a temporary checkout, using API tests as detectors.

Usage: python3.12 tests/mutation_check.py
Exit 0 means every mutation caused an assertion failure. Timeouts and collection
errors are reported as harness errors. Each run has a 60-second limit and an
8 MiB output cap. The working checkout is never modified.
"""

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# file, original, replacement, detector
MUTATIONS = [
    (
        "gdtensor/internal/api.py",
        "return replace(view, data_accessor=must_ok(session.init_data_accessor()))",
        "return view",
        "tests/io/test_io_contract.py::test_custom_sessions_and_accessors",
    ),
    (
        "gdtensor/internal/api.py",
        'except BaseException:\n        _cleanup(session, "finish")\n        raise',
        "except BaseException:\n        raise",
        "tests/io/test_io_contract.py::test_io_failure_closes_sessions",
    ),
    (
        "gdtensor/internal/api.py",
        "if read_size != src.size:",
        "if True:",
        "tests/io/test_io_contract.py::test_chunked_callbacks_use_independent_read_and_write_boundaries",
    ),
    (
        "gdtensor/internal/api.py",
        "if written != dst.size:",
        "if False:",
        "tests/io/test_io_contract.py::test_io_rejects_accessor_that_skips_callback",
    ),
    (
        "gdtensor/internal/torch_bridge.py",
        "return list(view.tensor_ids)",
        "return view.tensor_ids",
        "tests/core/test_api.py::test_bridge_plain_values_and_result_lookup",
    ),
    (
        "gdtensor/internal/plan.py",
        "RangeFromComputeView(tid, int(a) * isz, int(b - a) * isz)",
        "RangeFromComputeView(tid, (int(a) + 256) * isz, int(b - a) * isz)",
        "tests/demo/test_walkthrough.py::test_small_arange_crc_and_elastic_load",
    ),
    (
        "gdtensor/internal/dtensor.py",
        "c = -(-n // parts) if n else 0",
        "c = n // parts if n else 0",
        "tests/demo/test_walkthrough.py::test_small_arange_crc_and_elastic_load",
    ),
    (
        "gdtensor/builder.py",
        'return Err(\n        InvalidArgumentError(\n            argument_name="callback", msg="callback must return a Result"\n        )\n    )',
        "return Ok(value)",
        "tests/core/test_view_map.py::test_view_map_requires_result_callbacks",
    ),
    (
        "gdtensor/builder.py",
        "    ConstraintViolationError,\n",
        "    Error, ConstraintViolationError,\n",
        "tests/core/test_api_conformance.py::test_exports_are_exactly_the_design",
    ),
]


def _limit_output():
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))


def _run(snapshot, detector, log):
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
    )
    with log.open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "pytest", "-xq", "-p", "no:cacheprovider", detector],
            cwd=snapshot,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_limit_output,
        )
        try:
            return process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return -1


def _copy_checkout(snapshot):
    for name in ("gdtensor", "gdtensor_sim", "tests", "docs"):
        shutil.copytree(
            ROOT / name,
            snapshot / name,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )


def _check_baseline(snapshot, detector):
    log = snapshot / "baseline.log"
    print(f"BASELINE: {detector}", flush=True)
    if _run(snapshot, detector, log) != 0:
        print(log.read_text()[-6000:])
        raise SystemExit(f"baseline failed: {detector}")


def _apply_mutation(snapshot, index, mutation):
    file, original, replacement, detector = mutation
    path = snapshot / file
    source = path.read_text()
    assert source.count(original) == 1, (file, original)
    path.write_text(source.replace(original, replacement))
    log = snapshot / f"mutation-{index}.log"
    print(f"RUN: {file}: {original.splitlines()[0]} -> {detector}", flush=True)
    try:
        status = _run(snapshot, detector, log)
    finally:
        path.write_text(source)
    verdict = "KILLED" if status == 1 else "SURVIVED" if status == 0 else "ERROR"
    print(f"{verdict}: {file}: {original.splitlines()[0]}", flush=True)
    if status not in (0, 1):
        print(log.read_text()[-6000:])
    return status == 1


def main():
    with tempfile.TemporaryDirectory(prefix="gdtensor-mutations-") as directory:
        snapshot = Path(directory)
        _copy_checkout(snapshot)
        for detector in sorted({mutation[3] for mutation in MUTATIONS}):
            _check_baseline(snapshot, detector)
        killed = sum(
            _apply_mutation(snapshot, index, mutation)
            for index, mutation in enumerate(MUTATIONS)
        )
        print(f"{killed}/{len(MUTATIONS)} mutations killed", flush=True)
        return int(killed != len(MUTATIONS))


if __name__ == "__main__":
    sys.exit(main())

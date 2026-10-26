"""Run behavioral mutations in a temporary checkout, using API tests as detectors.

Usage: python3.12 tests/mutation_check.py [--group notorch|torch] [--allow-empty]
Exit 0 means every mutation caused an assertion failure. Timeouts and collection
errors are reported as harness errors. Each run has a 60-second limit and an
8 MiB output cap.

Safety model (honest): mutations are applied ONLY to a throwaway copy under a
system tempdir; the production tree is opened read-only and never written. The
real guarantee is "no live edit", not "finally always runs": a ``finally`` block
re-hashes the production tree on ordinary errors and KeyboardInterrupt, but a
hard SIGKILL of the harness process cannot run it. That is acceptable because
the mutation target lives in the snapshot, so killing mid-run leaves the
production tree bit-for-bit intact regardless. The start-of-run manifest also
verifies the frozen production baseline (including the known pinned
gdtensor/internal/api.py sha256) BEFORE any work, so an already-tampered tree is
rejected rather than treated as a clean baseline.
"""

import argparse
import hashlib
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Pinned frozen production hash for gdtensor/internal/api.py. If this ever drifts
# at harness start, a residual mutant exists BEFORE our run and we abort loudly.
# NOTE: this is a deliberate freeze guard for the current review, not a source
# contract. Any intentional future edit to gdtensor/internal/api.py must bump this
# constant in lockstep (two-place maintenance) -- otherwise the harness fails.
FROZEN_API_SHA256 = "d8f274eb6b22325e5173daedab5b77e4402ee3aa3e6a1072a72a413e5e5a8d07"


def _production_manifest(root=ROOT):
    """relpath -> sha256 of every production source file (the frozen tree)."""
    paths = [
        path
        for pkg in ("gdtensor", "gdtensor_sim")
        for path in (Path(root) / pkg).rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


# file, original, replacement, detector, [group]
# Group tag: notorch (no-torch checks job) or torch (torch job; importorskip torch).
# A group with zero wired mutations is a coverage lie and FAILS unless --allow-empty.
MUTATIONS = [
    (
        "gdtensor/internal/api.py",
        "return replace(view, data_accessor=must_ok(session.init_data_accessor()))",
        "return view",
        "tests/io/test_io_contract.py::test_custom_sessions_and_accessors",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        'except BaseException:\n        _cleanup(session, "finish")\n        raise',
        "except BaseException:\n        raise",
        "tests/io/test_io_contract.py::test_io_failure_closes_sessions",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "if read_size != src.size:",
        "if True:",
        "tests/io/test_io_contract.py::test_chunked_callbacks_use_independent_read_and_write_boundaries",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "if written != dst.size:",
        "if False:",
        "tests/io/test_io_contract.py::test_io_rejects_accessor_that_skips_callback",
        "notorch",
    ),
    (
        "gdtensor/internal/torch_bridge.py",
        "return list(view.tensor_ids)",
        "return view.tensor_ids",
        "tests/core/test_api.py::test_bridge_plain_values_and_result_lookup",
        "notorch",
    ),
    (
        "gdtensor/internal/plan.py",
        "RangeFromComputeView(tid, int(a) * isz, int(b - a) * isz)",
        "RangeFromComputeView(tid, (int(a) + 256) * isz, int(b - a) * isz)",
        "tests/demo/test_walkthrough.py::test_small_arange_crc_and_elastic_load",
        "notorch",
    ),
    (
        "gdtensor/internal/dtensor.py",
        "c = -(-n // parts) if n else 0",
        "c = n // parts if n else 0",
        "tests/demo/test_walkthrough.py::test_small_arange_crc_and_elastic_load",
        "notorch",
    ),
    (
        "gdtensor/builder.py",
        'return Err(\n        InvalidArgumentError(\n            argument_name="callback", msg="callback must return a Result"\n        )\n    )',
        "return Ok(value)",
        "tests/core/test_view_map.py::test_view_map_requires_result_callbacks",
        "notorch",
    ),
    (
        "gdtensor/builder.py",
        "    ConstraintViolationError,\n",
        "    Error, ConstraintViolationError,\n",
        "tests/core/test_api_conformance.py::test_exports_are_exactly_the_design",
        "notorch",
    ),
    # DCP contiguous guard: a chunk-packed owned range must be contiguous, else
    # get_file_offset must raise InvalidArgument. Inverting the guard (!= -> ==)
    # makes a gapped range look contiguous, so the detector (negative row offset
    # contract) flips from raise to Ok. importorskip("torch") -> torch group.
    (
        "gdtensor_sim/dcp_backend.py",
        "if local[-1] - local[0] + 1 != len(local):",
        "if local[-1] - local[0] + 1 == len(local):",
        "tests/sim/test_audit_dcp_regressions.py::test_col_shard_get_file_offset_must_raise_invalidarg",
        "torch",
    ),
    # File-backed C=1 and ACK detectors drive the public chain_load path.
    # Capacity mutation permits an extra file allocation; ACK mutation forces
    # a second disk fill of a slot that another consumer still needs.
    (
        "gdtensor/internal/api.py",
        "        if block and self.live >= self.cap:\n            return None\n        self.slots[key] = self._new_slot()\n        self.live += 1\n        return True\n\n    def _begin_existing",
        "        if False:\n            return None\n        self.slots[key] = self._new_slot()\n        self.live += 1\n        return True\n\n    def _begin_existing",
        "tests/io/test_cache_mutation_detectors.py::test_cap_guard_allows_one_fill_for_replicate_cohort",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        '        done = all(\n            self._covers(slot["cov"].get(r, set()), iv) for r, iv in required.items()\n        )',
        '        done = any(\n            self._covers(slot["cov"].get(r, set()), iv) for r, iv in required.items()\n        )',
        "tests/io/test_cache_mutation_detectors.py::test_ack_releases_slot_only_after_every_cohort_member",
        "notorch",
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
    file, original, replacement, detector = mutation[:4]
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


def _verify_no_source_residue(before):
    """Fail loudly if the frozen production tree differs from the start snapshot."""
    after = _production_manifest()
    if after != before:
        leaked = sorted(
            p for p in set(before) | set(after) if before.get(p) != after.get(p)
        )
        raise SystemExit(
            "FATAL: production source changed during mutation run (residual mutant): "
            + ", ".join(leaked)
        )


def _assert_frozen_baseline(before):
    """Reject an already-tampered tree as if it were a clean baseline."""
    api_rel = "gdtensor/internal/api.py"
    actual = before.get(api_rel)
    if actual != FROZEN_API_SHA256:
        raise SystemExit(
            f"FATAL: {api_rel} sha256={actual} != pinned {FROZEN_API_SHA256}. "
            "A residual mutant exists BEFORE this run; do not proceed."
        )


def _run_selected(snapshot, selected, group, allow_empty):
    if not selected and not allow_empty:
        raise SystemExit(
            f"FATAL: mutation group '{group}' has 0 wired mutations. "
            "An empty group exiting 0 would be a false coverage claim; "
            "wire at least one real mutant (or use --allow-empty for scaffolding)."
        )
    print(f"MUTATION GROUP: {group} ({len(selected)} mutations)", flush=True)
    for detector in sorted({m[3] for m in selected}):
        _check_baseline(snapshot, detector)
    killed = sum(
        _apply_mutation(snapshot, index, mutation)
        for index, mutation in enumerate(selected)
    )
    print(f"{killed}/{len(selected)} mutations killed", flush=True)
    return int(killed != len(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=("notorch", "torch"),
        default="notorch",
        help="notorch: no-torch detectors (checks job); torch: torch detectors",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="permit a group with zero wired mutations (basic scaffolding only; "
        "an advanced CI group must carry real mutations)",
    )
    args = parser.parse_args()

    before = _production_manifest()
    _assert_frozen_baseline(before)
    try:
        with tempfile.TemporaryDirectory(prefix="gdtensor-mutations-") as directory:
            snapshot = Path(directory)
            _copy_checkout(snapshot)
            selected = [
                m
                for m in MUTATIONS
                if (m[4] if len(m) > 4 else "notorch") == args.group
            ]
            return _run_selected(snapshot, selected, args.group, args.allow_empty)
    finally:
        # Always run the residue check, even on a failed/interrupted run.
        _verify_no_source_residue(before)


if __name__ == "__main__":
    sys.exit(main())

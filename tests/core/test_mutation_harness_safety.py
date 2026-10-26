"""Safety guarantees for tests/mutation_check.py (owned by the CI/mutation owner).

These tests do NOT touch real production state. They build a fake "production"
file in a pytest tmp dir, run a child process that mutates only a throwaway
snapshot copy, then kill the child mid-run (SIGKILL / SIGTERM). The invariant
under test is: even when the harness child dies so hard that no cleanup finally
runs, the original production file is bit-for-bit untouched, because mutations
are applied to the snapshot copy only.
"""

import hashlib
import signal
import subprocess
import sys
from pathlib import Path

import tests.mutation_check as harness

GUARD_LINE = "if block and self.live >= self.cap:\n"

# Child: copy src into snap/, mutate the copy, then suicide by the given signal.
# It deliberately never restores the copy and never writes back to src.
_CHILD = r"""
import shutil, sys, time
from pathlib import Path
src = Path(sys.argv[1]); snap = Path(sys.argv[2]); sig = int(sys.argv[3])
snap.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, snap / "api.py")
text = (snap / "api.py").read_text()
(snap / "api.py").write_text(text.replace(GUARD, "if False:"))
import os
os.killpg(os.getpid(), sig)
time.sleep(10)
""".replace("GUARD", repr(GUARD_LINE))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fake_production(root: Path) -> Path:
    pkg = root / "gdtensor" / "internal"
    pkg.mkdir(parents=True)
    api = pkg / "api.py"
    api.write_text("live = 0\n" + GUARD_LINE + "cap = 1\n")
    return api


def _run_child_killed(api: Path, snap: Path, sig: int) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(api), str(snap), str(sig)],
        start_new_session=True,
    )
    proc.wait(timeout=30)


def test_production_untouched_when_child_sigkilled(tmp_path):
    api = _make_fake_production(tmp_path / "root")
    before = _sha(api)
    snap = tmp_path / "snap"
    _run_child_killed(api, snap, signal.SIGKILL)  # no finally runs on SIGKILL
    assert _sha(api) == before, "production file was mutated by a snapshot run"
    # And the mutation really did land in the throwaway copy (snapshot isolation).
    assert "if False:" in (snap / "api.py").read_text()
    assert GUARD_LINE in api.read_text()


def test_production_untouched_when_child_sigterm(tmp_path):
    api = _make_fake_production(tmp_path / "root")
    before = _sha(api)
    snap = tmp_path / "snap"
    _run_child_killed(api, snap, signal.SIGTERM)
    assert _sha(api) == before, "production file was mutated by a snapshot run"


def test_harness_manifest_is_path_hash_and_frozen():
    manifest = harness._production_manifest()
    assert isinstance(manifest, dict)
    assert all(isinstance(v, str) and len(v) == 64 for v in manifest.values())
    # The pinned frozen api.py hash must match the real tree right now.
    assert manifest["gdtensor/internal/api.py"] == harness.FROZEN_API_SHA256

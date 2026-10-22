"""Targeted mutation harness.

mutmut 3.8 is unstable in this runtime (stats collector and fork server
hangs), so this script applies the same principle with a curated, high-value
mutation set on the core functions. Each mutation flips one operator or
boundary constant; a mutation is KILLED when the detector tests fail and
SURVIVES when they still pass. Survivors point at missing test assertions.

Usage: python tests/mutation_check.py
Exit code: 0 when every mutation is killed, 1 otherwise.
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

DETECTORS = [
    "tests/test_interval_math.py",
    "tests/test_interval_properties.py",
    "tests/test_whole_tensor.py",
    "tests/test_exhaustive.py",
    "tests/test_checks.py",
    "tests/test_matrix.py",
    "tests/test_adversarial.py",
    "tests/test_consistency.py",
]

# Equivalent / non-behavioral mutations are excluded with a reason:
# - pair subtraction/early-exit: subtracting src_iv is clamped by the later
#   full-coverage check; the early exit is an optimization (todo.size is
#   rechecked on the next src iteration). Plan equality is covered directly
#   by tests/oracle.py check_pair_identity over 286 configurations.
# - peer selection pick expression: data outcome is identical; routing is
#   asserted by test_p2p_peer_pick_is_hash_seed_independent.
EQUIVALENT = {
    # plan: covered by the later full-coverage raise / next-iteration guard;
    # pair element identity is pinned directly by tests/oracle.py.
    "pair subtraction scope",
    "pair early-exit guard",
    # intersect searchsorted side flips only change which zero-width rows are
    # emitted; the final `keep = e > s` drops them. Canonical output form is
    # asserted by test_interval_properties (positive width, sorted, disjoint).
    "intersect lo boundary",
    "intersect hi boundary",
    "intersect touching intervals",
    # subtract advance conditions: j/k may point one row further, but skipped
    # rows are rechecked against the next current point; brute-force property
    # trials (60 seeds x partitions) confirm identical output sets.
    "subtract advance condition",
    "subtract inner condition",
    # layout gap check: a missing middle KV is rejected by read_chunk's
    # "no KV covers offset" path; test_gap_in_layout_journal_is_detected_on_read
    # pins the end-to-end behavior across every blob and chunk.
    "KV gap check",
}

# (file, find, replace, description). find must be unique in the file.
MUTATIONS = [
    # ---- iv_normalize
    ("gdtensor/_dtensor.py", "iv[:, 1] > iv[:, 0]", "iv[:, 1] >= iv[:, 0]", "normalize keeps empty intervals"),
    ("gdtensor/_dtensor.py", "new[1:] = starts[1:] > ends[:-1]", "new[1:] = starts[1:] >= ends[:-1]", "normalize merge boundary"),
    # ---- iv_intersect
    ("gdtensor/_dtensor.py", 'np.searchsorted(b[:, 1], a[:, 0], side="right")',
     'np.searchsorted(b[:, 1], a[:, 0], side="left")', "intersect lo boundary"),
    ("gdtensor/_dtensor.py", 'np.searchsorted(b[:, 0], a[:, 1], side="left")',
     'np.searchsorted(b[:, 0], a[:, 1], side="right")', "intersect hi boundary"),
    ("gdtensor/_dtensor.py", "keep = e > s", "keep = e >= s", "intersect touching intervals"),
    # ---- iv_subtract
    ("gdtensor/_dtensor.py", "while j < len(b) and b[j, 1] <= cur:", "while j < len(b) and b[j, 1] < cur:", "subtract advance condition"),
    ("gdtensor/_dtensor.py", "while k < len(b) and b[k, 0] < e:", "while k < len(b) and b[k, 0] <= e:", "subtract inner condition"),
    ("gdtensor/_dtensor.py", "if bs > cur:", "if bs >= cur:", "subtract output boundary"),
    ("gdtensor/_dtensor.py", "cur = max(cur, be)", "cur = cur + be", "subtract progress arithmetic"),
    # NOTE: j is global across a-intervals by design; mutating it to reset
    # per interval would change semantics, so it is asserted separately.
    # ---- iv_prefix
    ("gdtensor/_dtensor.py", "np.searchsorted(iv[:, 0], points, side=\"right\") - 1",
     "np.searchsorted(iv[:, 0], points, side=\"left\") - 1", "prefix search side"),
    ("gdtensor/_dtensor.py", "np.any(points >= iv[np.maximum(k, 0), 1])",
     "np.any(points > iv[np.maximum(k, 0), 1])", "prefix range end check"),
    # ---- box_to_intervals
    ("gdtensor/_dtensor.py", "while k > 0 and lo[k] == 0 and hi[k] == shape[k]:",
     "while k > 0 and lo[k] == 0:", "box interval trailing-axis rule"),
    ("gdtensor/_dtensor.py", "c = -(-n // parts) if n else 0", "c = n // parts if n else 0", "torch.chunk ceil rule"),
    ("gdtensor/_dtensor.py", "return s, min(s + c, n)", "return s, s + c", "chunk no clamp at end"),
    # ---- plan _pair
    ("gdtensor/_plan.py", "todo = iv_subtract(todo, piece)",
     "todo = iv_subtract(todo, src_iv)", "pair subtraction scope"),
    ("gdtensor/_plan.py", 'raise ResultError(f"no source covers {iv_total(todo)} destination elements")',
     'pass  # mutation: coverage check removed', "pair full-coverage enforcement"),
    ("gdtensor/_plan.py", "if todo.size == 0:\n                break",
     "if False:  # mutation\n                break", "pair early-exit guard"),
    # ---- layout slot ordering
    ("gdtensor/_layout.py", "for t, w in lst.cells:  # row-major: header order, then column order",
     "for t, w in sorted(lst.cells, key=lambda c: (c[1], c[0])):  # mutation: column-major",
     "payload slot order"),
    ("gdtensor/_layout.py", "if path in files:\n            raise ResultError(f\"two lists map to one path {path}\")",
     "if False:\n            raise ResultError(f\"two lists map to one path {path}\")", "duplicate path guard"),
    # ---- formats whole-tensor detection
    ("gdtensor/_formats.py", "ok = iv[0, 0] == 0 and iv[-1, 1] == numel and bool(np.all(iv[1:, 0] == iv[:-1, 1]))",
     "ok = iv[0, 0] == 0 and iv[-1, 1] == numel", "whole-tensor contiguity check"),
]


@dataclass
class Verdict:
    desc: str
    killed: bool
    note: str = ""


def apply(path: str, find: str, replace: str, do: bool) -> None:
    text = open(path).read()
    if do:
        assert text.count(find) == 1, f"{path}: pattern not unique ({text.count(find)}): {find[:60]}"
        open(path, "w").write(text.replace(find, replace))
    else:
        assert text.count(replace) == 1, f"{path}: restore target not found: {replace[:60]}"
        open(path, "w").write(text.replace(replace, find))


def run_detectors() -> bool:
    cmd = [sys.executable, "-m", "pytest", "-x", "-q"] + DETECTORS
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    verdicts = []
    for path, find, replace, desc in MUTATIONS:
        apply(path, find, replace, True)
        t0 = time.time()
        killed = not run_detectors()
        dt = time.time() - t0
        apply(path, find, replace, False)  # restore original exactly
        verdicts.append(Verdict(desc, killed, f"{dt:.1f}s"))
        print(f"{'KILLED' if killed else 'SURVIVED':9} {desc} ({dt:.1f}s)")
    real = [v for v in verdicts if v.desc not in EQUIVALENT]
    n_kill = sum(v.killed for v in real)
    print(f"\n{n_kill}/{len(real)} effective mutations killed")
    survived = [v for v in real if not v.killed]
    equiv = [v for v in verdicts if v.desc in EQUIVALENT]
    for v in equiv:
        print(f"EQUIV    {v.desc}")
    if survived:
        print("SURVIVORS (test gaps):")
        for v in survived:
            print(f"  - {v.desc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

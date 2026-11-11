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
FROZEN_API_SHA256 = "5ef2460bbcda6731ecaa120a20ef4c75ef8240e1711360af03b8d7511e4f0b61"


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
        '_copy(plan, _session_view(storage_view, session), compute_view)\n    except BaseException:\n        _cleanup(session, "finish")\n        raise',
        "_copy(plan, _session_view(storage_view, session), compute_view)\n    except BaseException:\n        raise",
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
    # File-backed byte-budget and ACK detectors drive the public chain_load
    # path. Budget mutation permits an extra file allocation; ACK mutation
    # releases a segment after the first consumer, which forces a second disk
    # fill for the consumer that still needs it. The budget detector asserts
    # the resident peak never exceeds the advertised stat bound (cell budget
    # is summed over every tensor of a (step, slot) cell).
    (
        "gdtensor/internal/api.py",
        "        return self.resident + size <= self.budget",
        "        return False",
        "tests/io/test_cache_propagation.py::test_resident_cache_bytes_never_exceed_stat[8-2]",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        '        if not slot["owed"]:\n            self.release(key)',
        "        if True:\n            self.release(key)",
        "tests/io/test_cache_mutation_detectors.py::test_ack_releases_slot_only_after_every_cohort_member",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "        return self.rank in self.plan.fill_ranks[(layer, node, key)]",
        "        return True",
        "tests/io/test_colocated_cluster_sim.py::test_missing_local_filler_returns_error_without_remote_write",
        "notorch",
    ),
    # Randomized co-located stress: each mutant breaks one oracle axis (fill
    # owner, helper node, storage offset, generation refill, stat traffic).
    # The detector is the first seed that exposes the mutant on digest.
    (
        "gdtensor/internal/propagation.py",
        "eligible = [r for r in local if r in consumers]",
        "eligible = list(consumers)",
        "tests/io/test_random_cache_stress.py::test_random_cache_strategy_is_correct[0]",
        "notorch",
    ),
    (
        "gdtensor/internal/propagation.py",
        "helper = (local or plan.ranks)[0]",
        "helper = plan.ranks[0]",
        "tests/io/test_random_cache_stress.py::test_random_cache_strategy_is_correct[0]",
        "notorch",
    ),
    (
        "gdtensor/internal/plan.py",
        "RangeFromStorageView(path, cpo + int(l) * isz, int(b - a) * isz)",
        "RangeFromStorageView(path, cpo + int(a) * isz, int(b - a) * isz)",
        "tests/io/test_random_cache_stress.py::test_random_cache_strategy_is_correct[0]",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "if any(s.key in self.slots or s.key in self.completed for s in segs):",
        "if any(s.key in self.slots for s in segs):",
        "tests/io/test_random_cache_stress.py",
        "notorch",
    ),
    (
        "gdtensor/internal/propagation.py",
        "        _drain_traffic(rank // plan.k, pieces, inbound, outbound)\n",
        "        pass\n",
        "tests/io/test_random_cache_stress.py::test_random_cache_strategy_is_correct[0]",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        'if not slot["owed"]:',
        "if True:",
        "tests/io/test_cache_reuse_layout.py::test_two_layer_physical_reuse_waits_for_slow_consumer",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "rng = RangeFromComputeView(key[0], off, size)",
        "rng = RangeFromComputeView(key[0], off // 4, size)",
        "tests/io/test_cache_reuse_layout.py::test_two_layer_f32_uneven_column_shards",
        "notorch",
    ),
    (
        "gdtensor_sim/file_cache.py",
        "if len(raw) != size or hashlib.sha256(raw).digest() != digest:",
        "if len(raw) != size:",
        "tests/io/test_file_cache_correctness.py::test_corrupt_cache_fails_before_delivery[checksum]",
        "notorch",
    ),
    (
        "gdtensor_sim/file_cache.py",
        'raise OSError("cache short write made no progress")',
        "return",
        "tests/io/test_file_cache_correctness.py::test_zero_short_write_never_publishes",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "if identity != identity_of():",
        "if False:",
        "tests/io/test_file_cache_correctness.py::test_source_change_during_load_returns_error[0]",
        "notorch",
    ),
    (
        "gdtensor/internal/plan.py",
        "hit[1] == signature",
        "True",
        "tests/io/test_file_cache_correctness.py::test_public_plan_refreshes_inplace_layout",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "result = _primitive_subclass(value, int, memo)",
        "result = _type_name(value), repr(value)",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_distinguishes_primitive_subclass_values",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "return _type_name(value), primitive, state",
        "return _type_name(value), primitive, None",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_distinguishes_primitive_subclass_state",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "primitive = _hashlib.sha256(_pickle.dumps(value, protocol=5)).digest()\n"
        "        result = _type_name(value), primitive",
        "result = _type_name(value), value",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_supports_large_integers",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "return tuple(sorted((_canonical(v, memo) for v in values), key=repr))",
        "return tuple(_canonical(v, memo) for v in values)",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_normalizes_unordered_subclass_reductions",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "    if isinstance(reduced, str):\n"
        "        _pickle.dumps(value, protocol=5)\n"
        "        return _canonical(reduced, memo)",
        "    if isinstance(reduced, str):\n        return _canonical(reduced, memo)",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_rejects_unserializable_reductions",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "    elif isinstance(value, (list, tuple)) and type(value) not in (list, tuple):\n"
        "        result = _type_name(value), _reduction(value, memo)",
        '    elif hasattr(value, "__dataclass_fields__"):\n'
        "        result = _type_name(value), _canonical(value.__dict__, memo)\n"
        "    elif isinstance(value, (list, tuple)) and type(value) not in (list, tuple):\n"
        "        result = _type_name(value), _reduction(value, memo)",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_includes_dataclass_container_reduction",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "    elif isinstance(value, (list, tuple)) and type(value) not in (list, tuple):\n"
        "        result = _type_name(value), _reduction(value, memo)",
        "    elif isinstance(value, (list, tuple)) and type(value) not in (list, tuple):\n"
        "        result = _type_name(value), _canonical(list(value), memo)",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_includes_dataclass_container_reduction",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "            value.__qualname__,\n"
        "            _pickle.dumps(value, protocol=5),",
        "            value.__qualname__,\n            None,",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_rejects_unserializable_reductions",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "if not isinstance(reduced, tuple) or not 2 <= len(reduced) <= 6:",
        "if False:",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_rejects_unserializable_reductions",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "if not callable(reduced[0]) or not isinstance(reduced[1], tuple):",
        "if False:",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_rejects_unserializable_reductions",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "    else:\n"
        "        result = _type_name(value), _reduction(value, memo)\n"
        "    # Fixed-size child digests keep shared DAGs linear without encoding aliases.",
        "    else:\n"
        "        result = _type_name(value), _pickle.dumps(value, protocol=5)\n"
        "    # Fixed-size child digests keep shared DAGs linear without encoding aliases.",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_ignores_generic_object_aliases",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "    cached = memo.get(key)",
        "    cached = None",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_bounds_shared_subtree_expansion",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        '    result = _hashlib.sha256(repr(result).encode("utf-8")).digest()',
        '    result = repr(result).encode("utf-8")',
        "tests/io/test_cache_thread_coordination.py::test_content_digest_bounds_shared_subtree_expansion",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "    except RecursionError:",
        "    except TypeError:",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_supports_cycles_without_retaining_aliases",
        "notorch",
    ),
    (
        "gdtensor/access.py",
        "        return repr((_type_name(value), _pickle.dumps(value, protocol=5))).encode()",
        "        return repr(_type_name(value)).encode()",
        "tests/io/test_cache_thread_coordination.py::test_content_digest_supports_cycles_without_retaining_aliases",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "while hot and len(self.plan.ranks) > 1 and layer < self.last:",
        "while hot and layer < self.last:",
        "tests/io/test_cache_thread_coordination.py::test_all_fit_single_rank_terminal_hit_ignores_cold_parent",
        "notorch",
    ),
    (
        "gdtensor_sim/file_cache.py",
        "namespace = ga._content_digest(identity)[:20]",
        'namespace = "all-sources"',
        "tests/io/test_file_cache_correctness.py::test_source_file_identity_prevents_stale_hit[same_path]",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "hot = len(self._classify_drains(owned, layer, seal=False)) == len(owned)",
        "hot = True",
        "tests/io/test_cache_thread_coordination.py::test_all_fit_hot_owner_adopts_parent_for_cold_peer",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "if completed != seen:",
        "if False:",
        "tests/io/test_cache_thread_coordination.py::test_all_fit_wait_deadline_tracks_any_completed_segment",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "        nodes = dict.fromkeys(node for node, _key in probe_keys)\n"
        "        try:\n"
        "            self._inflight(1)",
        "        nodes = dict.fromkeys(node for node, _key in probe_keys)\n"
        "        try:\n"
        "            pass",
        "tests/io/test_cache_thread_coordination.py::test_all_fit_probe_marks_inflight",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "self._block(version, deadline, pending) if not moved else None",
        "self._block(self._version_checked(), deadline, pending) if not moved else None",
        "tests/io/test_cache_thread_coordination.py::test_loop_preserves_step_version_for_wait",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "        self._drain_hot(done)\n"
        "        for piece in done:\n"
        "            pending.remove(piece)",
        "        for piece in done:\n"
        "            self._drain(piece)\n"
        "            pending.remove(piece)",
        "tests/io/test_file_cache_correctness.py::test_non_all_fit_hot_uses_batch_reads",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "sealed = self._seal_drains(segs, layer, reserve=seal)",
        "sealed = self._seal_drains(segs, layer, reserve=True)",
        "tests/io/test_cache_thread_coordination.py::test_all_fit_hot_generation_never_reads_parent",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "self.error = replace(as_error(exc))",
        "self.error = as_error(exc)",
        "tests/io/test_cache_owner_lifecycle.py::test_probe_failure_reaches_peer_without_sharing_error_object",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "raise Fail(replace(self.error))",
        "raise Fail(self.error)",
        "tests/io/test_cache_owner_lifecycle.py::test_probe_failure_reaches_peer_without_sharing_error_object",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "        with self.t.cond:\n"
        "            sealed = self._seal_drains(segs, layer, reserve=seal)",
        "        self._inflight(-1)\n"
        "        with self.t.cond:\n"
        "            sealed = self._seal_drains(segs, layer, reserve=seal)\n"
        "        self._inflight(1)",
        "tests/io/test_cache_thread_coordination.py::test_all_fit_probe_stays_inflight_until_completion_is_published",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "        except BaseException as exc:\n"
        "            _error_cleanup(exc, self._fill_failed, layer, node, seg, exc)\n"
        "            raise",
        "        except BaseException as exc:\n"
        "            self._inflight(-1)\n"
        "            self._fail(layer, node, exc, seg.key)\n"
        "            raise",
        "tests/io/test_cache_owner_lifecycle.py::test_interrupted_error_publication_keeps_owner[fill]",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        'namespaces.append(("object", id(accessor)) if namespace is None else namespace)',
        "namespaces.append(namespace)",
        "tests/io/test_cache_thread_coordination.py::test_independent_memory_caches_do_not_share_coordination",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "    if view_id in users:",
        "    if True:",
        "tests/io/test_cache_thread_coordination.py::test_coordination_table_lives_until_the_last_view_is_gone",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "entry = _FINGERPRINTS.get(key)",
        "entry = None",
        "tests/io/test_cache_thread_coordination.py::test_slots_share_active_fingerprint_and_refresh_afterwards",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "    _FINGERPRINT_LOCK = threading.Lock()",
        "    _FINGERPRINT_LOCK = _FINGERPRINT_LOCK",
        "tests/io/test_cache_owner_lifecycle.py::test_fork_with_locked_fingerprint_state_starts_clean",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "    _FINGERPRINTS.clear()",
        "    _FINGERPRINTS.copy()",
        "tests/io/test_cache_owner_lifecycle.py::test_fork_with_pending_fingerprint_starts_clean",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "    return _content_digest(payload)[:20]",
        "    return _content_digest([id(table) for table in tables])[:20]",
        "tests/io/test_cache_thread_coordination.py::test_chain_fingerprint_ignores_equal_object_aliases",
        "notorch",
    ),
    (
        "gdtensor_sim/file_cache.py",
        "namespace = ga._content_digest(identity)[:20]",
        "namespace = repr(identity)",
        "tests/io/test_file_cache_correctness.py::test_cache_identity_content_prevents_repr_collision",
        "notorch",
    ),
    (
        "gdtensor_sim/file_cache.py",
        "digest_raw = os.pread(fd, record_size, self.bases[tid] + record_off)",
        "digest_raw = raw",
        "tests/io/test_file_cache_correctness.py::test_overlapping_write_refreshes_combined_integrity_record",
        "notorch",
    ),
    (
        "gdtensor_sim/file_cache.py",
        "records = sorted(self.integrity_keys.get((namespace, node, tid), ()))",
        "records = []",
        "tests/io/test_file_cache_correctness.py::test_adjacent_integrity_records_cover_one_read",
        "notorch",
    ),
    (
        "tests/support/cluster_sim.py",
        "            missing = (\n"
        "                self.identities.get((target, tid)) != identity\n"
        "                or not valid[off : off + size].all()\n"
        "            )",
        "            missing = not valid[off : off + size].all()",
        "tests/io/test_file_cache_correctness.py::test_memory_cache_read_rejects_a_different_source",
        "notorch",
    ),
    (
        "gdtensor_sim/ssh_cache.py",
        'namespace = ga._content_digest(getattr(rng, "source_identity", None))',
        'namespace = repr(getattr(rng, "source_identity", None))',
        "tests/io/test_ssh_cache.py::test_source_keys_use_identity_content_not_repr",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "        except BaseException as exc:\n"
        "            _error_cleanup(exc, self._fail_nodes, layer, nodes, exc, True)\n"
        "            raise",
        "        except BaseException as exc:\n"
        "            self._inflight(-1)\n"
        "            for node in nodes:\n"
        "                self._fail(layer, node, exc, None)\n"
        "            raise",
        "tests/io/test_cache_owner_lifecycle.py::test_interrupted_error_publication_keeps_owner[probe]",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "        except BaseException as exc:\n"
        "            _error_cleanup(exc, self._drain_failed, node, exc)\n"
        "            raise",
        "        except BaseException as exc:\n"
        "            self._inflight(-1)\n"
        "            self._fail(self.last, node, exc, None)\n"
        "            raise",
        "tests/io/test_cache_owner_lifecycle.py::test_interrupted_error_publication_keeps_owner[drain]",
        "notorch",
    ),
    (
        "gdtensor/internal/api.py",
        "        except BaseException as exc:\n"
        "            _error_cleanup(exc, self._fail, self.last, node, exc, None)\n"
        "            raise",
        "        except BaseException as exc:\n            raise",
        "tests/io/test_cache_owner_lifecycle.py::test_drain_entry_commit_failure_publishes_error",
        "notorch",
    ),
]


def _limit_output():
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))


def _budget():
    """Per-detector wall-clock limit. The stress file is the slowest detector,
    so leave head-room over its plain run time; a hang still trips this."""
    return int(os.environ.get("GDTENSOR_MUTATION_TIMEOUT", "120"))


def _run(snapshot, detector, log):
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
    )
    # Detector ids name default stress seeds; a caller seed range must not hide them.
    env.pop("GDTENSOR_STRESS_BASE", None)
    env.pop("GDTENSOR_STRESS_SEEDS", None)
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
            return process.wait(timeout=_budget())
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

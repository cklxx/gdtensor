"""Compare serial and concurrent IO without numeric merging.

One rank writes the baseline. Multiple ranks write the same arange tensors
under another layout. Independent file reconstruction must match the serial
baseline exactly. Consolidating the parallel files to the baseline layout
must also produce identical file bytes, including headers and metadata.

Read the serial files concurrently; read the parallel files with one rank,
then with multiple ranks sequentially and concurrently. Every result must
match the same baseline. A swapped-shard negative test checks this comparison
actually detects an incorrect assembly.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import zlib

import numpy as np
import pytest

import gdtensor.io as gio
import gdtensor.tag as gt
from tensor_oracle import ArrayFabric
from test_checkpoint_merge_crc import _fingerprint, _small_schema
from test_data_correctness import _regions, _storage, _storage_points


CASES = [
    ((1, 1, 3), 16, "rank_multi", "mirror"),
    ((2, 2, 2), 24, "single", "hf"),
    ((2, 2, 2), 24, "hf", "hf"),
    ((2, 2, 2), 24, "rank_multi", "mirror"),
    ((2, 2, 4), 32, "rank_multi", "mirror"),
    ((2, 2, 4), 32, "per_stage", "hf"),
]
ONE = (1, 1, 1)


def _ok(result):
    assert type(result) is gt.Ok, result


def _concurrent(world, action):
    ready = Barrier(world, timeout=10)
    order = np.random.default_rng(2026).permutation(world).tolist()

    def run(rank):
        ready.wait()
        return action(rank)

    with ThreadPoolExecutor(max_workers=world) as pool:
        results = list(pool.map(run, order))
    for result in results:
        _ok(result)


def _decode(root, fmt, table, parallel, mode, templates):
    points = _storage_points(table, parallel, mode)
    decoded = {tid: np.empty_like(value) for tid, value in templates.items()}
    counts = {
        tid: np.zeros(value.size, dtype=np.int64) for tid, value in templates.items()
    }
    for tid, slot, _, data, _, _ in _regions(root, fmt):
        indices = np.arange(templates[tid].size) if slot is None else points[tid, slot]
        assert len(data) == len(indices) * templates[tid].itemsize
        decoded[tid].reshape(-1)[indices] = np.frombuffer(
            data, dtype=templates[tid].dtype
        )
        counts[tid][indices] += 1
    for coverage in counts.values():
        np.testing.assert_array_equal(coverage, np.ones_like(coverage))
    return decoded


def _assert_values(actual, expected, label):
    assert set(actual) == set(expected)
    for tid, reference in expected.items():
        assert (
            actual[tid].dtype == reference.dtype
            and actual[tid].shape == reference.shape
        )
        assert actual[tid].tobytes() == reference.tobytes(), (
            f"{label}: tensor={tid} assembled bytes differ"
        )
    print(
        f"[CRC+ARANGE][SERIAL_PARALLEL][{label}][PASS] tensors={len(expected)} exact_bytes=equal",
        flush=True,
    )


def _assert_rank_buffers(actual, reference):
    for rank, tensors in reference.buffers.items():
        for tid, data in tensors.items():
            np.testing.assert_array_equal(
                actual.buffers[rank][tid], data, err_msg=f"rank={rank} tensor={tid}"
            )


def _read_one(table, points, storage):
    loaded = ArrayFabric(table, points)
    _ok(gio.rank_load(0, loaded.view(0), storage))
    return loaded


def _assert_single_rank(loaded, expected, label):
    actual = {
        tid: loaded.buffers[0][tid].view(value.dtype).reshape(value.shape)
        for tid, value in expected.items()
    }
    _assert_values(actual, expected, label)


def _read_schedules(table, points, storage, baseline, world):
    sequential, concurrent = ArrayFabric(table, points), ArrayFabric(table, points)
    for rank in reversed(range(world)):
        _ok(gio.rank_load(rank, sequential.view(rank), storage))
    _concurrent(world, lambda rank: gio.rank_load(rank, concurrent.view(rank), storage))
    sequential.assert_values(baseline)
    concurrent.assert_values(baseline)
    _assert_rank_buffers(concurrent, sequential)
    print(
        f"[CRC+ARANGE][SERIAL_PARALLEL][READ_SCHEDULES][PASS] ranks={world} sequential=reverse concurrent=shuffled local_bytes=equal",
        flush=True,
    )
    return concurrent


def _canonicalize(root, fmt, source, serial_table):
    destination = _storage(root, serial_table, ONE, "single", "hf", fmt, save=True)
    _ok(gio.rank_reshard(0, source, destination))


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
@pytest.mark.parametrize("parallel,tensor_count,kind,mode", CASES)
def test_serial_parallel_io_matches_exactly(
    tmp_path, fmt, parallel, tensor_count, kind, mode
):
    serial_table, serial_points, values = _small_schema(ONE, tensor_count)
    table, points, _ = _small_schema(parallel, tensor_count, column=True)
    world = int(np.prod(parallel))
    print(
        f"[CRC+ARANGE][SERIAL_PARALLEL][START] serial_ranks=1 parallel_ranks={world} tensors={tensor_count} layout={kind}/{mode} format={fmt} numeric_merge=none",
        flush=True,
    )
    serial_root, parallel_root = tmp_path / "serial", tmp_path / "parallel"
    serial_source, source = (
        ArrayFabric(serial_table, serial_points, values),
        ArrayFabric(table, points, values),
    )
    serial = _storage(serial_root, serial_table, ONE, "single", "hf", fmt, save=True)
    distributed = _storage(parallel_root, table, parallel, kind, mode, fmt, save=True)
    _ok(gio.rank_save(0, serial_source.view(0), serial))
    _concurrent(world, lambda rank: gio.rank_save(rank, source.view(rank), distributed))
    serial_bytes, parallel_bytes = (
        _fingerprint(serial_root),
        _fingerprint(parallel_root),
    )
    baseline = _decode(serial_root, fmt, serial_table, ONE, "hf", values)
    decoded = _decode(parallel_root, fmt, table, parallel, mode, values)
    _assert_values(baseline, values, "SERIAL_BASELINE")
    _assert_values(decoded, baseline, "PARALLEL_ASSEMBLY")
    _show_crc(decoded, baseline)
    serial_loaded = _read_one(serial_table, serial_points, serial)
    parallel_loaded = _read_one(serial_table, serial_points, distributed)
    _assert_single_rank(serial_loaded, baseline, "SERIAL_WRITE_SERIAL_READ")
    _assert_single_rank(parallel_loaded, baseline, "PARALLEL_WRITE_SERIAL_READ")
    from_serial = _read_schedules(table, points, serial, baseline, world)
    from_parallel = _read_schedules(table, points, distributed, baseline, world)
    _assert_rank_buffers(from_parallel, from_serial)
    canonical = tmp_path / "parallel_consolidated"
    _canonicalize(canonical, fmt, distributed, serial_table)
    assert _fingerprint(canonical) == serial_bytes, (
        "parallel consolidated files differ from serial files"
    )
    assert (
        _fingerprint(serial_root) == serial_bytes
        and _fingerprint(parallel_root) == parallel_bytes
    )
    serial_source.assert_values(values)
    source.assert_values(values)
    print(
        "[CRC+ARANGE][SERIAL_PARALLEL][PASS] reconstructed_tensors=byte_identical canonical_files=byte_identical all_read_schedules=equal source_files=unchanged",
        flush=True,
    )


def _show_crc(actual, baseline):
    for tid in baseline:
        serial_crc = zlib.crc32(baseline[tid].tobytes())
        parallel_crc = zlib.crc32(actual[tid].tobytes())
        assert serial_crc == parallel_crc
        if tid in list(baseline)[:3]:
            print(
                f"[CRC+ARANGE][SERIAL_PARALLEL][TENSOR] tensor={tid} crc_serial={serial_crc:08x} crc_parallel={parallel_crc:08x} serial_first12={baseline[tid].ravel()[:12].tolist()} parallel_first12={actual[tid].ravel()[:12].tolist()}",
                flush=True,
            )


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_serial_baseline_detects_swapped_parallel_shards(tmp_path, fmt):
    serial_table, serial_points, values = _small_schema(ONE, 16)
    table, points, _ = _small_schema((1, 1, 3), 16, column=True)
    serial = _storage(
        tmp_path / "serial", serial_table, ONE, "single", "hf", fmt, save=True
    )
    parallel_root = tmp_path / "parallel"
    distributed = _storage(
        parallel_root, table, (1, 1, 3), "rank_multi", "mirror", fmt, save=True
    )
    _ok(
        gio.rank_save(
            0, ArrayFabric(serial_table, serial_points, values).view(0), serial
        )
    )
    source = ArrayFabric(table, points, values)
    _concurrent(3, lambda rank: gio.rank_save(rank, source.view(rank), distributed))
    regions = [
        region for region in _regions(parallel_root, fmt) if region[0] == "matrix"
    ]
    left, right = regions[:2]
    assert left[4] != right[4] and len(left[3]) == len(right[3]) and left[3] != right[3]
    _swap_regions(left, right)
    baseline = _read_one(serial_table, serial_points, serial)
    broken = _read_one(serial_table, serial_points, distributed)
    dtype = values["matrix"].dtype
    print(
        f"[CRC+ARANGE][SERIAL_PARALLEL][SWAPPED_SHARDS] tensor=matrix serial_first12={baseline.buffers[0]['matrix'].view(dtype)[:12].tolist()} parallel_first12={broken.buffers[0]['matrix'].view(dtype)[:12].tolist()}",
        flush=True,
    )
    with pytest.raises(AssertionError):
        _assert_rank_buffers(broken, baseline)
    print(
        "[CRC+ARANGE][SERIAL_PARALLEL][EXPECTED_MISMATCH] swapped_parallel_shards=detected baseline=serial_read_bytes",
        flush=True,
    )


def _swap_regions(left, right):
    for region, data in ((left, right[3]), (right, left[3])):
        with region[4].open("r+b") as file:
            file.seek(region[5])
            assert file.write(data) == len(data)

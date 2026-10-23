"""Check a complete checkpoint pipeline through documented IO APIs.

Inputs are tiny arange tensors A, A+3 and A+9. Each checkpoint is split
across rank files, repacked into the other format and loaded on a different
rank topology. The caller averages the loaded local buffers. Equal weights
must produce A+4; weights (1, 2, 3) must produce A+5.5, cast to the input dtype.
The merged tensors are saved to multiple files, consolidated into one file
under a third topology and loaded again.

At every storage boundary, an independent dense oracle checks payload CRC,
exact bytes and one stored copy per global element. Read sources must remain
byte-for-byte unchanged. A separate negative test swaps equal-size regions
between files: both CRC verification and loaded-value verification must fail.

Ranks here are threads. Numeric merging belongs to the caller; file merging
uses rank_reshard and preserves values. README describes the steps and logs.
"""

import zlib

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.compat import rank_reshard
from tests.support.checkpoint import file_bytes, small_schema
from tests.support.oracle import ArrayFabric
from tests.support.schemas import payload_regions, run_ranks, storage_view, verify_crc

CASES = [
    ((1, 1, 3), (1, 1, 5), (1, 1, 2), 16),
    ((2, 2, 2), (2, 1, 3), (1, 2, 2), 24),
    ((2, 2, 4), (2, 2, 3), (2, 1, 4), 32),
]


def _check_files(label, root, fmt, table, parallel, mode, values, *, multiple=True):
    verify_crc(root, fmt, table, parallel, mode, values)
    payloads = {}
    tensors = {}
    for tid, _, _, data, path, _ in payload_regions(root, fmt):
        payloads.setdefault(path.name, bytearray()).extend(data)
        tensors.setdefault(path.name, set()).add(tid)
    assert len(payloads) > 1 if multiple else len(payloads) == 1
    assert any(len(ids) > 1 for ids in tensors.values())
    crcs = {name: f"{zlib.crc32(data):08x}" for name, data in sorted(payloads.items())}
    print(
        f"[CRC+ARANGE][{label}][PASS] checks=payload_crc,exact_bytes,coverage_once format={fmt} ranks={int(np.prod(parallel))} tensors={len(values)} files={len(payloads)} CRC={crcs}",
        flush=True,
    )


def _save_checkpoint(root, fmt, parallel, table, points, values):
    print(
        f"[CRC+ARANGE][SAVE_CHECKPOINT][START] checkpoint={root.name} format={fmt}",
        flush=True,
    )
    source = ArrayFabric(table, points, values)
    storage = storage_view(
        root, table, parallel, "rank_multi", "mirror", fmt, save=True
    )
    _assert_rank_multifile(storage)
    run_ranks(
        int(np.prod(parallel)),
        lambda rank: gio.rank_save(rank, source.view(rank), storage),
    )
    _check_files("SAVE_CHECKPOINT", root, fmt, table, parallel, "mirror", values)
    source.assert_values(values)
    return storage_view(root, table, parallel, "rank_multi", "mirror", fmt)


def _assert_rank_multifile(view):
    by_rank, by_tensor = {}, {}
    for pos in view.physical_layout.tensor_entries:
        result = gb.file_path_of(view.physical_layout, pos)
        assert type(result) is gb.Ok, result
        by_rank.setdefault(pos.rank, set()).add(result.value)
        by_tensor.setdefault(pos.tensor_id, set()).add(result.value)
    assert any(len(paths) > 1 for paths in by_rank.values())
    assert any(len(paths) > 1 for paths in by_tensor.values())


def _load_repacked(root, fmt, source, table, points, parallel, values):
    print(
        f"[CRC+ARANGE][REPACK_CHECKPOINT][START] output={root.name} format={fmt}",
        flush=True,
    )
    destination = storage_view(root, table, parallel, "hf", "hf", fmt, save=True)
    run_ranks(
        int(np.prod(parallel)), lambda rank: rank_reshard(rank, source, destination)
    )
    _check_files("REPACK_CHECKPOINT", root, fmt, table, parallel, "hf", values)
    loaded = ArrayFabric(table, points)
    read_view = storage_view(root, table, parallel, "hf", "hf", fmt)
    before_load = file_bytes(root)
    print(
        f"[CRC+ARANGE][LOAD_REPACKED][START] ranks={int(np.prod(parallel))}", flush=True
    )
    run_ranks(
        int(np.prod(parallel)),
        lambda rank: gio.rank_load(rank, loaded.view(rank), read_view),
    )
    loaded.assert_values(values)
    assert file_bytes(root) == before_load, "load modified the repacked checkpoint"
    print(
        "[CRC+ARANGE][LOAD_REPACKED][PASS] checks=every_rank_every_element,source_files_unchanged",
        flush=True,
    )
    return loaded


def _accumulate(sums, fabric, values, weight):
    for rank, tensors in fabric.buffers.items():
        for tid, data in tensors.items():
            local = data.view(values[tid].dtype).astype(np.float64)
            key = rank, tid
            sums[key] = sums.get(key, np.zeros_like(local)) + local * weight


def _merge_local_buffers(fabric, sums, values, denominator):
    for rank, tensors in fabric.buffers.items():
        for tid, data in tensors.items():
            merged = (sums[rank, tid] / denominator).astype(values[tid].dtype)
            data[:] = merged.view(np.uint8)


def _write_merge(root, fmt, fabric, parallel, values):
    print(f"[CRC+ARANGE][SAVE_MERGED][START] format={fmt}", flush=True)
    storage = storage_view(
        root, fabric.table, parallel, "rank_multi", "mirror", fmt, save=True
    )
    _assert_rank_multifile(storage)
    run_ranks(
        int(np.prod(parallel)),
        lambda rank: gio.rank_save(rank, fabric.view(rank), storage),
    )
    _check_files("SAVE_MERGED", root, fmt, fabric.table, parallel, "mirror", values)
    return storage_view(root, fabric.table, parallel, "rank_multi", "mirror", fmt)


def _consolidate(root, fmt, source, parallel, table, points, values):
    print(
        f"[CRC+ARANGE][CONSOLIDATE_FILES][START] format={fmt} target_files=1",
        flush=True,
    )
    target = storage_view(root, table, parallel, "single", "hf", fmt, save=True)
    run_ranks(int(np.prod(parallel)), lambda rank: rank_reshard(rank, source, target))
    _check_files(
        "CONSOLIDATE_FILES",
        root,
        fmt,
        table,
        parallel,
        "hf",
        values,
        multiple=False,
    )
    loaded = ArrayFabric(table, points)
    read_view = storage_view(root, table, parallel, "single", "hf", fmt)
    before_load = file_bytes(root)
    print(
        f"[CRC+ARANGE][LOAD_CONSOLIDATED][START] ranks={int(np.prod(parallel))}",
        flush=True,
    )
    run_ranks(
        int(np.prod(parallel)),
        lambda rank: gio.rank_load(rank, loaded.view(rank), read_view),
    )
    loaded.assert_values(values)
    assert file_bytes(root) == before_load, "load modified the consolidated checkpoint"
    print(
        "[CRC+ARANGE][LOAD_CONSOLIDATED][PASS] checks=every_rank_every_element,source_files_unchanged",
        flush=True,
    )
    return loaded


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
@pytest.mark.parametrize(
    "weights,mean_shift", [((1, 1, 1), 4.0), ((1, 2, 3), 5.5)], ids=["mean", "weighted"]
)
@pytest.mark.parametrize("source_p,merge_p,final_p,tensor_count", CASES)
def test_multifile_checkpoint_merge_crc(
    tmp_path, fmt, weights, mean_shift, source_p, merge_p, final_p, tensor_count
):
    source_table, source_points, base = small_schema(source_p, tensor_count)
    merge_table, merge_points, _ = small_schema(merge_p, tensor_count, column=True)
    final_table, final_points, _ = small_schema(final_p, tensor_count)
    formats = [fmt, "distcp" if fmt == "safetensors" else "safetensors"]
    # Use the global closed form independently of loaded bytes and local sums.
    expected = {
        tid: (data.astype(np.float64) + mean_shift).astype(data.dtype)
        for tid, data in base.items()
    }
    sums, originals = {}, {}
    first = next(iter(base))
    print(
        f"[CRC+ARANGE][CASE][START] rank_mode=threads worlds={tuple(int(np.prod(p)) for p in (source_p, merge_p, final_p))} tensors={tensor_count} shifts=(0,3,9) weights={weights}",
        flush=True,
    )
    print(
        f"[CRC+ARANGE][CASE][INPUT] tensor={first} dtype={base[first].dtype} base_arange_first12={base[first].ravel()[:12].tolist()}",
        flush=True,
    )
    for epoch, shift in enumerate((0, 3, 9)):
        values = {tid: (data + shift).astype(data.dtype) for tid, data in base.items()}
        print(
            f"[CRC+ARANGE][CHECKPOINT_INPUT] checkpoint={epoch} shift={shift} tensor={first} values_first12={values[first].ravel()[:12].tolist()}",
            flush=True,
        )
        root = tmp_path / f"checkpoint-{epoch}"
        source = _save_checkpoint(
            root, formats[epoch % 2], source_p, source_table, source_points, values
        )
        originals[root] = file_bytes(root)
        repacked = tmp_path / f"repacked-{epoch}"
        loaded = _load_repacked(
            repacked,
            formats[(epoch + 1) % 2],
            source,
            merge_table,
            merge_points,
            merge_p,
            values,
        )
        _accumulate(sums, loaded, values, weights[epoch])
    merged = ArrayFabric(merge_table, merge_points)
    print(
        f"[CRC+ARANGE][TENSOR_MEAN][START] inputs=A,A+3,A+9 weights={weights}",
        flush=True,
    )
    # The test caller owns this arithmetic; IO only moves the resulting bytes.
    _merge_local_buffers(merged, sums, base, sum(weights))
    merged.assert_values(expected)
    print(
        f"[CRC+ARANGE][TENSOR_MEAN][PASS] expected=A+{mean_shift} cast=original_dtype checks=every_rank_every_element",
        flush=True,
    )
    merged_root = tmp_path / "merged"
    saved = _write_merge(merged_root, fmt, merged, merge_p, expected)
    originals[merged_root] = file_bytes(merged_root)
    final = _consolidate(
        tmp_path / "consolidated",
        formats[1],
        saved,
        final_p,
        final_table,
        final_points,
        expected,
    )
    _assert_originals_unchanged(originals)
    print(
        f"[CRC+ARANGE][SOURCE_FILES][PASS] unchanged_directories={len(originals)}",
        flush=True,
    )
    print(
        f"[CRC+ARANGE][CASE][PASS] tensor={first} dtype={expected[first].dtype} merged_expected_first12={expected[first].ravel()[:12].tolist()} loaded_rank0_first12={final.buffers[0][first].view(expected[first].dtype)[:12].tolist()}",
        flush=True,
    )


def _assert_originals_unchanged(originals):
    for root, contents in originals.items():
        assert file_bytes(root) == contents, f"source checkpoint changed: {root.name}"


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_multifile_crc_detects_equal_size_payload_swap(tmp_path, fmt):
    print(
        f"[CRC+ARANGE][CORRUPTION][START] format={fmt} expected=crc_and_arange_detection",
        flush=True,
    )
    parallel = (1, 1, 3)
    table, points, values = small_schema(parallel, 16)
    read_view = _save_checkpoint(tmp_path, fmt, parallel, table, points, values)
    regions = list(payload_regions(tmp_path, fmt))
    left, right = next(
        (left, right)
        for i, left in enumerate(regions)
        for right in regions[i + 1 :]
        if left[4] != right[4] and len(left[3]) == len(right[3]) and left[3] != right[3]
    )
    assert zlib.crc32(left[3]) != zlib.crc32(right[3])
    _replace_payload(left, right[3])
    _replace_payload(right, left[3])
    with pytest.raises(AssertionError):
        verify_crc(tmp_path, fmt, table, parallel, "mirror", values)
    loaded = ArrayFabric(table, points)
    run_ranks(
        int(np.prod(parallel)),
        lambda rank: gio.rank_load(rank, loaded.view(rank), read_view),
    )
    with pytest.raises(AssertionError):
        loaded.assert_values(values)
    print(
        f"[CRC+ARANGE][CORRUPTION][PASS] crc=detected arange=detected equal_size_swap files={left[4].name},{right[4].name} bytes={len(left[3])}; independent CRC and loaded values both failed as expected",
        flush=True,
    )


def _replace_payload(region, data):
    with region[4].open("r+b") as file:
        file.seek(region[5])
        assert file.write(data) == len(data)

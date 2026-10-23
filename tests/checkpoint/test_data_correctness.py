"""Public IO verified with dense arange arrays and independent file decoding."""

from __future__ import annotations

import numpy as np
import pytest

import gdtensor.io as gio
from tests.support.checkpoint import corrupt_distcp_payload, repack_via_compute
from tests.support.oracle import ArrayFabric
from tests.support.schemas import (
    DST,
    LAYOUTS,
    SRC,
    payload_regions,
    run_ranks,
    storage_view,
    tensor_schema,
    verify_crc,
)


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
@pytest.mark.parametrize("kind,mode", LAYOUTS)
def test_io_arange_crc_elastic_and_reshard(tmp_path, fmt, kind, mode):
    ct, points, values = tensor_schema(SRC)
    target, target_points, _ = tensor_schema(DST, column=True)
    source = ArrayFabric(ct, points, values)
    root = tmp_path / "source"
    sv = storage_view(root, ct, SRC, kind, mode, fmt, save=True)
    run_ranks(int(np.prod(SRC)), lambda r: gio.rank_save(r, source.view(r), sv))
    verify_crc(root, fmt, ct, SRC, mode, values)

    loaded = ArrayFabric(target, target_points)
    read_view = storage_view(root, ct, SRC, kind, mode, fmt)
    run_ranks(int(np.prod(DST)), lambda r: gio.rank_load(r, loaded.view(r), read_view))
    loaded.assert_values(values)
    source.assert_values(values)

    for branch in ("compute", "storage"):
        destination = ArrayFabric(target, target_points)
        if branch == "compute":
            # The design cuts compute->compute reshard (§419); the live fabric
            # reaches the destination topology through the two-phase boundary:
            # persist it to a scratch checkpoint (rank_save), then elastically
            # load it (rank_load).
            scratch = storage_view(
                tmp_path / "scratch", ct, SRC, kind, mode, fmt, save=True
            )
            run_ranks(
                int(np.prod(SRC)),
                lambda r: gio.rank_save(r, source.view(r), scratch),
            )
            run_ranks(
                int(np.prod(DST)),
                lambda r: gio.rank_load(
                    r,
                    destination.view(r),
                    storage_view(tmp_path / "scratch", ct, SRC, kind, mode, fmt),
                ),
            )
        else:
            run_ranks(
                int(np.prod(DST)),
                lambda r: gio.rank_load(r, destination.view(r), read_view),
            )
        destination.assert_values(values)

    # Exercise compute->storage and the two-phase storage->storage repack,
    # then load both results back onto the source topology.
    for branch in ("compute", "storage"):
        out = tmp_path / branch
        output_fmt = "distcp" if fmt == "safetensors" else "safetensors"
        dst = storage_view(
            out, target, DST, "per_rank", "mirror", output_fmt, save=True
        )
        if branch == "compute":
            run_ranks(
                int(np.prod(DST)), lambda r: gio.rank_save(r, loaded.view(r), dst)
            )
        else:
            repack_via_compute(
                int(np.prod(DST)), read_view, dst, target, target_points, values
            )
        verify_crc(out, output_fmt, target, DST, "mirror", values)
        back = ArrayFabric(ct, points)
        output = storage_view(out, target, DST, "per_rank", "mirror", output_fmt)
        run_ranks(int(np.prod(SRC)), lambda r: gio.rank_load(r, back.view(r), output))
        back.assert_values(values)


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_io_crc_detects_payload_corruption(tmp_path, fmt):
    ct, points, values = tensor_schema(SRC)
    source = ArrayFabric(ct, points, values)
    sv = storage_view(tmp_path, ct, SRC, "single", "hf", fmt, save=True)
    run_ranks(int(np.prod(SRC)), lambda r: gio.rank_save(r, source.view(r), sv))
    verify_crc(tmp_path, fmt, ct, SRC, "hf", values)
    if fmt == "safetensors":
        tid, slot, info, data, path, start = next(payload_regions(tmp_path, fmt))
        with path.open("r+b") as f:
            f.seek(start)
            f.write(bytes([data[0] ^ 1]))
    else:
        corrupt_distcp_payload(tmp_path)
    with pytest.raises(AssertionError):
        verify_crc(tmp_path, fmt, ct, SRC, "hf", values)
    # Raw IO does not promise checksum validation. The independent oracle must fail.
    loaded = ArrayFabric(ct, points)
    read_view = storage_view(tmp_path, ct, SRC, "single", "hf", fmt)
    run_ranks(int(np.prod(SRC)), lambda r: gio.rank_load(r, loaded.view(r), read_view))
    with pytest.raises(AssertionError):
        loaded.assert_values(values)
    print(
        f"[CRC+ARANGE][CORRUPTION][PASS] format={fmt} fault=flipped_payload_byte crc=detected arange=detected",
        flush=True,
    )


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
@pytest.mark.parametrize(
    "src_p,dst_p,tensor_count,kind,mode",
    [
        ((1, 1, 3), (1, 1, 2), 16, "single", "hf"),
        ((2, 2, 4), (2, 2, 3), 32, "rank_multi", "mirror"),
        ((2, 4, 4), (2, 2, 4), 64, "per_rank", "mirror"),
    ],
)
def test_many_ranks_many_tensors(tmp_path, fmt, src_p, dst_p, tensor_count, kind, mode):
    ct, points, values = tensor_schema(src_p, tensor_count=tensor_count)
    target, target_points, _ = tensor_schema(
        dst_p, column=True, tensor_count=tensor_count
    )
    source = ArrayFabric(ct, points, values)
    saved = storage_view(tmp_path, ct, src_p, kind, mode, fmt, save=True)
    run_ranks(int(np.prod(src_p)), lambda r: gio.rank_save(r, source.view(r), saved))
    verify_crc(tmp_path, fmt, ct, src_p, mode, values)
    loaded = ArrayFabric(target, target_points)
    read_view = storage_view(tmp_path, ct, src_p, kind, mode, fmt)
    run_ranks(
        int(np.prod(dst_p)), lambda r: gio.rank_load(r, loaded.view(r), read_view)
    )
    loaded.assert_values(values)
    source.assert_values(values)

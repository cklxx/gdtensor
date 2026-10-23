"""Test-only moves expressed with the design's public IO API.

The latest design (5.2 / section 419) removed ``rank_reshard``:
* storage -> compute: ``chain_load`` with a ``DirectDTensorsReadChain``
  (equivalent to a single ``rank_load``);
* compute -> storage: ``rank_save``;
* storage -> storage and compute -> compute are unsupported and skipped.
"""

from __future__ import annotations


def _direct_load(rank: int, src_view, dst_view):
    import gdtensor.builder as gb
    import gdtensor.io as gio

    chain = gb.DirectDTensorsReadChain(src=src_view, dst=dst_view)
    return gio.chain_load(rank, chain)


def rank_reshard(rank: int, src_view, dst_view):
    import pytest
    import gdtensor.builder as gb

    if type(src_view) is gb.StorageDTensorsView:
        if type(dst_view) is gb.ComputeDTensorsView:
            return _direct_load(rank, src_view, dst_view)
        pytest.skip("storage->storage copy is not an IO API in the design")
    if type(dst_view) is gb.StorageDTensorsView:
        import gdtensor.io as gio

        return gio.rank_save(rank, src_view, dst_view)
    pytest.skip("compute->compute reshard is not an IO API in the design")


__all__ = ["rank_reshard"]

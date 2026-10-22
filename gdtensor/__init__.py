"""gdtensor: generic distributed tensors.

Public surface (design appendix C.3):

* top level: get_tensor_ids, get_gd_tensor, make_gd_tensor, get_d_tensor,
  reset_d_tensor, Result, Error
* gdtensor.io: rank_load / rank_save / rank_reshard
* gdtensor.builder: view types, meshes, layouts, errors, d_tensors_view_map
* gdtensor.session: StorageSaveSession / StorageLoadSession
* gdtensor.access: DTensorDataAccessor
* gdtensor.tag: Ok / Err / Shard / Replicate / Partial / WorldSize ...
"""
from __future__ import annotations

from .builder import (
    Error, get_d_tensor, get_gd_tensor, get_tensor_ids, make_gd_tensor, reset_d_tensor,
)
from .tag import Err, Ok
def _result_alias():
    import typing

    class _ResultAlias:
        # Result[T] = tag.Ok[T] | tag.Err[Error] (design C.2).
        def __getitem__(self, item):
            return typing.Union[Ok[item], Err[Error]]

    return _ResultAlias()


Result = _result_alias()
del _result_alias

__all__ = [
    "get_tensor_ids", "get_gd_tensor", "make_gd_tensor", "get_d_tensor",
    "reset_d_tensor", "Result", "Error", "Ok", "Err",
]

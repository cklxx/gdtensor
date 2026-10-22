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

from .internal.result import Error
from .internal.torch_bridge import (
    get_d_tensor,
    get_gd_tensor,
    get_tensor_ids,
    make_gd_tensor,
    reset_d_tensor,
)
from .tag import Err as _Err, Ok as _Ok

type Result[T] = _Ok[T] | _Err[Error]

__all__ = [
    "get_tensor_ids",
    "get_gd_tensor",
    "make_gd_tensor",
    "get_d_tensor",
    "reset_d_tensor",
    "Result",
    "Error",
]

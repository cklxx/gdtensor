"""gdtensor: generic distributed tensors, lite runtime.

Public surface follows DTensorsView design appendix C.3:

* top level: compute/torch bridge (get_tensor_ids, get_gd_tensor,
  make_gd_tensor, get_d_tensor, reset_d_tensor), Result, Error
* gdtensor.io: rank_load / rank_save / rank_reshard
* gdtensor.builder: view types, meshes, layouts, errors, d_tensors_view_map
* gdtensor.session: StorageSaveSession / StorageLoadSession / DTensorDataAccessor
* gdtensor.tag: Ok / Err / Shard / Replicate / Partial / WorldSize ...
"""
from __future__ import annotations

from .builder import (
    get_d_tensor,
    get_gd_tensor,
    get_tensor_ids,
    make_gd_tensor,
    reset_d_tensor,
)
from .tag import Err, Error, Ok, Result

__all__ = [
    "get_tensor_ids", "get_gd_tensor", "make_gd_tensor", "get_d_tensor",
    "reset_d_tensor", "Result", "Error", "Ok", "Err",
]

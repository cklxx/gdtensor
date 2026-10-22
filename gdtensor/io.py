"""gdtensor.io: thin IO executors (spec B.1.1, namespace C.3)."""

from __future__ import annotations

from .internal.api import (
    d_tensors_view_rank_load as rank_load,
    d_tensors_view_rank_reshard as rank_reshard,
    d_tensors_view_rank_save as rank_save,
)

__all__ = ["rank_load", "rank_save", "rank_reshard"]

"""gdtensor.io: thin IO executors (spec B.1.1, namespace C.3)."""

from __future__ import annotations

from .internal.api import (
    d_tensors_view_rank_chain_load as chain_load,
    d_tensors_view_rank_load as rank_load,
    d_tensors_view_rank_save as rank_save,
)
from .internal.chain import (
    get_stat_inbound_traffic_of_chain_load as chain_stat_inbound_traffic,
    get_stat_mem_overhead_of_chain_load as chain_stat_mem_overhead,
    get_stat_outbound_traffic_of_chain_load as chain_stat_outbound_traffic,
)

__all__ = [
    "rank_load",
    "rank_save",
    "chain_load",
    "chain_stat_mem_overhead",
    "chain_stat_outbound_traffic",
    "chain_stat_inbound_traffic",
]

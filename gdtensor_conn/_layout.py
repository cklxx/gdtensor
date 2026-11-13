"""Simulation file offsets derived from public layout metadata."""

from dataclasses import dataclass, field

import numpy as np

from gdtensor.builder import (
    TensorComponentPos,
    file_path_of,
    global_tensor_meta_of,
    leader_tensor_component_pos_of,
)
from gdtensor.builder import Err
from ._geometry import iv_total, nonempty_columns, tensor_ids_of
from ._result import Fail, invalid


def pos_key(pos):
    return pos.tensor_id, pos.rank


@dataclass
class SlotPayload:
    pos: TensorComponentPos
    tensor_order: int
    intervals: np.ndarray
    itemsize: int
    contiguous_payload_offset: int

    @property
    def nbytes(self):
        return iv_total(self.intervals) * self.itemsize


@dataclass
class FilePayload:
    path: str
    leader: TensorComponentPos
    slots: list = field(default_factory=list)
    size: int = 0

    @property
    def payload_size(self):
        return self.size


@dataclass
class PayloadIndex:
    files: dict
    slots: dict


def _file_payload(layout, pos, ids, files, leaders):
    path = file_path_of(layout, pos)
    if type(path) is Err:
        raise Fail(path.value)
    group = layout.tensor_entries[pos]
    # The leader is constant per physical list; recomputing it per slot is
    # quadratic in tensors x ranks.
    leader = leaders.get(id(group))
    if leader is None:
        leader = leader_tensor_component_pos_of(group)
        leaders[id(group)] = leader
    fp = files.setdefault(path.value, FilePayload(path.value, leader))
    if fp.leader != leader:
        raise invalid("file_paths", "two physical lists use one path")
    return fp


def build_payload_index(table, layout):
    files, slots, leaders = {}, {}, {}
    ids = tensor_ids_of(table)
    for order, nd in enumerate(table.named_d_tensors):
        for rank, intervals in nonempty_columns(nd).items():
            pos = TensorComponentPos(nd.global_tensor_id, rank)
            fp = _file_payload(layout, pos, ids, files, leaders)
            slot = SlotPayload(
                pos,
                order,
                intervals,
                global_tensor_meta_of(nd).itemsize,
                fp.payload_size,
            )
            fp.slots.append(slot)
            fp.size += slot.nbytes
            slots[pos] = fp.path, slot
    return PayloadIndex(files, slots)

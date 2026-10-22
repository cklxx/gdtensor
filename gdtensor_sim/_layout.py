"""Simulation file offsets derived from public layout metadata."""

from dataclasses import dataclass, field

import numpy as np

from gdtensor.builder import (
    TensorComponentPos,
    file_path_of,
    global_tensor_meta_of,
    leader_tensor_component_pos_of,
)
from gdtensor.tag import Err
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

    @property
    def payload_size(self):
        return sum(slot.nbytes for slot in self.slots)


@dataclass
class PayloadIndex:
    files: dict
    slots: dict


def _file_payload(layout, pos, ids, files):
    path = file_path_of(layout, pos)
    if type(path) is Err:
        raise Fail(path.value)
    group = layout.tensor_entries[pos]
    if group.tensor_ids != ids:
        raise invalid("tensor_ids", "physical list row labels differ from table order")
    leader = leader_tensor_component_pos_of(group)
    fp = files.setdefault(path.value, FilePayload(path.value, leader))
    if fp.leader != leader:
        raise invalid("file_paths", "two physical lists use one path")
    return fp


def build_payload_index(table, layout):
    files, slots = {}, {}
    ids = tensor_ids_of(table)
    for order, nd in enumerate(table.named_d_tensors):
        for rank, intervals in nonempty_columns(nd).items():
            pos = TensorComponentPos(nd.global_tensor_id, rank)
            fp = _file_payload(layout, pos, ids, files)
            slot = SlotPayload(
                pos,
                order,
                intervals,
                global_tensor_meta_of(nd).itemsize,
                fp.payload_size,
            )
            fp.slots.append(slot)
            slots[pos] = fp.path, slot
    return PayloadIndex(files, slots)

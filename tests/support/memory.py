"""In-memory implementations of the public session and accessor contracts."""

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.session as gs
import gdtensor.tag as gt


class Memory(ga.DTensorDataAccessor):
    def __init__(self, data, error=None, length=None):
        self.data = np.array(data, dtype=np.uint8)
        self.error, self.length = error, length

    def _call(self, rng, callback):
        if self.error is not None:
            return gb.Err(self.error)
        offset = (
            rng.global_tensor_offset
            if hasattr(rng, "global_tensor_offset")
            else rng.contiguous_payload_offset
        )
        data = self.data[offset : offset + rng.size]
        return callback(
            data.ctypes.data, data.nbytes if self.length is None else self.length
        )

    def read_to_buffer(self, global_range, read_to_buffer):
        return self._call(global_range, read_to_buffer)

    def write_from_buffer(self, global_range, write_from_buffer):
        return self._call(global_range, write_from_buffer)


class Save(gs.StorageSaveSession):
    def __init__(self, memory, events, init_error=None, commit_error=None):
        self.memory, self.events = memory, events
        self.init_error, self.commit_error = init_error, commit_error

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        self.events.append("init_save")
        return gb.Err(self.init_error) if self.init_error else gb.Ok(self.memory)

    def commit(self):
        self.events.append("commit")
        return gb.Err(self.commit_error) if self.commit_error else gb.Ok(None)

    def abort(self):
        self.events.append("abort")
        return gb.Ok(None)


class Load(gs.StorageLoadSession):
    def __init__(self, memory, events, init_error=None, finish_error=None):
        self.memory, self.events = memory, events
        self.init_error, self.finish_error = init_error, finish_error

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        self.events.append("init_load")
        return gb.Err(self.init_error) if self.init_error else gb.Ok(self.memory)

    def finish(self):
        self.events.append("finish")
        return gb.Err(self.finish_error) if self.finish_error else gb.Ok(None)


def compute_view(memory):
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([8], "U8", 1),
        [gb.Replicate()],
        gb.ArrangedMesh(gt.WorldSize(1)),
    )
    return gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), memory
    )


def storage_view(
    memory, events, *, init_error=None, commit_error=None, finish_error=None
):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    dt = gb.NaiveDTensor(gb.GlobalTensorMeta([8], "U8", 1), [gb.Replicate()] * 2, mesh)
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(1))
            )
        ]
    )
    pos = gb.TensorComponentPos("w", 0)
    layout = gb.PhysicalLayoutTable("safetensors", {pos: group}, {pos: "memory"})

    def begin_save(view, rank):
        events.append("begin_save")
        return gb.Ok(Save(memory, events, init_error, commit_error))

    def begin_load(view, rank):
        events.append("begin_load")
        return gb.Ok(Load(memory, events, init_error, finish_error))

    # Executors must use the accessor returned by the session.
    decoy = Memory([0] * 8, gb.GDTensorIOError(operation="decoy", msg="wrong accessor"))
    return gb.StorageDTensorsView(
        ["w"],
        gb.DTensorTable([gb.NamedDTensor("w", dt)]),
        decoy,
        layout,
        begin_save,
        begin_load,
    )

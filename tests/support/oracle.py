"""Test memory and element ownership, independent of internal interval code."""

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.tag as gt


def slice_points(shape, p, placements, stage, rank, *, storage=False, order=0):
    """Slice a dense index array with torch.chunk's ceiling-sized chunks."""
    pp, dp, tp = p
    coords = (rank // (dp * tp), rank // tp % dp, rank % tp)
    if coords[0] != stage:
        return np.array([], dtype=np.int64)
    points = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    for size, coord, placement in zip((1, dp, tp), (0, *coords[1:]), placements):
        if type(placement) is gt.Replicate and storage and coord != order % size:
            return np.array([], dtype=np.int64)
        if type(placement) is gt.Replicate:
            continue
        axis = placement.value
        chunk = (points.shape[axis] + size - 1) // size
        index = [slice(None)] * points.ndim
        index[axis] = slice(coord * chunk, (coord + 1) * chunk)
        points = points[tuple(index)]
    return points.ravel()


class ArrayFabric:
    """Each rank stores explicit global indices and separate local bytes."""

    def __init__(self, table, points, values=None):
        self.table = table
        self.points = points
        self.buffers = {}
        self.positions = {}
        for rank, tensors in points.items():
            self._allocate_rank(rank, tensors, values)

    def _allocate_rank(self, rank, tensors, values):
        self.buffers[rank] = {}
        for nd in self.table.named_d_tensors:
            self._allocate_tensor(rank, nd, tensors[nd.global_tensor_id], values)

    def _allocate_tensor(self, rank, nd, indices, values):
        tid = nd.global_tensor_id
        itemsize = gb.global_tensor_meta_of(nd).itemsize
        self.positions[rank, tid] = {int(e): i for i, e in enumerate(indices)}
        buf = np.full(len(indices) * itemsize, 0xA5, dtype=np.uint8)
        if values is not None:
            buf[:] = values[tid].reshape(-1)[indices].copy().view(np.uint8).ravel()
        self.buffers[rank][tid] = buf

    def view(self, rank, allow_remote_reads=False):
        ids = [nd.global_tensor_id for nd in self.table.named_d_tensors]
        return gb.ComputeDTensorsView(
            ids, self.table, _ArrayAccessor(self, rank, allow_remote_reads)
        )

    def assert_values(self, values):
        for rank, tensors in self.points.items():
            self._assert_rank(rank, tensors, values)

    def _assert_rank(self, rank, tensors, values):
        for tid, indices in tensors.items():
            want = values[tid].reshape(-1)[indices].copy().view(np.uint8).ravel()
            np.testing.assert_array_equal(
                self.buffers[rank][tid], want, err_msg=f"rank={rank}, tensor={tid}"
            )


class _ArrayAccessor(ga.DTensorDataAccessor):
    def __init__(self, fabric, rank, allow_remote_reads):
        self.fabric, self.rank = fabric, rank
        self.allow_remote_reads = allow_remote_reads

    def _locate(self, rng, rank):
        nd = next(
            nd
            for nd in self.fabric.table.named_d_tensors
            if nd.global_tensor_id == rng.global_tensor_id
        )
        itemsize = gb.global_tensor_meta_of(nd).itemsize
        assert rng.global_tensor_offset % itemsize == rng.size % itemsize == 0
        start = rng.global_tensor_offset // itemsize
        elements = range(start, start + rng.size // itemsize)
        positions = self.fabric.positions.get((rank, rng.global_tensor_id))
        if positions is None:
            return None
        if not all(e in positions for e in elements):
            return None
        local = [positions[e] for e in elements]
        return self.fabric.buffers[rank][rng.global_tensor_id].reshape(
            -1, itemsize
        ), local

    def read_to_buffer(self, global_range, read_to_buffer):
        ranks = [self.rank]
        if self.allow_remote_reads:
            ranks += [r for r in self.fabric.buffers if r != self.rank]
        return _read_from_ranks(self, global_range, read_to_buffer, ranks)

    def write_from_buffer(self, global_range, write_from_buffer):
        hit = self._locate(global_range, self.rank)
        assert hit is not None, f"rank {self.rank} does not own {global_range}"
        buf, local = hit
        data = np.empty(global_range.size, dtype=np.uint8)
        result = write_from_buffer(data.ctypes.data, data.nbytes)
        if type(result) is gt.Ok:
            buf[local] = data.reshape(-1, buf.shape[1])
        return result


def _read_from_ranks(accessor, rng, callback, ranks):
    for rank in ranks:
        hit = accessor._locate(rng, rank)
        if hit is not None:
            buf, local = hit
            data = buf[local].copy().ravel()
            return callback(data.ctypes.data, data.nbytes)
    raise AssertionError(f"no source owns {rng}")


class TorchAccessor(ga.DTensorDataAccessor):
    """Resolve the public bridge on every access, including after reset."""

    def __init__(self, view, points, torch):
        self.view, self.points, self.torch = view, points, torch

    def _buffer(self, rng):
        import gdtensor

        gd = gdtensor.get_gd_tensor(self.view, rng.global_tensor_id)
        assert type(gd) is gt.Ok
        dt = gdtensor.get_d_tensor(gd.value)
        local = dt.to_local()
        assert local.is_contiguous()
        itemsize = local.element_size()
        start = rng.global_tensor_offset // itemsize
        count = rng.size // itemsize
        indices = self.points[rng.global_tensor_id]
        positions = {int(e): i for i, e in enumerate(indices)}
        offsets = [positions[e] for e in range(start, start + count)]
        assert offsets == list(range(offsets[0], offsets[0] + count))
        data = local.reshape(-1).view(self.torch.uint8)
        return data[offsets[0] * itemsize : (offsets[0] + count) * itemsize]

    def read_to_buffer(self, global_range, read_to_buffer):
        buf = self._buffer(global_range)
        return read_to_buffer(buf.data_ptr(), buf.numel())

    def write_from_buffer(self, global_range, write_from_buffer):
        buf = self._buffer(global_range)
        return write_from_buffer(buf.data_ptr(), buf.numel())

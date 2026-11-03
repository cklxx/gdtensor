"""Co-located cluster simulation: every node has its own cache memory and NIC.

Deployment model: the cache data node (DN) of node n runs on the same machine
as ranks [n*k, (n+1)*k). A rank reaches the cache memory of its own node
without the network; it reaches any other node's cache memory through the two
NICs (sender tx, receiver rx).

The cache accessor receives only a byte range, so it derives the target node
from the calling rank with an independent oracle of the read-chain rule
(design 5.8): the last layer serves rank r from node r // k when that node is
in the layer, else round-robin over the sorted layer nodes; every lower layer
node reads its own node when present in the parent layer, else round-robin
over the sorted parent nodes. A rank therefore touches exactly one node per
layer.

Each node memory is a separate address space with a validity mask: a read of
bytes that were never written to that node fails, so a wrong-node read cannot
pass by accident.
"""

from __future__ import annotations

import threading
import time

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb


def parent_of(children, parents) -> dict:
    """Child -> parent node: a local parent wins, the rest go round-robin."""
    local = set(parents)
    remote = [c for c in sorted(children) if c not in local]
    out = {c: c for c in children if c in local}
    out.update({c: sorted(parents)[j % len(parents)] for j, c in enumerate(remote)})
    return out


class Nic:
    """Per-node network card: byte counters and an optional serial link."""

    def __init__(self, bandwidth: float = 0.0) -> None:
        self.tx = 0
        self.rx = 0
        self.bandwidth = bandwidth  # bytes/s; 0 = no delay
        self.link = threading.Lock()

    def wire(self, nbytes: int) -> None:
        if not self.bandwidth:
            return
        with self.link:
            time.sleep(nbytes / self.bandwidth)


class Cluster:
    """Nodes, rank placement and NICs of one simulated job."""

    def __init__(self, num_nodes: int, k: int, bandwidth: float = 0.0) -> None:
        self.k = k
        self.nics = {n: Nic(bandwidth) for n in range(num_nodes)}
        self.lock = threading.Lock()
        self.storage_in = {n: 0 for n in range(num_nodes)}  # storage -> actor node
        self._tl = threading.local()

    @property
    def rank(self) -> int:
        return self._tl.rank

    @rank.setter
    def rank(self, value: int) -> None:
        self._tl.rank = value

    def node_of_rank(self, rank: int) -> int:
        return rank // self.k

    def transfer(self, src: int, dst: int, nbytes: int) -> None:
        if src == dst:
            return
        self.nics[src].wire(nbytes)
        with self.lock:
            self.nics[src].tx += nbytes
            self.nics[dst].rx += nbytes

    def nic_bytes(self) -> tuple[list[int], list[int]]:
        nodes = sorted(self.nics)
        return [self.nics[n].rx for n in nodes], [self.nics[n].tx for n in nodes]


def _offset(rng) -> int:
    if hasattr(rng, "global_tensor_offset"):
        return rng.global_tensor_offset
    return rng.contiguous_payload_offset


class NodeCache(ga.DTensorDataAccessor):
    """One cache layer; every node keeps a private copy of the tensor bytes."""

    def __init__(self, cluster: Cluster, sizes: dict, target_of_rank) -> None:
        self.cluster = cluster
        self.sizes = sizes
        self.target_of_rank = target_of_rank
        self.mem: dict = {}  # (node, tid) -> (bytes, valid mask)
        self.lock = threading.Lock()
        self.wrong_node = 0
        self.writes = []
        self.fills = []  # (actor node, target node, tid, offset, size)
        self.reads = []  # (actor rank, target node, size)

    def _buf(self, node: int, tid: str):
        key = (node, tid)
        if key not in self.mem:
            size = self.sizes[tid]
            self.mem[key] = (np.zeros(size, np.uint8), np.zeros(size, bool))
        return self.mem[key]

    def _ends(self, rng):
        actor = self.cluster.node_of_rank(self.cluster.rank)
        target = getattr(rng, "cache_node", self.target_of_rank(self.cluster.rank))
        return actor, target

    def write_from_buffer(self, global_range, write_from_buffer):
        actor, target = self._ends(global_range)
        tid, off, size = (
            global_range.global_tensor_id,
            _offset(global_range),
            global_range.size,
        )
        tmp = np.zeros(size, np.uint8)
        res = write_from_buffer(tmp.ctypes.data, size)
        if type(res) is not gb.Ok:
            return res
        self.cluster.transfer(actor, target, size)
        with self.lock:
            self.writes.append((actor, target, off, size))
            self.fills.append((actor, target, tid, off, size))
            data, valid = self._buf(target, tid)
            data[off : off + size] = tmp
            valid[off : off + size] = True
        return res

    def read_to_buffer(self, global_range, read_to_buffer):
        actor, target = self._ends(global_range)
        tid, off, size = (
            global_range.global_tensor_id,
            _offset(global_range),
            global_range.size,
        )
        with self.lock:
            data, valid = self._buf(target, tid)
            missing = not valid[off : off + size].all()
            self.wrong_node += int(missing)
            self.reads.append((self.cluster.rank, target, size))
            view = data[off : off + size].copy()
        if missing:
            return gb.Err(
                gb.GDTensorIOError(
                    operation="read",
                    msg=f"node {target} has no bytes {tid}@{off}+{size}",
                )
            )
        self.cluster.transfer(target, actor, size)
        return read_to_buffer(view.ctypes.data, size)

    def has_bytes(self, global_range) -> bool:
        actor, target = self._ends(global_range)
        tid, off, size = (
            global_range.global_tensor_id,
            _offset(global_range),
            global_range.size,
        )
        with self.lock:
            _data, valid = self._buf(target, tid)
            return bool(valid[off : off + size].all())


class StorageMem(ga.DTensorDataAccessor):
    """Remote storage: every read lands on the calling rank's node."""

    def __init__(self, cluster: Cluster, data) -> None:
        self.cluster = cluster
        self.data = np.array(data, dtype=np.uint8)

    def read_to_buffer(self, global_range, read_to_buffer):
        off, size = _offset(global_range), global_range.size
        node = self.cluster.node_of_rank(self.cluster.rank)
        with self.cluster.lock:
            self.cluster.storage_in[node] += size
        view = self.data[off : off + size]
        return read_to_buffer(view.ctypes.data, size)

    def write_from_buffer(self, global_range, write_from_buffer):
        return gb.Err(gb.GDTensorIOError(operation="write", msg="read-only storage"))


class RankMem(ga.DTensorDataAccessor):
    """Compute memory of every rank: one flat buffer, local to its rank."""

    def __init__(self, cluster: Cluster, size: int, world: int | None = None) -> None:
        self.cluster = cluster
        self.buffers = {
            rank: np.full(size, 0xA5, np.uint8)
            for rank in range(
                world if world is not None else len(cluster.nics) * cluster.k
            )
        }

    @property
    def data(self):
        return self.buffers[self.cluster.rank]

    def write_from_buffer(self, global_range, write_from_buffer):
        off, size = _offset(global_range), global_range.size
        view = self.data[off : off + size]
        return write_from_buffer(view.ctypes.data, size)

    def read_to_buffer(self, global_range, read_to_buffer):
        off, size = _offset(global_range), global_range.size
        view = self.data[off : off + size]
        return read_to_buffer(view.ctypes.data, size)


def layer_targets(cluster: Cluster, layer_nodes: list) -> list:
    """Per layer: rank -> node the rank touches (independent oracle)."""
    k = cluster.k
    rank_nodes = sorted(cluster.nics)
    serve = parent_of(rank_nodes, layer_nodes[-1])
    ups = [
        parent_of(layer_nodes[i + 1], layer_nodes[i])
        for i in range(len(layer_nodes) - 1)
    ]

    def chain(rank: int) -> list:
        nodes = [serve[rank // k]]
        for i in range(len(layer_nodes) - 1, 0, -1):
            nodes.append(ups[i - 1][nodes[-1]])
        return nodes[::-1]

    return [(lambda r, i=i: chain(r)[i]) for i in range(len(layer_nodes))]

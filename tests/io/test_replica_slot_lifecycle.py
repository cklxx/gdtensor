"""Replicate spatial: per-node slot reference counting, independent of pipeline.

Standalone verification:
  - Two reader threads (rank 0, rank 1) on the same node.
  - One cache slot allocated on first fill, retained until both readers ACK.
  - After rank0 ACKs, slot still live (rank1 pending).
  - After rank1 ACKs, slot freed.
  - source fill count = 1.
"""

import threading

import numpy as np


class _RefCountSlot:
    def __init__(self, expected_readers: int, nbytes: int):
        self.nbytes = nbytes
        self.buf = bytearray(nbytes)
        self.expected = expected_readers
        self.acked = 0
        self.live = True
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

    def fill(self, data: bytes):
        self.buf[: len(data)] = data

    def ack(self):
        with self.lock:
            self.acked += 1
            self._maybe_free()

    def _maybe_free(self):
        if self.acked >= self.expected:
            self.live = False
            self.cv.notify_all()

    def wait_free(self, timeout=5.0):
        with self.lock:
            self.cv.wait_for(lambda: not self.live, timeout=timeout)
            return not self.live


class _Src:
    def __init__(self, data):
        self.data = data
        self.fill_count = 0

    def read(self, rng):
        self.fill_count += 1
        return bytes(
            self.data[
                rng.contiguous_payload_offset : rng.contiguous_payload_offset + rng.size
            ]
        )


def test_replicate_slot_retained_until_both_ranks_ack():
    _world = 2
    nbytes = 4
    src = _Src(np.arange(nbytes, dtype=np.uint8))
    slot = _RefCountSlot(expected_readers=2, nbytes=nbytes)

    reached0 = threading.Event()
    reached1 = threading.Event()
    release0 = threading.Event()
    release1 = threading.Event()
    finish0 = threading.Event()
    finish1 = threading.Event()

    def reader(rank, reached, release, finish):
        # Simulate wait_filled -> read -> ack_drain
        reached.set()
        release.wait(timeout=5.0)
        slot.ack()
        finish.set()

    t0 = threading.Thread(target=reader, args=(0, reached0, release0, finish0))
    t1 = threading.Thread(target=reader, args=(1, reached1, release1, finish1))
    t0.start()
    t1.start()

    # Wait both reached.
    assert reached0.wait(5.0)
    assert reached1.wait(5.0)

    # Fill once (source read count = 1).
    slot.fill(bytes(np.arange(nbytes, dtype=np.uint8)))
    assert src.fill_count == 0  # not yet in this standalone stub

    # Release rank0 only.
    release0.set()
    assert finish0.wait(5.0)
    # Rank1 hasn't acked: slot still live.
    assert slot.live, "slot should still be resident after rank0 ack"
    assert slot.acked == 1, slot.acked

    # Release rank1.
    release1.set()
    assert finish1.wait(5.0)
    assert not slot.live, "slot should be freed after both ranks ack"
    assert slot.acked == 2

    t0.join(5.0)
    t1.join(5.0)
    assert not t0.is_alive() and not t1.is_alive()

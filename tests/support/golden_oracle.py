"""Exact independent rank-window verification, including illegal writes."""

import numpy as np


def shard_window(size, world, rank):
    chunk = -(-size // world)
    return min(rank * chunk, size), min((rank + 1) * chunk, size)


def verify_tensor(parts, expected, rank, tid, lo, hi):
    pieces = sorted((off, buf) for (t, off), buf in parts.items() if t == tid)
    cursor = lo
    for off, buf in pieces:
        assert off == cursor, ("gap or overlap", rank, tid, off, cursor)
        assert len(buf) > 0 and off + len(buf) <= hi, ("outside window", rank, tid)
        assert np.array_equal(buf, expected[off : off + len(buf)]), (
            "byte mismatch",
            tid,
            rank,
        )
        cursor += len(buf)
    assert cursor == hi, ("unfilled tail", rank, tid, cursor, hi)


def verify_rank(parts, data, world, replicate, rank, itemsize):
    assert all(tid in data for tid, _ in parts), "unknown tensor write"
    for tid, expected in data.items():
        a, b = shard_window(len(expected) // itemsize, world, rank)
        lo, hi = (0, len(expected)) if replicate[tid] else (a * itemsize, b * itemsize)
        verify_tensor(parts, expected, rank, tid, lo, hi)


def verify_parts(parts_by_rank, data, world, replicate, itemsize=1):
    assert set(parts_by_rank) <= set(range(world)), "unexpected rank write"
    for rank in range(world):
        verify_rank(parts_by_rank.get(rank, {}), data, world, replicate, rank, itemsize)

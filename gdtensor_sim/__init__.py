"""Simulation / test harness. NOT part of the gdtensor release package.

This package plays the role of a training framework and storage layout
builder (design 4.3: builders live outside the core) plus a single
process multi-rank runner and the demo CLI.

The released gdtensor package contains only the design core. Run the demo
with ``python -m gdtensor_sim.sim``.
"""

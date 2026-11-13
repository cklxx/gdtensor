"""CPU smoke test for ``torchrun`` + ``torch.distributed`` + Gloo."""

from datetime import timedelta

import torch
from torch.distributed import (
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
)

from gdtensor_conn.coord import GlooCoordinator


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def main() -> None:
    torch.set_num_threads(1)
    init_process_group("gloo", init_method="env://", timeout=timedelta(seconds=120))
    try:
        rank = get_rank()
        world_size = get_world_size()
        coordinator = GlooCoordinator()
        _check(coordinator.vote(True), "all-success vote returned False")
        for failed_rank in range(world_size):
            _check(
                not coordinator.vote(rank != failed_rank),
                f"failure from rank {failed_rank} was not global",
            )
            coordinator.wait()
            _check(coordinator.vote(True), "coordinator did not recover")
            coordinator.wait()
        if rank == 0:
            print(
                f"PASS backend=gloo device=cpu world_size={world_size}",
                flush=True,
            )
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()

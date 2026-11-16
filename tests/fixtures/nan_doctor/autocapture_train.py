"""A training loop with NO capture wiring, for test_autocapture.py.

Plants a NaN batch at step PLANT_STEP on rank PLANT_RANK (env, -1 = never).
TWO_OPTIMIZERS=1 splits the parameters across two optimizers.  Runs on CPU,
single process or under torchrun (gloo).
"""
import os

import torch
import torch.distributed as dist


def main():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("gloo")
    plant_step = int(os.environ.get("PLANT_STEP", "-1"))
    plant_rank = int(os.environ.get("PLANT_RANK", "0"))
    torch.manual_seed(0)
    a, b = torch.nn.Linear(8, 8), torch.nn.Linear(8, 1)
    if os.environ.get("TWO_OPTIMIZERS") == "1":
        opts = [torch.optim.SGD(a.parameters(), lr=0.01), torch.optim.SGD(b.parameters(), lr=0.01)]
    else:
        opts = [torch.optim.SGD(list(a.parameters()) + list(b.parameters()), lr=0.01)]
    for step in range(8):
        x = torch.randn(4, 8)
        if step == plant_step and rank == plant_rank:
            x = x * float("nan")
        loss = b(torch.relu(a(x))).sum()
        loss.backward()
        if world > 1:
            for p in list(a.parameters()) + list(b.parameters()):
                dist.all_reduce(p.grad)
                p.grad.div_(world)
        for o in opts:
            o.step()
            o.zero_grad()
    print("TRAINING_DONE", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

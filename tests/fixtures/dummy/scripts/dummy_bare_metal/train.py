#!/usr/bin/env python3
"""A few local torch steps under torchrun.

No Hugging Face download. CUDA is used only when a one-element allocation
succeeds, so a prefix whose wheel does not match this GPU still finishes
and prints a performance line.
"""

import os
import time

import torch


def _cuda_device(local_rank: int):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device=torch.device("cuda", local_rank))
        torch.cuda.synchronize()
        return torch.device("cuda", local_rank)
    except Exception as exc:
        print(f"cuda unavailable, using cpu: {exc}", flush=True)
        return torch.device("cpu")


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = _cuda_device(local_rank)

    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)

    model = torch.nn.Linear(32, 32).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps = int(os.environ.get("MAD_TRAIN_STEPS", "2"))
    batch = int(os.environ.get("MAD_BATCH", "4"))
    model.train()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = 0
    for step in range(steps):
        inputs = torch.randn(batch, 32, device=device)
        loss = model(inputs).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tokens += batch * 32
        if local_rank == 0:
            print(f"step {step} loss {float(loss.detach()):.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)

    if local_rank == 0:
        device_name = (
            torch.cuda.get_device_name(local_rank) if device.type == "cuda" else "cpu"
        )
        print(f"device: {device_name}")
        print(f"world_size: {world_size}")
        print(f"performance: {tokens / elapsed:.2f} tokens_per_second")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

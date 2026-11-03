#!/usr/bin/env python3
"""Small multi-GPU training loop for validating madengine profiling tools.

Each step runs bf16 GEMMs (an MLP forward/backward) and a DDP gradient all-reduce, so
every collector has compute kernels and RCCL kernels to find within seconds.

The Kineto window follows the PyTorch profiler schedule: skipped steps, one warmup
step (recorded then discarded), then the active steps. Shapes and stacks stay off
unless asked for; they dominate trace size and are not required to see kernels or
collectives. Every rank is profiled so the collective report has a rank per trace.
The launcher names each trace rank<N>, which TraceLens reads directly.

Environment:
    DUMMY_PROF_STEPS            training steps (default 30)
    DUMMY_TORCH_PROFILE=1       capture a torch.profiler (Kineto) window
    DUMMY_TORCH_PROFILE_DIR     output dir for *.pt.trace.json (default ./torch_profiler_output)
    DUMMY_TORCH_PROFILE_SHAPES=1  record tensor shapes (off by default)
    DUMMY_PROF_WAIT / _WARMUP / _ACTIVE   schedule lengths (default 9 / 1 / 3)
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

STEPS = int(os.environ.get("DUMMY_PROF_STEPS", "30"))
PROFILE = os.environ.get("DUMMY_TORCH_PROFILE") == "1"
PROFILE_DIR = os.environ.get("DUMMY_TORCH_PROFILE_DIR", "torch_profiler_output")
PROFILE_WAIT = int(os.environ.get("DUMMY_PROF_WAIT", "9"))
PROFILE_WARMUP = int(os.environ.get("DUMMY_PROF_WARMUP", "1"))
PROFILE_ACTIVE = int(os.environ.get("DUMMY_PROF_ACTIVE", "3"))
RECORD_SHAPES = os.environ.get("DUMMY_TORCH_PROFILE_SHAPES") == "1"
BATCH, HIDDEN = 64, 4096


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    model = nn.Sequential(
        nn.Linear(HIDDEN, 4 * HIDDEN), nn.GELU(), nn.Linear(4 * HIDDEN, HIDDEN)
    ).cuda().to(torch.bfloat16)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(BATCH, HIDDEN, device="cuda", dtype=torch.bfloat16)

    prof = None
    if PROFILE:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        window = PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE
        if window > STEPS:
            raise SystemExit(
                f"dummy_profiling: profiler window is {window} steps but DUMMY_PROF_STEPS is {STEPS}"
            )
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=PROFILE_WAIT, warmup=PROFILE_WARMUP, active=PROFILE_ACTIVE, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(PROFILE_DIR, worker_name=f"rank{rank}"),
            record_shapes=RECORD_SHAPES,
            with_stack=False,
        )
        prof.start()

    times = []
    for step in range(STEPS):
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss = model(x).float().pow(2).mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        if prof:
            prof.step()
        if rank == 0:
            print(f"step {step + 1}/{STEPS} | step time (ms): {times[-1] * 1000:.2f}", flush=True)
    if prof:
        prof.stop()

    steady = sorted(times[5:]) or times
    median = steady[len(steady) // 2]
    if rank == 0:
        print(f"performance: {BATCH * world_size / median:.2f} samples_per_second", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

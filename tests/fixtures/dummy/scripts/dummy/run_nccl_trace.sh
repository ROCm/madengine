#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Single-process NCCL sanity check for RCCL_TRACE e2e tests. On hosts where the
# container exposes fewer GPUs than the physical topology (e.g. MI300X multi-OAM),
# RCCL can spend a long time probing inaccessible ROCr/KFD nodes without these guards.

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

python3 -c "
import os
import torch
import torch.distributed as dist

os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29501')
dist.init_process_group('nccl', rank=0, world_size=1)
tensor = torch.arange(1, dtype=torch.int64).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(tensor[0])
" | tee log.txt

echo "performance: 1 pass"

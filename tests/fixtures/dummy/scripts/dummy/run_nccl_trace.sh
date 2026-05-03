#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#

python -c "import torch; import torch.distributed as dist; import os; os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '29501'; dist.init_process_group('nccl', rank=0, world_size=1);tensor = torch.arange(1, dtype=torch.int64).cuda(); dist.all_reduce(tensor, op=dist.ReduceOp.SUM); print(tensor[0]); "  | tee log.txt

echo "performance: 1 pass"

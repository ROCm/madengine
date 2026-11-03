#!/bin/bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

# TheRock images ship python3 and not python.
if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

"$PY" -c "import torch; x = torch.ones(10,10).to('cuda'); l = torch.nn.Linear(10,30).cuda(); c = torch.nn.Conv2d(1, 20, 3).cuda(); out1 = l(x); out1 = out1[None, None, :, :] ; out2 = c(out1); print( 'performance=' + str(torch.cuda.memory_allocated(0)) )"  | tee log.txt
 
performance=$(grep -o "performance=[0-9]*" log.txt | tail -n 1 | sed 's/performance=//')
echo "performance: $performance bytes" 

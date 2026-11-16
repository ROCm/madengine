#!/bin/bash
export HSA_DISABLE_CACHE=1
torchrun --nproc-per-node=8 ddp_torchrun.py

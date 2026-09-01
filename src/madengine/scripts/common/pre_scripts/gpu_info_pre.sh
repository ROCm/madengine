#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

gpu_vendor=""
if [ -f "/usr/bin/nvidia-smi" ]; then
    echo "NVIDIA GPU detected."
    gpu_vendor="NVIDIA"
    gpu_architecture=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -m 1 -E -o ".{0,1}100"| xargs )
    python3 -m pip install nvidia-ml-py
elif [ -f "/opt/rocm/bin/rocm-smi" ]; then
    echo "AMD GPU detected."
    gpu_vendor="AMD"
    gpu_architecture=$(rocminfo | grep -o -m 1 'gfx.*' | xargs )
    MI200="gfx90a"
    MI100="gfx908"
    MI50="gfx906"
else
    echo "Unable to detect GPU vendor"
    exit 1
fi

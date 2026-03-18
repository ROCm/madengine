#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Simple ResNet50 Training Benchmark with TheRock
#
###############################################################################
set -ex

echo "========================================================================"
echo "ResNet50 Training Benchmark with TheRock"
echo "========================================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional: Check TheRock installation (informative, non-blocking)
echo ""
echo "=== TheRock Environment Check ==="
DETECT_SCRIPT="../scripts/common/tools/detect_therock.sh"
if [ -f "$DETECT_SCRIPT" ]; then
    bash "$DETECT_SCRIPT" || echo "⚠️  TheRock validation completed with warnings (continuing anyway)"
else
    echo "ℹ️  TheRock detector not available (skipping environment check)"
    echo "    To enable: Use --tools therock_check flag"
fi

# Show PyTorch configuration
echo ""
echo "=== PyTorch Configuration ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'HIP: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')"

echo ""
echo "========================================================================"
echo "Running Benchmark"
echo "========================================================================"

# Run training benchmark
python3 "$SCRIPT_DIR/train_resnet.py"

echo ""
echo "========================================================================"
echo "Benchmark completed!"
echo "========================================================================"


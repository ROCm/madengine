#!/bin/bash
#
# DeepSpeed Wrapper Script - Uses torchrun launcher
#
# This script launches DeepSpeed training using torchrun instead of MPI,
# which avoids the need for OpenMPI installation in the container.
#
set -e

echo "========================================================================"
echo "MADEngine DeepSpeed Wrapper Script"
echo "========================================================================"

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine launcher from environment or default to torchrun
LAUNCHER_CMD=${MAD_MULTI_NODE_RUNNER:-"torchrun --standalone --nproc_per_node=2"}

echo "========================================================================"
echo "Launcher Command:"
echo "$LAUNCHER_CMD"
echo "========================================================================"

# Launch training with torchrun
$LAUNCHER_CMD run_deepspeed.py --deepspeed_config ds_config.json

echo "========================================================================"
echo "Training script completed"
echo "========================================================================"


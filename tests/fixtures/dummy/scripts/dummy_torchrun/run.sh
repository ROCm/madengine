#!/bin/bash
#
# Bash wrapper for dummy_torchrun distributed training
# Uses MAD_MULTI_NODE_RUNNER for torchrun launcher
#

set -e

echo "========================================================================"
echo "madengine Torchrun Wrapper Script"
echo "========================================================================"

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine multi-node runner to use
# Default to standalone torchrun if not set
if [ -z "$MAD_MULTI_NODE_RUNNER" ]; then
    # Get number of GPUs from environment
    N_GPUS="${MAD_RUNTIME_NGPUS:-1}"
    
    echo "ℹ️  MAD_MULTI_NODE_RUNNER not set, using standalone torchrun"
    echo "ℹ️  Using $N_GPUS GPUs"
    
    MAD_MULTI_NODE_RUNNER="torchrun --standalone --nproc_per_node=$N_GPUS"
fi

echo "========================================================================"
echo "Launcher Command:"
echo "$MAD_MULTI_NODE_RUNNER"
echo "========================================================================"

# Create MIOpen cache directory if MIOPEN_USER_DB_PATH is set
# This prevents "Duplicate ID" errors in multi-GPU training
if [ -n "$MIOPEN_USER_DB_PATH" ]; then
    # Extract base directory (before LOCAL_RANK expansion)
    MIOPEN_BASE_DIR=$(dirname "$MIOPEN_USER_DB_PATH")
    mkdir -p "$MIOPEN_BASE_DIR"
    echo "ℹ️  MIOpen cache directory: $MIOPEN_USER_DB_PATH"
    echo "   (will be created per-process with LOCAL_RANK)"
fi

# Execute the Python training script with torchrun
echo "Executing: $MAD_MULTI_NODE_RUNNER run_torchrun.py"
$MAD_MULTI_NODE_RUNNER run_torchrun.py
PYTHON_EXIT_CODE=$?

echo "========================================================================"
echo "Training script completed with exit code: $PYTHON_EXIT_CODE"
echo "========================================================================"

# Exit with the Python script's exit code
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training script failed with exit code $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi

#!/bin/bash
#
# Bash wrapper for dummy_torchrun_multi: torchrun + multiple_results CSV output
# Uses MAD_MULTI_NODE_RUNNER for torchrun launcher; writes perf_dummy_torchrun.csv
#

set -e

echo "========================================================================"
echo "madengine Torchrun Multi (multiple_results) Wrapper Script"
echo "========================================================================"

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine multi-node runner to use
if [ -z "$MAD_MULTI_NODE_RUNNER" ]; then
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
if [ -n "$MIOPEN_USER_DB_PATH" ]; then
    MIOPEN_BASE_DIR=$(dirname "$MIOPEN_USER_DB_PATH")
    mkdir -p "$MIOPEN_BASE_DIR"
    echo "ℹ️  MIOpen cache directory: $MIOPEN_USER_DB_PATH"
fi

# Execute the Python script (multiple_results variant)
echo "Executing: $MAD_MULTI_NODE_RUNNER run_torchrun_multi.py"
$MAD_MULTI_NODE_RUNNER run_torchrun_multi.py
PYTHON_EXIT_CODE=$?

echo "========================================================================"
echo "Training script completed with exit code: $PYTHON_EXIT_CODE"
echo "========================================================================"

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training script failed with exit code $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi

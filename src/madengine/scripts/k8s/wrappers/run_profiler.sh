#!/bin/bash
# madengine K8s Wrapper - GPU Info Profiler
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Wrapper for gpu_info_profiler.py to work in K8s environment
# Usage: run_profiler.sh [power|vram]

set -e

MODE=${1:-power}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/profiler_results}

echo "=== GPU Info Profiler (K8s) ==="
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"

# Verify the Python tool exists
PROFILER_SCRIPT="/workspace/scripts/common/tools/gpu_info_profiler.py"
if [ ! -f "$PROFILER_SCRIPT" ]; then
    echo "Error: gpu_info_profiler.py not found at $PROFILER_SCRIPT"
    echo "Available scripts:"
    ls -la /workspace/scripts/common/tools/ 2>/dev/null || echo "  scripts/common/tools/ not found"
    exit 1
fi

# Set environment variables for the profiler
export DEVICE=${DEVICE:-all}
export SAMPLING_RATE=${SAMPLING_RATE:-0.1}
export MODE=$MODE
export DUAL_GCD=${DUAL_GCD:-false}

# Create output directory
mkdir -p $OUTPUT_DIR

# Change to workspace to match expected paths
cd /workspace

# Run the profiler (reusing the same Python script as local execution!)
echo "Starting profiler..."
python3 $PROFILER_SCRIPT

echo "✓ GPU profiler completed"
echo "Results saved to: $OUTPUT_DIR"

# List output files
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output files:"
    ls -lh $OUTPUT_DIR
fi


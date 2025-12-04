#!/bin/bash
# MADEngine K8s Wrapper - rocEnvTool
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Wrapper for rocEnvTool to work in K8s environment
# Usage: run_rocenv.sh [output_name]

set -e

OUTPUT_NAME=${1:-sys_config_info}

echo "=== rocEnvTool (K8s) ==="
echo "Output: $OUTPUT_NAME"

# Verify rocEnvTool exists
ROCENV_DIR="/workspace/scripts/common/pre_scripts/rocEnvTool"
if [ ! -d "$ROCENV_DIR" ]; then
    echo "Error: rocEnvTool not found at $ROCENV_DIR"
    echo "Available pre_scripts:"
    ls -la /workspace/scripts/common/pre_scripts/ 2>/dev/null || echo "  pre_scripts/ not found"
    exit 1
fi

# Change to workspace
cd /workspace

# Copy rocEnvTool to working directory (same as local execution)
echo "Copying rocEnvTool..."
cp -r scripts/common/pre_scripts/rocEnvTool .

# Run rocEnvTool (same command as local!)
echo "Running rocEnvTool..."
cd rocEnvTool
python3 rocenv_tool.py --lite --dump-csv --print-csv --output-name $OUTPUT_NAME

# Copy results back to workspace
echo "Copying results..."
OUT_DIR=".$OUTPUT_NAME"
OUT_CSV="$OUTPUT_NAME.csv"

if [ -d "$OUT_DIR" ]; then
    cp -r $OUT_DIR /workspace/
    echo "✓ Copied directory: /workspace/$OUT_DIR"
fi

if [ -f "$OUT_CSV" ]; then
    cp $OUT_CSV /workspace/
    echo "✓ Copied CSV: /workspace/$OUT_CSV"
fi

cd /workspace

echo "✓ rocEnvTool completed"
echo "Results saved to: /workspace/$OUTPUT_NAME.csv"

# List output files
if [ -f "/workspace/$OUT_CSV" ]; then
    echo "CSV file size: $(du -h /workspace/$OUT_CSV | cut -f1)"
fi


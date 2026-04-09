#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

OUTPUT_FILE_NAME=${1:-"sys_config_info"}

# Determine the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if rocEnvTool is in the same directory (K8s execution)
if [ -d "$SCRIPT_DIR/rocEnvTool" ]; then
    # K8s execution: rocEnvTool is already in place
    cd "$SCRIPT_DIR/rocEnvTool"
    python3 rocenv_tool.py --lite --dump-csv --print-csv --output-name $OUTPUT_FILE_NAME
    out_dir="."$OUTPUT_FILE_NAME
    out_csv=$OUTPUT_FILE_NAME".csv"
    # Copy results back to workspace root
    if [ -d "$out_dir" ]; then
        cp -r "$out_dir" /workspace/
    fi
    if [ -f "$out_csv" ]; then
        cp "$out_csv" /workspace/
    fi
else
    # Local execution: copy rocEnvTool from relative path
    cp -r ../scripts/common/pre_scripts/rocEnvTool .
    cd rocEnvTool
    python3 rocenv_tool.py --lite --dump-csv --print-csv --output-name $OUTPUT_FILE_NAME
    out_dir="."$OUTPUT_FILE_NAME
    out_csv=$OUTPUT_FILE_NAME".csv"
    cp -r $out_dir ../../
    cp $out_csv ../../
    cd ..
fi

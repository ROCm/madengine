#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

OUTPUT_FILE_NAME=${1:-"sys_config_info"}
ROCENV_MODE=${2:-"lite"}

LITE_FLAG="--lite"
if [ "$ROCENV_MODE" = "full" ]; then
    LITE_FLAG=""
    # Install diagnostic tools on-demand if missing (best-effort)
    # These are needed for hardware_information, bios_settings,
    # dmsg_gpu_drm_atom_logs, and amdgpu_modinfo sections
    MISSING_PKGS=""
    command -v lshw      >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS lshw"
    command -v dmidecode >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS dmidecode"
    command -v modinfo   >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS kmod"
    command -v dmesg     >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS util-linux"
    if [ -n "$MISSING_PKGS" ]; then
        echo "rocenv full mode: installing missing diagnostic tools:$MISSING_PKGS"
        apt-get update -qq >/dev/null 2>&1 && \
            apt-get install -y -qq --no-install-recommends $MISSING_PKGS >/dev/null 2>&1 || \
            echo "Warning: could not install some diagnostic tools (network or permissions issue)"
    fi
fi

# Determine the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if rocEnvTool is in the same directory (K8s execution)
if [ -d "$SCRIPT_DIR/rocEnvTool" ]; then
    # K8s execution: rocEnvTool is already in place
    cd "$SCRIPT_DIR/rocEnvTool"
    python3 rocenv_tool.py $LITE_FLAG --dump-csv --print-csv --output-name $OUTPUT_FILE_NAME
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
    python3 rocenv_tool.py $LITE_FLAG --dump-csv --print-csv --output-name $OUTPUT_FILE_NAME
    out_dir="."$OUTPUT_FILE_NAME
    out_csv=$OUTPUT_FILE_NAME".csv"
    cp -r $out_dir ../../
    cp $out_csv ../../
    cd ..
fi

#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Lightweight env-var mode wrapper for rocm-trace-lite (RTL).
# Unlike rtl_trace_wrapper.sh (which uses `rtl trace` CLI), this sets
# HSA_TOOLS_LIB + LD_PRELOAD + RTL_OUTPUT directly and exec's the model
# command — no Python subprocess wrapping, no blocking post-processing.
#
# Post-processing (merge, summary, Perfetto) is deferred to the post-script,
# which runs AFTER madengine has captured the performance metric and wall time.
#
# Usage (from model run_directory, as prepended by madengine):
#   bash ../scripts/common/tools/rtl_envvar_wrapper.sh <application and arguments>
#
# Environment (optional):
#   RTL_WRAPPER_OUTPUT_DIR   Output directory (default: rocm_trace_lite_output)
#   RTL_MODE                 Profiling mode: lite (default), standard, full
#                            See: https://sunway513.github.io/rocm-trace-lite/quickstart.html

set -euo pipefail

RTL_OUT_DIR="${RTL_WRAPPER_OUTPUT_DIR:-rocm_trace_lite_output}"
mkdir -p "${RTL_OUT_DIR}"

# Resolve librtl.so path via RTL's Python API
RTL_LIB=$(python3 -c 'from rocm_trace_lite import get_lib_path; print(get_lib_path())' 2>/dev/null) || true
if [ -z "$RTL_LIB" ] || [ ! -f "$RTL_LIB" ]; then
	echo "Error: cannot resolve librtl.so path (rocm-trace-lite not installed?)." >&2
	echo "Install: pip install a release wheel from https://github.com/sunway513/rocm-trace-lite/releases" >&2
	exit 127
fi

# Set env vars for HSA runtime interception (no rtl trace CLI)
export HSA_TOOLS_LIB="$RTL_LIB"
export LD_PRELOAD="${RTL_LIB}${LD_PRELOAD:+:$LD_PRELOAD}"
export RTL_OUTPUT="${RTL_OUT_DIR}/trace_%p.db"

# Add librtl.so dir and ROCm paths to LD_LIBRARY_PATH
LIB_DIR=$(dirname "$RTL_LIB")
EXTRA_DIRS="$LIB_DIR"
for d in /opt/rocm/lib /opt/rocm/lib64; do
	[ -d "$d" ] && EXTRA_DIRS="${EXTRA_DIRS}:${d}"
done
ROCM_P="${ROCM_PATH:-${HIP_PATH:-}}"
[ -n "$ROCM_P" ] && EXTRA_DIRS="${EXTRA_DIRS}:${ROCM_P}/lib"
export LD_LIBRARY_PATH="${EXTRA_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "rtl (env-var mode): HSA_TOOLS_LIB=$RTL_LIB RTL_MODE=${RTL_MODE:-lite} RTL_OUTPUT=$RTL_OUTPUT" >&2

exec "$@"

#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Wrapper for rocm-trace-lite (RTL).
# Docs: https://sunway513.github.io/rocm-trace-lite/quickstart.html
#
# Usage (from model run_directory, as prepended by madengine):
#   bash ../scripts/common/tools/rtl_trace_wrapper.sh <application and arguments>
#
# Writes SQLite and companion artifacts under rocm_trace_lite_output/ (configurable)
# so scripts/common/post_scripts/trace.sh can collect them into /myworkspace/.
#
# Environment (optional):
#   RTL_WRAPPER_OUTPUT_DIR   Output directory (default: rocm_trace_lite_output)
#   RTL_WRAPPER_TRACE_DB     Full path to trace DB (default: $RTL_WRAPPER_OUTPUT_DIR/trace.db)

set -euo pipefail

RTL_OUT_DIR="${RTL_WRAPPER_OUTPUT_DIR:-rocm_trace_lite_output}"
RTL_DB="${RTL_WRAPPER_TRACE_DB:-${RTL_OUT_DIR}/trace.db}"

mkdir -p "${RTL_OUT_DIR}"

# Prefer rtl on PATH; else Python module after pip install (same entry point as rtl CLI).
if command -v rtl >/dev/null 2>&1; then
	exec rtl trace -o "${RTL_DB}" "$@"
fi
if python3 -c 'import rocm_trace_lite' 2>/dev/null; then
	exec python3 -m rocm_trace_lite.cli trace -o "${RTL_DB}" "$@"
fi

echo "Error: rocm-trace-lite not available (no 'rtl' and no Python package rocm_trace_lite)." >&2
echo "Install: run pre_scripts/trace.sh rocm_trace_lite, or pip install a release wheel from https://github.com/sunway513/rocm-trace-lite/releases" >&2
exit 127

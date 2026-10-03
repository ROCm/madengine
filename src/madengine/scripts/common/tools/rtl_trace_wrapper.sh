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
# After the workload exits, the trace is checked for GPU operations. A trace with
# none (or only roctx UserMarker ranges) means RTL never intercepted kernel
# dispatches, e.g. on ROCm runtimes that bypass HSA_TOOLS_LIB, and the wrapper
# exits non-zero instead of reporting an empty profile as a successful run.
#
# Environment (optional):
#   RTL_WRAPPER_OUTPUT_DIR   Output directory (default: rocm_trace_lite_output)
#   RTL_WRAPPER_TRACE_DB     Full path to trace DB (default: $RTL_WRAPPER_OUTPUT_DIR/trace.db)
#   RTL_WRAPPER_ALLOW_EMPTY  Set to 1 to keep exit code 0 when no GPU ops were captured
#   RTL_MODE                 Profiling mode for `rtl trace --mode` (e.g. lite, default, full).
#                            Used only if `rtl trace --help` (or the Python CLI --help) lists --mode;
#                            otherwise a warning is printed and tracing runs without --mode.
#                            When unset, `rtl trace` uses the RTL CLI default (version-dependent).
#                            See: https://github.com/sunway513/rocm-trace-lite

set -euo pipefail

RTL_OUT_DIR="${RTL_WRAPPER_OUTPUT_DIR:-rocm_trace_lite_output}"
RTL_DB="${RTL_WRAPPER_TRACE_DB:-${RTL_OUT_DIR}/trace.db}"

mkdir -p "${RTL_OUT_DIR}"
# Absolute paths: the workload may change directory before it starts GPU work.
RTL_OUT_DIR="$(cd "${RTL_OUT_DIR}" && pwd)"
[[ "${RTL_DB}" != /* ]] && RTL_DB="$PWD/${RTL_DB#./}"

# Prefer rtl on PATH; else Python module after pip install (same entry point as rtl CLI).
if command -v rtl >/dev/null 2>&1; then
	RTL_CLI=(rtl trace)
elif python3 -c 'import rocm_trace_lite' 2>/dev/null; then
	RTL_CLI=(python3 -m rocm_trace_lite.cli trace)
else
	echo "Error: rocm-trace-lite not available (no 'rtl' and no Python package rocm_trace_lite)." >&2
	echo "Install: run pre_scripts/trace.sh rocm_trace_lite, or pip install a release wheel from https://github.com/sunway513/rocm-trace-lite/releases" >&2
	exit 127
fi

if [[ -n "${RTL_MODE:-}" ]]; then
	if "${RTL_CLI[@]}" --help 2>&1 | grep -q -- '--mode'; then
		RTL_CLI+=(--mode "${RTL_MODE}")
	else
		echo "Warning: RTL_MODE is set, but installed '${RTL_CLI[*]}' does not support --mode; continuing without it." >&2
	fi
fi

rc=0
"${RTL_CLI[@]}" -o "${RTL_DB}" "$@" || rc=$?
if [[ $rc -ne 0 ]]; then
	exit $rc
fi

# Count captured GPU operations; roctx UserMarker ranges are CPU-side annotations.
gpu_ops=$(python3 - "${RTL_DB}" <<'EOF'
import os, sqlite3, sys
db = sys.argv[1]
if not os.path.exists(db):
    print(0)
    sys.exit()
con = sqlite3.connect(db)
try:
    n = con.execute(
        "select count(*) from rocpd_op o left join rocpd_string s on o.opType_id = s.id "
        "where coalesce(s.string, '') != 'UserMarker'"
    ).fetchone()[0]
except sqlite3.Error:
    n = 0
print(n)
EOF
)

if [[ "${gpu_ops}" -eq 0 ]]; then
	echo "Error: rocm-trace-lite captured no GPU operations (trace: ${RTL_DB})." >&2
	echo "  The workload ran, but RTL did not intercept any kernel dispatch. Known cause: ROCm" >&2
	echo "  runtimes whose HSA loads tools via rocprofiler-register and ignores HSA_TOOLS_LIB" >&2
	echo "  (e.g. TheRock rocm-sdk wheels). Use a rocprofv3 preset instead, or set" >&2
	echo "  RTL_WRAPPER_ALLOW_EMPTY=1 to accept an empty trace." >&2
	if [[ "${RTL_WRAPPER_ALLOW_EMPTY:-0}" != "1" ]]; then
		exit 3
	fi
fi
exit 0

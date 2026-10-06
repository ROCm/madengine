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
#   RTL_WRAPPER_TRACE_DB     Full path to the merged trace DB (default: $RTL_WRAPPER_OUTPUT_DIR/trace.db)
#   RTL_WRAPPER_ALLOW_EMPTY  Set to 1 to keep exit code 0 when no GPU ops were captured
#   RTL_WRAPPER_KEEP_REGISTER  Set to 1 to leave rocprofiler-register in charge of HSA tool loading
#   HSA_TOOLS_LIB            librtl.so. Exported when unset so torchrun workers inherit it.
#   RTL_OUTPUT               Per-process trace path (default: $RTL_WRAPPER_OUTPUT_DIR/trace_%p.db)
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

# `rtl trace` attaches HSA_TOOLS_LIB to the process it execs. A launcher that then
# starts GPU workers (torchrun, Primus run.sh) only keeps that attachment when the
# variable is already in the environment those workers inherit.
if [[ -z "${HSA_TOOLS_LIB:-}" ]]; then
	if [[ -f /usr/local/lib/librtl.so ]]; then
		export HSA_TOOLS_LIB=/usr/local/lib/librtl.so
	else
		_rtl_lib=$(python3 -c 'from rocm_trace_lite import get_lib_path; print(get_lib_path())' 2>/dev/null || true)
		if [[ -n "${_rtl_lib}" && -f "${_rtl_lib}" ]]; then
			export HSA_TOOLS_LIB="${_rtl_lib}"
		fi
	fi
fi
# TheRock HSA loads tools through rocprofiler-register and never dlopens HSA_TOOLS_LIB
# unless the register is turned off for this process. No effect when the register is absent.
if [[ -n "${HSA_TOOLS_LIB:-}" && "${RTL_WRAPPER_KEEP_REGISTER:-0}" != "1" ]]; then
	export HSA_TOOLS_DISABLE_REGISTER="${HSA_TOOLS_DISABLE_REGISTER:-1}"
fi
# One db per GPU worker. %p is replaced with the process id. The -o path is the merge.
if [[ -z "${RTL_OUTPUT:-}" ]]; then
	export RTL_OUTPUT="${RTL_OUT_DIR}/trace_%p.db"
fi
# rtl trace overwrites HSA_TOOLS_LIB with get_lib_path(), which prefers the wheel
# copy. The pre-script's native library is what matches this image's HSA.
if [[ -f /usr/local/lib/librtl.so ]]; then
	_rtl_pkg=$(python3 -c 'from rocm_trace_lite import get_lib_path; print(get_lib_path())' 2>/dev/null || true)
	if [[ -n "${_rtl_pkg}" && "${_rtl_pkg}" != /usr/local/lib/librtl.so && -f "${_rtl_pkg}" ]]; then
		if ! cmp -s /usr/local/lib/librtl.so "${_rtl_pkg}"; then
			install -m 755 /usr/local/lib/librtl.so "${_rtl_pkg}"
			echo "rocm-trace-lite: replaced ${_rtl_pkg} with the library built for this image." >&2
		fi
	fi
	echo "rocm-trace-lite: librtl $(sha256sum /usr/local/lib/librtl.so | awk '{print $1}')" >&2
fi

rc=0
"${RTL_CLI[@]}" -o "${RTL_DB}" "$@" || rc=$?
if [[ $rc -ne 0 ]]; then
	exit $rc
fi

# Count captured GPU operations across the merged db and per-process dbs.
# roctx UserMarker ranges are CPU-side annotations.
gpu_ops=$(python3 - "${RTL_OUT_DIR}" <<'EOF'
import glob, os, sqlite3, sys
root = sys.argv[1]
total = 0
for db in sorted(set(glob.glob(os.path.join(root, "trace*.db")))):
    if os.path.getsize(db) == 0:
        continue
    try:
        con = sqlite3.connect(db)
        total += con.execute(
            "select count(*) from rocpd_op o left join rocpd_string s on o.opType_id = s.id "
            "where coalesce(s.string, '') != 'UserMarker'"
        ).fetchone()[0]
    except sqlite3.Error:
        pass
print(total)
EOF
)

if [[ "${gpu_ops}" -eq 0 ]]; then
	echo "Error: rocm-trace-lite captured no GPU operations (trace dir: ${RTL_OUT_DIR})." >&2
	echo "  The workload ran, but RTL did not intercept any kernel dispatch." >&2
	echo "  HSA_TOOLS_LIB=${HSA_TOOLS_LIB:-unset}" >&2
	echo "  On TheRock images the prebuilt wheel's librtl.so does not match the" >&2
	echo "  pip-wheel HSA ABI; trace.sh rebuilds it against ROCM_PATH when headers exist." >&2
	echo "  Set RTL_WRAPPER_ALLOW_EMPTY=1 to accept an empty trace." >&2
	if [[ "${RTL_WRAPPER_ALLOW_EMPTY:-0}" != "1" ]]; then
		exit 3
	fi
fi
exit 0

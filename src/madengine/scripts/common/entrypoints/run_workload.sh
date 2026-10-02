#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Agnostic workload entrypoint dispatcher.
#
# A "workload" is a real-world use case for whichever model/container madengine
# has set up: training, inference, finetuning, serving, benchmarking, etc. This
# script is model-agnostic: it is wired in as madengine's `encapsulate_script`,
# so it wraps the model's normal run command and selects what to actually run
# based on the requested workload.
#
# Usage (as an encapsulate_script; madengine appends the model command):
#   bash ../scripts/common/entrypoints/run_workload.sh --workload <name> -- <model command...>
#
# Everything before `--` is parsed as workload options; everything after `--`
# is the model's base command (e.g. `bash run.sh --batch-size 32`).
#
# Resolution order for what gets executed (first match wins):
#   1. $MAD_WORKLOAD_CMD               -- explicit command override (env var)
#   2. ./<workload>.sh                 -- model-provided workload script (convention)
#   3. ./run_<workload>.sh             -- alternate convention
#   4. the base command after `--`     -- the model's default run, with
#                                         MAD_WORKLOAD exported so cooperating
#                                         model scripts can branch on it
#
# In all cases MAD_WORKLOAD (and MAD_WORKLOAD_ARGS) are exported so downstream
# scripts can adapt their behavior.

set -euo pipefail

WORKLOAD="run"
WORKLOAD_ARGS=()
BASE_CMD=()
found_separator=false

while [ "$#" -gt 0 ]; do
    if [ "$found_separator" = true ]; then
        BASE_CMD+=("$1")
        shift
        continue
    fi
    case "$1" in
        --)
            found_separator=true
            shift
            ;;
        --workload)
            WORKLOAD="${2:-run}"
            shift 2
            ;;
        --workload=*)
            WORKLOAD="${1#*=}"
            shift
            ;;
        *)
            # Any other pre-separator token is treated as a workload argument.
            WORKLOAD_ARGS+=("$1")
            shift
            ;;
    esac
done

export MAD_WORKLOAD="${WORKLOAD}"
export MAD_WORKLOAD_ARGS="${WORKLOAD_ARGS[*]:-}"

echo "=============================================================="
echo "madengine workload: ${MAD_WORKLOAD}"
[ -n "${MAD_WORKLOAD_ARGS}" ] && echo "workload args      : ${MAD_WORKLOAD_ARGS}"
echo "model command      : ${BASE_CMD[*]:-<none>}"
echo "=============================================================="

run_convention_script() {
    local script="$1"
    if [ -f "${script}" ]; then
        echo "[run_workload] dispatching to model-provided script: ${script}"
        # Guarded expansion: passing an empty array under `set -u` errors on bash < 4.4.
        exec bash "${script}" ${WORKLOAD_ARGS[@]+"${WORKLOAD_ARGS[@]}"}
    fi
    return 1
}

# 1. Explicit command override.
if [ -n "${MAD_WORKLOAD_CMD:-}" ]; then
    echo "[run_workload] running MAD_WORKLOAD_CMD override"
    exec bash -c "${MAD_WORKLOAD_CMD}"
fi

# 2 & 3. Model-provided convention scripts.
run_convention_script "./${WORKLOAD}.sh" || true
run_convention_script "./run_${WORKLOAD}.sh" || true

# 4. Fall back to the model's base command with MAD_WORKLOAD exported.
if [ "${#BASE_CMD[@]}" -gt 0 ]; then
    echo "[run_workload] running model base command with MAD_WORKLOAD=${MAD_WORKLOAD}"
    exec "${BASE_CMD[@]}"
fi

echo "[run_workload] ERROR: no model command provided and no workload script found for '${WORKLOAD}'." >&2
exit 1

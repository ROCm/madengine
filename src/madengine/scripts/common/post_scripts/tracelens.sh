#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Generate TraceLens reports for whatever trace artifacts the co-selected
# profiling tool produced. Stack this after a profiler, for example:
#   --additional-context '{"tools":[{"name":"rocprofv3_perfetto"},{"name":"tracelens"}]}'

set -x

TRACELENS_VENV=${TRACELENS_VENV:-/opt/madengine-tracelens-venv}
OUTPUT_DIR=${TRACELENS_OUTPUT_DIR:-tracelens_output}
MODE=${TRACELENS_MODE:-auto}

if [ ! -x "${TRACELENS_VENV}/bin/python3" ]; then
    echo "Error: TraceLens venv missing at ${TRACELENS_VENV}. The tracelens pre-script must run first." >&2
    exit 1
fi

if [ -f "scripts/common/tools/tracelens_analyze.py" ]; then
    ANALYZER="scripts/common/tools/tracelens_analyze.py"
elif [ -f "../scripts/common/tools/tracelens_analyze.py" ]; then
    ANALYZER="../scripts/common/tools/tracelens_analyze.py"
else
    echo "Error: Cannot find tracelens_analyze.py" >&2
    exit 1
fi

ARGS=(
    --root .
    --output-dir "$OUTPUT_DIR"
    --mode "$MODE"
    --python "${TRACELENS_VENV}/bin/python3"
    --json-summary "${OUTPUT_DIR}/tracelens_summary.json"
)
if [ -n "${TRACELENS_GPU_ARCH:-}" ]; then
    ARGS+=(--gpu-arch "$TRACELENS_GPU_ARCH")
fi
if [ -n "${TRACELENS_WORLD_SIZE:-}" ]; then
    ARGS+=(--world-size "$TRACELENS_WORLD_SIZE")
fi
if [ -n "${TRACELENS_MAX_TRACES:-}" ]; then
    ARGS+=(--max-traces "$TRACELENS_MAX_TRACES")
fi

mkdir -p "$OUTPUT_DIR"

# Analysis is reporting, not the workload: a TraceLens failure must not turn a
# passing model run into a failure.
if ! python3 "$ANALYZER" "${ARGS[@]}"; then
    echo "WARNING: TraceLens analysis reported failures; see ${OUTPUT_DIR}/tracelens_summary.csv"
fi

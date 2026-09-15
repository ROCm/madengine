#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Stop the dynolog daemon and its trace trigger, then report what was captured.
# Artifact collection into /myworkspace is handled by post_scripts/trace.sh.

set -x

echo "Stopping dynolog daemon..."

DYNOLOG_PID_FILE="/tmp/madengine_dynolog.pid"
TRIGGER_PID_FILE="/tmp/madengine_dynolog_trigger.pid"
DYNOLOG_START_FILE="/tmp/madengine_dynolog.started"
RESULT_FILE="/tmp/madengine_dynolog_trigger.result"

OUTPUT_DIR=${TORCH_PROFILE_OUTPUT_DIR:-torch_profiler_output}

if [ ! -f "$DYNOLOG_START_FILE" ]; then
    echo "⚠️  Warning: dynolog was not started - skipping"
    exit 0
fi

stop_pid() {
    local name=$1
    local pid_file=$2
    if [ ! -f "$pid_file" ]; then
        echo "⚠️  Warning: $name PID file not found"
        return 0
    fi
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ $waited -lt 20 ]; do
            sleep 0.5
            waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "⚠️  $name did not stop gracefully, force killing..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "✓ $name stopped (PID: $pid)"
    else
        echo "⚠️  $name (PID: $pid) was no longer running"
    fi
    rm -f "$pid_file"
}

# Stop the trigger first so it cannot issue a request against a dying daemon.
stop_pid "dynolog trace trigger" "$TRIGGER_PID_FILE"
stop_pid "dynolog daemon" "$DYNOLOG_PID_FILE"
rm -f "$DYNOLOG_START_FILE"

# Kineto appends the process id to the requested filename, so a multi-rank run
# produces one file per rank.
trace_count=0
if [ -d "$OUTPUT_DIR" ]; then
    trace_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.json.gz' \) 2>/dev/null | wc -l)
fi

if [ "$trace_count" -gt 0 ]; then
    echo "✓ Captured ${trace_count} torch.profiler trace(s) in ${OUTPUT_DIR}"
    ls -la "$OUTPUT_DIR" || true
else
    echo "⚠️  No torch.profiler traces were captured in ${OUTPUT_DIR}"
    if [ -f "$RESULT_FILE" ] && [ "$(cat "$RESULT_FILE")" = "no_process" ]; then
        echo "⚠️  The trigger never matched a PyTorch process. Most likely causes:"
        echo "⚠️    - the workload is not PyTorch, or predates PyTorch 1.13"
        echo "⚠️    - the model run finished before TORCH_PROFILE_WARMUP_S elapsed"
    fi
fi

for log in /tmp/madengine_dynolog.log /tmp/madengine_dynolog_trigger.log; do
    if [ -f "$log" ]; then
        echo "=== $(basename "$log") ==="
        tail -40 "$log" || true
        echo "=========================="
    fi
done

echo "✓ dynolog cleanup complete"

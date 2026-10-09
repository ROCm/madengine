#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Start the dynolog daemon and arm a background trigger that requests a
# torch.profiler trace once the workload's PyTorch processes have registered.

set -x

echo "Starting dynolog daemon for on-demand torch.profiler tracing..."

PORT=${DYNOLOG_PORT:-1778}
OUTPUT_DIR=${TORCH_PROFILE_OUTPUT_DIR:-torch_profiler_output}

DYNOLOG_PID_FILE="/tmp/madengine_dynolog.pid"
TRIGGER_PID_FILE="/tmp/madengine_dynolog_trigger.pid"
DYNOLOG_START_FILE="/tmp/madengine_dynolog.started"

if ! command -v dynolog >/dev/null 2>&1 || ! command -v dyno >/dev/null 2>&1; then
    echo "Error: dynolog/dyno not on PATH. The dynolog pre-script must run first."
    exit 1
fi

# Traces are written by the workload process itself (via Kineto), so the output
# directory has to exist before the trigger fires.
mkdir -p "$OUTPUT_DIR"

# --enable_ipc_monitor is what allows the daemon to talk to PyTorch/Kineto.
nohup dynolog --enable_ipc_monitor --port "$PORT" \
    > /tmp/madengine_dynolog.log 2>&1 &
DYNOLOG_PID=$!
echo "$DYNOLOG_PID" > "$DYNOLOG_PID_FILE"

# Give the daemon time to bind its port before the workload tries to register.
sleep 3

if ! kill -0 "$DYNOLOG_PID" 2>/dev/null; then
    echo "Error: dynolog daemon exited immediately. Log follows:"
    cat /tmp/madengine_dynolog.log || true
    rm -f "$DYNOLOG_PID_FILE"
    exit 1
fi
echo "✓ dynolog daemon started (PID: $DYNOLOG_PID, port: $PORT)"

# The trigger has to run alongside the workload, because `dyno gputrace` can only
# match PyTorch processes that have already started and registered.
if [ -f "scripts/common/tools/dynolog_trigger.sh" ]; then
    TRIGGER_SCRIPT="scripts/common/tools/dynolog_trigger.sh"
elif [ -f "../scripts/common/tools/dynolog_trigger.sh" ]; then
    TRIGGER_SCRIPT="../scripts/common/tools/dynolog_trigger.sh"
else
    echo "Error: Cannot find dynolog_trigger.sh"
    exit 1
fi

# Iteration capture counts optimizer.step(), and PyTorch registers that hook
# only when torch.profiler is imported. Unmodified workloads never import it,
# so the daemon matches the process and then waits forever. A site hook imports
# it for this run; dynolog_stop.sh removes the hook. Install it before the
# trigger is armed: the import check is itself a PyTorch process, and a trigger
# that is already polling will accept that short-lived process and exit.
install_kineto_hook() {
    local hook_src=""
    if [ -f "scripts/common/tools/madengine_kineto_hook.py" ]; then
        hook_src="scripts/common/tools/madengine_kineto_hook.py"
    elif [ -f "../scripts/common/tools/madengine_kineto_hook.py" ]; then
        hook_src="../scripts/common/tools/madengine_kineto_hook.py"
    else
        echo "Warning: madengine_kineto_hook.py not found; iteration capture needs torch.profiler imported."
        return 0
    fi
    # This check imports torch. With the daemon already running that registers a
    # process which exits immediately, and the worker then segfaults on flush.
    if ! env -u KINETO_USE_DAEMON python3 -c 'import torch' >/dev/null 2>&1; then
        echo "Warning: torch is not importable; skipping the Kineto optimizer-step hook."
        return 0
    fi
    local site
    site=$(env -u KINETO_USE_DAEMON python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)
    if [ -z "$site" ] || [ ! -d "$site" ]; then
        echo "Warning: could not find site-packages; skipping the Kineto optimizer-step hook."
        return 0
    fi
    cp "$hook_src" "${site}/madengine_kineto_hook.py"
    printf '%s\n' 'import madengine_kineto_hook' > "${site}/madengine_kineto_hook.pth"
    echo "$site" > /tmp/madengine_kineto_hook.site
    echo "✓ installed Kineto optimizer-step hook in ${site}"
}
install_kineto_hook

nohup bash "$TRIGGER_SCRIPT" > /tmp/madengine_dynolog_trigger.log 2>&1 &
TRIGGER_PID=$!
echo "$TRIGGER_PID" > "$TRIGGER_PID_FILE"
echo "✓ dynolog trace trigger armed (PID: $TRIGGER_PID)"

touch "$DYNOLOG_START_FILE"
echo "✓ dynolog initialization complete"

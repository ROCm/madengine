#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Request a torch.profiler trace from the running workload via dynolog.
#
# Runs in the background for the lifetime of the model run. `dyno gputrace` can
# only configure PyTorch processes that have already registered with the daemon,
# and there is no way to know when the workload reaches steady state, so this
# polls until a request is accepted.
#
# `dyno gputrace` exits 0 whether or not it matched anything, so the outcome has
# to be read from its output: the response carries the matched pids, and an empty
# `processesMatched` list means the workload has not registered yet.

set -u

PORT=${DYNOLOG_PORT:-1778}
OUTPUT_DIR=${TORCH_PROFILE_OUTPUT_DIR:-torch_profiler_output}
LOG_NAME=${TORCH_PROFILE_LOG_FILE:-libkineto_trace.json}
ITERATIONS=${TORCH_PROFILE_ITERATIONS:-5}
DURATION_MS=${TORCH_PROFILE_DURATION_MS:-500}
WARMUP_S=${TORCH_PROFILE_WARMUP_S:-60}
RETRY_INTERVAL_S=${TORCH_PROFILE_RETRY_INTERVAL_S:-15}
MAX_ATTEMPTS=${TORCH_PROFILE_MAX_ATTEMPTS:-40}
# Upstream defaults to 3, which silently drops most ranks of a multi-GPU job.
PROCESS_LIMIT=${TORCH_PROFILE_PROCESS_LIMIT:-64}
JOB_ID=${TORCH_PROFILE_JOB_ID:-${SLURM_JOB_ID:-0}}

RESULT_FILE="/tmp/madengine_dynolog_trigger.result"
rm -f "$RESULT_FILE"

# TraceLens needs input shapes and CPU call stacks for per-op and roofline
# analysis, and modules for the nn.Module breakdown.
OPTS=()
[ "${TORCH_PROFILE_RECORD_SHAPES:-1}" = "1" ] && OPTS+=(--record-shapes)
[ "${TORCH_PROFILE_WITH_STACKS:-1}" = "1" ] && OPTS+=(--with-stacks)
[ "${TORCH_PROFILE_WITH_MODULES:-1}" = "1" ] && OPTS+=(--with-modules)
[ "${TORCH_PROFILE_WITH_FLOPS:-0}" = "1" ] && OPTS+=(--with-flops)
[ "${TORCH_PROFILE_PROFILE_MEMORY:-0}" = "1" ] && OPTS+=(--profile-memory)

# Iteration-based capture counts optimizer.step(). That hook is registered only
# when torch.profiler is imported; dynolog_start.sh installs it for the run.
# Workloads with no optimizer step need TORCH_PROFILE_ITERATIONS=0.
if [ "$ITERATIONS" -gt 0 ] 2>/dev/null; then
    OPTS+=(--iterations "$ITERATIONS")
else
    OPTS+=(--duration-ms "$DURATION_MS")
fi

mkdir -p "$OUTPUT_DIR"
# Kineto renames a temporary file onto the log path when the window ends.
# That rename leaves no file when the path is on the workspace bind mount, so
# the trace is written on the container's own disk. dynolog_stop.sh copies it
# into OUTPUT_DIR, which is what gets collected.
KINETO_DIR=${TORCH_PROFILE_KINETO_DIR:-/tmp/madengine_kineto}
if [ "$KINETO_DIR" = "/tmp/madengine_kineto" ]; then
    rm -rf "$KINETO_DIR"
fi
mkdir -p "$KINETO_DIR"
LOG_FILE="${KINETO_DIR}/${LOG_NAME}"

# The daemon remembers every process that registered, including pre-scripts
# that imported torch and exited, and the torchrun launcher, which never calls
# optimizer.step(). A gputrace that names either is reported as installed and
# then never finalizes the worker's trace. Ask only for processes that are
# still running and are not that launcher. Zombies count as dead: kill -0
# still succeeds for them. A script path that merely contains "torchrun"
# (run_torchrun.py) is the workload and must stay. Only the first pid on each
# registration line registered; the rest are ancestors. Requesting an ancestor
# makes dynolog trace the launcher again.
live_registered_pids() {
    local log=/tmp/madengine_dynolog.log
    if [ ! -f "$log" ]; then
        return 0
    fi
    local raw pid state cmd live=""
    raw=$(
        grep "Registered process" "$log" 2>/dev/null \
            | sed -n 's/.*Registered process (\([0-9][0-9]*\).*/\1/p' \
            || true
    )
    if [ -z "$raw" ]; then
        return 0
    fi
    for pid in $raw; do
        kill -0 "$pid" 2>/dev/null || continue
        if [ -r "/proc/$pid/stat" ]; then
            state=$(sed 's/.*) //' "/proc/$pid/stat" 2>/dev/null | cut -d' ' -f1)
            [ "$state" = "Z" ] && continue
        fi
        if [ -r "/proc/$pid/cmdline" ]; then
            cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
            case "$cmd" in
                *torch.distributed.run*|*torch.distributed.elastic*)
                    continue
                    ;;
                *torch/distributed/run.py*)
                    continue
                    ;;
            esac
            if printf '%s\n' "$cmd" | grep -Eq '(^|[[:space:]/])torchrun([[:space:]]|$)'; then
                continue
            fi
        fi
        case ",${live}," in
            *",${pid},"*) continue ;;
        esac
        if [ -n "$live" ]; then
            live="${live},${pid}"
        else
            live=$pid
        fi
    done
    printf '%s\n' "$live"
}

echo "[dynolog-trigger] waiting ${WARMUP_S}s for the workload to reach steady state"
sleep "$WARMUP_S"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    pid_args=()
    live_pids=$(live_registered_pids || true)
    if [ -n "$live_pids" ]; then
        pid_args=(--pids "$live_pids")
        echo "[dynolog-trigger] attempt ${attempt}/${MAX_ATTEMPTS}: live pids ${live_pids}"
    elif [ -f /tmp/madengine_dynolog.log ] && grep -q "Registered process" /tmp/madengine_dynolog.log 2>/dev/null; then
        echo "[dynolog-trigger] attempt ${attempt}/${MAX_ATTEMPTS}: no runnable workload process yet; retrying in ${RETRY_INTERVAL_S}s"
        sleep "$RETRY_INTERVAL_S"
        continue
    fi
    echo "[dynolog-trigger] attempt ${attempt}/${MAX_ATTEMPTS}: requesting trace -> ${LOG_FILE}"
    response=$(dyno --port "$PORT" gputrace \
        --job-id "$JOB_ID" \
        --log-file "$LOG_FILE" \
        --process-limit "$PROCESS_LIMIT" \
        "${pid_args[@]}" \
        "${OPTS[@]}" 2>&1)
    echo "$response"

    if echo "$response" | grep -q '"processesMatched":\[[0-9]'; then
        echo "[dynolog-trigger] trace request accepted on attempt ${attempt}"
        echo "accepted" > "$RESULT_FILE"
        exit 0
    fi

    # A response that reports no matches is the expected case while the workload
    # is still starting up. Anything else means dyno rejected the request itself
    # (an unsupported flag, an unreachable daemon), which retrying cannot fix.
    if ! echo "$response" | grep -q 'processesMatched'; then
        echo "[dynolog-trigger] dyno rejected the request; not retrying."
        echo "[dynolog-trigger] Check the dyno output above against the installed"
        echo "[dynolog-trigger] dynolog version ('dyno gputrace --help')."
        echo "request_rejected" > "$RESULT_FILE"
        exit 1
    fi

    echo "[dynolog-trigger] no PyTorch process matched yet; retrying in ${RETRY_INTERVAL_S}s"
    sleep "$RETRY_INTERVAL_S"
done

echo "[dynolog-trigger] gave up after ${MAX_ATTEMPTS} attempts: no PyTorch process registered."
echo "[dynolog-trigger] Confirm the workload runs PyTorch >= 1.13 with KINETO_USE_DAEMON=1,"
echo "[dynolog-trigger] and raise TORCH_PROFILE_WARMUP_S / TORCH_PROFILE_MAX_ATTEMPTS for slow starts."
echo "no_process" > "$RESULT_FILE"
exit 1

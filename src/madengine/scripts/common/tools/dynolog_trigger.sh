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

# Iteration-based capture needs an optimizer step hook; PyTorch falls back to a
# duration-based trace on its own when it cannot count iterations.
if [ "$ITERATIONS" -gt 0 ] 2>/dev/null; then
    OPTS+=(--iterations "$ITERATIONS")
else
    OPTS+=(--duration-ms "$DURATION_MS")
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$(cd "$OUTPUT_DIR" && pwd)/${LOG_NAME}"

echo "[dynolog-trigger] waiting ${WARMUP_S}s for the workload to reach steady state"
sleep "$WARMUP_S"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    echo "[dynolog-trigger] attempt ${attempt}/${MAX_ATTEMPTS}: requesting trace -> ${LOG_FILE}"
    response=$(dyno --port "$PORT" gputrace \
        --job-id "$JOB_ID" \
        --log-file "$LOG_FILE" \
        --process-limit "$PROCESS_LIMIT" \
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

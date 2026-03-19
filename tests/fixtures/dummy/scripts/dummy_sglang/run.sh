#!/bin/bash
#
# SGLang Distributed Inference Script
#
# Single-node: One SGLang instance with Tensor Parallelism (TP) across GPUs on the node.
# Multi-node: One SGLang instance per node (TP only on that node). No shared Ray across nodes
#   = Data Parallelism: each node runs the same benchmark independently; SLURM runs one task per node.
# Aligns with dummy_vllm pattern: pipeline-based tee (exit 0 on SIGPIPE), one serve per node.
#
set -e

# Ignore SIGPIPE so the script does not exit 141 when the stdout reader closes the pipe
# (e.g. docker exec or SLURM log capture closing before cleanup finishes).
trap '' PIPE

# When run under docker exec (non-TTY pipe), re-exec with line-buffered stdout/stderr for streaming logs.
if [ -z "${RUNSH_UNBUFFERED:-}" ]; then
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  if command -v stdbuf >/dev/null 2>&1; then
    export RUNSH_UNBUFFERED=1
    exec stdbuf -oL -eL env RUNSH_UNBUFFERED=1 bash "$SCRIPT_PATH" "$@"
  fi
  if command -v script >/dev/null 2>&1; then
    export RUNSH_UNBUFFERED=1
    exec script -q -c "env RUNSH_UNBUFFERED=1 bash $(printf '%q' "$SCRIPT_PATH")" /dev/null
  fi
  if command -v python3 >/dev/null 2>&1; then
    export RUNSH_UNBUFFERED=1
    exec python3 -u -c "
import os, subprocess, sys
os.environ['RUNSH_UNBUFFERED'] = '1'
p = subprocess.Popen(
    ['bash', sys.argv[1]],
    bufsize=1,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=os.environ,
    text=True,
)
for line in iter(p.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
sys.exit(p.wait())
" "$SCRIPT_PATH"
  fi
  echo "madengine SGLang: stdout may be buffered; progress will be in run_directory/sglang_run.log" >&2
fi

# Cleanup: terminate SGLang/GPU processes on exit
cleanup() {
    EXIT_CODE=$?
    echo ""
    echo "========================================================================"
    echo "Cleanup: Terminating SGLang/GPU processes..."
    echo "========================================================================"
    pkill -9 -f "sglang" 2>/dev/null || true
    if command -v rocm-smi &> /dev/null; then
        echo "Final GPU state:"
        rocm-smi 2>/dev/null || true
    elif command -v amd-smi &> /dev/null; then
        amd-smi list 2>/dev/null || true
    fi
    echo "Cleanup completed (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
}

trap cleanup EXIT INT TERM SIGINT SIGTERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Deployment configuration (same as dummy_vllm)
NNODES=${NNODES:-1}
GPUS_PER_NODE=${MAD_RUNTIME_NGPUS:-${NPROC_PER_NODE:-1}}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# One SGLang per node: TP = GPUs on this node only (data parallel across nodes)
TP_SIZE=${SGLANG_TENSOR_PARALLEL_SIZE:-$GPUS_PER_NODE}
MODEL_NAME=${SGLANG_MODEL_PATH:-facebook/opt-125m}
EXECUTION_MODE=${SGLANG_EXECUTION_MODE:-offline}

export PYTHONUNBUFFERED=1

# Run the entire workload in a pipeline with tee so that when the consumer (docker exec / SLURM)
# closes the pipe, we capture PIPESTATUS[0] and treat 141 (SIGPIPE) as success.
{
  trap '' PIPE
  set -e
  echo "========================================================================"
  echo "madengine SGLang Inference (one serve per node, TP only)"
  echo "========================================================================"
  echo "Deployment Configuration:"
  echo "  Model: $MODEL_NAME"
  echo "  Nodes: $NNODES"
  echo "  GPUs per node: $GPUS_PER_NODE"
  echo "  Node rank: $NODE_RANK"
  echo "  Tensor Parallel Size: $TP_SIZE"
  echo "  Execution mode: $EXECUTION_MODE"
  echo "========================================================================"

  if [ "$EXECUTION_MODE" = "server" ]; then
    echo "Running in SERVER mode (OpenAI-compatible API)"
    if [ "$NNODES" -gt 1 ]; then
      python3 -m sglang.launch_server \
        --model-path "$MODEL_NAME" \
        --tp "$TP_SIZE" \
        --nnodes "$NNODES" \
        --node-rank "$NODE_RANK" \
        --nccl-init-addr "${MASTER_ADDR}:${MASTER_PORT}" \
        --host 0.0.0.0 \
        --port 30000
    else
      python3 -m sglang.launch_server \
        --model-path "$MODEL_NAME" \
        --tp "$TP_SIZE" \
        --host 0.0.0.0 \
        --port 30000
    fi
  else
    echo "Running in OFFLINE mode (batch inference benchmark)"
    # One run per node: TP on this node only (nnodes=1 from this process's perspective)
    python3 run_sglang_inference.py \
      --model "$MODEL_NAME" \
      --tp-size "$TP_SIZE" \
      --nnodes 1 \
      --node-rank 0 \
      --master-addr "$MASTER_ADDR" \
      --master-port "$MASTER_PORT"
  fi

  echo "========================================================================"
  echo "Node ${NODE_RANK}: Inference completed successfully"
  echo "========================================================================"
} | tee -a sglang_run.log

# When tee is killed (SIGPIPE), the left side of the pipeline may exit 141; treat as success
EXIT=${PIPESTATUS[0]}
[ "$EXIT" = "141" ] && EXIT=0
exit $EXIT

#!/bin/bash
#
# vLLM V1 Engine Distributed Inference Script
#
# Single-node: One vLLM instance with Tensor Parallelism (TP) across GPUs on the node.
# Multi-node: One vLLM instance per node (TP only on that node). No shared Ray across nodes
#   = Data Parallelism: each node runs the same benchmark independently; SLURM runs one task per node.
# Aligns with vLLM docs "External Load Balancing" / independent servers per node.
#
set -e

# Ignore SIGPIPE so the script does not exit 141 when the stdout reader closes the pipe
# (e.g. docker exec or SLURM log capture closing before cleanup finishes).
trap '' PIPE

# When run under docker exec (non-TTY pipe), bash buffers stdout. Re-exec with line-buffered
# stdout/stderr so logs stream to the host and SLURM logs show progress.
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
  echo "madengine vLLM: stdout may be buffered; progress will be in run_directory/vllm_run.log" >&2
fi

# Cleanup function to ensure GPU processes are properly terminated (no Ray in data-parallel mode)
cleanup() {
    EXIT_CODE=$?
    echo ""
    echo "========================================================================"
    echo "Cleanup: Terminating GPU processes..."
    echo "========================================================================"
    echo "Killing vLLM processes..."
    pkill -9 -f "vllm" 2>/dev/null || true
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

# Detect deployment configuration from environment
NNODES=${NNODES:-1}
GPUS_PER_NODE=${MAD_RUNTIME_NGPUS:-${GPUS_PER_NODE:-1}}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# One vLLM per node: TP = GPUs on this node only; no Ray
TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-$GPUS_PER_NODE}
PIPELINE_PARALLEL_SIZE=1
VLLM_DISTRIBUTED_BACKEND=${VLLM_DISTRIBUTED_BACKEND:-none}

MODEL_NAME=${MODEL_NAME:-facebook/opt-350m}
export PYTHONUNBUFFERED=1

# Run the entire workload in a pipeline with tee so that when the consumer (docker exec / SLURM)
# closes the pipe, only tee gets SIGPIPE; we capture PIPESTATUS[0] and treat 141 (SIGPIPE) as success.
# This prevents "Subprocess failed with exit code 1" when inference actually completed.
# Subshell (pipeline left side) ignores SIGPIPE so a broken pipe does not kill the script.
{
  trap '' PIPE
  set -e
  echo "========================================================================"
  echo "madengine vLLM V1 Engine Inference Script (one serve per node, TP only)"
  echo "========================================================================"
  echo "Deployment Configuration:"
  echo "  Model: $MODEL_NAME"
  echo "  Nodes: $NNODES"
  echo "  GPUs per node: $GPUS_PER_NODE"
  echo "  Node rank: $NODE_RANK"
  echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
  echo "  Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
  echo "  Distributed backend: $VLLM_DISTRIBUTED_BACKEND (no Ray = data parallel across nodes)"
  echo "========================================================================"

  # GPU visibility
  if command -v rocm-smi &> /dev/null || command -v rocminfo &> /dev/null; then
    unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
    unset ROCR_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES
    echo "GPU environment (AMD): HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
    unset HIP_VISIBLE_DEVICES
    unset ROCR_VISIBLE_DEVICES
    echo "GPU environment (NVIDIA): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  fi

  # Single-node and multi-node (data parallel): same command — one vLLM per node, TP only
  echo "Running vLLM: TP=$TENSOR_PARALLEL_SIZE PP=1 backend=$VLLM_DISTRIBUTED_BACKEND"
  python3 run_vllm_inference.py \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size 1 \
    --distributed-backend none

  echo "========================================================================"
  echo "Node ${NODE_RANK}: Inference completed successfully"
  echo "========================================================================"
} | tee -a vllm_run.log

# When tee is killed (SIGPIPE), the left side of the pipeline may exit 141; treat as success
EXIT=${PIPESTATUS[0]}
[ "$EXIT" = "141" ] && EXIT=0
exit $EXIT

#!/bin/bash
#
# vLLM V1 Engine Distributed Inference Script - Data Parallelism Mode
#
# Multi-node Data Parallelism Strategy:
# - Each node runs an INDEPENDENT vLLM replica (no shared Ray cluster)
# - Each replica uses Tensor Parallelism across GPUs within the node
# - Benefits: Simpler, faster init, more robust, better for benchmarking
#
set -e

echo "========================================================================"
echo "madengine vLLM V1 Engine Inference Script"
echo "========================================================================"

# Cleanup function to ensure Ray and GPU processes are properly terminated
cleanup() {
    EXIT_CODE=$?
    echo ""
    echo "========================================================================"
    echo "Cleanup: Terminating Ray cluster and GPU processes..."
    echo "========================================================================"
    
    # Stop Ray cluster
    if command -v ray &> /dev/null; then
        echo "Stopping Ray cluster..."
        ray stop --force 2>/dev/null || true
        sleep 2
    fi
    
    # Kill any lingering Ray processes
    echo "Killing lingering Ray processes..."
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "RayWorkerWrapper" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    
    # Kill any vLLM processes
    echo "Killing vLLM processes..."
    pkill -9 -f "vllm" 2>/dev/null || true
    
    # Display final GPU state
    if command -v rocm-smi &> /dev/null; then
        echo "Final GPU state:"
        rocm-smi 2>/dev/null || true
    elif command -v amd-smi &> /dev/null; then
        amd-smi list 2>/dev/null || true
    fi
    
    echo "Cleanup completed (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
}

# Register cleanup function to run on script exit (success, failure, or interruption)
trap cleanup EXIT INT TERM SIGINT SIGTERM

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect deployment configuration from environment
NNODES=${NNODES:-1}
GPUS_PER_NODE=${MAD_RUNTIME_NGPUS:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Model selection
MODEL_NAME=${MODEL_NAME:-facebook/opt-125m}

echo "========================================================================"
echo "Deployment Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Nodes: $NNODES"
echo "  GPUs available: $GPUS_PER_NODE"
echo "  Node rank: $NODE_RANK"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "========================================================================"

# Determine parallelism strategy
if [ "$NNODES" -eq 1 ]; then
    # Single node with multiple GPUs: use tensor parallelism
    TENSOR_PARALLEL_SIZE=$GPUS_PER_NODE
    PIPELINE_PARALLEL_SIZE=1
    DISTRIBUTED_BACKEND="auto"  # Will use default (no Ray needed)
    
    echo "Single-node mode: Using Tensor Parallelism"
    echo "  TP Size: $TENSOR_PARALLEL_SIZE"
else
    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-NODE DATA PARALLELISM MODE
    # ═══════════════════════════════════════════════════════════════════════
    # Strategy: Each node runs an INDEPENDENT vLLM replica
    # - No shared Ray cluster across nodes
    # - Each node: Local Ray + Tensor Parallelism
    # - Benefits: Simpler, faster init, more robust, better for benchmarking
    # ═══════════════════════════════════════════════════════════════════════
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║          MULTI-NODE DATA PARALLELISM MODE                          ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Total nodes: ${NNODES}"
    echo "  Current node rank: ${NODE_RANK}"
    echo "  GPUs per node: ${GPUS_PER_NODE}"
    echo "  Data Parallelism: ${NNODES} independent replicas"
    echo "  Tensor Parallelism: ${GPUS_PER_NODE} GPUs per replica"
    echo "  Total GPUs: $((NNODES * GPUS_PER_NODE))"
    echo ""
    
    # Data Parallelism: TP per node, NO Pipeline Parallelism
    TENSOR_PARALLEL_SIZE=$GPUS_PER_NODE
    PIPELINE_PARALLEL_SIZE=1  # No pipeline parallelism in DP mode!
    DISTRIBUTED_BACKEND="ray"
    
    # Set GPU environment variables for visibility
    # CRITICAL: Ray requires ONLY ONE visibility variable
    # - AMD GPUs: Use ONLY HIP_VISIBLE_DEVICES
    # - NVIDIA GPUs: Use ONLY CUDA_VISIBLE_DEVICES
    # Setting both causes Ray error: "Inconsistent values found"
    if command -v rocm-smi &> /dev/null || command -v rocminfo &> /dev/null; then
        # AMD GPU detected - use HIP_VISIBLE_DEVICES ONLY
        # CRITICAL: Unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES which is set by rocm/vllm image
        # This variable tells Ray to ignore HIP_VISIBLE_DEVICES, causing conflicts
        unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
        export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
        unset ROCR_VISIBLE_DEVICES  # Unset to avoid Ray conflicts
        unset CUDA_VISIBLE_DEVICES  # Unset to avoid "Inconsistent values" error
        echo "  GPU environment (AMD): HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
    else
        # NVIDIA GPU - use CUDA_VISIBLE_DEVICES ONLY
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
        unset HIP_VISIBLE_DEVICES   # Unset to avoid Ray conflicts
        unset ROCR_VISIBLE_DEVICES
        echo "  GPU environment (NVIDIA): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
    echo ""
    
    # Get current node IP
    CURRENT_NODE_IP=$(getent hosts $(hostname) | awk '{print $1}' | head -1)
    echo "  Node $(hostname) IP: $CURRENT_NODE_IP"
    export VLLM_HOST_IP="$CURRENT_NODE_IP"
    
    # Clean any existing Ray processes from previous jobs
    echo "  Cleaning any existing Ray processes..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    sleep 2
    
    # Start INDEPENDENT Ray cluster on THIS node only
    # NOTE: Each node starts its own Ray cluster (NOT shared across nodes!)
    echo "  Starting independent Ray cluster on Node ${NODE_RANK}..."
    ray start --head --port=6379 --node-ip-address="$CURRENT_NODE_IP" --num-gpus=$GPUS_PER_NODE
    
    sleep 3
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Ray cluster ready on Node ${NODE_RANK}"
    echo "═══════════════════════════════════════════════════════════════════"
    ray status
    echo ""
fi

echo "========================================================================"
echo "vLLM V1 Configuration:"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
echo "  Distributed Backend: $DISTRIBUTED_BACKEND"
if [ "$NNODES" -gt 1 ]; then
    echo "  Data Parallel Size: $NNODES"
fi
echo "========================================================================"

# Export environment for vLLM
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT

# Data Parallelism: ALL nodes run inference independently
echo ""
echo "Node ${NODE_RANK}: Launching vLLM inference..."
python3 run_vllm_inference.py \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --distributed-backend "$DISTRIBUTED_BACKEND"

# Note: cleanup() trap handler will run automatically on exit
echo "========================================================================"
echo "Node ${NODE_RANK}: Inference completed successfully"
echo "========================================================================"


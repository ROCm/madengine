#!/bin/bash
#
# vLLM V1 Engine Distributed Inference Script
#
# vLLM V1 manages its own process spawning - DO NOT use torchrun!
# The V1 engine automatically handles:
# - Tensor parallelism (TP) within a node
# - Data parallelism (DP) across replicas
# - Multi-node coordination via Ray
#
set -e

echo "========================================================================"
echo "MADEngine vLLM V1 Engine Inference Script"
echo "========================================================================"

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
# Single-node scenarios:
if [ "$NNODES" -eq 1 ]; then
    # Single node with multiple GPUs: use tensor parallelism
    TENSOR_PARALLEL_SIZE=$GPUS_PER_NODE
    PIPELINE_PARALLEL_SIZE=1
    DISTRIBUTED_BACKEND="auto"  # Will use default (no Ray needed)
    
    echo "Single-node mode: Using Tensor Parallelism"
    echo "  TP Size: $TENSOR_PARALLEL_SIZE"
else
    # Multi-node: use pipeline parallelism + tensor parallelism
    # TP within node, PP across nodes
    TENSOR_PARALLEL_SIZE=$GPUS_PER_NODE
    PIPELINE_PARALLEL_SIZE=$NNODES
    DISTRIBUTED_BACKEND="ray"  # Ray required for multi-node
    
    echo "Multi-node mode: Using Pipeline + Tensor Parallelism"
    echo "  TP Size (per node): $TENSOR_PARALLEL_SIZE"
    echo "  PP Size (nodes): $PIPELINE_PARALLEL_SIZE"
    echo "  Backend: Ray"
    
    # Initialize Ray cluster if multi-node
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "Initializing Ray head node..."
        ray start --head --port=6379 --node-ip-address="$MASTER_ADDR" || true
    else
        echo "Connecting to Ray head node at $MASTER_ADDR..."
        ray start --address="$MASTER_ADDR:6379" || true
    fi
fi

echo "========================================================================"
echo "vLLM V1 Configuration:"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
echo "  Distributed Backend: $DISTRIBUTED_BACKEND"
echo "========================================================================"

# Export environment for vLLM
export NNODES
export MASTER_ADDR
export MASTER_PORT

# Launch vLLM inference - DIRECT PYTHON, NO TORCHRUN!
# vLLM V1 handles its own multiprocessing
python3 run_vllm_inference.py \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --distributed-backend "$DISTRIBUTED_BACKEND"

EXIT_CODE=$?

# Cleanup Ray if multi-node
if [ "$NNODES" -gt 1 ]; then
    echo "Stopping Ray..."
    ray stop || true
fi

echo "========================================================================"
echo "Inference script completed with exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE


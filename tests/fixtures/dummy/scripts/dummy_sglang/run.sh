#!/bin/bash
#
# SGLang Distributed Inference Script
#
# SGLang has its own native launcher (sglang.launch_server) - NO torchrun needed!
# Uses Ray for distributed coordination internally
#
set -e

echo "========================================================================"
echo "madengine SGLang Inference Wrapper Script"
echo "========================================================================"

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect deployment configuration from environment
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "========================================================================"
echo "Deployment Configuration:"
echo "  Nodes: $NNODES"
echo "  GPUs per node: $NPROC_PER_NODE"
echo "  Node rank: $NODE_RANK"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "========================================================================"

# SGLang-specific parallelism
# - Tensor Parallelism (TP): Split model across GPUs within a node
# - Data Parallelism (DP): Distribute requests across nodes (via multi-node setup)
TP_SIZE=$NPROC_PER_NODE  # Tensor parallel within node
TOTAL_GPUS=$((TP_SIZE * NNODES))

echo "========================================================================"
echo "SGLang Parallelism Configuration:"
echo "  Tensor Parallel (TP) Size: $TP_SIZE (GPUs per node)"
echo "  Number of Nodes: $NNODES"
echo "  Total GPUs: $TOTAL_GPUS"
echo "========================================================================"

# Choose execution mode: server or offline batch inference
# Server mode: Launches SGLang server for OpenAI-compatible API
# Offline mode: Runs batch inference directly (better for benchmarking)
EXECUTION_MODE=${SGLANG_EXECUTION_MODE:-offline}

if [ "$EXECUTION_MODE" = "server" ]; then
    echo "========================================================================"
    echo "Running in SERVER mode (OpenAI-compatible API)"
    echo "========================================================================"
    
    if [ $NNODES -gt 1 ]; then
        echo "Multi-node server setup - using SGLang native launcher"
        
        # SGLang multi-node server launch
        # Each node must run this command with appropriate node_rank
        python3 -m sglang.launch_server \
            --model-path "facebook/opt-125m" \
            --tp $TP_SIZE \
            --nnodes $NNODES \
            --node-rank $NODE_RANK \
            --nccl-init-addr "${MASTER_ADDR}:${MASTER_PORT}" \
            --host 0.0.0.0 \
            --port 30000
    else
        echo "Single-node server setup - using SGLang native launcher"
        
        # SGLang single-node server launch
        python3 -m sglang.launch_server \
            --model-path "facebook/opt-125m" \
            --tp $TP_SIZE \
            --host 0.0.0.0 \
            --port 30000
    fi
else
    echo "========================================================================"
    echo "Running in OFFLINE mode (batch inference benchmark)"
    echo "========================================================================"
    
    # For offline batch inference, we use SGLang's Runtime directly
    # No need for torchrun - SGLang handles distributed setup via Ray
    python3 run_sglang_inference.py \
        --model "facebook/opt-125m" \
        --tp-size $TP_SIZE \
        --nnodes $NNODES \
        --node-rank $NODE_RANK \
        --master-addr $MASTER_ADDR \
        --master-port $MASTER_PORT
fi

echo "========================================================================"
echo "Inference script completed"
echo "========================================================================"

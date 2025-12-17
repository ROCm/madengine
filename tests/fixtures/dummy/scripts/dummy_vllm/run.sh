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
    echo "  Total GPUs: $((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))"
    
    # Set GPU environment variables for visibility
    export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0,1,2,3}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    echo "GPU environment: ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
    
    # vLLM Best Practice for Multi-Node: Start Ray manually, vLLM auto-connects
    # This allows proper per-node GPU allocation
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "Starting Ray head node..."
        ray start --head --port=6379 --node-ip-address="$MASTER_ADDR" --num-gpus=$GPUS_PER_NODE --block &
        # Longer delay to ensure Ray head is fully initialized
        sleep 15
        echo "Ray head node started and ready"
    else
        echo "Worker node: connecting to Ray head at $MASTER_ADDR..."
        # Longer initial delay to ensure Ray head is ready
        sleep 20
        
        # Track connection success
        RAY_CONNECTED=false
        
        for i in {1..10}; do
            echo "Attempt $i to connect to Ray cluster..."
            ray start --address="$MASTER_ADDR:6379" --num-gpus=$GPUS_PER_NODE --block &
            sleep 5
            if ray status > /dev/null 2>&1; then
                echo "✓ Connected to Ray cluster successfully"
                RAY_CONNECTED=true
                break
            fi
            echo "  Retry $i/10 failed, waiting..."
            sleep 5
        done
        
        # Fail fast if connection failed after all retries
        if [ "$RAY_CONNECTED" = false ]; then
            echo ""
            echo "========================================================================"
            echo "❌ ERROR: Failed to connect to Ray cluster after 10 attempts"
            echo "========================================================================"
            echo "Possible causes:"
            echo "  - Ray head node failed or crashed"
            echo "  - Network connectivity issues"
            echo "  - Ray head not fully initialized"
            echo "  - Incorrect MASTER_ADDR: $MASTER_ADDR"
            echo ""
            echo "This worker node will now exit to prevent indefinite hanging."
            echo "========================================================================"
            exit 1
        fi
    fi
    
    # Verify Ray cluster is ready
    echo "Verifying Ray cluster status..."
    ray status || echo "Warning: Ray status check failed, proceeding anyway"
    
    # Start a watchdog process to detect unhealthy Ray cluster
    # This prevents indefinite hanging if nodes fail during execution
    echo "Starting Ray health watchdog..."
    (
        sleep 60  # Initial grace period for initialization
        while true; do
            if ! ray status > /dev/null 2>&1; then
                echo ""
                echo "========================================================================"
                echo "❌ WATCHDOG: Ray cluster became unhealthy"
                echo "========================================================================"
                echo "The Ray cluster is no longer responding."
                echo "This usually means a node has failed or network connectivity was lost."
                echo "Terminating vLLM processes to prevent indefinite hanging..."
                echo "========================================================================"
                
                # Kill vLLM inference processes
                pkill -9 -f "python.*run_vllm_inference.py" 2>/dev/null || true
                
                # Exit this script
                exit 1
            fi
            sleep 30  # Check every 30 seconds
        done
    ) &
    WATCHDOG_PID=$!
    echo "Ray health watchdog started (PID: $WATCHDOG_PID)"
    
    # vLLM will auto-detect the local Ray cluster (no RAY_ADDRESS needed)
    echo "Ray cluster ready. vLLM will auto-connect to local Ray instance."
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
echo "Launching vLLM inference..."
python3 run_vllm_inference.py \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --distributed-backend "$DISTRIBUTED_BACKEND"

# Note: cleanup() trap handler will run automatically on exit
echo "========================================================================"
echo "Inference completed successfully"
echo "========================================================================"


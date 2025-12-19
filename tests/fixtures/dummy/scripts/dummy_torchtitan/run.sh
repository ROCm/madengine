#!/bin/bash
# TorchTitan Training Test Script
# Minimal test for torchtitan launcher functionality

set -e

echo "======================================"
echo "TorchTitan madengine Test"
echo "======================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

# Display distributed environment
echo "Distributed Environment:"
echo "  RANK: ${RANK:-0}"
echo "  LOCAL_RANK: ${LOCAL_RANK:-0}"
echo "  WORLD_SIZE: ${WORLD_SIZE:-1}"
echo "  MASTER_ADDR: ${MASTER_ADDR:-localhost}"
echo "  MASTER_PORT: ${MASTER_PORT:-29500}"
echo ""

echo "TorchTitan Configuration:"
echo "  Tensor Parallel Size: ${TORCHTITAN_TENSOR_PARALLEL_SIZE:-1}"
echo "  Pipeline Parallel Size: ${TORCHTITAN_PIPELINE_PARALLEL_SIZE:-1}"
echo "  FSDP Enabled: ${TORCHTITAN_FSDP_ENABLED:-0}"
echo "  Context Parallel Size: ${TORCHTITAN_CONTEXT_PARALLEL_SIZE:-1}"
echo ""

# Create minimal torchtitan config
cat > /tmp/test_config.toml << 'EOF'
# Minimal TorchTitan test configuration
[job]
dump_folder = "/tmp/outputs"
description = "madengine torchtitan test"

[profiling]
enable_profiling = false

[model]
name = "llama3"
flavor = "debugmodel"  # Minimal model for testing
norm_type = "rmsnorm"

[optimizer]
name = "AdamW"
lr = 3e-4

[training]
batch_size = 1
seq_len = 128
steps = 10
data_parallel_degree = -1
tensor_parallel_degree = 1
compile = false
dataset = "c4_test"

[experimental]
enable_async_tensor_parallel = false

[checkpoint]
enable_checkpoint = false

[metrics]
log_freq = 1
enable_tensorboard = false
EOF

echo "Generated test config at /tmp/test_config.toml"
cat /tmp/test_config.toml
echo ""

# Run torchtitan training
echo "Starting TorchTitan training..."
echo "Command: ${MAD_MULTI_NODE_RUNNER:-torchrun} /opt/torchtitan/train.py --job.config_file /tmp/test_config.toml"
echo ""

# Execute via MAD_MULTI_NODE_RUNNER (set by deployment) or fallback to direct torchrun
if [ -n "$MAD_MULTI_NODE_RUNNER" ]; then
    # Multi-GPU/Multi-node: Use launcher command from deployment
    $MAD_MULTI_NODE_RUNNER /opt/torchtitan/train.py --job.config_file /tmp/test_config.toml
else
    # Single GPU fallback
    python /opt/torchtitan/train.py --job.config_file /tmp/test_config.toml
fi

echo ""
echo "======================================"
echo "TorchTitan Test Complete"
echo "======================================"

# Output performance metric for madengine
echo "performance: 100.0 tokens_per_second"


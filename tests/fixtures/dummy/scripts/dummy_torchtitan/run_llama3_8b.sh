#!/bin/bash
# TorchTitan Llama 3.1 8B Training Script
# Full training example with model download and checkpointing

set -e

echo "======================================"
echo "TorchTitan Llama 3.1 8B Training"
echo "======================================"

# Ensure torchtitan is available
if [ ! -d "/opt/torchtitan" ]; then
    echo "Error: torchtitan not found at /opt/torchtitan"
    exit 1
fi

cd /opt/torchtitan

# Download tokenizer if not present (requires HF_TOKEN environment variable)
if [ -n "$HF_TOKEN" ] && [ ! -f "tokenizer.model" ]; then
    echo "Downloading Llama 3.1 tokenizer..."
    python scripts/download_hf_assets.py \
        --repo_id meta-llama/Llama-3.1-8B \
        --assets tokenizer \
        --hf_token=$HF_TOKEN
fi

# Use config file if provided, otherwise use default 8B config
CONFIG_FILE=${TORCHTITAN_CONFIG:-"./torchtitan/models/llama3/train_configs/llama3_8b.toml"}

echo "Using config: $CONFIG_FILE"
echo "Distributed setup: ${WORLD_SIZE:-1} GPUs across ${NNODES:-1} nodes"
echo ""

# Run training via MAD launcher
if [ -n "$MAD_MULTI_NODE_RUNNER" ]; then
    echo "Launching via: $MAD_MULTI_NODE_RUNNER"
    $MAD_MULTI_NODE_RUNNER train.py --job.config_file $CONFIG_FILE
else
    # Fallback to direct execution
    python train.py --job.config_file $CONFIG_FILE
fi

echo ""
echo "Training complete!"

# Parse and output performance metric
if [ -f "/tmp/outputs/metrics.txt" ]; then
    TOKENS_PER_SEC=$(grep "tokens/sec" /tmp/outputs/metrics.txt | tail -1 | awk '{print $NF}')
    echo "performance: ${TOKENS_PER_SEC} tokens_per_second"
else
    echo "performance: 0.0 tokens_per_second"
fi


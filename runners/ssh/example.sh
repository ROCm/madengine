#!/bin/bash
# Example script showing how to use the SSH multi-node runner

# Configuration
MODEL="pyt_megatron_lm_train_llama2_7b"
NODES="192.168.1.1,192.168.1.2"
MASTER_ADDR="192.168.0.1"
MASTER_PORT="4000"
SSH_USER="username"  # Replace with your SSH username
SSH_KEY="~/.ssh/id_rsa"
SHARED_DATA="/nfs/data"
NCCL_INTERFACE="ens14np0"
GLOO_INTERFACE="ens14np0"

echo "🚀 Starting multi-node training with SSH runner"
echo "📋 Model: $MODEL"
echo "🔗 Nodes: $NODES"
echo "🏠 Master: $MASTER_ADDR:$MASTER_PORT"

# Install requirements if not already installed
if ! python -c "import paramiko" 2>/dev/null; then
    echo "📦 Installing required packages..."
    pip install -r requirements.txt
fi

# Run the SSH multi-node runner
python run.py \
    --model "$MODEL" \
    --nodes "$NODES" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --ssh-user "$SSH_USER" \
    --ssh-key "$SSH_KEY" \
    --shared-data-path "$SHARED_DATA" \
    --nccl-interface "$NCCL_INTERFACE" \
    --gloo-interface "$GLOO_INTERFACE" \
    --timeout 7200

echo "✅ Multi-node training completed!"

#!/bin/bash
# Example script showing how to use the SSH multi-node runner

# Configuration
MODEL="pyt_megatron_lm_train_llama2_7b"
NODES="192.168.0.1,192.168.0.2"
MASTER_ADDR="10.227.23.63"
MASTER_PORT="4000"
SSH_USER="username"  # Replace with your SSH username
SSH_KEY="$HOME/.ssh/id_ed25519"
SHARED_DATA="/nfs/data"
NCCL_INTERFACE="ens14np0"
GLOO_INTERFACE="ens14np0"

echo "🚀 Starting multi-node training with SSH runner"
echo "📋 Model: $MODEL"
echo "🔗 Nodes: $NODES"
echo "🏠 Master: $MASTER_ADDR:$MASTER_PORT"

# Validate SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key file not found: $SSH_KEY"
    echo "💡 Available SSH keys in ~/.ssh/:"
    ls -la ~/.ssh/*.pub 2>/dev/null || echo "   No SSH keys found"
    echo ""
    echo "🔧 To generate a new SSH key, run:"
    echo "   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519"
    echo "   ssh-copy-id -i ~/.ssh/id_ed25519.pub $SSH_USER@<node-ip>"
    exit 1
fi

echo "✅ SSH key validated: $SSH_KEY"

# Detect Python command
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
else
    echo "❌ Python is not installed or not in PATH"
    echo "💡 If you're using a virtual environment, make sure it's activated"
    exit 1
fi

# Install requirements if not already installed
if ! $PYTHON_CMD -c "import paramiko" 2>/dev/null; then
    echo "📦 Installing required packages..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Run the SSH multi-node runner
$PYTHON_CMD run.py \
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

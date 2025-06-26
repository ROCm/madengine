#!/bin/bash
# Quick start script for SSH Multi-Node Runner

set -e

echo "🚀 SSH Multi-Node Runner for MAD Engine"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

echo "✅ Python is available"

# Check if paramiko is installed
if ! python -c "import paramiko" 2>/dev/null; then
    echo "📦 Installing paramiko..."
    pip install paramiko
else
    echo "✅ paramiko is already installed"
fi

echo ""
echo "🎯 Quick Start Examples:"
echo ""
echo "1. SSH Key Authentication:"
echo "   python run.py --model pyt_megatron_lm_train_llama2_7b \\"
echo "                  --nodes 192.168.1.1,192.168.1.2 \\"
echo "                  --master-addr 192.168.0.1 \\"
echo "                  --ssh-user ubuntu \\"
echo "                  --ssh-key ~/.ssh/id_rsa"
echo ""
echo "2. Configuration File:"
echo "   python run.py --config config.ini"
echo ""
echo "3. Run Tests:"
echo "   python test_runner.py"
echo ""
echo "📖 For detailed documentation, see README.md"
echo "✨ Ready to run multi-node training!"

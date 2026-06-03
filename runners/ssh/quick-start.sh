#!/bin/bash
# Quick start script for SSH Multi-Node Runner

set -e

echo "🚀 SSH Multi-Node Runner for MAD Engine"
echo "========================================"
echo ""

# Check if Python is available (try python3, python, or VIRTUAL_ENV python)
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

echo "✅ Python is available ($PYTHON_CMD)"

# Check if paramiko is installed
if ! $PYTHON_CMD -c "import paramiko" 2>/dev/null; then
    echo "📦 Installing paramiko..."
    $PYTHON_CMD -m pip install paramiko
else
    echo "✅ paramiko is already installed"
fi

echo ""
echo "🎯 Quick Start Examples:"
echo ""
echo "1. SSH Key Authentication:"
echo "   $PYTHON_CMD run.py --model pyt_megatron_lm_train_llama2_7b \\"
echo "                  --nodes 192.168.1.1,192.168.1.2 \\"
echo "                  --master-addr 192.168.0.1 \\"
echo "                  --ssh-user ubuntu \\"
echo "                  --ssh-key ~/.ssh/id_rsa"
echo ""
echo "2. Configuration File:"
echo "   $PYTHON_CMD run.py --config config.ini"
echo ""
echo "3. Run Tests:"
echo "   $PYTHON_CMD test_runner.py"
echo ""
echo "📖 For detailed documentation, see README.md"
echo "✨ Ready to run multi-node training!"

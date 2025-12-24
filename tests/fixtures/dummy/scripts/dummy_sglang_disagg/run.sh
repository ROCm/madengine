#!/bin/bash
# SGLang Disaggregated Dummy Test Script
# Tests disaggregated prefill/decode architecture with minimal model

set -e

echo "============================================"
echo "SGLang Disaggregated Dummy Test"
echo "============================================"

# Check if disagg mode is enabled
if [ "${SGLANG_DISAGG_MODE:-}" = "enabled" ]; then
    echo "✓ Disaggregated mode detected"
    echo "  Node Rank: ${SGLANG_NODE_RANK:-unknown}"
    echo "  Prefill Nodes: ${SGLANG_DISAGG_PREFILL_NODES:-unknown}"
    echo "  Decode Nodes: ${SGLANG_DISAGG_DECODE_NODES:-unknown}"
    
    # Run Python script that handles node roles
    python3 run_sglang_disagg_inference.py
else
    echo "❌ ERROR: SGLANG_DISAGG_MODE not set"
    echo "This test requires SGLang Disaggregated launcher"
    exit 1
fi

echo "============================================"
echo "✓ SGLang Disagg Test Complete"
echo "============================================"


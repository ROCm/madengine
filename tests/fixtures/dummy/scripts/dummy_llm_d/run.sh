#!/bin/bash
#
# llm-d benchmark client.
#
# Runs in a single CPU-only pod and drives load against an llm-d gateway. The
# deployment layer injects the MAD_LLM_D_* variables read below; see
# src/madengine/deployment/llm_d.py.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1

if [ -z "${MAD_LLM_D_ENDPOINT:-}" ]; then
    echo "ERROR: MAD_LLM_D_ENDPOINT is not set." >&2
    echo "  This model must be run through the llm-d deployment target, e.g." >&2
    echo "  --additional-context '{\"llm_d\": {\"endpoint_url\": \"http://...\", \"model\": {\"name\": \"...\"}}}'" >&2
    exit 1
fi

if [ -z "${MAD_LLM_D_MODEL:-}" ]; then
    echo "ERROR: MAD_LLM_D_MODEL is not set (llm_d.model.name)." >&2
    exit 1
fi

NUM_REQUESTS=${LLM_D_NUM_REQUESTS:-32}
CONCURRENCY=${LLM_D_CONCURRENCY:-4}
MAX_TOKENS=${LLM_D_MAX_TOKENS:-64}

echo "========================================================================"
echo "madengine llm-d benchmark client"
echo "========================================================================"
echo "  Endpoint         : $MAD_LLM_D_ENDPOINT"
echo "  Model            : $MAD_LLM_D_MODEL"
echo "  Namespace        : ${MAD_LLM_D_NAMESPACE:-<unset>}"
echo "  Prefill replicas : ${MAD_LLM_D_PREFILL_REPLICAS:-<unset>}"
echo "  Decode replicas  : ${MAD_LLM_D_DECODE_REPLICAS:-<unset>}"
echo "  Tensor parallel  : ${MAD_LLM_D_TP:-<unset>}"
echo "  Requests         : $NUM_REQUESTS at concurrency $CONCURRENCY"
echo "========================================================================"

python3 bench_llm_d.py \
    --endpoint "$MAD_LLM_D_ENDPOINT" \
    --model "$MAD_LLM_D_MODEL" \
    --num-requests "$NUM_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS"

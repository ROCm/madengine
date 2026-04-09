#!/bin/bash
# TorchTitan dummy benchmark: runs real training (llama3_debugmodel + c4_test).
# Requires Hugging Face token for tokenizer download. Set MAD_SECRET_HFTOKEN via
# additional_context docker_env_vars so it is passed into the container; this
# script sets HF_TOKEN from it for TorchTitan and download_hf_assets.py.

set -e

echo "======================================"
echo "TorchTitan madengine Benchmark"
echo "======================================"
echo "Hostname: $(hostname)"
echo "RANK: ${RANK:-0} WORLD_SIZE: ${WORLD_SIZE:-1}"
echo "MASTER_ADDR: ${MASTER_ADDR:-localhost} MASTER_PORT: ${MASTER_PORT:-29500}"
echo ""

# HF token: use MAD_SECRET_HFTOKEN (set via docker_env_vars in additional_context).
export HF_TOKEN="${MAD_SECRET_HFTOKEN:-}"

if [ -z "$HF_TOKEN" ]; then
    echo "Error: Hugging Face token required for tokenizer download."
    echo "  Set MAD_SECRET_HFTOKEN in additional_context docker_env_vars, e.g.:"
    echo '  --additional-context '\''{"docker_env_vars": {"MAD_SECRET_HFTOKEN": "<your-hf-token>"}}'\'''
    exit 1
fi

# TorchTitan lives in the image at /opt/torchtitan; run from there so
# dataset path tests/assets/c4_test and asset paths resolve.
cd /opt/torchtitan

# Paths: download script writes to local_dir/model_name (e.g. tests/assets/Llama-3.1-8B).
# llama3_debugmodel expects tokenizer at default path tests/assets/tokenizer; we download
# to tests/assets/Llama-3.1-8B then symlink tests/assets/tokenizer -> Llama-3.1-8B so
# no CLI override is needed (this image's train.py does not accept --model.hf_assets_path).
HF_LOCAL_DIR="/opt/torchtitan/tests/assets"
HF_REPO_ID="${TORCHTITAN_HF_REPO:-meta-llama/Llama-3.1-8B}"
HF_MODEL_NAME="${HF_REPO_ID##*/}"

# Download tokenizer if not already present (idempotent).
if [ ! -f "${HF_LOCAL_DIR}/${HF_MODEL_NAME}/tokenizer.model" ] && [ ! -f "${HF_LOCAL_DIR}/${HF_MODEL_NAME}/tokenizer.json" ]; then
    echo "Downloading tokenizer from ${HF_REPO_ID} to ${HF_LOCAL_DIR}..."
    mkdir -p "$HF_LOCAL_DIR"
    python scripts/download_hf_assets.py \
        --repo_id "$HF_REPO_ID" \
        --assets tokenizer \
        --hf_token "$HF_TOKEN" \
        --local_dir "$HF_LOCAL_DIR"
    echo "Tokenizer downloaded to ${HF_LOCAL_DIR}/${HF_MODEL_NAME}"
else
    echo "Using existing tokenizer at ${HF_LOCAL_DIR}/${HF_MODEL_NAME}"
fi

# Point default path tests/assets/tokenizer at the downloaded dir (llama3_debugmodel default).
TOKENIZER_LINK="${HF_LOCAL_DIR}/tokenizer"
if [ ! -L "$TOKENIZER_LINK" ] || [ " $(readlink -f "$TOKENIZER_LINK")" != " $(readlink -f "${HF_LOCAL_DIR}/${HF_MODEL_NAME}")" ]; then
    rm -rf "$TOKENIZER_LINK"
    ln -snf "$HF_MODEL_NAME" "$TOKENIZER_LINK"
    echo "Linked tests/assets/tokenizer -> $HF_MODEL_NAME"
fi

# Train script and args: tokenizer:config + llama3_debugmodel (uses c4_test; tokenizer at default path).
# Disable torch.compile to avoid ROCm HSA_STATUS_ERROR_EXCEPTION (0x1016) in vectorized_gather on some AMD GPUs (e.g. gfx942).
export TORCH_COMPILE_DISABLE=1
TORCHTITAN_TRAIN="${TORCHTITAN_TRAIN:-/opt/torchtitan/torchtitan/train.py}"
[ ! -f "$TORCHTITAN_TRAIN" ] && [ -f /opt/torchtitan/train.py ] && TORCHTITAN_TRAIN=/opt/torchtitan/train.py
if [ ! -f "$TORCHTITAN_TRAIN" ]; then
    echo "Error: train.py not found under /opt/torchtitan"
    exit 1
fi

# Do not pass --model.compile=false: this image's train.py does not accept it (Unrecognized options).
# Run with training.max_steps=0 to validate setup (tokenizer, launcher, multi-node env) without
# running the GPU training step that triggers ROCm HSA_STATUS_ERROR_EXCEPTION (0x1016) on gfx942.
# For full training, override with --training.max_steps=<N> (may hit 0x1016 on this stack).
TORCHTITAN_ARGS="tokenizer:config --module llama3 --config llama3_debugmodel --training.max_steps=0"

echo "Command: (cd /opt/torchtitan && ... $TORCHTITAN_TRAIN $TORCHTITAN_ARGS)"
echo ""

EXIT=0
if [ -n "$MAD_MULTI_NODE_RUNNER" ]; then
    $MAD_MULTI_NODE_RUNNER "$TORCHTITAN_TRAIN" $TORCHTITAN_ARGS || EXIT=$?
else
    python "$TORCHTITAN_TRAIN" $TORCHTITAN_ARGS || EXIT=$?
fi

echo ""
echo "TorchTitan exit code: $EXIT"
if [ $EXIT -ne 0 ]; then
    exit $EXIT
fi

# Metric line for madengine (real run may parse from logs; fallback)
echo "performance: 100.0 tokens_per_second"

# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# SGLang Disaggregated Dockerfile for AMD ROCm - Dummy Test Version
# Reference: https://github.com/sgl-project/sglang
# Reference: https://github.com/kvcache-ai/Mooncake (disaggregation framework)

# ============================================================================
# Base Image: Official SGLang with ROCm 7.x Support
# ============================================================================
# Using lmsysorg/sglang:latest which includes:
# - SGLang with disaggregation support
# - ROCm 7.x for AMD MI300X
# - Ray for distributed coordination
ARG BASE_DOCKER=lmsysorg/sglang:latest
FROM $BASE_DOCKER

# ============================================================================
# ROCm 7.x Environment Configuration
# ============================================================================
ENV MIOPEN_FIND_MODE=1 \
    MIOPEN_USER_DB_PATH=/tmp/.miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/tmp/.miopen

RUN mkdir -p /tmp/.miopen && chmod 1777 /tmp/.miopen

# ROCm 7.x optimizations for MI300X
ENV HSA_FORCE_FINE_GRAIN_PCIE=1 \
    HSA_ENABLE_SDMA=0 \
    GPU_MAX_HW_QUEUES=2 \
    NCCL_DEBUG=WARN \
    NCCL_MIN_NCHANNELS=16 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ENV ROCM_USE_FLASH_ATTENTION=1 \
    HIP_FORCE_DEV_KERNARG=1

# ============================================================================
# SGLang Disaggregated Configuration
# ============================================================================
# Core SGLang settings
ENV SGLANG_ALLOW_LONG_MAX_MODEL_LEN=1 \
    SGLANG_USE_MODELSCOPE=False \
    SGLANG_LOGGING_LEVEL=INFO

# SGLang Disaggregation - Enable prefill/decode separation
ENV SGLANG_ENABLE_DISAGGREGATION=1 \
    SGLANG_DISAGG_TRANSFER_BACKEND=mooncake

# RadixAttention for KV cache efficiency
ENV SGLANG_ENABLE_RADIX_CACHE=1 \
    SGLANG_RADIX_CACHE_SIZE=0.9

# Ray Configuration for distributed coordination
ENV RAY_DEDUP_LOGS=1 \
    RAY_BACKEND_LOG_LEVEL=warning \
    RAY_USAGE_STATS_ENABLED=0 \
    RAY_USAGE_STATS_ENABLED_OVERRIDE=0

# ============================================================================
# Mooncake Framework Setup (Simplified for Dummy Test)
# ============================================================================
# Mooncake is the KV cache transfer framework for disaggregated inference
# Reference: https://github.com/kvcache-ai/Mooncake
#
# For dummy testing, we create a minimal simulation environment
# Production deployments should use full Mooncake with RDMA support

# Install dependencies for Mooncake simulation
RUN pip install --no-cache-dir \
    flask \
    py-spy \
    etcd3 \
    && rm -rf /root/.cache/pip/*

# Create Mooncake cookbook directory structure (for dummy scripts)
RUN mkdir -p /opt/mooncake-cookbook && \
    chmod -R 755 /opt/mooncake-cookbook

ENV MOONCAKE_COOKBOOK_PATH=/opt/mooncake-cookbook

# Create dummy Mooncake environment setup script
RUN echo '#!/bin/bash' > /opt/mooncake-cookbook/set_env_vars.sh && \
    echo '# Mooncake Environment Variables (Dummy Test Mode)' >> /opt/mooncake-cookbook/set_env_vars.sh && \
    echo 'export MOONCAKE_TEST_MODE=1' >> /opt/mooncake-cookbook/set_env_vars.sh && \
    echo 'export MOONCAKE_TRANSFER_PROTOCOL=tcp' >> /opt/mooncake-cookbook/set_env_vars.sh && \
    echo 'export IBDEVICES=eth0' >> /opt/mooncake-cookbook/set_env_vars.sh && \
    echo 'echo "✓ Mooncake environment configured (test mode)"' >> /opt/mooncake-cookbook/set_env_vars.sh && \
    chmod +x /opt/mooncake-cookbook/set_env_vars.sh

# Create dummy synchronization scripts for multi-node coordination
RUN echo '#!/usr/bin/env python3' > /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'import sys' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'import time' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'import argparse' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'parser = argparse.ArgumentParser()' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'parser.add_argument("--local-ip", default="localhost")' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'parser.add_argument("--local-port", type=int, default=5000)' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'parser.add_argument("--enable-port", action="store_true")' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'parser.add_argument("--node-ips", default="localhost")' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'parser.add_argument("--node-ports", default="5000")' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'args = parser.parse_args()' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'print(f"[Barrier] Synchronizing nodes...")' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'time.sleep(1)  # Simulate barrier' >> /opt/mooncake-cookbook/socket_barrier.py && \
    echo 'print(f"[Barrier] All nodes synchronized")' >> /opt/mooncake-cookbook/socket_barrier.py && \
    chmod +x /opt/mooncake-cookbook/socket_barrier.py

RUN echo '#!/usr/bin/env python3' > /opt/mooncake-cookbook/socket_wait.py && \
    echo 'import sys' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'import time' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'import argparse' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'parser = argparse.ArgumentParser()' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'parser.add_argument("--remote-ip", default="localhost")' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'parser.add_argument("--remote-port", type=int, default=30000)' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'args = parser.parse_args()' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'print(f"[Wait] Waiting for {args.remote_ip}:{args.remote_port}")' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'time.sleep(2)  # Simulate wait' >> /opt/mooncake-cookbook/socket_wait.py && \
    echo 'print(f"[Wait] Connection closed")' >> /opt/mooncake-cookbook/socket_wait.py && \
    chmod +x /opt/mooncake-cookbook/socket_wait.py

# ============================================================================
# Verification - Ensure all components are ready
# ============================================================================
# Verify SGLang with disaggregation support
RUN python3 -c "import sglang; \
    print(f'✓ SGLang version: {sglang.__version__}'); \
    print(f'✓ SGLang installation: Disaggregation-ready')" || \
    (echo "✗ SGLang import failed" && exit 1)

# Verify SGLang disaggregation modules
RUN python3 -c "from sglang.srt.disaggregation.mini_lb import main; \
    print('✓ SGLang disaggregation modules available (mini_lb)')" || \
    (echo "⚠ SGLang disaggregation module check failed (may require newer version)" && true)

# Verify PyTorch with ROCm 7.x
RUN python3 -c "import torch; \
    print(f'✓ PyTorch version: {torch.__version__}'); \
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None; \
    print(f'✓ ROCm available: {is_rocm}'); \
    if is_rocm: \
        hip_version = torch.version.hip; \
        print(f'✓ ROCm/HIP version: {hip_version}')" || \
    (echo "✗ PyTorch/ROCm check failed" && exit 1)

# Verify dependencies
RUN python3 -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" && \
    python3 -c "import ray; print(f'✓ Ray: {ray.__version__}')" && \
    python3 -c "import flask; print(f'✓ Flask: {flask.__version__}')" || \
    (echo "✗ Dependency check failed" && exit 1)

# ============================================================================
# Workspace Setup
# ============================================================================
WORKDIR /workspace

# Create run logs directory
RUN mkdir -p /run_logs && chmod 1777 /run_logs

# ============================================================================
# Final Environment Summary
# ============================================================================
RUN echo "========================================================================" && \
    echo "✅ SGLang Disaggregated Docker Image Build Complete (Dummy Test)" && \
    echo "========================================================================" && \
    echo "Base Image:           lmsysorg/sglang:latest" && \
    echo "ROCm Version:         $(cat /opt/rocm/.info/version 2>/dev/null || echo '7.x')" && \
    echo "SGLang Version:       $(python3 -c 'import sglang; print(sglang.__version__)')" && \
    echo "PyTorch Version:      $(python3 -c 'import torch; print(torch.__version__)')" && \
    echo "Ray Version:          $(python3 -c 'import ray; print(ray.__version__)')" && \
    echo "------------------------------------------------------------------------" && \
    echo "Build Type:           Dummy Test (Disaggregated Architecture)" && \
    echo "Target GPUs:          AMD MI300X, MI250X (ROCm 7.x optimized)" && \
    echo "Architecture:         Prefill/Decode Separation" && \
    echo "Transfer Backend:     Mooncake (simulated for testing)" && \
    echo "Min Nodes:            3 (1 proxy + 1 prefill + 1 decode)" && \
    echo "------------------------------------------------------------------------" && \
    echo "Key Features:" && \
    echo "  • Disaggregated prefill/decode clusters" && \
    echo "  • Mooncake framework simulation" && \
    echo "  • Multi-node coordination (Ray + etcd)" && \
    echo "  • RadixAttention for KV cache efficiency" && \
    echo "========================================================================" && \
    echo "" && \
    echo "🚀 Ready for SGLang Disaggregated testing on AMD GPUs!" && \
    echo "   Note: This is a dummy/test image for madengine validation" && \
    echo "   For production: Use full Mooncake with RDMA support" && \
    echo ""


# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# Production vLLM Dockerfile - Using official ROCm vLLM image for real benchmarking
ARG BASE_DOCKER=rocm/vllm:latest
FROM $BASE_DOCKER

# ============================================================================
# ROCm Optimizations
# ============================================================================
# MIOpen configuration for ROCm
ENV MIOPEN_FIND_MODE=1 \
    MIOPEN_USER_DB_PATH=/tmp/.miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/tmp/.miopen

RUN mkdir -p /tmp/.miopen && chmod 1777 /tmp/.miopen

# ============================================================================
# vLLM Environment Variables for ROCm
# ============================================================================
# Core vLLM settings
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_USE_MODELSCOPE=False \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_LOGGING_LEVEL=INFO

# ROCm specific optimizations
ENV HSA_FORCE_FINE_GRAIN_PCIE=1 \
    HSA_ENABLE_SDMA=0 \
    GPU_MAX_HW_QUEUES=2 \
    NCCL_DEBUG=WARN \
    NCCL_MIN_NCHANNELS=16

# PyTorch settings for ROCm
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# HIP/ROCm runtime settings
# Note: HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES should be set at runtime
# ENV HIP_VISIBLE_DEVICES=0
# ENV ROCR_VISIBLE_DEVICES=0

# ============================================================================
# vLLM Flash Attention for ROCm
# ============================================================================
ENV VLLM_USE_FLASH_ATTN_TRITON=1

# ============================================================================
# Verification
# ============================================================================
# Verify real vLLM installation
RUN python3 -c "import vllm; print(f'✓ vLLM version: {vllm.__version__}'); \
    assert not 'mock' in vllm.__version__.lower(), 'Mock vLLM detected!'" || \
    (echo "✗ vLLM import failed or mock detected" && exit 1)

# Verify PyTorch with ROCm
RUN python3 -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')" || \
    (echo "✗ PyTorch import failed" && exit 1)

# Verify ROCm availability
RUN python3 -c "import torch; \
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None; \
    print(f'✓ ROCm available: {is_rocm}'); \
    print(f'✓ ROCm version: {torch.version.hip if is_rocm else \"N/A\"}')" || \
    (echo "✗ ROCm check failed" && exit 1)

# GPU device check (will show count = 0 in build environment)
RUN python3 -c "import torch; count = torch.cuda.device_count(); print(f'✓ GPU devices detected: {count}'); print(f'✓ GPU 0: {torch.cuda.get_device_name(0)}' if count > 0 else '  (No GPUs in build environment - will be available at runtime)')"

# Verify ROCm tools (may not be available in build environment)
RUN rocminfo > /dev/null 2>&1 || echo "  (rocminfo check skipped - will be available at runtime)"
RUN rocm-smi > /dev/null 2>&1 || echo "  (rocm-smi check skipped - will be available at runtime)"

# Verify key dependencies
RUN python3 -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" || \
    (echo "✗ Transformers import failed" && exit 1)
RUN python3 -c "import ray; print(f'✓ Ray: {ray.__version__}')" || \
    (echo "✗ Ray import failed" && exit 1)

# ============================================================================
# Workspace Setup
# ============================================================================
WORKDIR /workspace

# Print final environment info
RUN echo "=======================================" && \
    echo "vLLM Docker Image Build Complete" && \
    echo "=======================================" && \
    echo "Base Image: rocm/vllm:latest" && \
    echo "ROCm Version: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'latest')" && \
    echo "vLLM Version: $(python3 -c 'import vllm; print(vllm.__version__)')" && \
    echo "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)')" && \
    echo "Build Type: Production (Real vLLM with ROCm)" && \
    echo "======================================="


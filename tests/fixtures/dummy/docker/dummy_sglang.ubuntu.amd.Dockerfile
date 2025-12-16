# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# Production SGLang Dockerfile - Using official SGLang image for real benchmarking
ARG BASE_DOCKER=lmsysorg/sglang:latest
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
# SGLang Environment Variables for ROCm
# ============================================================================
# Core SGLang settings
ENV SGLANG_ALLOW_LONG_MAX_MODEL_LEN=1 \
    SGLANG_USE_MODELSCOPE=False \
    SGLANG_ENABLE_FLASHINFER=1 \
    SGLANG_LOGGING_LEVEL=INFO

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
# SGLang RadixAttention Configuration
# ============================================================================
# SGLang uses RadixAttention for efficient KV cache with automatic prefix caching
ENV SGLANG_ENABLE_RADIX_CACHE=1 \
    SGLANG_RADIX_CACHE_SIZE=0.9

# ============================================================================
# Ray Configuration for Distributed Inference
# ============================================================================
# Ray is used for distributed coordination in SGLang
ENV RAY_DEDUP_LOGS=1 \
    RAY_BACKEND_LOG_LEVEL=warning

# ============================================================================
# Verification
# ============================================================================
# Verify real SGLang installation
RUN python3 -c "import sglang; print(f'✓ SGLang version: {sglang.__version__}'); \
    assert not 'mock' in sglang.__version__.lower(), 'Mock SGLang detected!'" || \
    (echo "✗ SGLang import failed or mock detected" && exit 1)

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
RUN python3 -c "import torch; \
    print(f'✓ GPU devices detected: {torch.cuda.device_count()}'); \
    if torch.cuda.device_count() > 0: \
        print(f'✓ GPU 0: {torch.cuda.get_device_name(0)}') \
    else: \
        print('  (No GPUs in build environment - will be available at runtime)')"

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
    echo "SGLang Docker Image Build Complete" && \
    echo "=======================================" && \
    echo "Base Image: lmsysorg/sglang:latest" && \
    echo "ROCm Version: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'latest')" && \
    echo "SGLang Version: $(python3 -c 'import sglang; print(sglang.__version__)')" && \
    echo "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)')" && \
    echo "Build Type: Production (Real SGLang with ROCm)" && \
    echo "======================================="


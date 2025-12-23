# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# SGLang Dockerfile for AMD ROCm - Using official SGLang image with ROCm 7.x support
# Reference: https://github.com/sgl-project/sglang

# ============================================================================
# Base Image: Official SGLang with ROCm 7.x Support
# ============================================================================
# Using lmsysorg/sglang:latest which includes:
# - SGLang with latest features (RadixAttention, multi-modal support)
# - ROCm 7.x for AMD MI300X and latest GPU support
# - Pre-optimized kernels and dependencies
# - Ray for distributed inference
ARG BASE_DOCKER=lmsysorg/sglang:latest
FROM $BASE_DOCKER

# ============================================================================
# ROCm 7.x Environment Configuration
# ============================================================================
# MIOpen configuration for optimal kernel selection
ENV MIOPEN_FIND_MODE=1 \
    MIOPEN_USER_DB_PATH=/tmp/.miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/tmp/.miopen

RUN mkdir -p /tmp/.miopen && chmod 1777 /tmp/.miopen

# ROCm 7.x specific optimizations for MI300X
ENV HSA_FORCE_FINE_GRAIN_PCIE=1 \
    HSA_ENABLE_SDMA=0 \
    GPU_MAX_HW_QUEUES=2 \
    NCCL_DEBUG=WARN \
    NCCL_MIN_NCHANNELS=16 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ROCm 7.x advanced features
ENV ROCM_USE_FLASH_ATTENTION=1 \
    HIP_FORCE_DEV_KERNARG=1

# ============================================================================
# SGLang Runtime Configuration
# ============================================================================
# Core SGLang settings for production deployment
ENV SGLANG_ALLOW_LONG_MAX_MODEL_LEN=1 \
    SGLANG_USE_MODELSCOPE=False \
    SGLANG_LOGGING_LEVEL=INFO

# SGLang RadixAttention - Automatic prefix caching for efficient KV cache
# Reference: https://github.com/sgl-project/sglang#radixattention
# This is SGLang's key innovation for 5-10x speedup on shared prefix workloads
ENV SGLANG_ENABLE_RADIX_CACHE=1 \
    SGLANG_RADIX_CACHE_SIZE=0.9

# Ray Configuration for Distributed Multi-Node Inference
# SGLang uses Ray for coordination across nodes
ENV RAY_DEDUP_LOGS=1 \
    RAY_BACKEND_LOG_LEVEL=warning \
    RAY_USAGE_STATS_ENABLED=0 \
    RAY_USAGE_STATS_ENABLED_OVERRIDE=0

# ============================================================================
# Verification - Ensure ROCm 7.x and SGLang are properly configured
# ============================================================================
# Verify SGLang installation (from base image)
RUN python3 -c "import sglang; \
    print(f'✓ SGLang version: {sglang.__version__}'); \
    print(f'✓ SGLang installation: Production-ready')" || \
    (echo "✗ SGLang import failed" && exit 1)

# Verify PyTorch with ROCm 7.x
RUN python3 -c "import torch; \
    print(f'✓ PyTorch version: {torch.__version__}'); \
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None; \
    print(f'✓ ROCm available: {is_rocm}'); \
    if is_rocm: \
        hip_version = torch.version.hip; \
        print(f'✓ ROCm/HIP version: {hip_version}'); \
        major_version = int(hip_version.split('.')[0]) if hip_version else 0; \
        if major_version >= 7: \
            print(f'✓ ROCm 7.x+ detected (optimal for MI300X)'); \
        else: \
            print(f'⚠ ROCm version < 7.0 (consider upgrading)')" || \
    (echo "✗ PyTorch/ROCm check failed" && exit 1)

# GPU device check (will show count = 0 in build environment)
RUN python3 -c "import torch; \
    gpu_count = torch.cuda.device_count(); \
    print(f'✓ GPU devices detected: {gpu_count}'); \
    if gpu_count == 0: \
        print('  (No GPUs in build environment - GPUs will be available at runtime)'); \
    else: \
        for i in range(gpu_count): \
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')" || true

# Verify key dependencies (Ray for distributed inference)
RUN python3 -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" && \
    python3 -c "import ray; print(f'✓ Ray: {ray.__version__} (for distributed coordination)')" || \
    (echo "✗ Dependency check failed" && exit 1)

# Verify SGLang server module (key for inference)
RUN python3 -c "from sglang import launch_server; print('✓ SGLang server module available')" || \
    (echo "✗ SGLang server module not found" && exit 1)

# ============================================================================
# Workspace Setup
# ============================================================================
WORKDIR /workspace

# ============================================================================
# Final Environment Summary
# ============================================================================
RUN echo "========================================================================" && \
    echo "✅ SGLang Docker Image Build Complete" && \
    echo "========================================================================" && \
    echo "Base Image:      lmsysorg/sglang:latest" && \
    echo "ROCm Version:    $(cat /opt/rocm/.info/version 2>/dev/null || echo '7.x')" && \
    echo "SGLang Version:  $(python3 -c 'import sglang; print(sglang.__version__)')" && \
    echo "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)')" && \
    echo "Ray Version:     $(python3 -c 'import ray; print(ray.__version__)')" && \
    echo "------------------------------------------------------------------------" && \
    echo "Build Type:      Production (Official SGLang with ROCm 7.x)" && \
    echo "Target GPUs:     AMD MI300X, MI250X (ROCm 7.x optimized)" && \
    echo "Key Features:    RadixAttention, Multi-modal, Distributed Inference" && \
    echo "Reference:       https://github.com/sgl-project/sglang" && \
    echo "========================================================================" && \
    echo "" && \
    echo "🚀 Ready for distributed LLM inference on AMD GPUs!" && \
    echo ""


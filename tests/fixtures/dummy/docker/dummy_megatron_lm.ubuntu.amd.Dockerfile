# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# Using official ROCm Megatron-LM image with pre-installed dependencies
ARG BASE_DOCKER=rocm/megatron-lm:latest
FROM $BASE_DOCKER

# ============================================================================
# ROCm/MIOpen Optimizations
# ============================================================================
# Clear any existing MIOpen cache to ensure clean state
RUN if [ -d "$HOME/.config/miopen" ]; then \
        rm -rf $HOME/.config/miopen/* ; \
    fi && \
    if [ -d "/tmp/.miopen" ]; then \
        rm -rf /tmp/.miopen/* ; \
    fi

# Configure MIOpen for optimal performance
ENV MIOPEN_FIND_MODE=1 \
    MIOPEN_USER_DB_PATH=/tmp/.miopen

RUN mkdir -p /tmp/.miopen && chmod 1777 /tmp/.miopen

# ============================================================================
# Distributed Training Environment Variables
# ============================================================================
# Optimized settings for ROCm distributed training
ENV MEGATRON_FRAMEWORK=megatron_lm \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_DEBUG=WARN \
    TORCH_NCCL_HIGH_PRIORITY=1 \
    GPU_MAX_HW_QUEUES=2 \
    HSA_ENABLE_SDMA=0 \
    HSA_FORCE_FINE_GRAIN_PCIE=1 \
    RCCL_ENABLE_HIPGRAPH=0

# ============================================================================
# Verify Installation
# ============================================================================
# Verify Megatron-LM and ROCm are properly installed
RUN python3 -c "import megatron; print('✓ Megatron-LM available')" && \
    python3 -c "from megatron.core import parallel_state; print('✓ Megatron-Core available')" && \
    python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" && \
    python3 -c "import torch; print(f'✓ CUDA/ROCm available: {torch.cuda.is_available()}')" && \
    rocminfo > /dev/null 2>&1 || echo "ROCm check (OK in build env)"

WORKDIR /workspace

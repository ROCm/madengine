# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch
FROM $BASE_DOCKER

# ============================================================================
# Install Dependencies for ROCm/Megatron-LM
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages for Megatron-LM
RUN pip install --no-cache-dir \
    regex \
    pybind11 \
    nltk \
    einops \
    tensorstore==0.1.45 \
    zarr

# ============================================================================
# Install ROCm-optimized Megatron-LM
# ============================================================================
WORKDIR /opt
RUN git clone --depth 1 --branch rocm_dev https://github.com/ROCm/Megatron-LM.git && \
    cd Megatron-LM && \
    pip install --no-cache-dir -e .

# Set PYTHONPATH to include Megatron-LM
ENV PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH

# ============================================================================
# ROCm/MIOpen Optimizations
# ============================================================================
RUN if [ -d "$HOME/.config/miopen" ]; then \
        rm -rf $HOME/.config/miopen/* ; \
    fi && \
    if [ -d "/tmp/.miopen" ]; then \
        rm -rf /tmp/.miopen/* ; \
    fi

ENV MIOPEN_FIND_MODE=1 \
    MIOPEN_USER_DB_PATH=/tmp/.miopen

RUN mkdir -p /tmp/.miopen && chmod 1777 /tmp/.miopen

# ============================================================================
# Megatron-LM Environment Variables
# ============================================================================
# Environment variables for Megatron-LM training
ENV MEGATRON_FRAMEWORK=megatron_lm \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=eth0

# Verify installations
RUN python3 -c "import megatron; print('✓ Megatron-LM installed')" && \
    rocminfo > /dev/null 2>&1 || echo "ROCm check (OK in build env)"

WORKDIR /workspace

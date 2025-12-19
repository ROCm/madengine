# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch:latest
FROM $BASE_DOCKER

# ============================================================================
# Install TorchTitan Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages for TorchTitan
RUN pip install --no-cache-dir \
    tomli \
    tomli-w \
    psutil \
    tensorboard

# ============================================================================
# Install TorchTitan
# ============================================================================
WORKDIR /opt
RUN git clone https://github.com/pytorch/torchtitan.git && \
    cd torchtitan && \
    pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to include TorchTitan
ENV PYTHONPATH=/opt/torchtitan:$PYTHONPATH

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
# TorchTitan Environment Variables
# ============================================================================
# Default environment variables for TorchTitan training
# These will be overridden by madengine deployment configs
ENV TORCHTITAN_TENSOR_PARALLEL_SIZE=1 \
    TORCHTITAN_PIPELINE_PARALLEL_SIZE=1 \
    TORCHTITAN_FSDP_ENABLED=0 \
    TORCHTITAN_CONTEXT_PARALLEL_SIZE=1

# ============================================================================
# Verification
# ============================================================================
# Verify TorchTitan installation
RUN python3 -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')" && \
    test -f /opt/torchtitan/train.py && echo "✓ TorchTitan installed" || echo "⚠ TorchTitan not found" && \
    rocminfo > /dev/null 2>&1 || echo "ROCm check (OK in build env)"

WORKDIR /workspace


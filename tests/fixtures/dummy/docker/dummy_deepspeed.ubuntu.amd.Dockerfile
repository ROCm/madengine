# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch
FROM $BASE_DOCKER

# ============================================================================
# Install DeepSpeed
# ============================================================================
RUN pip install deepspeed

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
# DeepSpeed Environment
# ============================================================================
ENV DEEPSPEED_LAUNCHER=deepspeed

# Verify installations
RUN python3 -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
RUN rocminfo > /dev/null 2>&1 || echo "ROCm check (OK in build env)"

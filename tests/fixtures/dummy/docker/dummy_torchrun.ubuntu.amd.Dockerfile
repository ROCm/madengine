# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch
FROM $BASE_DOCKER

# Install any additional dependencies for torchrun testing
# (rocm/pytorch already has PyTorch with distributed support)

# ============================================================================
# ROCm/MIOpen Optimizations (Optional - reduces warnings)
# ============================================================================

# Clean MIOpen find-db to avoid duplicate kernel warnings
RUN if [ -d "$HOME/.config/miopen" ]; then \
        rm -rf $HOME/.config/miopen/* ; \
    fi && \
    if [ -d "/tmp/.miopen" ]; then \
        rm -rf /tmp/.miopen/* ; \
    fi

# Set MIOpen environment variables for better performance
# Disable cache to avoid "Duplicate ID" warnings completely
ENV MIOPEN_FIND_MODE=1 \
    MIOPEN_USER_DB_PATH=/tmp/.miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/tmp/.miopen \
    MIOPEN_DISABLE_CACHE=1 \
    MIOPEN_ENABLE_LOGGING=0

# Pre-create MIOpen cache directory with proper permissions
RUN mkdir -p /tmp/.miopen && chmod 1777 /tmp/.miopen

# ============================================================================
# Optional: Install additional utilities for debugging
# ============================================================================
# Uncomment if you need debugging tools:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     lshw \
#     pciutils \
#     && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Verification (Optional - useful for debugging)
# ============================================================================
# Verify ROCm installation
RUN rocminfo > /dev/null 2>&1 || echo "ROCm info check failed (expected in non-GPU build environment)"

# Note: The K8s deployment config should override these env vars if needed:
# - MIOPEN_FIND_MODE is already set in deployment_config.env_vars
# - MIOPEN_USER_DB_PATH is already set in deployment_config.env_vars


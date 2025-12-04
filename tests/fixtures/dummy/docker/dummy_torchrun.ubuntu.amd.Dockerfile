# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch
FROM $BASE_DOCKER

# Install any additional dependencies for torchrun testing
# (rocm/pytorch already has PyTorch with distributed support)


# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Dockerfile for PyTorch Benchmarking with TheRock ROCm Distribution
# TheRock provides HIP and ROCm components via Python pip packages
# Reference: https://github.com/ROCm/TheRock
#
###############################################################################
ARG BASE_DOCKER=ubuntu:24.04
FROM ${BASE_DOCKER}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gfortran \
    git \
    ninja-build \
    cmake \
    g++ \
    pkg-config \
    xxd \
    patchelf \
    automake \
    libtool \
    python3-venv \
    python3-dev \
    python3-pip \
    libegl1-mesa-dev \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Clone TheRock repository
ARG THEROCK_BRANCH=main
RUN git clone https://github.com/ROCm/TheRock.git /workspace/TheRock && \
    cd /workspace/TheRock && \
    git checkout ${THEROCK_BRANCH}

WORKDIR /workspace/TheRock

# Setup Python virtual environment and install dependencies
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Fetch sources (includes submodules and patches)
RUN . .venv/bin/activate && \
    python3 ./build_tools/fetch_sources.py

# Configure build with CMake
# Default to gfx942 (MI300 series), can be overridden with build arg
ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942

# Enable components needed for PyTorch:
# - CORE_RUNTIME: Essential ROCm runtime
# - HIP_RUNTIME: HIP runtime for GPU execution
# - BLAS: rocBLAS for linear algebra operations
# - PRIM: rocPRIM for parallel primitives
# - RAND: rocRAND for random number generation
# This is much faster than building all components
RUN . .venv/bin/activate && \
    cmake -B build -GNinja . \
    -DTHEROCK_AMDGPU_TARGETS=${MAD_SYSTEM_GPU_ARCHITECTURE} \
    -DTHEROCK_ENABLE_ALL=OFF \
    -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
    -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
    -DTHEROCK_ENABLE_BLAS=ON \
    -DTHEROCK_ENABLE_PRIM=ON \
    -DTHEROCK_ENABLE_RAND=ON \
    -DBUILD_TESTING=ON

# Build TheRock components
# This will take some time depending on enabled components
RUN . .venv/bin/activate && \
    cmake --build build

# Install built components
RUN . .venv/bin/activate && \
    cmake --install build --prefix /opt/rocm

# Set up runtime environment
ENV PATH=/opt/rocm/bin:/workspace/TheRock/.venv/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm

# Install PyTorch with ROCm support
# Using PyTorch's official ROCm wheels that work with TheRock's ROCm distribution
RUN . /workspace/TheRock/.venv/bin/activate && \
    pip3 install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installations
RUN . /workspace/TheRock/.venv/bin/activate && \
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'ROCm/HIP: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')"

# Create entrypoint script to activate venv
RUN echo '#!/bin/bash\n\
source /workspace/TheRock/.venv/bin/activate\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# Labels
LABEL maintainer="AMD ROCm <mad.support@amd.com>"
LABEL description="TheRock PyTorch Benchmark - The HIP Environment and ROCm Kit with PyTorch"
LABEL version="nightly"
LABEL gpu_architecture="${MAD_SYSTEM_GPU_ARCHITECTURE}"
LABEL components="core_runtime,hip_runtime,blas,prim,rand,pytorch"


# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
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

# Download submodules and apply patches
# Note: dvc is optional but recommended for faster builds
RUN apt-get update && apt-get install -y snapd && \
    rm -rf /var/lib/apt/lists/* || true

# Fetch sources (includes submodules and patches)
RUN . .venv/bin/activate && \
    python3 ./build_tools/fetch_sources.py

# Configure build with CMake
# Default to gfx942 (MI300 series), can be overridden with build arg
ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942
ARG THEROCK_ENABLE_ALL=ON
ARG THEROCK_ENABLE_PYTORCH=OFF

# Create build directory and configure
RUN . .venv/bin/activate && \
    cmake -B build -GNinja . \
    -DTHEROCK_AMDGPU_TARGETS=${MAD_SYSTEM_GPU_ARCHITECTURE} \
    -DTHEROCK_ENABLE_ALL=${THEROCK_ENABLE_ALL} \
    -DBUILD_TESTING=ON

# Build TheRock components
# This will take a significant amount of time depending on enabled components
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

# Create entrypoint script
RUN echo '#!/bin/bash\n\
source /workspace/TheRock/.venv/bin/activate\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# Labels
LABEL maintainer="AMD ROCm"
LABEL description="TheRock - The HIP Environment and ROCm Kit"
LABEL version="nightly"
LABEL gpu_architecture="${MAD_SYSTEM_GPU_ARCHITECTURE}"

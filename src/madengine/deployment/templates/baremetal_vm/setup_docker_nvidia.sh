#!/bin/bash
#
# Setup script for Docker Engine with NVIDIA CUDA GPU support in VM.
#
# This script is executed inside the VM to install Docker and configure
# GPU access for NVIDIA GPUs with CUDA.
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#

set -e

echo "========================================"
echo "Setting up Docker Engine with NVIDIA CUDA"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root"
    exit 1
fi

# Update package lists
echo "[1/7] Updating package lists..."
apt-get update -qq

# Install prerequisites
echo "[2/7] Installing prerequisites..."
apt-get install -y -qq \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common

# Add Docker's official GPG key
echo "[3/7] Adding Docker GPG key..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "[4/7] Adding Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
echo "[5/7] Installing Docker Engine..."
apt-get update -qq
apt-get install -y -qq \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# Start and enable Docker service
systemctl start docker
systemctl enable docker

# Install NVIDIA Container Toolkit
echo "[6/7] Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit

# Configure Docker for NVIDIA GPU
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Verify Docker installation
echo "[7/7] Verifying Docker installation..."
docker --version

# Verify GPU access
echo ""
echo "Checking GPU access..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found, checking GPU visibility..."
    nvidia-smi || echo "Warning: nvidia-smi failed, GPU may not be visible yet"
else
    echo "⚠ nvidia-smi not found (install CUDA drivers if needed)"
fi

# Test Docker with GPU
echo ""
echo "Testing Docker with GPU access..."
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi || \
    echo "Warning: GPU test failed"

echo ""
echo "========================================"
echo "✓ Docker setup complete!"
echo "========================================"
echo ""
echo "Docker version: $(docker --version)"
echo "Docker is running and configured for NVIDIA GPUs"
echo ""

# Cleanup
apt-get clean
rm -rf /var/lib/apt/lists/*

exit 0

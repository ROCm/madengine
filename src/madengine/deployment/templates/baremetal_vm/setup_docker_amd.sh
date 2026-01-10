#!/bin/bash
#
# Setup script for Docker Engine with AMD ROCm GPU support in VM.
#
# This script is executed inside the VM to install Docker and configure
# GPU access for AMD GPUs with ROCm.
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#

set -e

echo "========================================"
echo "Setting up Docker Engine with AMD ROCm"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root"
    exit 1
fi

# Update package lists
echo "[1/6] Updating package lists..."
apt-get update -qq

# Install prerequisites
echo "[2/6] Installing prerequisites..."
apt-get install -y -qq \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common

# Add Docker's official GPG key
echo "[3/6] Adding Docker GPG key..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "[4/6] Adding Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
echo "[5/6] Installing Docker Engine..."
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

# Verify Docker installation
echo "[6/6] Verifying Docker installation..."
docker --version

# Configure Docker for AMD ROCm GPU access
echo ""
echo "Configuring Docker for AMD ROCm GPU access..."

# Create Docker daemon config
cat > /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-runtime": "runc",
  "runtimes": {
    "rocm": {
      "path": "/usr/bin/rocm-runtime"
    }
  }
}
EOF

# Restart Docker to apply config
systemctl restart docker

# Verify GPU access (if rocm-smi is available)
echo ""
echo "Checking GPU access..."
if command -v rocm-smi &> /dev/null; then
    echo "✓ rocm-smi found, checking GPU visibility..."
    rocm-smi || echo "Warning: rocm-smi failed, GPU may not be visible yet"
else
    echo "⚠ rocm-smi not found (install ROCm if needed)"
fi

# Test Docker with a simple container
echo ""
echo "Testing Docker with hello-world..."
docker run --rm hello-world

echo ""
echo "========================================"
echo "✓ Docker setup complete!"
echo "========================================"
echo ""
echo "Docker version: $(docker --version)"
echo "Docker is running and configured for AMD GPUs"
echo ""

# Cleanup
apt-get clean
rm -rf /var/lib/apt/lists/*

exit 0

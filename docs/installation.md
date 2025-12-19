# Installation Guide

Complete installation instructions for madengine.

## Prerequisites

- **Python 3.8+** with pip
- **Docker** with GPU support (ROCm for AMD, CUDA for NVIDIA)
- **Git** for repository management
- **MAD package** - Required for model discovery and execution

## Quick Install

### From GitHub

```bash
# Basic installation
pip install git+https://github.com/ROCm/madengine.git

# With Kubernetes support
pip install "madengine[kubernetes] @ git+https://github.com/ROCm/madengine.git"

# With all optional dependencies
pip install "madengine[all] @ git+https://github.com/ROCm/madengine.git"
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/ROCm/madengine.git
cd madengine

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks (optional, for contributors)
pre-commit install
```

## Optional Dependencies

| Extra | Install Command | Use Case |
|-------|----------------|----------|
| `kubernetes` | `pip install madengine[kubernetes]` | Kubernetes deployment support |
| `dev` | `pip install madengine[dev]` | Development tools (pytest, black, mypy, etc.) |
| `all` | `pip install madengine[all]` | All optional dependencies |

**Note**: SLURM deployment requires no additional Python dependencies (uses CLI commands).

## MAD Package Setup

madengine requires the MAD package for model definitions and execution scripts.

```bash
# Clone MAD package
git clone https://github.com/ROCm/MAD.git
cd MAD

# Install madengine within MAD directory
pip install git+https://github.com/ROCm/madengine.git

# Verify installation
madengine-cli --version
madengine discover  # Test model discovery
```

## Docker GPU Setup

### AMD ROCm

```bash
# Test ROCm GPU access
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  rocm/pytorch:latest rocm-smi

# Verify with madengine
madengine-cli run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

### NVIDIA CUDA

```bash
# Test CUDA GPU access
docker run --rm --gpus all nvidia/cuda:latest nvidia-smi

# Verify with madengine
madengine-cli run --tags dummy \
  --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "UBUNTU"}'
```

## Verify Installation

```bash
# Check installation
madengine-cli --version
madengine --version

# Test basic functionality (requires MAD package)
cd /path/to/MAD
madengine discover --tags dummy
madengine-cli run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

## Troubleshooting

### Import Errors

If you get import errors, ensure your virtual environment is activated and madengine is installed:

```bash
pip list | grep madengine
```

### Docker Permission Issues

If you encounter Docker permission errors:

```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### ROCm GPU Not Detected

```bash
# Check ROCm installation
rocm-smi

# Verify devices are accessible
ls -la /dev/kfd /dev/dri
```

### MAD Package Not Found

Ensure you're running madengine commands from within a MAD package directory:

```bash
cd /path/to/MAD
export MODEL_DIR=$(pwd)
madengine discover
```

## Next Steps

- [User Guide](user-guide.md) - Learn how to use madengine
- [Deployment Guide](deployment.md) - Deploy to Kubernetes or SLURM
- [Quick Start](how-to-quick-start.md) - Run your first model


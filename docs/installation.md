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
| `baremetal_vm` | `pip install libvirt-python` | Bare Metal VM execution support |
| `dev` | `pip install madengine[dev]` | Development tools (pytest, black, mypy, etc.) |
| `all` | `pip install madengine[all]` | All optional dependencies |

**Note**: 
- SLURM deployment requires no additional Python dependencies (uses CLI commands)
- Bare Metal VM requires `libvirt-python` and system packages (KVM/QEMU, libvirt)

## MAD Package Setup

madengine requires the MAD package for model definitions and execution scripts.

```bash
# Clone MAD package
git clone https://github.com/ROCm/MAD.git
cd MAD

# Install madengine within MAD directory
pip install git+https://github.com/ROCm/madengine.git

# Verify installation
madengine --version
madengine discover  # Test model discovery
```

## Docker GPU Setup

### AMD ROCm

```bash
# Test ROCm GPU access
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  rocm/pytorch:latest rocm-smi

# Verify with madengine
madengine run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

### NVIDIA CUDA

```bash
# Test CUDA GPU access
docker run --rm --gpus all nvidia/cuda:latest nvidia-smi

# Verify with madengine
madengine run --tags dummy \
  --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "UBUNTU"}'
```

## Bare Metal VM Setup (Optional)

For VM-based execution with guaranteed isolation and cleanup on bare metal nodes.

### Prerequisites

**Hardware:**
- CPU with virtualization support (Intel VT-x or AMD-V)
- IOMMU enabled (Intel VT-d or AMD-Vi)
- AMD MI200/MI300 GPU with SR-IOV support or NVIDIA GPU with VFIO support
- At least 128GB RAM for typical workloads

**Software:**
```bash
# Install KVM/QEMU and libvirt
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils

# Install Python package
pip install libvirt-python

# Enable IOMMU in /etc/default/grub
GRUB_CMDLINE_LINUX="amd_iommu=on iommu=pt"  # For AMD
GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt"  # For Intel

sudo update-grub && sudo reboot
```

### Verify Setup

```bash
# Check KVM module
lsmod | grep kvm  # Should show kvm_amd or kvm_intel

# Verify IOMMU
dmesg | grep -i iommu  # Should show "IOMMU enabled"

# Check libvirt
systemctl status libvirtd
```

### Prepare Base Image

Create a base VM image with Ubuntu and GPU drivers (one-time setup):

```bash
# Create base image
sudo qemu-img create -f qcow2 \
  /var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2 50G

# Install Ubuntu 22.04 + ROCm/CUDA drivers in a temporary VM
# Then shut down and use as base image
```

### Test Bare Metal VM

```bash
cat > baremetal-vm-test.json << 'EOF'
{
  "baremetal_vm": {
    "enabled": true,
    "base_image": "/var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2",
    "vcpus": 16,
    "memory": "64G",
    "gpu_passthrough": {
      "mode": "sriov",
      "gpu_vendor": "AMD"
    }
  },
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU"
}
EOF

madengine run --tags dummy \
  --additional-context-file baremetal-vm-test.json \
  --timeout 3600 \
  --live-output
```

For complete setup instructions, see **[Bare Metal VM Guide](baremetal-vm.md)**.

## Verify Installation

```bash
# Check installation
madengine --version
madengine --version

# Test basic functionality (requires MAD package)
cd /path/to/MAD
madengine discover --tags dummy
madengine run --tags dummy \
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

- [Usage Guide](usage.md) - Learn how to use madengine
- [Deployment Guide](deployment.md) - Deploy to Kubernetes, SLURM, or Bare Metal VM
- [Bare Metal VM Guide](baremetal-vm.md) - VM-based execution with isolation
- [CLI Reference](cli-reference.md) - Complete command options


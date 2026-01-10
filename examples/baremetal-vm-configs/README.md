# Bare Metal VM Configuration Examples

Example configurations for running madengine on bare metal nodes using VM-based isolation.

## Overview

Bare metal VM execution mode provides:
- **Isolation**: Complete environment isolation via ephemeral VMs
- **Cleanup**: Guaranteed restoration to clean state after execution  
- **Compatibility**: Reuses 100% of existing Docker images and workflows
- **Performance**: Near-native GPU performance with SR-IOV/VFIO passthrough

## Architecture

```
Bare Metal Node (KVM host)
└── Ephemeral VM (Ubuntu + Docker)
    └── Docker Container (existing madengine images)
        └── Model execution
```

## Prerequisites

### System Requirements

1. **Hardware**:
   - CPU with virtualization extensions (Intel VT-x or AMD-V)
   - IOMMU support (Intel VT-d or AMD-Vi)
   - GPU with SR-IOV or VFIO support (AMD MI200/MI300 recommended)
   - At least 128GB RAM for typical workloads

2. **Software**:
   - Linux host OS (Ubuntu 22.04+ recommended)
   - KVM/QEMU installed (`apt install qemu-kvm libvirt-daemon-system`)
   - libvirt-python (`pip install libvirt-python`)
   - Base VM image with GPU drivers pre-installed

### Base Image Creation

Create a base VM image with GPU drivers pre-installed:

```bash
# For AMD GPUs with ROCm
qemu-img create -f qcow2 /var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2 50G

# Install Ubuntu 22.04 and ROCm drivers in the VM
# Then shutdown and use as base image

# For NVIDIA GPUs with CUDA
qemu-img create -f qcow2 /var/lib/libvirt/images/ubuntu-22.04-cuda.qcow2 50G
# Install Ubuntu 22.04 and CUDA drivers
```

### Enable IOMMU

Add to kernel boot parameters in `/etc/default/grub`:

```bash
# For Intel CPUs
GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt"

# For AMD CPUs
GRUB_CMDLINE_LINUX="amd_iommu=on iommu=pt"

# Update grub and reboot
sudo update-grub
sudo reboot
```

Verify IOMMU is enabled:

```bash
dmesg | grep -i iommu
# Should show "IOMMU enabled" or similar
```

## Configuration Files

### Single GPU AMD (SR-IOV)

**File**: `single-gpu-amd.json`

Basic configuration for single AMD GPU using SR-IOV Virtual Functions.

```bash
madengine run --tags llama2_7b \
  --additional-context-file examples/baremetal-vm-configs/single-gpu-amd.json
```

### Multi-GPU AMD (SR-IOV)

**File**: `multi-gpu-amd.json`

Configuration for multi-GPU training with AMD GPUs.

```bash
madengine run --tags llama2_70b \
  --additional-context-file examples/baremetal-vm-configs/multi-gpu-amd.json
```

### Single GPU NVIDIA (VFIO)

**File**: `single-gpu-nvidia.json`

Configuration for NVIDIA GPU using full VFIO passthrough.

```bash
madengine run --tags model \
  --additional-context-file examples/baremetal-vm-configs/single-gpu-nvidia.json
```

## Configuration Options

### Main Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `enabled` | Enable bare metal VM mode | `false` | Yes |
| `hypervisor` | Hypervisor type | `"kvm"` | No |
| `base_image` | Path to base VM image | - | Yes |
| `vcpus` | Number of virtual CPUs | `32` | No |
| `memory` | VM memory (e.g., "128G") | `"128G"` | No |
| `disk_size` | VM disk size | `"100G"` | No |

### GPU Passthrough Options

| Option | Description | Options | Required |
|--------|-------------|---------|----------|
| `mode` | Passthrough mode | `"sriov"`, `"vfio"`, `"vgpu"` | Yes |
| `gpu_vendor` | GPU vendor | `"AMD"`, `"NVIDIA"` | Yes |
| `gpu_architecture` | GPU architecture | `"gfx90a"`, `"sm_80"`, etc. | No |
| `gpu_ids` | PCI addresses of GPUs | Array of strings | No (auto-discovers) |

### Cleanup Options

| Option | Description | Default |
|--------|-------------|---------|
| `mode` | Cleanup mode | `"destroy"` |
| `verify_clean` | Verify clean state | `true` |
| `timeout` | Cleanup timeout (seconds) | `300` |

## Usage Workflow

### 1. SSH to Bare Metal Node

```bash
ssh admin@baremetal-gpu-node-01.example.com
```

### 2. Prepare Workspace

```bash
cd /workspace
git clone https://github.com/ROCm/MAD.git
cd MAD
```

### 3. Run madengine

```bash
madengine run --tags model_name \
  --additional-context-file /path/to/baremetal-vm-config.json \
  --timeout 3600 \
  --live-output
```

### 4. What Happens

1. madengine creates ephemeral VM from base image
2. Configures GPU passthrough (SR-IOV or VFIO)
3. Starts VM and waits for SSH
4. Installs Docker Engine inside VM
5. Runs existing Docker workflow (same as local execution!)
6. Collects results (perf_entry.csv, etc.)
7. Destroys VM completely
8. Verifies bare metal restored to clean state

### 5. View Results

```bash
cat perf_entry.csv
madengine report to-html --csv-file perf_entry.csv
```

## GPU Passthrough Modes

### SR-IOV (Recommended for AMD)

**Best for**: AMD MI200/MI300 series GPUs

**Advantages**:
- Share single GPU among multiple VMs
- Better resource utilization
- Dynamic VF creation/destruction

**Requirements**:
- GPU must support SR-IOV
- IOMMU enabled in kernel

**Example**:
```json
{
  "gpu_passthrough": {
    "mode": "sriov",
    "gpu_vendor": "AMD"
  }
}
```

### VFIO (Full Passthrough)

**Best for**: NVIDIA GPUs, or when full GPU access needed

**Advantages**:
- Full GPU access to VM
- Maximum performance
- Works with most GPUs

**Requirements**:
- IOMMU enabled
- GPU bound to vfio-pci driver

**Example**:
```json
{
  "gpu_passthrough": {
    "mode": "vfio",
    "gpu_vendor": "NVIDIA"
  }
}
```

### vGPU

**Best for**: NVIDIA GRID or AMD MxGPU

**Advantages**:
- Hardware-accelerated GPU sharing
- Best for inference workloads

**Requirements**:
- NVIDIA GRID license or AMD MxGPU support
- Vendor-specific drivers

## Troubleshooting

### VM Creation Fails

```bash
# Check KVM is loaded
lsmod | grep kvm

# Check libvirtd is running
systemctl status libvirtd

# Check base image exists
ls -lh /var/lib/libvirt/images/
```

### IOMMU Not Enabled

```bash
# Check kernel parameters
cat /proc/cmdline

# Should show intel_iommu=on or amd_iommu=on

# If not, edit /etc/default/grub and update
sudo update-grub
sudo reboot
```

### GPU Not Visible in VM

```bash
# Check IOMMU groups
find /sys/kernel/iommu_groups/ -type l

# Check GPU PCI address
lspci | grep -i vga

# For SR-IOV, check VFs created
cat /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs
```

### Docker Installation Fails

```bash
# SSH into VM manually
ssh root@<vm-ip>

# Check internet connectivity
ping google.com

# Manually install Docker
/tmp/setup_docker.sh
```

### Performance Issues

- Ensure IOMMU is in pass-through mode (`iommu=pt` in kernel params)
- Use CPU pinning for better performance
- Allocate more vCPUs/memory if needed
- Check GPU is not overcommitted

## Advanced Configuration

### Custom Base Image Path

```json
{
  "baremetal_vm": {
    "base_image": "/custom/path/to/base-image.qcow2"
  }
}
```

### SSH Key Authentication

```json
{
  "baremetal_vm": {
    "ssh_user": "ubuntu",
    "ssh_key": "/home/user/.ssh/id_rsa"
  }
}
```

### Preserve VM for Debugging

```json
{
  "baremetal_vm": {
    "cleanup": {
      "mode": "preserve"
    }
  }
}
```

Then manually inspect and cleanup:

```bash
virsh list --all
virsh destroy madengine-vm-xxxxx
virsh undefine madengine-vm-xxxxx
```

## Performance Comparison

Expected performance vs bare metal:

| Metric | Bare Metal | VM (SR-IOV) | VM (VFIO) |
|--------|-----------|-------------|-----------|
| Training throughput | 100% | 96-98% | 94-97% |
| Inference latency | Baseline | +50-100μs | +100-200μs |
| Memory bandwidth | 100% | 98-99% | 98-99% |
| GPU utilization | 100% | 95-98% | 95-98% |

The 2-5% overhead is acceptable given the isolation and cleanup benefits.

## See Also

- [madengine Documentation](../../docs/)
- [Deployment Guide](../../docs/deployment.md)
- [Bare Metal VM Design Proposal](../../docs/baremetal-vm-proposal.md)

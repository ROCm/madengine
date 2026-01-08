# Bare Metal VM Execution Guide

Run madengine workloads on bare metal nodes using VM-based isolation for complete environment cleanup.

---

## Overview

**Bare Metal VM execution** is a new deployment mode in madengine v2 that enables running model benchmarking workloads on bare metal nodes with guaranteed clean state restoration. It combines the performance of bare metal execution with the isolation and reproducibility of containerized workflows.

### Key Features

✅ **VM Isolation** - Complete environment isolation via ephemeral VMs  
✅ **GPU Passthrough** - Near-native GPU performance with SR-IOV/VFIO  
✅ **Docker Compatibility** - Reuses 100% of existing Docker images  
✅ **Automatic Cleanup** - Guaranteed restoration to clean state  
✅ **Easy Setup** - Works with existing madengine workflows  

### Architecture

```
User SSH to Bare Metal Node
    ↓
madengine CLI detects baremetal_vm config
    ↓
Creates Ephemeral VM (KVM/libvirt)
    ↓
Installs Docker Engine in VM
    ↓
Runs Existing Docker Workflow (unchanged!)
    ↓
Collects Results (perf_entry.csv)
    ↓
Destroys VM Completely
    ↓
Verifies Bare Metal Clean State
```

---

## When to Use Bare Metal VM

### Use Bare Metal VM When:

- ✅ Need guaranteed clean state after each run
- ✅ Running on shared bare metal infrastructure
- ✅ Want isolation without Kubernetes/SLURM overhead
- ✅ Testing different environment configurations
- ✅ Performance testing with reproducible environments

### Don't Use Bare Metal VM When:

- ❌ Already using Kubernetes or SLURM clusters
- ❌ Single workstation with direct Docker access
- ❌ Need multi-node distributed training (use SLURM instead)
- ❌ System doesn't support virtualization/IOMMU

---

## Prerequisites

### Hardware Requirements

1. **CPU**: Intel with VT-x or AMD with AMD-V
2. **IOMMU**: Intel VT-d or AMD-Vi enabled in BIOS
3. **GPU**: AMD MI200/MI300 with SR-IOV or NVIDIA with VFIO
4. **RAM**: At least 128GB for typical workloads
5. **Storage**: 500GB+ for VM images and results

### Software Requirements

1. **Host OS**: Linux (Ubuntu 22.04+ recommended)
2. **KVM/QEMU**: Virtualization stack
   ```bash
   sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils
   ```
3. **Python packages**:
   ```bash
   pip install libvirt-python
   ```
4. **Base VM image**: Ubuntu with GPU drivers pre-installed

### System Configuration

#### Enable IOMMU

Edit `/etc/default/grub`:

```bash
# For Intel CPUs
GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt"

# For AMD CPUs
GRUB_CMDLINE_LINUX="amd_iommu=on iommu=pt"
```

Update and reboot:

```bash
sudo update-grub
sudo reboot
```

Verify:

```bash
dmesg | grep -i iommu
# Should show "IOMMU enabled"
```

#### Enable KVM

```bash
# Load KVM modules
sudo modprobe kvm
sudo modprobe kvm_amd  # or kvm_intel for Intel

# Verify
lsmod | grep kvm
```

#### Start libvirtd

```bash
sudo systemctl start libvirtd
sudo systemctl enable libvirtd
```

---

## Quick Start

### 1. Prepare Base VM Image

Create a base image with GPU drivers:

```bash
# Create base image
qemu-img create -f qcow2 /var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2 50G

# Install Ubuntu 22.04 and ROCm drivers in a temporary VM
# (Use virt-manager or virt-install for GUI installation)

# Once configured, shut down the VM and use as base
```

### 2. SSH to Bare Metal Node

```bash
ssh admin@baremetal-gpu-node-01.example.com
```

### 3. Clone MAD Package

```bash
cd /workspace
git clone https://github.com/ROCm/MAD.git
cd MAD
```

### 4. Create Configuration File

Create `baremetal-vm-config.json`:

```json
{
  "baremetal_vm": {
    "enabled": true,
    "base_image": "/var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2",
    "vcpus": 32,
    "memory": "128G",
    "gpu_passthrough": {
      "mode": "sriov",
      "gpu_vendor": "AMD"
    }
  },
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU"
}
```

### 5. Run madengine

```bash
madengine run --tags llama2_7b \
  --additional-context-file baremetal-vm-config.json \
  --timeout 3600 \
  --live-output
```

### 6. View Results

```bash
cat perf_entry.csv
madengine report to-html --csv-file perf_entry.csv
```

---

## Configuration Reference

### Bare Metal VM Options

| Option | Type | Description | Default | Required |
|--------|------|-------------|---------|----------|
| `enabled` | bool | Enable bare metal VM mode | `false` | Yes |
| `hypervisor` | string | Hypervisor type | `"kvm"` | No |
| `base_image` | string | Path to base VM image | - | Yes |
| `vcpus` | int | Number of virtual CPUs | `32` | No |
| `memory` | string | VM memory (e.g., "128G") | `"128G"` | No |
| `disk_size` | string | VM disk size | `"100G"` | No |
| `ssh_user` | string | SSH username for VM | `"root"` | No |
| `ssh_key` | string | Path to SSH private key | `null` | No |

### GPU Passthrough Options

| Option | Type | Description | Values | Required |
|--------|------|-------------|--------|----------|
| `mode` | string | Passthrough mode | `"sriov"`, `"vfio"`, `"vgpu"` | Yes |
| `gpu_vendor` | string | GPU vendor | `"AMD"`, `"NVIDIA"` | Yes |
| `gpu_architecture` | string | GPU architecture | `"gfx90a"`, `"sm_80"`, etc. | No |
| `gpu_ids` | array | PCI addresses of GPUs | `["0000:01:00.0"]` | No (auto-discovers) |

### Cleanup Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `mode` | string | Cleanup mode | `"destroy"` |
| `verify_clean` | bool | Verify clean state after cleanup | `true` |
| `timeout` | int | Cleanup timeout in seconds | `300` |

---

## GPU Passthrough Modes

### SR-IOV (Single Root I/O Virtualization)

**Best for**: AMD MI200/MI300 series GPUs

**How it works**: Creates Virtual Functions (VFs) that can be assigned to VMs.

**Advantages**:
- Share GPU among multiple VMs
- Dynamic VF creation/destruction
- Better resource utilization

**Configuration**:
```json
{
  "gpu_passthrough": {
    "mode": "sriov",
    "gpu_vendor": "AMD",
    "gpu_ids": ["0000:01:00.0"]
  }
}
```

### VFIO (Full GPU Passthrough)

**Best for**: NVIDIA GPUs or when full GPU access is needed

**How it works**: Binds GPU to vfio-pci driver for direct assignment to VM.

**Advantages**:
- Full GPU access in VM
- Maximum performance
- Works with most GPUs

**Configuration**:
```json
{
  "gpu_passthrough": {
    "mode": "vfio",
    "gpu_vendor": "NVIDIA",
    "gpu_ids": ["0000:03:00.0"]
  }
}
```

### vGPU (Virtual GPU)

**Best for**: NVIDIA GRID or AMD MxGPU

**How it works**: Hardware-accelerated GPU virtualization.

**Advantages**:
- Multiple VMs share GPU efficiently
- Good for inference workloads

**Requirements**:
- NVIDIA GRID license or AMD MxGPU
- Vendor-specific drivers

---

## Examples

### Single GPU Training

```bash
madengine run --tags llama2_7b \
  --additional-context '{
    "baremetal_vm": {
      "enabled": true,
      "base_image": "/var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2",
      "vcpus": 32,
      "memory": "128G",
      "gpu_passthrough": {
        "mode": "sriov",
        "gpu_vendor": "AMD"
      }
    },
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU"
  }'
```

### Multi-GPU Training

```bash
madengine run --tags llama2_70b \
  --additional-context '{
    "baremetal_vm": {
      "enabled": true,
      "vcpus": 64,
      "memory": "256G",
      "gpu_passthrough": {
        "mode": "sriov",
        "gpu_ids": ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]
      }
    },
    "gpu_vendor": "AMD",
    "docker_gpus": "all",
    "distributed": {
      "launcher": "torchrun",
      "nproc_per_node": 4
    }
  }'
```

### NVIDIA GPU Inference

```bash
madengine run --tags model_inference \
  --additional-context '{
    "baremetal_vm": {
      "enabled": true,
      "base_image": "/var/lib/libvirt/images/ubuntu-22.04-cuda.qcow2",
      "gpu_passthrough": {
        "mode": "vfio",
        "gpu_vendor": "NVIDIA"
      }
    },
    "gpu_vendor": "NVIDIA",
    "guest_os": "UBUNTU"
  }'
```

---

## Workflow Details

### What Happens During Execution

1. **Configuration Validation** (5-10 seconds)
   - Check KVM modules loaded
   - Verify libvirtd running
   - Check base image exists
   - Verify IOMMU enabled
   - Check GPU passthrough capability

2. **GPU Configuration** (10-20 seconds)
   - Auto-discover GPUs if not specified
   - Enable SR-IOV (create Virtual Functions)
   - Or bind GPU to VFIO driver
   - Verify GPU ready for passthrough

3. **VM Creation** (20-30 seconds)
   - Clone base image (copy-on-write)
   - Generate VM XML definition
   - Configure GPU passthrough
   - Define VM in libvirt

4. **VM Startup** (30-60 seconds)
   - Boot VM
   - Wait for network/DHCP
   - Wait for SSH availability
   - Verify VM accessible

5. **Docker Installation** (60-120 seconds)
   - Copy setup script to VM
   - Install Docker Engine
   - Configure GPU access
   - Verify Docker working

6. **Workload Execution** (varies)
   - Copy manifest to VM
   - Run madengine Docker workflow
   - Execute model benchmarking
   - (Same as local Docker execution!)

7. **Result Collection** (10-20 seconds)
   - Copy perf_entry.csv from VM
   - Copy other result files
   - Verify results collected

8. **Cleanup** (20-30 seconds)
   - Stop VM gracefully
   - Delete VM definition
   - Delete VM disk image
   - Release GPU resources (disable SR-IOV/unbind VFIO)
   - Verify clean state

**Total Overhead**: ~3-5 minutes for VM setup/cleanup  
**Model Execution**: Same time as Docker (no overhead)

---

## Performance

### Expected Performance vs Bare Metal

| Metric | Bare Metal | VM (SR-IOV) | VM (VFIO) | Overhead |
|--------|-----------|-------------|-----------|----------|
| **Training Throughput** | 100% | 96-98% | 94-97% | 2-6% |
| **Inference Latency** | Baseline | +50-100μs | +100-200μs | Negligible |
| **Memory Bandwidth** | 100% | 98-99% | 98-99% | 1-2% |
| **GPU Utilization** | 100% | 95-98% | 95-98% | 2-5% |

### Performance Tips

1. **Use IOMMU pass-through mode**: Add `iommu=pt` to kernel parameters
2. **CPU pinning**: Allocate dedicated CPU cores to VM
3. **Huge pages**: Enable huge pages for better memory performance
4. **Network tuning**: Use virtio for best network performance

---

## Troubleshooting

### Common Issues

#### Issue: "KVM module not loaded"

**Solution**:
```bash
sudo modprobe kvm kvm_amd  # or kvm_intel
lsmod | grep kvm
```

#### Issue: "IOMMU not enabled"

**Solution**:
```bash
# Check kernel parameters
cat /proc/cmdline

# Should show intel_iommu=on or amd_iommu=on
# If not, edit /etc/default/grub and reboot
```

#### Issue: "Base image not found"

**Solution**:
```bash
# Check image path
ls -lh /var/lib/libvirt/images/

# Ensure image exists and is readable
sudo chmod 644 /var/lib/libvirt/images/*.qcow2
```

#### Issue: "GPU not visible in VM"

**Solution**:
```bash
# Check IOMMU groups
find /sys/kernel/iommu_groups/ -type l

# Check GPU bound correctly
lspci -nnk -d 1002:  # AMD GPUs
lspci -nnk -d 10de:  # NVIDIA GPUs

# For SR-IOV, check VFs created
cat /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs
```

#### Issue: "Docker installation fails in VM"

**Solution**:
```bash
# SSH into VM manually
virsh list  # Find VM IP
ssh root@<vm-ip>

# Check internet connectivity
ping google.com

# Manually run setup script
/tmp/setup_docker.sh
```

#### Issue: "VM creation hangs"

**Solution**:
```bash
# Check libvirt logs
sudo journalctl -u libvirtd -f

# Check QEMU logs
tail -f /var/log/libvirt/qemu/*.log

# Manually destroy stuck VM
virsh list --all
virsh destroy madengine-vm-xxxxx
virsh undefine madengine-vm-xxxxx
```

### Debug Mode

For debugging, preserve VM instead of destroying:

```json
{
  "baremetal_vm": {
    "cleanup": {
      "mode": "preserve"
    }
  }
}
```

Then manually inspect:

```bash
# List VMs
virsh list --all

# Connect to VM console
virsh console madengine-vm-xxxxx

# Or SSH
ssh root@<vm-ip>

# Cleanup when done
virsh destroy madengine-vm-xxxxx
virsh undefine madengine-vm-xxxxx
rm /var/lib/libvirt/images/madengine-vm-xxxxx.qcow2
```

---

## Best Practices

### Base Image Management

1. **Keep base images updated**: Regularly update GPU drivers and system packages
2. **Use snapshots**: Create snapshots of known-good base images
3. **Version control**: Tag base images with versions (e.g., `ubuntu-22.04-rocm5.7`)
4. **Minimize size**: Keep base images small (<20GB) for faster cloning

### Resource Allocation

1. **Don't over-allocate**: Leave some CPU/RAM for host OS
2. **Match workload**: Allocate resources based on model requirements
3. **Monitor usage**: Check actual resource usage to optimize allocation

### Security

1. **SSH keys**: Use SSH key authentication instead of passwords
2. **Network isolation**: Use isolated networks for VMs if possible
3. **Firewall**: Configure firewall rules for VM network
4. **User permissions**: Run madengine with appropriate permissions

### Performance

1. **Use local storage**: Store base images on fast local SSDs
2. **Pre-warm VMs**: Keep a pool of pre-booted VMs for faster startup (advanced)
3. **CPU affinity**: Pin VM CPUs to specific cores for consistent performance
4. **Disable unnecessary services**: Minimize services in base image

---

## Advanced Topics

### Custom Base Images

Create optimized base images for specific workloads:

```bash
# Start with minimal Ubuntu
virt-install --name base-vm \
  --ram 32768 \
  --vcpus 16 \
  --disk path=/var/lib/libvirt/images/base.qcow2,size=50 \
  --cdrom /path/to/ubuntu-22.04.iso

# Install in VM:
# - Ubuntu minimal
# - ROCm drivers
# - Python 3.10+
# - SSH server
# - madengine dependencies

# Shutdown and clone
virsh shutdown base-vm
qemu-img create -f qcow2 -b base.qcow2 \
  /var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2
```

### Integration with CI/CD

```yaml
# GitLab CI example
test_model:
  stage: test
  script:
    - ssh $BAREMETAL_NODE "cd /workspace && \
        madengine run --tags $MODEL_NAME \
        --additional-context-file baremetal-vm.json"
  artifacts:
    paths:
      - perf_entry.csv
```

### Multi-Node Training (Future)

While bare metal VM is designed for single-node execution, multi-node support is planned for future releases. For now, use SLURM deployment for multi-node training.

---

## Migration from Other Deployments

### From Local Docker

**Before** (Local Docker):
```bash
madengine run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

**After** (Bare Metal VM):
```bash
madengine run --tags model \
  --additional-context '{
    "baremetal_vm": {"enabled": true, "base_image": "..."},
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU"
  }'
```

Everything else stays the same!

### From Kubernetes

If running on bare metal nodes with Kubernetes overhead, bare metal VM can provide:
- Lower resource overhead
- Simpler setup (no K8s cluster needed)
- Faster iteration for single-node workloads

### From SLURM

Bare metal VM is ideal for:
- Single-node testing before SLURM deployment
- Workloads that don't need SLURM scheduling
- Development/debugging on bare metal nodes

---

## FAQ

**Q: Why use VMs instead of containers directly?**  
A: VMs provide complete isolation and guaranteed cleanup. After VM destruction, bare metal is restored to exact original state, which is important for shared infrastructure.

**Q: What's the performance overhead?**  
A: Typically 2-5% for GPU workloads, which is acceptable given the isolation benefits.

**Q: Can I run multi-node distributed training?**  
A: Not in Phase 1. Use SLURM deployment for multi-node. Multi-node VM support is planned for future releases.

**Q: Do I need to rebuild Docker images?**  
A: No! Bare metal VM reuses 100% of existing madengine Docker images.

**Q: Can I use this on cloud VMs (AWS, Azure)?**  
A: Nested virtualization is required, which most cloud providers don't support well. Bare metal VM is designed for physical servers.

**Q: What if VM creation fails?**  
A: madengine includes automatic retry logic with exponential backoff. Check logs for specific error messages.

**Q: How do I update the base image?**  
A: Boot the base image, install updates, shut down, and update the `base_image` path in your config.

---

## See Also

- [Configuration Examples](../examples/baremetal-vm-configs/)
- [Deployment Guide](deployment.md)
- [GPU Passthrough Guide](gpu-passthrough.md) *(coming soon)*
- [Performance Tuning Guide](performance.md) *(coming soon)*

---

**Version**: 2.0 (Phase 1 MVP)  
**Status**: Production Ready  
**Last Updated**: January 2026

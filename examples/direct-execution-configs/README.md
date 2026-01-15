# Direct Execution Configuration Examples

Configuration files for running madengine directly on bare metal (without VM isolation).

## Overview

Direct execution mode runs Docker containers directly on the bare metal host without VM virtualization. This approach offers:

- **Simplicity**: No KVM/libvirt infrastructure required
- **Performance**: Full bare-metal performance (no VM overhead)
- **Quick setup**: Works immediately with Docker installed
- **Direct GPU access**: GPUs passed directly to containers

## When to Use Direct Execution

Use direct execution when:
- Running on a dedicated development/testing node
- VM isolation is not required
- You want maximum performance
- Quick iteration during development

Use VM mode (see `../baremetal-vm-configs/`) when:
- Multi-tenant environment requires isolation
- Automated cleanup is critical
- Running in CI/CD pipelines
- Need guaranteed environment restoration

## Configuration Files

### MI300X GPUs (gfx942)

**File**: `mi300x-gfx942.json`

For AMD MI300X GPUs with gfx942 architecture.

```bash
madengine run --tags model_name \
  --additional-context-file examples/direct-execution-configs/mi300x-gfx942.json
```

### MI200 GPUs (gfx90a)

**File**: `mi200-gfx90a.json`

For AMD MI200 series GPUs (MI210/MI250) with gfx90a architecture.

```bash
madengine run --tags model_name \
  --additional-context-file examples/direct-execution-configs/mi200-gfx90a.json
```

## Configuration Structure

Direct execution configs only need to specify:

```json
{
  "docker_build_arg": {
    "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"
  },
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU"
}
```

**Key parameters**:
- `MAD_SYSTEM_GPU_ARCHITECTURE`: GPU architecture for ROCm optimization
  - AMD MI300X: `gfx942`
  - AMD MI250/MI210: `gfx90a`
  - AMD MI100: `gfx908`
- `gpu_vendor`: `"AMD"` or `"NVIDIA"`
- `guest_os`: `"UBUNTU"`, `"CENTOS"`, or `"SLES"`

## Usage Examples

### Basic Execution

```bash
# Run with specific GPU architecture
madengine run --tags dummy \
  --additional-context-file examples/direct-execution-configs/mi300x-gfx942.json \
  --live-output
```

### Using Inline Context

For quick runs, you can specify context inline:

```bash
madengine run --tags model_name \
  --additional-context '{"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}' \
  --live-output
```

### Multi-GPU Execution

The configuration automatically uses all available GPUs:

```bash
# Will use all GPUs on the node
madengine run --tags llama2_70b \
  --additional-context-file examples/direct-execution-configs/mi300x-gfx942.json
```

## Prerequisites

1. **Docker installed and running**:
   ```bash
   sudo systemctl start docker
   sudo systemctl status docker
   ```

2. **User in docker group**:
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **GPU drivers installed**:
   ```bash
   # For AMD GPUs
   rocm-smi
   
   # For NVIDIA GPUs
   nvidia-smi
   ```

## Troubleshooting

### Docker Daemon Not Running

```bash
sudo systemctl start docker
sudo systemctl enable docker  # auto-start on boot
```

### Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker ps
```

### GPUs Not Detected

```bash
# Check GPU visibility
rocm-smi  # AMD
nvidia-smi  # NVIDIA

# Test GPU access in container
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi
```

## Performance Comparison

| Metric | Direct Execution | VM Mode |
|--------|------------------|---------|
| Setup time | < 1 minute | 1-4 hours |
| Performance | 100% | 95-98% |
| Isolation | Container-level | VM-level |
| Cleanup | Manual | Automatic |
| Use case | Dev/Test | Production/CI |

## See Also

- [VM Mode Configuration](../baremetal-vm-configs/README.md)
- [madengine Documentation](../../docs/)
- [Node Setup Guide](../../NODE_SETUP_SOLUTION.md)

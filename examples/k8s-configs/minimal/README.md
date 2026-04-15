# Minimal Kubernetes Configuration Examples

These are minimal configuration examples that leverage madengine's built-in defaults.

## 🎯 Philosophy

With madengine v2.0+, you only need to specify what's unique to your deployment:
- **GPU count** (required)
- **Distributed settings** (if using multiple GPUs)
- **Overrides** (only if you need to change defaults)

Everything else is automatically configured based on best practices.

## 🚀 Key Feature: Auto-Inferred Deployment Type

**No `deploy` field needed!** Deployment type is automatically inferred:
- Presence of `k8s` field → K8s deployment
- Presence of `slurm` field → SLURM deployment
- Neither present → Local execution

This follows the **Convention over Configuration** principle.

## 📁 Examples

### [single-gpu-minimal.json](single-gpu-minimal.json)
**Just 1 field:** GPU count
```json
{
  "k8s": {
    "gpu_count": 1
  }
}
```
**Auto-applied:**
- Memory: 16Gi / 32Gi limit
- CPU: 8 / 16 limit
- AMD optimizations
- Standard env vars

**Usage:**
```bash
madengine run --tags model \
  --additional-context-file examples/k8s-configs/minimal/single-gpu-minimal.json
```

---

### [multi-gpu-minimal.json](multi-gpu-minimal.json)
**Multi-GPU training** with minimal config
```json
{
  "k8s": {
    "gpu_count": 2
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 1,
    "nproc_per_node": 2
  }
}
```
**Auto-applied:**
- Memory: 64Gi / 128Gi limit
- CPU: 16 / 32 limit
- All AMD multi-GPU optimizations
- NCCL/RCCL environment variables
- ROCm performance tuning

---

### [multi-node-minimal.json](multi-node-minimal.json)
**Multi-node distributed** training (2 nodes × 2 GPUs = 4 GPUs total)
```json
{
  "k8s": {
    "gpu_count": 2
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 2,
    "nproc_per_node": 2
  }
}
```
**Auto-applied:**
- All multi-GPU optimizations
- `host_ipc: true` for shared memory
- Multi-node NCCL settings
- Timeout and async error handling

---

### [nvidia-gpu-minimal.json](nvidia-gpu-minimal.json)
**NVIDIA GPUs** get different optimizations
```json
{
  "gpu_vendor": "NVIDIA",
  "k8s": {
    "gpu_count": 4
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 1,
    "nproc_per_node": 4
  }
}
```
**Auto-applied:**
- `gpu_resource_name: nvidia.com/gpu`
- NVIDIA-specific NCCL settings
- P2P optimizations
- NVLink configuration

---

### [custom-namespace-minimal.json](custom-namespace-minimal.json)
**Override defaults** when needed
```json
{
  "k8s": {
    "gpu_count": 1,
    "namespace": "ml-team",
    "memory": "32Gi"
  }
}
```
**Shows:** You can override any default while keeping others

---

### [primus-minimal.json](primus-minimal.json)
**Primus pretrain** with `distributed.launcher: "primus"` and `distributed.primus.config_path` pointing at a Primus experiment YAML (Megatron, TorchTitan, or MaxText layout under the Primus repo).

```json
{
  "k8s": {
    "gpu_count": 2
  },
  "distributed": {
    "launcher": "primus",
    "nnodes": 1,
    "nproc_per_node": 2,
    "primus": {
      "config_path": "examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml",
      "cli_extra": ""
    }
  }
}
```

**See:** [Launchers Guide](../../../docs/launchers.md#5-primus) and [examples/k8s-configs/README.md](../README.md#primus-on-kubernetes) (Primus on Kubernetes).

---

## 🔄 Comparison: Old vs New

### Before (Full Config Required)
```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "deploy": "k8s",
  "k8s": {
    "kubeconfig": "~/.kube/config",
    "namespace": "default",
    "gpu_count": 1,
    "memory": "16Gi",
    "memory_limit": "32Gi",
    "cpu": "8",
    "cpu_limit": "16",
    "image_pull_policy": "Always",
    "backoff_limit": 3
  },
  "env_vars": {
    "OMP_NUM_THREADS": "8"
  },
  "debug": false
}
```

### After (Minimal)
```json
{
  "k8s": {
    "gpu_count": 1
  }
}
```

**Both produce identical results!**

---

## 🚀 Quick Start

1. **Copy a minimal config:**
   ```bash
   cp examples/k8s-configs/minimal/single-gpu-minimal.json my-config.json
   ```

2. **Customize if needed:**
   ```bash
   # Edit my-config.json to add namespace, memory overrides, etc.
   ```

3. **Build and run:**
   ```bash
   MODEL_DIR=tests/fixtures/dummy madengine build \
     --tags my_model \
     --additional-context-file my-config.json
   
   madengine run \
     --manifest-file build_manifest.json \
     --live-output
   ```

---

## 💡 Tips

### Use CLI for one-off overrides
```bash
madengine run --tags model \
  --additional-context-file minimal/single-gpu-minimal.json \
  --additional-context '{"debug": true}'
```

### View resolved configuration
```bash
madengine config show \
  --additional-context-file my-config.json
```
(Shows all defaults that will be applied)

### Start minimal, add as needed
1. Start with minimal config
2. Test and validate
3. Add overrides only when necessary
4. Advanced features (PVCs, tolerations, node selectors) work the same

---

## 📚 See Full Examples

For advanced use cases with PVCs, tolerations, node selectors, etc., see:
- [../01-single-node-single-gpu.json](../01-single-node-single-gpu.json)
- [../04-multi-node-advanced.json](../04-multi-node-advanced.json)
- [../06-data-provider-with-pvc.json](../06-data-provider-with-pvc.json)

These full configs still work exactly as before - no breaking changes!


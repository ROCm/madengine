# Kubernetes Configuration Examples - Quick Index

## 🎯 Choose Your Configuration

### By GPU Count

| GPUs | Nodes | File | Description |
|------|-------|------|-------------|
| 1 | 1 | [`00-minimal.json`](00-minimal.json) | Quickstart (uses defaults) |
| 1 | 1 | [`01-single-node-single-gpu.json`](01-single-node-single-gpu.json) | Basic configuration |
| 8 | 1 | [`02-single-node-multi-gpu.json`](02-single-node-multi-gpu.json) | Single-node data parallelism |
| 16 | 2 | [`03-multi-node-basic.json`](03-multi-node-basic.json) | Multi-node distributed |
| 32 | 4 | [`04-multi-node-advanced.json`](04-multi-node-advanced.json) | Production multi-node |
| 4 | 1 | [`05-nvidia-gpu-example.json`](05-nvidia-gpu-example.json) | NVIDIA GPUs |

### By Use Case

| Use Case | Recommended File |
|----------|-----------------|
| **Quick testing** | [`00-minimal.json`](00-minimal.json) |
| **Small models (BERT, ResNet)** | [`01-single-node-single-gpu.json`](01-single-node-single-gpu.json) |
| **Large models (GPT-2, SD)** | [`02-single-node-multi-gpu.json`](02-single-node-multi-gpu.json) |
| **Very large models (LLaMA-70B)** | [`03-multi-node-basic.json`](03-multi-node-basic.json) |
| **Production training** | [`04-multi-node-advanced.json`](04-multi-node-advanced.json) |
| **NVIDIA clusters** | [`05-nvidia-gpu-example.json`](05-nvidia-gpu-example.json) |

### By GPU Vendor

| Vendor | Configuration | GPUs |
|--------|---------------|------|
| **AMD MI300X** | [`02-single-node-multi-gpu.json`](02-single-node-multi-gpu.json) | 8 |
| **AMD MI250X** | [`02-single-node-multi-gpu.json`](02-single-node-multi-gpu.json) | 8 |
| **NVIDIA A100** | [`05-nvidia-gpu-example.json`](05-nvidia-gpu-example.json) | 4 |
| **NVIDIA H100** | [`05-nvidia-gpu-example.json`](05-nvidia-gpu-example.json) | 4 |

---

## 🚀 Quick Start (3 Steps)

```bash
# 1. Copy example
cp examples/k8s-configs/01-single-node-single-gpu.json my-config.json

# 2. Edit for your cluster
vim my-config.json  # Update kubeconfig, namespace, node_selector

# 3. Build and run
madengine-cli build --tags model --registry dockerhub --additional-context-file my-config.json
madengine-cli run --manifest-file build_manifest.json
```

---

## 📋 Full Documentation

See [`README.md`](README.md) for complete configuration reference, troubleshooting, and performance tips.

---

## 🔍 Decision Tree

```
Start Here
    │
    ├─ Testing/Debugging?
    │   └─→ Use: 00-minimal.json (fastest)
    │
    ├─ Single GPU sufficient?
    │   └─→ Use: 01-single-node-single-gpu.json
    │
    ├─ Model fits in single node (≤8 GPUs)?
    │   ├─ Yes → Use: 02-single-node-multi-gpu.json
    │   └─ No  → Continue...
    │
    ├─ Need distributed training (>8 GPUs)?
    │   ├─ Basic (2 nodes) → Use: 03-multi-node-basic.json
    │   └─ Advanced (4+ nodes) → Use: 04-multi-node-advanced.json
    │
    └─ Using NVIDIA GPUs?
        └─→ Use: 05-nvidia-gpu-example.json
```

---

## 💾 File Contents at a Glance

### 00-minimal.json
```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "deploy": "k8s",
  "k8s": { "gpu_count": 1 }
}
```

### 01-single-node-single-gpu.json
- 1 GPU, 16Gi RAM, 8 CPUs
- Basic configuration with explicit defaults

### 02-single-node-multi-gpu.json
- 8 GPUs, 256Gi RAM, 64 CPUs
- Includes distributed config (torchrun)
- Node selector for GPU instance type

### 03-multi-node-basic.json
- 2 nodes × 8 GPUs = 16 GPUs total
- NCCL configuration
- Network interface specification

### 04-multi-node-advanced.json
- 4 nodes × 8 GPUs = 32 GPUs total
- PVCs for data and results
- Tolerations and advanced node selection
- Full NCCL tuning

### 05-nvidia-gpu-example.json
- 4 NVIDIA GPUs
- `nvidia.com/gpu` resource name
- CUDA environment variables

---

## 📝 Key Differences

| Feature | Minimal | Single GPU | Multi-GPU | Multi-Node | Advanced |
|---------|---------|------------|-----------|------------|----------|
| **GPU Count** | 1 | 1 | 8 | 16 | 32 |
| **Nodes** | 1 | 1 | 1 | 2 | 4 |
| **Memory** | Default | 16Gi | 256Gi | 256Gi | 512Gi |
| **Distributed** | No | No | Yes | Yes | Yes |
| **Node Selector** | No | No | Yes | Yes | Yes |
| **Tolerations** | No | No | No | No | Yes |
| **PVCs** | No | No | No | No | Yes |
| **NCCL Tuning** | No | No | Basic | Yes | Advanced |

---

## 🎓 Learning Path

1. **Beginner**: Start with `00-minimal.json`
2. **Intermediate**: Try `01-single-node-single-gpu.json` with custom settings
3. **Advanced**: Scale to `02-single-node-multi-gpu.json`
4. **Expert**: Deploy `03-multi-node-basic.json` or `04-multi-node-advanced.json`

---

## 🔗 Related Documentation

- [`README.md`](README.md) - Complete configuration reference
- `../../K8S_DEPLOYMENT_GUIDE.md` - Full deployment guide
- `../../PERF_CSV_UNIFIED_FORMAT.md` - Understanding results

---

**Last Updated**: December 1, 2025


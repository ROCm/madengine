# Kubernetes Configuration Examples

This directory contains example Kubernetes configuration files for `madengine-cli` covering various deployment scenarios.

---

## 📁 Available Examples

| File | GPUs | Nodes | Use Case |
|------|------|-------|----------|
| [`00-minimal.json`](00-minimal.json) | 1 | 1 | Quickstart with defaults |
| [`01-single-node-single-gpu.json`](01-single-node-single-gpu.json) | 1 | 1 | Basic single GPU testing |
| [`02-single-node-multi-gpu.json`](02-single-node-multi-gpu.json) | 8 | 1 | Data parallelism, high performance |
| [`03-multi-node-basic.json`](03-multi-node-basic.json) | 16 | 2 | Distributed training basics |
| [`04-multi-node-advanced.json`](04-multi-node-advanced.json) | 32 | 4 | Production multi-node with all features |
| [`05-nvidia-gpu-example.json`](05-nvidia-gpu-example.json) | 4 | 1 | NVIDIA GPU configuration |

---

## 🚀 Quick Start

### 1. Choose a Configuration

```bash
# For single GPU testing
cp examples/k8s-configs/01-single-node-single-gpu.json my-k8s-config.json

# For multi-GPU on single node
cp examples/k8s-configs/02-single-node-multi-gpu.json my-k8s-config.json

# For multi-node distributed training
cp examples/k8s-configs/03-multi-node-basic.json my-k8s-config.json
```

### 2. Edit Configuration

Update these fields for your environment:

```json
{
  "k8s": {
    "kubeconfig": "/path/to/your/.kube/config",  // Your kubeconfig path
    "namespace": "your-namespace",                 // Your K8s namespace
    "node_selector": {                            // Your node labels
      "node.kubernetes.io/instance-type": "your-instance-type"
    }
  }
}
```

### 3. Build and Run

```bash
# Build with K8s config
madengine-cli build --tags model_name --registry dockerhub \
  --additional-context-file my-k8s-config.json

# Run on Kubernetes
madengine-cli run --manifest-file build_manifest.json
```

---

## 📖 Configuration Reference

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `gpu_vendor` | string | **Yes** | GPU vendor: `"AMD"` or `"NVIDIA"` |
| `guest_os` | string | **Yes** | Operating system: `"UBUNTU"`, `"RHEL"`, etc. |
| `deploy` | string | **Yes** | Deployment target: `"k8s"` for Kubernetes |
| `k8s` | object | **Yes** | Kubernetes-specific configuration |
| `distributed` | object | No | Distributed training configuration |
| `env_vars` | object | No | Environment variables for containers |

### `k8s` Object Fields

#### Required

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_count` | integer | - | **Number of GPUs per pod** |

#### Optional - Basic

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kubeconfig` | string | `~/.kube/config` | Path to kubeconfig file |
| `namespace` | string | `"default"` | Kubernetes namespace |
| `gpu_resource_name` | string | `"amd.com/gpu"` | GPU resource name (`"nvidia.com/gpu"` for NVIDIA) |

#### Optional - Resources

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory` | string | `"128Gi"` | Memory request (e.g., `"16Gi"`, `"256Gi"`) |
| `memory_limit` | string | `"256Gi"` | Memory limit |
| `cpu` | string | `"32"` | CPU cores request |
| `cpu_limit` | string | `"64"` | CPU cores limit |

#### Optional - Job Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_pull_policy` | string | `"Always"` | Image pull policy: `"Always"`, `"IfNotPresent"`, `"Never"` |
| `backoff_limit` | integer | `3` | Number of retries before marking job as failed |

#### Optional - Node Selection

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_selector` | object | `{}` | Node selector labels for pod placement |
| `tolerations` | array | `[]` | Tolerations for pod scheduling |

#### Optional - Storage

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `results_pvc` | string | `null` | PersistentVolumeClaim name for results storage |
| `data_pvc` | string | `null` | PersistentVolumeClaim name for dataset storage |

#### Optional - Debugging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | string | `"./k8s_manifests"` | Directory to save rendered K8s manifests |

### `distributed` Object Fields

For multi-GPU and multi-node training:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable distributed training |
| `backend` | string | `"nccl"` | Communication backend: `"nccl"`, `"gloo"`, `"mpi"` |
| `launcher` | string | `"torchrun"` | Launcher: `"torchrun"`, `"deepspeed"`, `"accelerate"` |
| `nnodes` | integer | `1` | Number of nodes |
| `nproc_per_node` | integer | GPU count | Number of processes per node (usually = GPU count) |
| `master_addr` | string | `"$(hostname)"` | Master node address |
| `master_port` | integer | `29500` | Master node port |
| `rdzv_backend` | string | `"c10d"` | Rendezvous backend for elastic training |
| `rdzv_endpoint` | string | - | Rendezvous endpoint |

### `env_vars` Object

Custom environment variables passed to containers:

```json
{
  "env_vars": {
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_DISABLE": "0",
    "OMP_NUM_THREADS": "8"
  }
}
```

---

## 🎯 Use Case Guide

### Single GPU (Testing, Small Models)

**Configuration**: [`01-single-node-single-gpu.json`](01-single-node-single-gpu.json)

```json
{
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 1,
    "memory": "16Gi",
    "cpu": "8"
  }
}
```

**Best for**:
- Quick testing and validation
- Small models (BERT-base, ResNet-50)
- Debugging model scripts
- Cost-effective experimentation

---

### Single Node, Multiple GPUs (Data Parallelism)

**Configuration**: [`02-single-node-multi-gpu.json`](02-single-node-multi-gpu.json)

```json
{
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 8,
    "memory": "256Gi",
    "cpu": "64"
  },
  "distributed": {
    "enabled": true,
    "launcher": "torchrun",
    "nnodes": 1,
    "nproc_per_node": 8
  }
}
```

**Best for**:
- Large models that fit in single-node memory
- Data parallel training
- Maximum single-node performance
- GPT-2, BERT-large, Stable Diffusion

---

### Multi-Node (Model Parallelism, Very Large Models)

**Configuration**: [`03-multi-node-basic.json`](03-multi-node-basic.json)

```json
{
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 8,
    "memory": "256Gi"
  },
  "distributed": {
    "enabled": true,
    "launcher": "torchrun",
    "nnodes": 2,
    "nproc_per_node": 8
  },
  "env_vars": {
    "NCCL_SOCKET_IFNAME": "eth0",
    "GLOO_SOCKET_IFNAME": "eth0"
  }
}
```

**Best for**:
- Very large models (LLaMA-70B, GPT-3)
- Models requiring pipeline parallelism
- Tensor parallelism across nodes
- Maximum cluster utilization

---

## 📝 Common Configurations

### AMD MI300X (8 GPUs)

```json
{
  "gpu_vendor": "AMD",
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 8,
    "gpu_resource_name": "amd.com/gpu",
    "memory": "512Gi",
    "cpu": "96",
    "node_selector": {
      "node.kubernetes.io/instance-type": "mi300x-8gpu"
    }
  }
}
```

### AMD MI250X (8 GPUs)

```json
{
  "gpu_vendor": "AMD",
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 8,
    "gpu_resource_name": "amd.com/gpu",
    "memory": "256Gi",
    "cpu": "64",
    "node_selector": {
      "accelerator": "mi250x"
    }
  }
}
```

### NVIDIA A100 (8 GPUs)

```json
{
  "gpu_vendor": "NVIDIA",
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 8,
    "gpu_resource_name": "nvidia.com/gpu",
    "memory": "256Gi",
    "cpu": "64",
    "node_selector": {
      "accelerator": "nvidia-tesla-a100"
    }
  }
}
```

### NVIDIA H100 (8 GPUs)

```json
{
  "gpu_vendor": "NVIDIA",
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 8,
    "gpu_resource_name": "nvidia.com/gpu",
    "memory": "640Gi",
    "cpu": "112",
    "node_selector": {
      "accelerator": "nvidia-h100-80gb-hbm3"
    }
  }
}
```

---

## 🔧 Advanced Features

### Node Affinity (Pin to Specific Nodes)

```json
{
  "k8s": {
    "node_selector": {
      "node.kubernetes.io/instance-type": "mi300x-8gpu",
      "topology.kubernetes.io/zone": "us-west-2a",
      "workload-type": "ml-training"
    }
  }
}
```

### Tolerations (Schedule on Tainted Nodes)

```json
{
  "k8s": {
    "tolerations": [
      {
        "key": "gpu",
        "operator": "Equal",
        "value": "amd",
        "effect": "NoSchedule"
      },
      {
        "key": "workload",
        "operator": "Equal",
        "value": "training",
        "effect": "NoSchedule"
      }
    ]
  }
}
```

### Shared Storage (PersistentVolumeClaims)

```json
{
  "k8s": {
    "results_pvc": "ml-results-pvc",
    "data_pvc": "ml-datasets-pvc"
  }
}
```

**Benefits**:
- Share datasets across multiple jobs
- Persist results to shared storage
- Use pre-downloaded datasets

### NCCL Tuning for Multi-Node

```json
{
  "env_vars": {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_HCA": "mlx5_0,mlx5_1,mlx5_2,mlx5_3",
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_NET_GDR_LEVEL": "5",
    "NCCL_P2P_LEVEL": "NVL"
  }
}
```

---

## 🐛 Troubleshooting

### Job Fails to Schedule

**Symptom**: Job stays in `Pending` state

**Check**:
1. GPU availability: `kubectl get nodes -o json | jq '.items[].status.capacity'`
2. Node selector labels: `kubectl get nodes --show-labels`
3. Resource requests vs. node capacity

**Fix**:
- Reduce `gpu_count`, `memory`, or `cpu`
- Update `node_selector` to match your nodes
- Add appropriate `tolerations`

### Out of Memory (OOM)

**Symptom**: Pod crashes with OOM killed

**Check**: `kubectl describe pod <pod-name>`

**Fix**:
```json
{
  "k8s": {
    "memory": "512Gi",      // Increase memory request
    "memory_limit": "768Gi" // Increase memory limit
  }
}
```

### NCCL Timeout (Multi-Node)

**Symptom**: Training hangs or timeout errors

**Check**: Network connectivity between nodes

**Fix**:
```json
{
  "env_vars": {
    "NCCL_DEBUG": "INFO",
    "NCCL_SOCKET_IFNAME": "eth0",  // Specify correct interface
    "NCCL_IB_TIMEOUT": "23",
    "NCCL_BLOCKING_WAIT": "1"
  }
}
```

### Image Pull Failures

**Symptom**: `ImagePullBackOff` or `ErrImagePull`

**Fix**:
1. Check registry credentials: `kubectl get secret`
2. Use `"image_pull_policy": "IfNotPresent"` for local images
3. Verify image exists: `docker pull <image>`

---

## 📊 Performance Tips

### Single Node

1. **Use all available GPUs**: Set `gpu_count` to match node capacity
2. **Optimize CPU allocation**: Typically 8-12 CPUs per GPU
3. **Memory**: 32-64 GiB per GPU for most models

### Multi-Node

1. **Enable NCCL optimizations**: Set appropriate `NCCL_*` env vars
2. **Use InfiniBand**: `"NCCL_IB_DISABLE": "0"`
3. **Pin processes to cores**: Set `OMP_NUM_THREADS`
4. **Use same availability zone**: Reduces network latency

### General

1. **Cache images**: Use `"image_pull_policy": "IfNotPresent"`
2. **Use PVCs**: Avoid re-downloading datasets
3. **Monitor resources**: `kubectl top pods`

---

## 📚 Additional Resources

### Kubernetes Documentation
- [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Node Affinity](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/)
- [Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)
- [PersistentVolumeClaims](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)

### madengine-cli Documentation
- `K8S_DEPLOYMENT_GUIDE.md` - Complete K8s deployment guide
- `K8S_CREDENTIALS_GUIDE.md` - Kubeconfig handling
- `PERF_CSV_UNIFIED_FORMAT.md` - Performance results format

### GPU Device Plugins
- [AMD GPU Device Plugin](https://github.com/ROCm/k8s-device-plugin)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)

---

## 🔍 Validation

Test your configuration before running expensive jobs:

```bash
# 1. Validate K8s connection
kubectl get nodes

# 2. Check GPU availability
kubectl get nodes -o json | jq '.items[].status.capacity."amd.com/gpu"'

# 3. Dry-run build
madengine-cli build --tags dummy --dry-run \
  --additional-context-file my-k8s-config.json

# 4. Check rendered manifests
ls -la k8s_manifests/
cat k8s_manifests/job.yaml
```

---

## 💡 Tips

1. **Start small**: Use `00-minimal.json` or `01-single-node-single-gpu.json` first
2. **Iterate**: Test single GPU → multi-GPU → multi-node progressively
3. **Debug locally**: Run models locally before deploying to K8s
4. **Save manifests**: Set `"output_dir"` to inspect generated YAML files
5. **Use namespaces**: Isolate experiments with different namespaces
6. **Monitor costs**: Track GPU usage with `kubectl top nodes`

---

**Created**: December 1, 2025  
**madengine-cli Version**: Compatible with v2.1+  
**Status**: Production Ready ✅


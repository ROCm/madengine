# Kubernetes Configuration Guide

Complete reference for deploying MADEngine workloads on Kubernetes clusters.

---

## 📋 Table of Contents

- [Minimal Configuration (NEW!)](#-minimal-configuration-new)
- [Quick Start](#-quick-start)
- [Available Configurations](#-available-configurations)
- [Decision Matrix](#-decision-matrix-which-config-to-use)
- [Usage Examples](#-usage-examples)
- [Data Providers](#-data-providers-with-kubernetes)
- [Configuration Reference](#-configuration-reference)
- [Best Practices](#-best-practices)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Minimal Configuration (NEW!)

**MADEngine v2.0+ includes built-in presets!** You only need to specify what's unique:

### Single GPU - Just 1 Field!
```json
{
  "k8s": {
    "gpu_count": 1
  }
}
```
**Note**: No `"deploy": "k8s"` needed - automatically inferred from `k8s` field presence!

### Multi-GPU (2 GPUs)
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

### Multi-Node (2 nodes × 2 GPUs)
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

**Auto-Applied Defaults:**
- ✅ Deployment type (k8s/slurm/local) inferred from config structure
- ✅ Resource limits (memory, CPU) based on GPU count
- ✅ AMD/NVIDIA-specific optimizations
- ✅ ROCm/CUDA environment variables
- ✅ NCCL/RCCL configuration
- ✅ Multi-node settings (host_ipc, etc.)

**See:** [minimal/](minimal/) directory for more examples and documentation.

---

## 🚀 Quick Start

### Option 1: Minimal Configuration (Recommended)

```bash
# Create minimal config
cat > my-config.json << EOF
{
  "k8s": {
    "gpu_count": 1
  }
}
EOF

# Build and run
MODEL_DIR=tests/fixtures/dummy madengine-cli build \
  --tags my_model \
  --additional-context-file my-config.json \
  --registry dockerhub

MODEL_DIR=tests/fixtures/dummy madengine-cli run \
  --manifest-file build_manifest.json \
  --live-output
```

### Option 2: Full Configuration (Advanced)

#### 1. Choose a Configuration

```bash
# For single GPU testing
cp examples/k8s-configs/01-single-node-single-gpu.json my-config.json

# For multi-GPU (2 GPUs)
cp examples/k8s-configs/02-single-node-multi-gpu.json my-config.json

# For multi-node distributed (2 nodes × 2 GPUs)
cp examples/k8s-configs/03-multi-node-basic.json my-config.json

# For data provider with auto-PVC
cp examples/k8s-configs/06-data-provider-with-pvc.json my-config.json
```

#### 2. Customize for Your Cluster (Optional)

With built-in defaults, customization is optional. Override only what you need:

```json
{
  "k8s": {
    "namespace": "my-namespace",         // Override default "default"
    "memory": "32Gi",                    // Override auto-calculated memory
    "node_selector": {                   // Optional: target specific nodes
      "node.kubernetes.io/instance-type": "Standard_ND96isr_H100_v5"
    }
  }
}
```

#### 3. Build and Deploy

```bash
# Build container image
MODEL_DIR=tests/fixtures/dummy madengine-cli build \
  --tags my_model \
  --additional-context-file my-config.json \
  --registry dockerhub

# Deploy and run
MODEL_DIR=tests/fixtures/dummy madengine-cli run \
  --manifest-file build_manifest.json \
  --live-output
```

---

## 📁 Available Configurations

### Minimal Configs (NEW - Recommended for Most Users)

Located in [`minimal/`](minimal/) directory:

| File | Description | GPU Count |
|------|-------------|-----------|
| [`minimal/single-gpu-minimal.json`](minimal/single-gpu-minimal.json) | Single GPU with auto-defaults | 1 |
| [`minimal/multi-gpu-minimal.json`](minimal/multi-gpu-minimal.json) | Multi-GPU with auto-defaults | 2 |
| [`minimal/multi-node-minimal.json`](minimal/multi-node-minimal.json) | Multi-node with auto-defaults | 2×2 |
| [`minimal/nvidia-gpu-minimal.json`](minimal/nvidia-gpu-minimal.json) | NVIDIA GPUs with auto-defaults | 4 |
| [`minimal/custom-namespace-minimal.json`](minimal/custom-namespace-minimal.json) | Shows override examples | 1 |

**See [minimal/README.md](minimal/README.md) for detailed documentation.**

### Full Configs (Reference Examples)

Complete configurations showing all available fields:

| File | GPUs | Nodes | Launcher | Use Case |
|------|------|-------|----------|----------|
| [`01-single-node-single-gpu.json`](01-single-node-single-gpu.json) | 1 | 1 | None | Basic testing, small models |
| [`01-single-node-single-gpu-tools.json`](01-single-node-single-gpu-tools.json) | 1 | 1 | None | Single GPU + monitoring |
| [`02-single-node-multi-gpu.json`](02-single-node-multi-gpu.json) | 2 | 1 | torchrun | Multi-GPU training |
| [`02-single-node-multi-gpu-tools.json`](02-single-node-multi-gpu-tools.json) | 2 | 1 | torchrun | Multi-GPU + monitoring |
| [`03-multi-node-basic.json`](03-multi-node-basic.json) | 2/node | 2 | torchrun | Multi-node basics (4 GPUs total) |
| [`04-multi-node-advanced.json`](04-multi-node-advanced.json) | 2/node | 4 | torchrun | Production multi-node (8 GPUs) |
| [`05-nvidia-gpu-example.json`](05-nvidia-gpu-example.json) | 4 | 1 | torchrun | NVIDIA GPUs (A100, H100) |
| [`06-data-provider-with-pvc.json`](06-data-provider-with-pvc.json) | 2 | 1+ | torchrun | **Data provider with auto-PVC** |

---

## 🎯 Decision Matrix: Which Config to Use?

### By GPU Requirements

| Scenario | Config File | GPUs | Nodes |
|----------|-------------|------|-------|
| **Quick test** | `01-single-node-single-gpu.json` | 1 | 1 |
| **Single GPU benchmark** | `01-single-node-single-gpu-tools.json` | 1 | 1 |
| **Multi-GPU (2 GPUs)** | `02-single-node-multi-gpu.json` | 2 | 1 |
| **Multi-GPU + monitoring** | `02-single-node-multi-gpu-tools.json` | 2 | 1 |
| **Multi-node (4 GPUs)** | `03-multi-node-basic.json` | 2×2 | 2 |
| **Multi-node (8 GPUs)** | `04-multi-node-advanced.json` | 2×4 | 4 |
| **NVIDIA GPUs** | `05-nvidia-gpu-example.json` | 4 | 1 |
| **With data download** | `06-data-provider-with-pvc.json` | 2 | 1+ |

### By Use Case

| Use Case | Recommended Config |
|----------|-------------------|
| **Development/Testing** | `01-single-node-single-gpu.json` |
| **Small models (BERT, ResNet)** | `01-single-node-single-gpu.json` |
| **Medium models (GPT-2, Stable Diffusion)** | `02-single-node-multi-gpu.json` |
| **Large models (LLaMA-13B)** | `03-multi-node-basic.json` |
| **Very large models (LLaMA-70B+)** | `04-multi-node-advanced.json` |
| **Models requiring datasets** | `06-data-provider-with-pvc.json` |
| **Busy/shared clusters** | `02-single-node-multi-gpu.json` (2 GPUs) |

---

## 💻 Usage Examples

### Example 1: Single GPU Test

```bash
MODEL_DIR=tests/fixtures/dummy madengine-cli build \
  --tags dummy \
  --additional-context-file examples/k8s-configs/01-single-node-single-gpu.json \
  --registry dockerhub

MODEL_DIR=tests/fixtures/dummy madengine-cli run \
  --manifest-file build_manifest.json \
  --live-output
```

### Example 2: Multi-GPU Training (2 GPUs)

```bash
MODEL_DIR=tests/fixtures/dummy madengine-cli build \
  --tags dummy_torchrun \
  --additional-context-file examples/k8s-configs/02-single-node-multi-gpu.json \
  --registry dockerhub

MODEL_DIR=tests/fixtures/dummy madengine-cli run \
  --manifest-file build_manifest.json \
  --live-output
```

### Example 3: Multi-Node Training (2 nodes, 4 GPUs)

```bash
MODEL_DIR=tests/fixtures/dummy madengine-cli build \
  --tags dummy_torchrun \
  --additional-context-file examples/k8s-configs/03-multi-node-basic.json \
  --registry dockerhub

MODEL_DIR=tests/fixtures/dummy madengine-cli run \
  --manifest-file build_manifest.json \
  --live-output
```

### Example 4: With Data Provider (Auto-PVC)

```bash
MODEL_DIR=tests/fixtures/dummy madengine-cli build \
  --tags dummy_torchrun_data_minio \
  --additional-context-file examples/k8s-configs/06-data-provider-with-pvc.json \
  --registry dockerhub

MODEL_DIR=tests/fixtures/dummy madengine-cli run \
  --manifest-file build_manifest.json \
  --live-output

# Verify PVC was auto-created
kubectl get pvc madengine-shared-data
```

---

## 📦 Data Providers with Kubernetes

**NEW:** MADEngine automatically handles data provisioning for K8s deployments!

### ✨ Auto-PVC Feature

**No manual PVC creation needed!** MADEngine automatically:
1. Creates `madengine-shared-data` PVC if it doesn't exist
2. Selects appropriate access mode (RWO for single-node, RWX for multi-node)
3. Downloads data on first run
4. Reuses data on subsequent runs

### Quick Setup

**Step 1: Use data provider config**
```bash
madengine-cli build --tags dummy_torchrun_data_minio \
  --additional-context-file examples/k8s-configs/06-data-provider-with-pvc.json \
  --registry dockerhub
```

**Step 2: Run (PVC auto-created)**
```bash
madengine-cli run --manifest-file build_manifest.json --live-output

# Output shows:
# 📦 Data provider detected: Will auto-create shared data PVC
#    PVC name: madengine-shared-data (reusable across runs)
#    Access mode: RWO for single-node, RWX for multi-node (auto-selected)
```

**Step 3: Verify (optional)**
```bash
# Check PVC status
kubectl get pvc madengine-shared-data

# Check PVC contents
kubectl exec -it <pod-name> -- ls -lh /data/
```

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. MADEngine detects data provider in model config          │
├─────────────────────────────────────────────────────────────┤
│  2. Auto-creates madengine-shared-data PVC (if not exists)  │
│     • Single-node: ReadWriteOnce (RWO)                      │
│     • Multi-node: ReadWriteMany (RWX)                       │
├─────────────────────────────────────────────────────────────┤
│  3. Mounts PVC at /data in pod                               │
├─────────────────────────────────────────────────────────────┤
│  4. Downloads data from MinIO/S3/NAS to /data               │
├─────────────────────────────────────────────────────────────┤
│  5. Training starts with data at /data/<filename>           │
├─────────────────────────────────────────────────────────────┤
│  6. PVC persists - subsequent runs skip download! ✅         │
└─────────────────────────────────────────────────────────────┘
```

### Supported Data Providers

| Provider | Protocol | Configuration |
|----------|----------|---------------|
| **MinIO** | S3-compatible | Automatic (credentials from `credential.json`) |
| **AWS S3** | S3 | AWS credentials in environment or `credential.json` |
| **NAS** | SSH/rsync | NAS credentials in `credential.json` |
| **Local** | Filesystem | Pre-mounted PVC |

### Storage Classes

**Single-Node (RWO)**:
- ✅ `local-path` (Rancher)
- ✅ AWS EBS (`gp3`, `io2`)
- ✅ Azure Disk
- ✅ Any RWO storage class

**Multi-Node (RWX)**:
- ✅ NFS (`nfs-client`)
- ✅ CephFS
- ✅ GlusterFS
- ✅ AWS EFS
- ✅ Azure Files
- ❌ `local-path` (RWO only)

### Custom PVC (Optional)

To use an existing PVC instead of auto-creation:

```json
{
  "k8s": {
    "data_pvc": "my-existing-pvc"  // Skip auto-creation
  }
}
```

---

## 📖 Configuration Reference

### Configuration Structure

```json
{
  "_comment": "Description of this configuration",
  "gpu_vendor": "AMD|NVIDIA",
  "guest_os": "UBUNTU",
  "deploy": "k8s",
  
  "k8s": {
    "kubeconfig": "~/.kube/config",
    "namespace": "default",
    "gpu_count": 2,
    
    "memory": "64Gi",
    "memory_limit": "128Gi",
    "cpu": "16",
    "cpu_limit": "32",
    
    "image_pull_policy": "Always",
    "backoff_limit": 3,
    
    "node_selector": {},
    "tolerations": [],
    
    "data_pvc": null,        // Optional: for data providers
    "results_pvc": null      // Optional: custom results storage
  },
  
  "distributed": {
    "enabled": true,
    "backend": "nccl",
    "launcher": "torchrun",
    "nnodes": 1,
    "nproc_per_node": 2,
    "master_port": 29500
  },
  
  "env_vars": {
    "NCCL_DEBUG": "WARN",
    "NCCL_IB_DISABLE": "1",
    "OMP_NUM_THREADS": "8"
  },
  
  "debug": false
}
```

### Field Reference

#### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `gpu_vendor` | string | **Yes** | `"AMD"` or `"NVIDIA"` |
| `guest_os` | string | **Yes** | `"UBUNTU"`, `"RHEL"`, etc. |
| `deploy` | string | **Yes** | Must be `"k8s"` |
| `k8s` | object | **Yes** | Kubernetes configuration |
| `distributed` | object | No | Distributed training (for torchrun) |
| `env_vars` | object | No | Custom environment variables |
| `debug` | boolean | No | Enable debug mode (saves manifests) |

#### K8s Configuration Fields

**Required:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_count` | integer | - | **Number of GPUs per pod** |

**Optional - Basic:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kubeconfig` | string | `~/.kube/config` | Path to kubeconfig |
| `namespace` | string | `"default"` | Kubernetes namespace |
| `gpu_resource_name` | string | `"amd.com/gpu"` | GPU resource (`"nvidia.com/gpu"` for NVIDIA) |

**Optional - Resources:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory` | string | `"128Gi"` | Memory request (e.g., `"16Gi"`, `"64Gi"`) |
| `memory_limit` | string | `"256Gi"` | Memory limit (typically 2× memory) |
| `cpu` | string | `"32"` | CPU cores request |
| `cpu_limit` | string | `"64"` | CPU cores limit (typically 2× cpu) |

**Optional - Job Control:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_pull_policy` | string | `"Always"` | `"Always"`, `"IfNotPresent"`, or `"Never"` |
| `backoff_limit` | integer | `3` | Retry attempts before marking failed |
| `host_ipc` | boolean | `false` | Enable shared memory (required for multi-node) |

**Optional - Node Selection:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_selector` | object | `{}` | Label selectors for pod placement |
| `tolerations` | array | `[]` | Tolerations for tainted nodes |

**Optional - Storage:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_pvc` | string | `null` | Data PVC name (auto-created if using data provider) |
| `results_pvc` | string | `null` | Results PVC name (auto-created by default) |

#### Distributed Execution Fields

Configuration for distributed workloads (training with torchrun/deepspeed or inference with vLLM/SGLang):

For multi-GPU and multi-node (torchrun):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `launcher` | string | - | Launcher type: `torchrun`, `vllm`, `sglang`, `deepspeed` |
| `enabled` | boolean | `false` | Enable distributed execution (legacy, prefer `launcher`) |
| `backend` | string | `"nccl"` | `"nccl"`, `"gloo"`, or `"mpi"` |
| `launcher` | string | `"torchrun"` | `"torchrun"`, `"deepspeed"`, `"accelerate"` |
| `nnodes` | integer | `1` | Number of nodes |
| `nproc_per_node` | integer | gpu_count | Processes per node (= GPUs per node) |
| `master_port` | integer | `29500` | Master communication port |

#### Environment Variables

Custom environment variables for containers:

```json
{
  "env_vars": {
    // NCCL/RCCL (AMD distributed execution)
    "NCCL_DEBUG": "WARN",              // "INFO" for debugging, "WARN" for production
    "NCCL_IB_DISABLE": "1",            // Disable InfiniBand (required for K8s)
    "NCCL_SOCKET_IFNAME": "eth0",      // Network interface
    "TORCH_NCCL_HIGH_PRIORITY": "1",   // RCCL optimization for FSDP
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",  // Multi-node error handling
    
    // AMD ROCm optimizations
    "GPU_MAX_HW_QUEUES": "2",          // MI series optimization
    "HSA_ENABLE_SDMA": "0",            // Disable SDMA for multi-GPU
    "HSA_FORCE_FINE_GRAIN_PCIE": "1",  // Multi-node communication
    "RCCL_ENABLE_HIPGRAPH": "0",       // Disable for compatibility
    
    // MIOpen
    "MIOPEN_FIND_MODE": "1",           // Use compiled kernels
    "MIOPEN_USER_DB_PATH": "/tmp/.miopen",  // Writable cache location
    
    // General
    "OMP_NUM_THREADS": "8"             // OpenMP threads
  }
}
```

---

## 🎓 Best Practices

### Resource Sizing

**Single GPU:**
```
GPUs: 1
Memory: 16Gi (request), 32Gi (limit)
CPU: 8 (request), 16 (limit)
```

**Multi-GPU (2 GPUs):**
```
GPUs: 2
Memory: 64Gi (request), 128Gi (limit)
CPU: 16 (request), 32 (limit)
```

**Multi-Node (2 nodes × 2 GPUs):**
```
GPUs: 2 per node (4 total)
Memory: 64Gi per node
CPU: 16 per node
host_ipc: true (required!)
```

**Multi-Node Advanced (4 nodes × 2 GPUs):**
```
GPUs: 2 per node (8 total)
Memory: 128Gi per node
CPU: 24 per node
host_ipc: true
PVCs: Recommended for data and results
```

### When to Use torchrun

✅ **Use torchrun when:**
- Multi-GPU on single node (2+ GPUs)
- Multi-node distributed workloads
- Testing distributed infrastructure
- Data parallelism or model parallelism

❌ **Don't use torchrun when:**
- Single GPU workloads
- Simple benchmarks without distributed execution
- Minimal testing scenarios

### AMD ROCm Optimizations

**Always set in K8s:**
- `NCCL_IB_DISABLE=1` - InfiniBand not available in K8s
- `NCCL_SOCKET_IFNAME=eth0` - Use Ethernet interface
- `MIOPEN_FIND_MODE=1` - Avoid MIOpen find-db warnings
- `MIOPEN_USER_DB_PATH=/tmp/.miopen` - Writable cache

**For multi-GPU:**
- `TORCH_NCCL_HIGH_PRIORITY=1` - RCCL optimization
- `GPU_MAX_HW_QUEUES=2` - MI series GPUs
- `HSA_ENABLE_SDMA=0` - Disable SDMA for better P2P

**For multi-node:**
- `host_ipc: true` - Required for shared memory
- `HSA_FORCE_FINE_GRAIN_PCIE=1` - Cross-node communication
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` - Better error handling

### For Busy/Shared Clusters

✅ **Recommendations:**
- Use 1-2 GPUs instead of 8 to avoid scheduling conflicts
- Test with single-GPU first, then scale up
- Monitor GPU availability: `kubectl describe nodes | grep amd.com/gpu`
- Use node selectors to target specific node types
- Consider resource quotas and limits

---

## 🐛 Troubleshooting

### Pod Stuck in Pending

**Symptoms:**
```bash
kubectl get pods
# NAME                    READY   STATUS    RESTARTS   AGE
# madengine-job-xxxxx     0/1     Pending   0          5m
```

**Solutions:**

1. **Check GPU availability:**
```bash
kubectl describe nodes | grep -A5 "amd.com/gpu\|nvidia.com/gpu"
# Shows: Allocatable vs Allocated
```

2. **Reduce GPU count:**
```json
{
  "k8s": {
    "gpu_count": 1  // Try 1 instead of 2
  }
}
```

3. **Check node selectors:**
```bash
kubectl get nodes --show-labels | grep instance-type
# Verify your node_selector matches actual node labels
```

### NCCL/RCCL Errors

**Error: "Duplicate GPU detected"**
```
Solution: gpu_count in config must match nproc_per_node in distributed config
```

**Error: "Network connection failed"**
```
Solution: Verify NCCL_SOCKET_IFNAME matches your network interface
Check: kubectl exec <pod> -- ip addr
```

**Error: "NCCL initialization failed"**
```
Solution: Ensure these are set:
  NCCL_IB_DISABLE=1
  NCCL_SOCKET_IFNAME=eth0
Enable debug: NCCL_DEBUG=INFO
```

### Out of Memory (OOM)

**Symptoms:**
```bash
kubectl get pods
# NAME                    READY   STATUS      RESTARTS   AGE
# madengine-job-xxxxx     0/1     OOMKilled   0          2m
```

**Solutions:**

1. **Increase memory limit:**
```json
{
  "k8s": {
    "memory": "128Gi",      // Increase request
    "memory_limit": "256Gi" // Increase limit (2× request)
  }
}
```

2. **Reduce batch size** (in model config)

3. **Enable gradient checkpointing** (model-specific)

### Job Failed

**Check logs:**
```bash
kubectl logs <pod-name>
kubectl describe pod <pod-name>
```

**Common issues:**
- Image pull failed: Check registry credentials
- Permission denied: Check security context and PVC permissions
- Command not found: Verify scripts are in container
- Timeout: Increase `backoff_limit` or job timeout

### Multi-Node Communication Fails

**Symptoms:**
```
NCCL WARN ... Connection refused
NCCL WARN ... Unable to find NCCL communicator
```

**Solutions:**

1. **Enable host_ipc:**
```json
{
  "k8s": {
    "host_ipc": true  // Required for multi-node!
  }
}
```

2. **Verify headless service:**
```bash
kubectl get svc | grep madengine
# Should show ClusterIP: None (headless)
```

3. **Check DNS resolution:**
```bash
kubectl exec <pod> -- nslookup madengine-job-name.default.svc.cluster.local
```

4. **Increase timeout:**
```json
{
  "env_vars": {
    "NCCL_TIMEOUT": "600"  // 10 minutes
  }
}
```

### Data Provider Issues

**Error: "Read-only file system"**
```
Solution: Bug in template - should be fixed in latest version
The data PVC mount must have readOnly: false
```

**Error: "Data file not found"**
```
Check:
1. PVC exists: kubectl get pvc madengine-shared-data
2. PVC is Bound: kubectl describe pvc madengine-shared-data
3. Data downloaded: kubectl exec <pod> -- ls -lh /data/
4. MAD_DATAHOME=/data set correctly
```

**Error: "PVC pending"**
```
Solution: Storage class issue
Check: kubectl describe pvc madengine-shared-data
Fix: Ensure your cluster has NFS storage class for RWX
For single-node: Any storage class works (uses RWO)
```

---

## 🔍 Configuration Comparison

| Feature | Single GPU | Multi-GPU (2) | Multi-Node (2×2) | Advanced (4×2) |
|---------|------------|---------------|------------------|----------------|
| **GPUs** | 1 | 2 | 4 | 8 |
| **Nodes** | 1 | 1 | 2 | 4 |
| **Memory** | 16Gi | 64Gi | 64Gi/node | 128Gi/node |
| **CPU** | 8 | 16 | 16/node | 24/node |
| **torchrun** | ❌ | ✅ | ✅ | ✅ |
| **host_ipc** | ❌ | ❌ | ✅ | ✅ |
| **NCCL Vars** | Basic | Yes | Full | Advanced |
| **PVCs** | No | No | Optional | Recommended |
| **Tolerations** | No | No | No | Yes |
| **Complexity** | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 📚 Advanced Topics

### Node Selectors

Target specific node types:

```json
{
  "k8s": {
    "node_selector": {
      "node.kubernetes.io/instance-type": "Standard_ND96isr_H100_v5",
      "gpu-type": "mi300x",
      "zone": "us-west-2a"
    }
  }
}
```

Check available labels:
```bash
kubectl get nodes --show-labels
```

### Tolerations

Schedule on tainted nodes:

```json
{
  "k8s": {
    "tolerations": [
      {
        "key": "gpu",
        "operator": "Equal",
        "value": "true",
        "effect": "NoSchedule"
      }
    ]
  }
}
```

### Custom Storage Classes

For multi-node with custom NFS:

```json
{
  "k8s": {
    "storage_class": "nfs-client"  // Your NFS storage class
  }
}
```

Check available storage classes:
```bash
kubectl get storageclass
```

### Debug Mode

Save rendered K8s manifests for inspection:

```json
{
  "debug": true,
  "k8s": {
    "output_dir": "./debug_manifests"
  }
}
```

Manifests saved to:
- `./debug_manifests/job.yaml`
- `./debug_manifests/configmap.yaml`
- `./debug_manifests/service.yaml` (multi-node only)

---

## 📊 Resource Scaling Guide

### Single GPU (Development/Testing)
```
GPUs: 1
Memory: 16Gi (request), 32Gi (limit)
CPU: 8 (request), 16 (limit)
Use Case: Small models, debugging, cost-effective testing
```

### 2 GPUs (Recommended for Shared Clusters)
```
GPUs: 2
Memory: 64Gi (request), 128Gi (limit)
CPU: 16 (request), 32 (limit)
Use Case: Multi-GPU training, testing on busy clusters
```

### 4 GPUs (Multi-Node Testing)
```
Configuration: 2 nodes × 2 GPUs per node
Memory: 64Gi per node
CPU: 16 per node
host_ipc: true (required!)
Use Case: Distributed training development
```

### 8 GPUs (Production Multi-Node)
```
Configuration: 4 nodes × 2 GPUs per node
Memory: 128Gi per node
CPU: 24 per node
host_ipc: true
PVCs: Recommended
Use Case: Large-scale production training
```

---

## 🎯 Examples by Scenario

### Scenario 1: Quick Smoke Test

```bash
# Use minimal config (defaults for everything)
madengine-cli build --tags dummy \
  --additional-context-file examples/k8s-configs/01-single-node-single-gpu.json \
  --registry dockerhub

madengine-cli run --manifest-file build_manifest.json
```

### Scenario 2: Benchmark on Busy Cluster

```bash
# Use 2 GPUs to avoid scheduling conflicts
madengine-cli build --tags resnet50 \
  --additional-context-file examples/k8s-configs/02-single-node-multi-gpu.json \
  --registry dockerhub

madengine-cli run --manifest-file build_manifest.json --live-output
```

### Scenario 3: Large Model Training

```bash
# Multi-node for large models
madengine-cli build --tags llama_13b \
  --additional-context-file examples/k8s-configs/03-multi-node-basic.json \
  --registry dockerhub

madengine-cli run --manifest-file build_manifest.json --live-output
```

### Scenario 4: Production with Datasets

```bash
# Data provider with auto-PVC
madengine-cli build --tags bert_large \
  --additional-context-file examples/k8s-configs/06-data-provider-with-pvc.json \
  --registry dockerhub

madengine-cli run --manifest-file build_manifest.json --live-output

# Verify PVC
kubectl get pvc madengine-shared-data
kubectl exec <pod> -- ls -lh /data/
```

### Scenario 5: GPU Profiling

```bash
# Use *-tools.json variant for monitoring
madengine-cli build --tags model \
  --additional-context-file examples/k8s-configs/02-single-node-multi-gpu-tools.json \
  --registry dockerhub

madengine-cli run --manifest-file build_manifest.json --live-output

# Profiling results in PVC
kubectl cp <pod>:/results/gpu_info_*.csv ./
```

---

## 🔧 Customization Guide

### Start from Example

```bash
# Copy closest match
cp examples/k8s-configs/02-single-node-multi-gpu.json my-custom-config.json

# Edit
vim my-custom-config.json
```

### Common Customizations

**Change GPU count:**
```json
{
  "k8s": {
    "gpu_count": 4  // Change from 2 to 4
  },
  "distributed": {
    "nproc_per_node": 4  // Must match gpu_count
  }
}
```

**Target specific node type:**
```json
{
  "k8s": {
    "node_selector": {
      "gpu-type": "mi300x"
    }
  }
}
```

**Increase memory:**
```json
{
  "k8s": {
    "memory": "128Gi",
    "memory_limit": "256Gi"  // 2× memory
  }
}
```

**Add custom environment variables:**
```json
{
  "env_vars": {
    "MY_CUSTOM_VAR": "value",
    "BATCH_SIZE": "256"
  }
}
```

---

## 📈 Performance Tips

### Multi-GPU Scaling

**Expected Scaling Efficiency:**
- 2 GPUs: ~95-100% (ideal: 2× single GPU)
- 4 GPUs: ~85-95% (network overhead)
- 8 GPUs: ~80-90% (more communication)

**Factors affecting scaling:**
- Model size (larger = better scaling)
- Batch size (larger = less communication)
- Network bandwidth (faster = better)
- NCCL configuration (optimized = better)

### NCCL Tuning for AMD

**Basic (included in examples):**
```json
{
  "NCCL_DEBUG": "WARN",
  "NCCL_IB_DISABLE": "1",
  "TORCH_NCCL_HIGH_PRIORITY": "1",
  "GPU_MAX_HW_QUEUES": "2"
}
```

**Advanced (for production):**
```json
{
  "NCCL_DEBUG": "WARN",
  "NCCL_IB_DISABLE": "1",
  "NCCL_SOCKET_IFNAME": "eth0",
  "TORCH_NCCL_HIGH_PRIORITY": "1",
  "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
  "GPU_MAX_HW_QUEUES": "2",
  "HSA_ENABLE_SDMA": "0",
  "HSA_FORCE_FINE_GRAIN_PCIE": "1",
  "RCCL_ENABLE_HIPGRAPH": "0",
  "MIOPEN_FIND_MODE": "1"
}
```

### Monitoring During Training

```bash
# Watch pod status
kubectl get pods -w

# Monitor resource usage
kubectl top pods

# Stream logs
kubectl logs -f <pod-name>

# Check GPU utilization (from pod)
kubectl exec <pod> -- rocm-smi

# Check NCCL communication (multi-node)
kubectl logs <pod> | grep NCCL
```

---

## 🎓 Learning Path

### Level 1: Beginner
1. Start with `01-single-node-single-gpu.json`
2. Test on single GPU
3. Understand basic K8s concepts
4. Monitor logs and results

### Level 2: Intermediate
1. Try `02-single-node-multi-gpu.json`
2. Learn distributed execution with torchrun (training workloads)
3. Understand NCCL configuration
4. Profile GPU utilization

### Level 3: Advanced
1. Deploy `03-multi-node-basic.json`
2. Master multi-node networking
3. Optimize NCCL parameters
4. Use PVCs for data and results

### Level 4: Expert
1. Customize `04-multi-node-advanced.json`
2. Fine-tune for your cluster
3. Implement node affinity and tolerations
4. Scale to 8+ nodes

---

## 📋 Configuration Checklist

Before deploying to production:

- [ ] Tested on single GPU first
- [ ] Verified GPU availability on cluster
- [ ] Set appropriate memory and CPU limits
- [ ] Configured node selectors (if needed)
- [ ] Set NCCL environment variables
- [ ] Enabled `host_ipc` for multi-node
- [ ] Tested with small batch size first
- [ ] Configured PVCs for data (if using data providers)
- [ ] Set up monitoring and logging
- [ ] Tested failure scenarios (backoff_limit)

---

## 🔗 Related Documentation

- **Main Documentation**: `../../README.md`
- **Data Provider Guide**: `../../docs/K8S_DATA_PROVIDER_GUIDE.md` (if exists)
- **Deployment Guide**: `../../K8S_DEPLOYMENT_GUIDE.md` (if exists)
- **Performance CSV Format**: `../../PERF_CSV_UNIFIED_FORMAT.md` (if exists)

---

## 📝 File Structure

```
examples/k8s-configs/
├── README.md                               # This file
├── 01-single-node-single-gpu.json         # 1 GPU, basic
├── 01-single-node-single-gpu-tools.json   # 1 GPU + monitoring
├── 02-single-node-multi-gpu.json          # 2 GPUs, distributed
├── 02-single-node-multi-gpu-tools.json    # 2 GPUs + monitoring
├── 03-multi-node-basic.json               # 2 nodes × 2 GPUs
├── 04-multi-node-advanced.json            # 4 nodes × 2 GPUs
├── 05-nvidia-gpu-example.json             # NVIDIA GPUs
└── 06-data-provider-with-pvc.json         # Data provider + auto-PVC
```

---

## ✅ Summary

- **8 configuration files** covering all common scenarios
- **Auto-PVC creation** for data providers - no manual setup!
- **Production-ready** with best practices
- **Well-documented** with inline comments
- **Tested** on AMD MI300X and NVIDIA clusters
- **Ready to use** - just copy and customize!

---

**Last Updated**: December 6, 2025  
**Status**: Production Ready ✅

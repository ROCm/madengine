# Deployment Guide

Deploy madengine workloads to Kubernetes or SLURM clusters for distributed execution.

## Overview

madengine supports two deployment backends:

- **Kubernetes** - Cloud-native container orchestration
- **SLURM** - HPC cluster job scheduling

Deployment is configured via `--additional-context` and happens automatically during the run phase.

## Deployment Workflow

```
┌─────────────────────────────────────────────┐
│  1. Build Phase (Local or CI/CD)           │
│     madengine-cli build --tags model       │
│     → Creates Docker image                  │
│     → Pushes to registry                    │
│     → Generates build_manifest.json         │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  2. Deploy Phase (Run with Context)         │
│     madengine-cli run                       │
│       --manifest-file build_manifest.json   │
│       --additional-context '{"deploy":...}' │
│     → Detects deployment target             │
│     → Creates K8s Job or SLURM script       │
│     → Submits and monitors execution        │
└─────────────────────────────────────────────┘
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster with GPU support
- GPU device plugin installed ([AMD](https://github.com/ROCm/k8s-device-plugin) or [NVIDIA](https://github.com/NVIDIA/k8s-device-plugin))
- Kubeconfig configured (`~/.kube/config` or in-cluster)
- Docker registry accessible from cluster

### Quick Start

#### Minimal Configuration (Recommended)

```json
{
  "k8s": {
    "gpu_count": 1
  }
}
```

This automatically applies intelligent defaults for namespace, resources, image pull policy, etc.

#### Build and Deploy

```bash
# 1. Build image
madengine-cli build --tags my_model \
  --registry my-registry.io \
  --additional-context-file k8s-config.json

# 2. Deploy to Kubernetes
madengine-cli run \
  --manifest-file build_manifest.json \
  --timeout 3600
```

The deployment target is automatically detected from the `k8s` key in the config.

### Configuration Options

**k8s-config.json:**

```json
{
  "k8s": {
    "gpu_count": 2,
    "namespace": "ml-team",
    "gpu_vendor": "AMD",
    "memory": "32Gi",
    "cpu": "16",
    "service_account": "madengine-sa",
    "image_pull_policy": "Always"
  }
}
```

**Configuration Priority:**
1. User config (`--additional-context-file`)
2. Profile presets (single-gpu/multi-gpu)
3. GPU vendor presets (AMD/NVIDIA)
4. Base defaults

See [examples/k8s-configs/](../examples/k8s-configs/) for complete examples.

### Multi-Node Training

For distributed training across multiple nodes:

```json
{
  "k8s": {
    "gpu_count": 8
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 2,
    "nproc_per_node": 4
  }
}
```

This creates:
- Kubernetes Indexed Job with 2 completions
- Headless service for pod discovery
- Automatic rank assignment via `JOB_COMPLETION_INDEX`
- `MAD_MULTI_NODE_RUNNER` environment variable with torchrun command

**Supported Launchers:**
- `torchrun` - PyTorch DDP/FSDP
- `deepspeed` - ZeRO optimization
- `torchtitan` - LLM pre-training
- `vllm` - LLM inference
- `sglang` - Structured generation

See [Distributed Launchers Guide](distributed-launchers.md) for details.

### Monitoring

```bash
# Check job status
kubectl get jobs -n your-namespace

# View pod logs
kubectl logs -f job/madengine-job-xxx -n your-namespace

# Check pod status
kubectl get pods -n your-namespace
```

### Cleanup

Jobs are automatically cleaned up after completion (configurable via `ttlSecondsAfterFinished`).

Manual cleanup:

```bash
kubectl delete job madengine-job-xxx -n your-namespace
```

## SLURM Deployment

### Prerequisites

- Access to SLURM login node
- SLURM commands available (`sbatch`, `squeue`, `scontrol`)
- Shared filesystem for MAD package and results
- Module system or container runtime (Singularity/Apptainer)

### Quick Start

#### Configuration

**slurm-config.json:**

```json
{
  "slurm": {
    "partition": "gpu",
    "gpus_per_node": 4,
    "time": "02:00:00",
    "account": "my_account"
  }
}
```

#### Build and Deploy

```bash
# 1. Build image (on build node or locally)
madengine-cli build --tags my_model \
  --registry my-registry.io \
  --additional-context-file slurm-config.json

# 2. SSH to SLURM login node
ssh user@hpc-login.example.com

# 3. Deploy to SLURM
cd /shared/workspace
madengine-cli run \
  --manifest-file build_manifest.json \
  --timeout 7200
```

The deployment target is automatically detected from the `slurm` key in the config.

### Configuration Options

**slurm-config.json:**

```json
{
  "slurm": {
    "partition": "gpu",
    "account": "research_group",
    "qos": "normal",
    "gpus_per_node": 8,
    "nodes": 1,
    "time": "24:00:00",
    "mail_user": "user@example.com",
    "mail_type": "ALL"
  }
}
```

**Common SLURM Options:**
- `partition`: SLURM partition name
- `account`: Billing account
- `qos`: Quality of Service
- `gpus_per_node`: Number of GPUs per node
- `nodes`: Number of nodes (for multi-node)
- `time`: Wall time limit (HH:MM:SS)
- `mem`: Memory per node (e.g., "64G")

See [examples/slurm-configs/](../examples/slurm-configs/) for complete examples.

### Multi-Node Training

For distributed training across SLURM nodes:

```json
{
  "slurm": {
    "partition": "gpu",
    "nodes": 4,
    "gpus_per_node": 8,
    "time": "48:00:00"
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 4,
    "nproc_per_node": 8
  }
}
```

SLURM automatically provides:
- Node list via `$SLURM_JOB_NODELIST`
- Master address detection
- Network interface configuration
- Rank assignment via `$SLURM_PROCID`

### Monitoring

```bash
# Check job queue
squeue -u $USER

# Monitor job progress
squeue -j <job_id>

# View job details
scontrol show job <job_id>

# Check output logs
tail -f slurm-<job_id>.out
```

### Cancellation

```bash
# Cancel job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Deployment Comparison

| Feature | Kubernetes | SLURM |
|---------|-----------|-------|
| **Environment** | Cloud, on-premise | HPC clusters |
| **Orchestration** | Automatic | Job scheduler |
| **Dependencies** | Python library (`kubernetes`) | CLI commands only |
| **Multi-node Setup** | Headless service + DNS | SLURM env vars |
| **Resource Management** | Declarative (YAML) | Batch script |
| **Best For** | Cloud deployments, microservices | Academic HPC, supercomputers |

## Configuration Examples

### Single-GPU Development (K8s)

```json
{
  "k8s": {
    "gpu_count": 1,
    "namespace": "dev"
  }
}
```

### Multi-GPU Training (K8s)

```json
{
  "k8s": {
    "gpu_count": 4,
    "memory": "64Gi",
    "cpu": "32"
  },
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 1,
    "nproc_per_node": 4
  }
}
```

### Multi-Node Training (K8s)

```json
{
  "k8s": {
    "gpu_count": 8,
    "namespace": "ml-training"
  },
  "distributed": {
    "launcher": "torchtitan",
    "nnodes": 4,
    "nproc_per_node": 8
  }
}
```

### Single-Node SLURM

```json
{
  "slurm": {
    "partition": "gpu",
    "gpus_per_node": 8,
    "time": "12:00:00"
  }
}
```

### Multi-Node SLURM

```json
{
  "slurm": {
    "partition": "gpu",
    "nodes": 8,
    "gpus_per_node": 8,
    "time": "72:00:00",
    "account": "research_proj"
  },
  "distributed": {
    "launcher": "deepspeed",
    "nnodes": 8,
    "nproc_per_node": 8
  }
}
```

## Troubleshooting

### Kubernetes Issues

**Image Pull Failures:**
```bash
# Check image exists
docker pull <registry>/<image>:<tag>

# Verify image pull secrets
kubectl get secrets -n your-namespace

# Check pod events
kubectl describe pod <pod-name> -n your-namespace
```

**Resource Issues:**
```bash
# Check node resources
kubectl describe nodes | grep -A5 "Allocated resources"

# Check GPU availability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.'amd\.com/gpu'
```

### SLURM Issues

**Job Pending:**
```bash
# Check reason
squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Check partition status
sinfo -p gpu
```

**Out of Resources:**
```bash
# Check available resources
sinfo -o "%P %.5a %.10l %.6D %.6t %N"

# Adjust resource requests in config
```

## Best Practices

### For Kubernetes

1. Use minimal configs with intelligent defaults
2. Specify resource limits to prevent over-allocation
3. Use appropriate namespaces for isolation
4. Configure image pull policies based on registry location
5. Monitor pod resource usage with `kubectl top`

### For SLURM

1. Start with conservative time limits
2. Use appropriate QoS for priority
3. Monitor job efficiency with `seff <job_id>`
4. Use shared filesystem for input/output
5. Test with single node before scaling

## Next Steps

- [Distributed Launchers Guide](distributed-launchers.md) - Multi-node training frameworks
- [K8s Examples](../examples/k8s-configs/) - Complete Kubernetes configurations
- [SLURM Examples](../examples/slurm-configs/) - Complete SLURM configurations
- [User Guide](user-guide.md) - General usage instructions


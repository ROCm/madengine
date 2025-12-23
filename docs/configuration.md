# Configuration Guide

Complete guide to configuring madengine for various use cases and environments.

## Configuration Methods

### 1. Inline JSON String

```bash
madengine run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

### 2. Configuration File

```bash
madengine run --tags model --additional-context-file config.json
```

**config.json:**
```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "timeout_multiplier": 2.0
}
```

## Basic Configuration

### Required for Local Execution

```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU"
}
```

**gpu_vendor** (case-insensitive):
- `"AMD"` - AMD ROCm GPUs
- `"NVIDIA"` - NVIDIA CUDA GPUs

**guest_os** (case-insensitive):
- `"UBUNTU"` - Ubuntu Linux
- `"CENTOS"` - CentOS Linux

## Build Configuration

### Batch Manifest

Use batch manifest files for selective builds with per-model configuration:

```bash
madengine build --batch-manifest batch.json \
  --registry my-registry.com \
  --additional-context-file config.json
```

**Batch manifest structure** (`batch.json`):

```json
[
  {
    "model_name": "model1",
    "build_new": true,
    "registry": "registry1.io",
    "registry_image": "namespace/model1"
  },
  {
    "model_name": "model2",
    "build_new": false,
    "registry": "registry2.io",
    "registry_image": "namespace/model2"
  }
]
```

**Fields:**
- `model_name` (string, required): Model tag to include
- `build_new` (boolean, optional, default: `false`): Whether to build this model
  - `true`: Build the model from source
  - `false`: Reference existing image without rebuilding
- `registry` (string, optional): Per-model registry override
- `registry_image` (string, optional): Custom registry image name/namespace

**Key Behaviors:**
- Only models with `"build_new": true` are built
- Models with `"build_new": false` are included in output manifest without building
- Per-model `registry` overrides the global `--registry` flag
- Cannot use `--batch-manifest` and `--tags` together (mutually exclusive)

**Use Case - CI/CD Incremental Builds:**

```json
[
  {"model_name": "changed_model", "build_new": true},
  {"model_name": "stable_model1", "build_new": false},
  {"model_name": "stable_model2", "build_new": false}
]
```

This allows you to rebuild only changed models while maintaining references to existing stable images in a single manifest.

## Docker Configuration

### Environment Variables

Pass environment variables to containers:

```json
{
  "docker_env_vars": {
    "HSA_ENABLE_SDMA": "0",
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "NCCL_DEBUG": "INFO"
  }
}
```

### Custom Base Image

Override Docker base image:

```json
{
  "MAD_CONTAINER_IMAGE": "rocm/pytorch:custom-tag"
}
```

Or override BASE_DOCKER in FROM line:

```json
{
  "docker_build_arg": {
    "BASE_DOCKER": "rocm/pytorch:rocm6.1_ubuntu22.04_py3.10"
  }
}
```

### Build Arguments

Pass build-time variables:

```json
{
  "docker_build_arg": {
    "ROCM_VERSION": "6.1",
    "PYTHON_VERSION": "3.10",
    "CUSTOM_ARG": "value"
  }
}
```

### Mount Host Directories

Mount host directories inside containers:

```json
{
  "docker_mounts": {
    "/data-inside-container": "/data-on-host",
    "/models": "/home/user/models"
  }
}
```

### Select GPUs and CPUs

Specify GPU and CPU subsets:

```json
{
  "docker_gpus": "0,2-4,7",
  "docker_cpus": "0-15,32-47"
}
```

Format: Comma-separated list with hyphen ranges.

## Performance Configuration

### Timeout Settings

```json
{
  "timeout_multiplier": 2.0
}
```

Or use command-line option:

```bash
madengine run --tags model --timeout 7200
```

### Local Data Mirroring

Force local data caching:

```json
{
  "mirrorlocal": "/tmp/local_mirror"
}
```

Or use command-line option:

```bash
madengine run --tags model --force-mirror-local /tmp/mirror
```

## Kubernetes Deployment

### Minimal Configuration

```json
{
  "k8s": {
    "gpu_count": 1
  }
}
```

Automatically applies:
- Namespace: `default`
- Resource limits based on GPU count
- Image pull policy: `IfNotPresent`
- Service account: `default`
- GPU vendor detection from context

### Full Configuration

```json
{
  "k8s": {
    "gpu_count": 2,
    "namespace": "ml-team",
    "gpu_vendor": "AMD",
    "memory": "32Gi",
    "memory_limit": "64Gi",
    "cpu": "16",
    "cpu_limit": "32",
    "service_account": "madengine-sa",
    "image_pull_policy": "Always",
    "image_pull_secrets": ["my-registry-secret"]
  }
}
```

**K8s Options:**
- `gpu_count` - Number of GPUs (required)
- `namespace` - Kubernetes namespace (default: `default`)
- `gpu_vendor` - GPU vendor override (auto-detected from context)
- `memory` - Memory request (default: auto-scaled by GPU count)
- `memory_limit` - Memory limit (default: 2× memory request)
- `cpu` - CPU cores request (default: auto-scaled by GPU count)
- `cpu_limit` - CPU cores limit (default: 2× CPU request)
- `service_account` - Service account name
- `image_pull_policy` - `Always`, `IfNotPresent`, or `Never`
- `image_pull_secrets` - List of image pull secrets

### Multi-Node Kubernetes

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

## SLURM Deployment

### Basic Configuration

```json
{
  "slurm": {
    "partition": "gpu",
    "gpus_per_node": 4,
    "time": "02:00:00"
  }
}
```

### Full Configuration

```json
{
  "slurm": {
    "partition": "gpu",
    "account": "research_group",
    "qos": "normal",
    "gpus_per_node": 8,
    "nodes": 1,
    "time": "24:00:00",
    "mem": "64G",
    "mail_user": "user@example.com",
    "mail_type": "ALL"
  }
}
```

**SLURM Options:**
- `partition` - SLURM partition name (required)
- `account` - Billing account
- `qos` - Quality of Service
- `gpus_per_node` - GPUs per node (default: 1)
- `nodes` - Number of nodes (default: 1)
- `time` - Wall time limit HH:MM:SS (required)
- `mem` - Memory per node (e.g., "64G")
- `mail_user` - Email for notifications
- `mail_type` - Notification types (BEGIN, END, FAIL, ALL)

### Multi-Node SLURM

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

## Distributed Training

### Launcher Configuration

```json
{
  "distributed": {
    "launcher": "torchrun",
    "nnodes": 2,
    "nproc_per_node": 4,
    "master_port": 29500
  }
}
```

**Launcher Options:**
- `launcher` - Framework name (required)
- `nnodes` - Number of nodes
- `nproc_per_node` - Processes/GPUs per node
- `master_port` - Master communication port (default: 29500)

**Supported Launchers:**
- `torchrun` - PyTorch DDP/FSDP
- `deepspeed` - ZeRO optimization
- `megatron` - Large transformers (K8s + SLURM)
- `torchtitan` - LLM pre-training
- `vllm` - LLM inference
- `sglang` - Structured generation

See [Launchers Guide](launchers.md) for details.

### TorchTitan Configuration

```json
{
  "distributed": {
    "launcher": "torchtitan",
    "nnodes": 4,
    "nproc_per_node": 8
  },
  "env_vars": {
    "TORCHTITAN_TENSOR_PARALLEL_SIZE": "8",
    "TORCHTITAN_PIPELINE_PARALLEL_SIZE": "4",
    "TORCHTITAN_FSDP_ENABLED": "1"
  }
}
```

### vLLM Configuration

```json
{
  "distributed": {
    "launcher": "vllm",
    "nnodes": 2,
    "nproc_per_node": 4
  },
  "vllm": {
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 1
  }
}
```

## Profiling Configuration

### Basic Profiling

```json
{
  "tools": [
    {"name": "rocprof"}
  ]
}
```

### Custom Tool Configuration

```json
{
  "tools": [
    {
      "name": "rocprof",
      "cmd": "rocprof --timestamp on",
      "env_vars": {
        "NCCL_DEBUG": "INFO"
      }
    }
  ]
}
```

### Multiple Tools (Stackable)

```json
{
  "tools": [
    {"name": "rocprof"},
    {"name": "miopen_trace"},
    {"name": "rocblas_trace"}
  ]
}
```

**Available Tools:**
- `rocprof` - GPU profiling
- `rpd` - ROCm Profiler Data
- `rocblas_trace` - rocBLAS library tracing
- `miopen_trace` - MIOpen library tracing
- `tensile_trace` - Tensile library tracing
- `rccl_trace` - RCCL communication tracing
- `gpu_info_power_profiler` - Power consumption profiling
- `gpu_info_vram_profiler` - VRAM usage profiling

See [Profiling Guide](profiling.md) for details.

## Pre/Post Execution Scripts

Run scripts before and after model execution:

```json
{
  "pre_scripts": [
    {
      "path": "scripts/common/pre_scripts/setup.sh",
      "args": "-v"
    }
  ],
  "encapsulate_script": "scripts/common/wrapper.sh",
  "post_scripts": [
    {
      "path": "scripts/common/post_scripts/cleanup.sh",
      "args": "-r"
    }
  ]
}
```

## Model Arguments

Pass arguments to model execution script:

```json
{
  "model_args": "--model_name_or_path bigscience/bloom --batch_size 32"
}
```

## Data Provider Configuration

Configure in `data.json` (MAD package root):

```json
{
  "data_sources": {
    "model_data": {
      "nas": {"path": "/home/datum"},
      "minio": {"path": "s3://datasets/datum"},
      "aws": {"path": "s3://datasets/datum"}
    }
  },
  "mirrorlocal": "/tmp/local_mirror"
}
```

## Credential Configuration

Configure in `credential.json` (MAD package root):

```json
{
  "dockerhub": {
    "username": "your_username",
    "password": "your_token",
    "repository": "myorg"
  },
  "AMD_GITHUB": {
    "username": "github_username",
    "password": "github_token"
  },
  "MAD_AWS_S3": {
    "username": "aws_access_key",
    "password": "aws_secret_key"
  }
}
```

### Environment Variable Override

```bash
export MAD_DOCKERHUB_USER=myusername
export MAD_DOCKERHUB_PASSWORD=mytoken
export MAD_DOCKERHUB_REPO=myorg
```

## Configuration Priority

For Kubernetes/SLURM deployments:
1. CLI overrides (`--additional-context`) - Highest
2. User config file (`--additional-context-file`)
3. Profile presets (single-gpu/multi-gpu/multi-node)
4. GPU vendor presets (AMD/NVIDIA optimizations)
5. Base defaults (k8s/defaults.json)
6. Environment variables
7. Built-in fallbacks - Lowest

## Complete Examples

### Local GPU Development

```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "docker_gpus": "0",
  "docker_env_vars": {
    "PYTORCH_TUNABLEOP_ENABLED": "1"
  }
}
```

### Kubernetes Single-GPU

```json
{
  "k8s": {
    "gpu_count": 1,
    "namespace": "dev"
  }
}
```

### Kubernetes Multi-GPU Training

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

### SLURM Multi-Node

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

### Production with Profiling

```json
{
  "k8s": {
    "gpu_count": 2,
    "namespace": "production",
    "memory": "32Gi"
  },
  "tools": [
    {"name": "rocprof"},
    {"name": "gpu_info_power_profiler"}
  ],
  "docker_env_vars": {
    "NCCL_DEBUG": "INFO",
    "PYTORCH_TUNABLEOP_ENABLED": "1"
  }
}
```

## Troubleshooting

### Configuration Not Applied

```bash
# Verify configuration is valid JSON
python -m json.tool config.json

# Use verbose logging
madengine run --tags model \
  --additional-context-file config.json \
  --verbose
```

### Environment Variables Not Set

```bash
# Check environment variables
env | grep MAD

# Verify Docker receives env vars
docker inspect container_name | grep -A 10 Env
```

### GPU Vendor Auto-Detection

madengine auto-detects GPU vendor if not specified:
- Looks for ROCm drivers → AMD
- Looks for CUDA drivers → NVIDIA
- Falls back to configuration or fails

Override with explicit configuration:

```json
{
  "gpu_vendor": "AMD"
}
```

## Best Practices

1. **Use configuration files** for complex settings
2. **Start with minimal configs** and add as needed
3. **Validate JSON syntax** before running
4. **Use environment variables** for sensitive data
5. **Test locally first** before deploying
6. **Enable verbose logging** when debugging
7. **Document custom configurations** for team use

## Next Steps

- [Usage Guide](usage.md) - Using madengine commands
- [Deployment Guide](deployment.md) - Deploy to clusters
- [Profiling Guide](profiling.md) - Performance analysis
- [Launchers Guide](launchers.md) - Distributed training frameworks


# Usage Guide

Complete guide to using madengine-cli for running AI models locally and in distributed environments.

## Quick Start

### Prerequisites

- Python 3.8+ with madengine installed
- Docker with GPU support  
- MAD package cloned locally

```bash
git clone https://github.com/ROCm/MAD.git
cd MAD
pip install git+https://github.com/ROCm/madengine.git
```

### Your First Model

```bash
# Discover models
madengine-cli discover --tags dummy

# Run locally
madengine-cli run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

Results are saved to `perf_entry.csv`.

## Commands

### discover - Find Available Models

List models in the MAD package:

```bash
# All models
madengine-cli discover

# Specific models
madengine-cli discover --tags dummy pyt_huggingface_bert

# With verbose output
madengine-cli discover --tags model --verbose
```

### build - Create Docker Images

Build Docker images for models:

```bash
# Basic build
madengine-cli build --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Build with registry
madengine-cli build --tags model \
  --registry docker.io/myorg \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Multiple models
madengine-cli build --tags model1 model2 model3 \
  --registry localhost:5000

# Clean rebuild (no cache)
madengine-cli build --tags model --clean-docker-cache

# Custom manifest output
madengine-cli build --tags model --manifest-output my_manifest.json
```

**Options:**
- `--tags, -t` - Model tags to build
- `--registry, -r` - Docker registry URL
- `--additional-context, -c` - Configuration JSON string
- `--additional-context-file, -f` - Configuration file path
- `--clean-docker-cache` - Rebuild without Docker cache
- `--manifest-output, -m` - Output manifest file (default: build_manifest.json)
- `--verbose, -v` - Verbose logging

### run - Execute Models

Run models locally or deploy to clusters:

```bash
# Run locally
madengine-cli run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Run with manifest (pre-built images)
madengine-cli run --manifest-file build_manifest.json

# Real-time output
madengine-cli run --tags model --live-output --verbose

# Custom timeout (seconds)
madengine-cli run --tags model --timeout 7200

# Keep container alive for debugging
madengine-cli run --tags model --keep-alive
```

**Options:**
- `--tags, -t` - Model tags to run
- `--manifest-file, -m` - Build manifest (for pre-built images)
- `--registry, -r` - Docker registry URL
- `--timeout` - Execution timeout in seconds
- `--additional-context, -c` - Configuration JSON string
- `--additional-context-file, -f` - Configuration file path
- `--keep-alive` - Keep containers alive after run
- `--live-output, -l` - Real-time output streaming
- `--verbose, -v` - Verbose logging

## Model Discovery

madengine supports three discovery methods:

### 1. Root Models (models.json)

Central model definitions in MAD package root:

```bash
madengine-cli discover --tags dummy pyt_huggingface_bert
```

### 2. Directory-Specific Models

Models organized in subdirectories (`scripts/{dir}/models.json`):

```bash
madengine-cli discover --tags dummy2:dummy_2
```

### 3. Dynamic Models with Parameters

Python-generated models (`scripts/{dir}/get_models_json.py`):

```bash
madengine-cli discover --tags dummy3:dummy_3:batch_size=512:in=32
```

## Build Workflow

### Basic Build

Create Docker images and manifest:

```bash
madengine-cli build --tags model \
  --registry localhost:5000 \
  --additional-context-file config.json
```

Creates `build_manifest.json`:

```json
{
  "models": [
    {
      "model_name": "my_model",
      "image": "localhost:5000/my_model:20240115_123456",
      "tag": "my_model"
    }
  ],
  "registry": "localhost:5000",
  "build_timestamp": "2024-01-15T12:34:56Z"
}
```

### Build with Deployment Config

Include deployment configuration:

```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "k8s": {
    "gpu_count": 2,
    "namespace": "ml-team"
  }
}
```

```bash
madengine-cli build --tags model \
  --registry docker.io/myorg \
  --additional-context-file k8s-config.json
```

The deployment config is saved in `build_manifest.json` and used during run phase.

### Registry Authentication

Configure in `credential.json` (MAD package root):

```json
{
  "dockerhub": {
    "username": "your_username",
    "password": "your_token",
    "repository": "myorg"
  }
}
```

Or use environment variables:

```bash
export MAD_DOCKERHUB_USER=your_username
export MAD_DOCKERHUB_PASSWORD=your_token
export MAD_DOCKERHUB_REPO=myorg
```

## Run Workflow

### Local Execution

Run on local machine:

```bash
madengine-cli run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

**Required for Local:**
- `gpu_vendor`: "AMD", "NVIDIA"
- `guest_os`: "UBUNTU", "CENTOS"

### Deploy to Kubernetes

```bash
# Build phase
madengine-cli build --tags model \
  --registry gcr.io/myproject \
  --additional-context '{"k8s": {"gpu_count": 2}}'

# Deploy phase
madengine-cli run --manifest-file build_manifest.json
```

Deployment target is automatically detected from `k8s` key in configuration.

### Deploy to SLURM

```bash
# Build phase (local or CI)
madengine-cli build --tags model \
  --registry my-registry.io \
  --additional-context '{"slurm": {"partition": "gpu", "gpus_per_node": 4}}'

# Deploy phase (on SLURM login node)
ssh user@hpc-login.example.com
madengine-cli run --manifest-file build_manifest.json
```

Deployment target is automatically detected from `slurm` key in configuration.

## Common Usage Patterns

### Configuration Files

Use configuration files for complex settings:

**config.json:**
```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "timeout_multiplier": 2.0,
  "docker_env_vars": {
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "HSA_ENABLE_SDMA": "0"
  }
}
```

```bash
madengine-cli run --tags model --additional-context-file config.json
```

### Custom Timeouts

```bash
# Override default timeout
madengine-cli run --tags model --timeout 7200

# No timeout (run indefinitely)
madengine-cli run --tags model --timeout 0
```

### Debugging

```bash
# Keep containers alive
madengine-cli run --tags model --keep-alive

# Verbose output
madengine-cli run --tags model --verbose --live-output

# Both
madengine-cli run --tags model --keep-alive --verbose --live-output
```

### Clean Rebuild

```bash
# Rebuild without Docker cache
madengine-cli build --tags model --clean-docker-cache
```

## Performance Profiling

Profile GPU usage and library calls:

```bash
# GPU profiling
madengine-cli run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{"name": "rocprof"}]
  }'

# Library tracing
madengine-cli run --tags model \
  --additional-context '{"tools": [{"name": "rocblas_trace"}]}'

# Multiple tools (stackable)
madengine-cli run --tags model \
  --additional-context '{"tools": [
    {"name": "rocprof"},
    {"name": "miopen_trace"}
  ]}'
```

See [Profiling Guide](profiling.md) for details.

## Multi-Node Training

Configure distributed training:

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

**Supported Launchers:**
- `torchrun` - PyTorch DDP/FSDP
- `deepspeed` - ZeRO optimization
- `megatron` - Large transformers (SLURM only)
- `torchtitan` - LLM pre-training
- `vllm` - LLM inference
- `sglang` - Structured generation

See [Launchers Guide](launchers.md) for details.

## Output and Results

### Performance CSV

Results are saved to `perf_entry.csv`:

```csv
model_name,execution_time,gpu_utilization,memory_used,...
my_model,125.3,98.5,15.2,...
```

### Build Manifest

`build_manifest.json` contains:
- Built image names and tags
- Model configurations
- Deployment configuration
- Build timestamp

Use this manifest to run pre-built images:

```bash
madengine-cli run --manifest-file build_manifest.json
```

## Troubleshooting

### Model Not Found

```bash
# Ensure you're in MAD directory
cd /path/to/MAD
madengine-cli discover --tags your_model
```

### Docker Permission Denied

```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### GPU Not Detected

```bash
# AMD GPUs
rocm-smi

# NVIDIA GPUs
nvidia-smi

# Test with Docker
docker run --rm --device=/dev/kfd --device=/dev/dri \
  rocm/pytorch:latest rocm-smi
```

### Build Failures

```bash
# Check Docker daemon
docker ps

# Rebuild without cache
madengine-cli build --tags model --clean-docker-cache --verbose
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_DIR` | MAD package directory | `/path/to/MAD` |
| `MAD_VERBOSE_CONFIG` | Verbose config logging | `"true"` |
| `MAD_DOCKERHUB_USER` | Docker Hub username | `"myusername"` |
| `MAD_DOCKERHUB_PASSWORD` | Docker Hub password | `"mytoken"` |
| `MAD_DOCKERHUB_REPO` | Docker Hub repository | `"myorg"` |

## Best Practices

1. **Use configuration files** for complex settings
2. **Separate build and run** for distributed deployments
3. **Test locally first** before deploying to clusters
4. **Use registries** for distributed execution
5. **Enable verbose logging** when debugging
6. **Start with small timeouts** and increase as needed

## Next Steps

- [Configuration Guide](configuration.md) - Advanced configuration options
- [Deployment Guide](deployment.md) - Kubernetes and SLURM deployment
- [Profiling Guide](profiling.md) - Performance analysis
- [Launchers Guide](launchers.md) - Multi-node training frameworks


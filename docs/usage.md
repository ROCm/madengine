# Usage Guide

Complete guide to using madengine for running AI models locally and in distributed environments.

> **📖 Quick Reference:** For detailed command options and flags, see the **[CLI Command Reference](cli-reference.md)**.

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
madengine discover --tags dummy

# Run locally (full workflow: discover/build/run as configured by the model)
madengine run --tags dummy

# Or with explicit configuration
madengine run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

> **Note**: `gpu_vendor` defaults to `AMD` and `guest_os` defaults to `UBUNTU` for build operations. For production or non-AMD/Ubuntu environments, specify these values explicitly.

Results are saved to `perf_entry.csv`.

## Commands Overview

madengine provides five main commands:

| Command | Purpose | Common Options |
|---------|---------|----------------|
| `discover` | Find available models | `--tags`, `--verbose` |
| `build` | Build Docker images | `--tags`, `--registry`, `--batch-manifest` |
| `run` | Execute models | `--tags`, `--manifest-file`, `--timeout` |
| `report` | Generate HTML reports | `to-html`, `to-email` |
| `database` | Upload to MongoDB | `--csv-file`, `--database-name` |

For complete command options and detailed examples, see **[CLI Command Reference](cli-reference.md)**.

### Quick Command Examples

```bash
# Discover models
madengine discover --tags dummy

# Build image (uses AMD/UBUNTU defaults)
madengine build --tags model

# Run model
madengine run --tags model

# For NVIDIA or other configurations, specify explicitly:
# madengine build --tags model --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'

# Generate HTML report
madengine report to-html --csv-file perf_entry.csv

# Upload to MongoDB
madengine database --csv-file perf_entry.csv \
  --database-name mydb --collection-name results
```

## Model Discovery

madengine supports three discovery methods:

### 1. Root Models (models.json)

Central model definitions in MAD package root:

```bash
madengine discover --tags dummy pyt_huggingface_bert
```

### 2. Directory-Specific Models

Models organized in subdirectories (`scripts/{dir}/models.json`):

```bash
madengine discover --tags dummy2:dummy_2
```

### 3. Dynamic Models with Parameters

Python-generated models (`scripts/{dir}/get_models_json.py`):

```bash
madengine discover --tags dummy3:dummy_3:batch_size=512:in=32
```

## Build Workflow

### Basic Build

Create Docker images and manifest:

```bash
madengine build --tags model \
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
madengine build --tags model \
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

### Batch Build Mode

Batch build mode enables selective builds with per-model configuration, ideal for CI/CD pipelines where you need fine-grained control over which models to rebuild.

#### Batch Manifest Format

Create a JSON file (e.g., `batch.json`) with a list of model entries:

```json
[
  {
    "model_name": "model1",
    "build_new": true,
    "registry": "my-registry.com",
    "registry_image": "custom-namespace/model1"
  },
  {
    "model_name": "model2",
    "build_new": false,
    "registry": "my-registry.com",
    "registry_image": "custom-namespace/model2"
  },
  {
    "model_name": "model3",
    "build_new": true
  }
]
```

**Fields:**
- `model_name` (required): Model tag to include
- `build_new` (optional, default: false): If true, build this model; if false, reference existing image
- `registry` (optional): Per-model registry override
- `registry_image` (optional): Custom registry image name/namespace

#### Usage Example

```bash
# Basic batch build
madengine build --batch-manifest batch.json \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# With global registry (can be overridden per model)
madengine build --batch-manifest batch.json \
  --registry localhost:5000 \
  --additional-context-file config.json

# Verbose output
madengine build --batch-manifest batch.json \
  --registry my-registry.com \
  --verbose
```

#### Key Features

**Selective Building**: Only models with `"build_new": true` are built. Models with `"build_new": false` are added to the output manifest without building, useful for referencing existing images.

**Per-Model Registry Override**: Each model can specify its own `registry` and `registry_image`, overriding the global `--registry` flag.

**Mutually Exclusive**: Cannot use `--batch-manifest` and `--tags` together.

#### Use Cases

**CI/CD Incremental Builds**:
```json
[
  {"model_name": "changed_model", "build_new": true},
  {"model_name": "unchanged_model1", "build_new": false},
  {"model_name": "unchanged_model2", "build_new": false}
]
```

**Multi-Registry Deployment**:
```json
[
  {
    "model_name": "public_model",
    "build_new": true,
    "registry": "docker.io/myorg"
  },
  {
    "model_name": "private_model",
    "build_new": true,
    "registry": "gcr.io/myproject"
  }
]
```

**Development vs Production**:
```json
[
  {
    "model_name": "dev_model",
    "build_new": true,
    "registry": "localhost:5000"
  },
  {
    "model_name": "prod_model",
    "build_new": false,
    "registry": "prod-registry.com",
    "registry_image": "production/model"
  }
]
```

### Pre-built Image Mode

Skip Docker build entirely and use an existing image:

```bash
# Auto-detect image from model card's DOCKER_IMAGE_NAME env var
madengine build --tags sglang_disagg \
  --use-image \
  --additional-context-file slurm-config.json

# Or explicitly specify the image
madengine build --tags sglang_disagg \
  --use-image lmsysorg/sglang:v0.5.5.post3-rocm700-mi30x \
  --additional-context-file slurm-config.json

# Then run normally
madengine run --manifest-file build_manifest.json
```

**Image Resolution:**
1. If `--use-image <name>` provided → use that image
2. If `--use-image` (no value) → auto-detect from model card's `DOCKER_IMAGE_NAME`
3. If no image found → error with helpful message

**Mutual Exclusivity:**
- Cannot use with `--registry` (push requires local build)
- Cannot use with `--build-on-compute` (skip vs. build)

**Use cases:**
- Official framework images (SGLang, vLLM, PyTorch NGC)
- Pre-cached images on compute nodes
- Quick testing without rebuild time
- CI/CD with external registries

The manifest marks the image as `"prebuilt": true` with zero build time.

### Build on Compute Node

For SLURM environments where login nodes have limited resources:

```bash
# Build on 1 compute node, push to registry, pull in parallel at runtime
madengine build --tags model \
  --build-on-compute \
  --registry docker.io/myorg \
  --additional-context '{"slurm": {"reservation": "my-res"}}'
```

**Required:** `--registry` must be specified.

**SLURM Config Merging:**
- Model card's `slurm` section provides base configuration
- `--additional-context` overrides specific fields
- Only specify what's missing or needs override

**How it works:**

*Build Phase:*
1. Builds Docker image on **1 compute node**
2. Pushes image to registry
3. Stores registry image name in manifest

*Run Phase:*
1. Pulls image **in parallel on ALL nodes** via `srun docker pull`
2. Executes model script

**Benefits:**
- Offloads heavy build to compute resources
- Build once, distribute via registry pull
- Respects login node resource policies
- Parallel pull scales to many nodes

## Run Workflow

### Skip model run after build

When `madengine run` **builds** in the same invocation (no pre-existing `--manifest-file`), you can pass **`--skip-model-run`** to produce images and `build_manifest.json` **without** running model containers.

- **Ignored** when `--manifest-file` points at an existing manifest (execution-only mode): use plain `madengine run --manifest-file ...` to run later.
- **Ignored** with a warning if this invocation did not perform a build (for example a manifest was already present and no rebuild occurred).

```bash
madengine run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}' \
  --skip-model-run
```

See [CLI Reference — `run`](cli-reference.md#run---execute-models) and `madengine run --help`.

### Local Execution

Run on local machine:

```bash
madengine run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

**Required for Local:**
- `gpu_vendor`: "AMD", "NVIDIA"
- `guest_os`: "UBUNTU", "CENTOS"

### ROCm path (non-default installs)

When ROCm is not installed under `/opt/rocm` (e.g. [TheRock](https://github.com/ROCm/TheRock) or pip), set the ROCm root so GPU detection and container environment use the correct paths:

```bash
# Via environment variable
export ROCM_PATH=/path/to/rocm
madengine run --tags model --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Via CLI (overrides ROCM_PATH)
madengine run --tags model --rocm-path /path/to/rocm \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

`--rocm-path` applies only to the **run** command (not build). See [CLI Reference - run](cli-reference.md#run---execute-models).

### Deploy to Kubernetes

```bash
# Build phase
madengine build --tags model \
  --registry gcr.io/myproject \
  --additional-context '{"k8s": {"gpu_count": 2}}'

# Deploy phase
madengine run --manifest-file build_manifest.json
```

Deployment target is automatically detected from `k8s` key in configuration.

### Deploy to SLURM

```bash
# Build phase (local or CI)
madengine build --tags model \
  --registry my-registry.io \
  --additional-context '{"slurm": {"partition": "gpu", "gpus_per_node": 4}}'

# Deploy phase (on SLURM login node)
ssh user@hpc-login.example.com
madengine run --manifest-file build_manifest.json
```

Deployment target is automatically detected from `slurm` key in configuration. To run on specific nodes, set `slurm.nodelist` (e.g. `"nodelist": "node01,node02"`); see [Configuration](configuration.md#slurm-deployment) and [examples/slurm-configs/basic/03-multi-node-basic-nodelist.json](../examples/slurm-configs/basic/03-multi-node-basic-nodelist.json).

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
madengine run --tags model --additional-context-file config.json
```

### Custom Timeouts

```bash
# Override default timeout
madengine run --tags model --timeout 7200

# No timeout (run indefinitely)
madengine run --tags model --timeout 0
```

### Debugging

```bash
# Keep containers alive
madengine run --tags model --keep-alive

# Verbose output
madengine run --tags model --verbose --live-output

# Both
madengine run --tags model --keep-alive --verbose --live-output
```

If the run is marked `FAILURE` because the log contains benign substrings (for example `RuntimeError:`) while the workload actually passed, configure [log error pattern scan](configuration.md#run-phase-log-error-pattern-scan) (`log_error_pattern_scan`, `log_error_benign_patterns`).

### Clean Rebuild

```bash
# Rebuild without Docker cache
madengine build --tags model --clean-docker-cache
```

## Performance Profiling

Profile GPU usage and library calls:

```bash
# GPU profiling
madengine run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{"name": "rocprof"}]
  }'

# Library tracing
madengine run --tags model \
  --additional-context '{"tools": [{"name": "rocblas_trace"}]}'

# Multiple tools (stackable)
madengine run --tags model \
  --additional-context '{"tools": [
    {"name": "rocprof"},
    {"name": "miopen_trace"}
  ]}'
```

See [Profiling Guide](profiling.md) and [CLI Reference - run command](cli-reference.md#run---execute-models) for details.

## Reporting and Database Integration

### Generate HTML Reports

Convert performance CSV files to viewable HTML reports:

```bash
# Single CSV to HTML
madengine report to-html --csv-file perf_entry.csv

# Result: Creates perf_entry.html in same directory
```

### Consolidated Email Reports

Generate a single HTML report from multiple CSV files:

```bash
# Process all CSV files in current directory
madengine report to-email

# Specify directory
madengine report to-email --directory ./results

# Custom output filename
madengine report to-email --dir ./results --output weekly_summary.html
```

**Use Cases:**
- Weekly performance summaries
- CI/CD result reports
- Team email distributions
- Performance trend analysis

### Upload to MongoDB

Store performance data in MongoDB for long-term tracking:

```bash
# Configure MongoDB connection
export MONGO_HOST=mongodb.example.com
export MONGO_PORT=27017
export MONGO_USER=performance_user
export MONGO_PASSWORD=secretpassword

# Upload results
madengine database \
  --csv-file perf_entry.csv \
  --database-name performance_tracking \
  --collection-name model_runs

# Upload specific results
madengine database \
  --csv-file results/perf_mi300.csv \
  --db benchmarks \
  --collection mi300_results
```

**Integration Workflow:**

```bash
# 1. Run benchmarks
madengine run --tags model1 model2 model3 \
  --output perf_entry.csv

# 2. Generate HTML report
madengine report to-html --csv-file perf_entry.csv

# 3. Upload to database
madengine database \
  --csv-file perf_entry.csv \
  --db benchmarks \
  --collection daily_runs

# 4. Send email report
madengine report to-email --output daily_summary.html
# (Then use your email tool to send daily_summary.html)
```

See [CLI Reference](cli-reference.md#report---generate-reports) and [CLI Reference](cli-reference.md#database---upload-to-mongodb) for complete options.

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
- `megatron` - Large transformers (K8s + SLURM)
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
madengine run --manifest-file build_manifest.json
```

## Troubleshooting

### Model Not Found

```bash
# Ensure you're in MAD directory
cd /path/to/MAD
madengine discover --tags your_model
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
madengine build --tags model --clean-docker-cache --verbose
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_DIR` | MAD package directory | `/path/to/MAD` |
| `ROCM_PATH` | ROCm installation root (used when `--rocm-path` not set). Use when ROCm is not in `/opt/rocm` (e.g. Rock, pip). | `/path/to/rocm` |
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

## Command-Line Tips

### Using Configuration Files

For complex configurations, use JSON files:

```bash
# Create config.json
cat > config.json << 'EOF'
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "docker_gpus": "0,1,2,3",
  "timeout_multiplier": 2.0,
  "distributed": {
    "launcher": "torchrun",
    "nproc_per_node": 4
  }
}
EOF

# Use with commands
madengine build --tags model --additional-context-file config.json
madengine run --tags model --additional-context-file config.json
```

### Multiple Tags

Specify tags in multiple ways:

```bash
# Space-separated
madengine run --tags model1 --tags model2 --tags model3

# Comma-separated
madengine run --tags model1,model2,model3

# Mix both
madengine run --tags model1 --tags model2,model3
```

### Debugging Commands

```bash
# Full verbose output with real-time logs
madengine run --tags model --verbose --live-output

# Keep container alive for inspection
madengine run --tags model --keep-alive

# Check what will be discovered
madengine discover --tags model --verbose
```

### CI/CD Integration

madengine uses consistent exit codes (0=success, 2=build failure, 3=run failure, 4=invalid args). Failed runs are still written to `perf.csv` with status `FAILURE`. See [CLI Reference — Exit Codes](cli-reference.md#exit-codes) for the full table.

```bash
#!/bin/bash
# Example CI script

set -e  # Exit on error

# Build images
madengine build --batch-manifest batch.json \
  --registry docker.io/myorg \
  --verbose

# Run tests
madengine run --manifest-file build_manifest.json \
  --timeout 3600

# Check exit code (0=success, 2=build failure, 3=run failure; see CLI Reference)
if [ $? -eq 0 ]; then
  echo "✅ Tests passed"
  
  # Generate and upload results
  madengine report to-email --output ci_results.html
  madengine database \
    --csv-file perf.csv \
    --db ci_results \
    --collection ${CI_BUILD_ID}
else
  echo "❌ Tests failed"
  exit $?
fi
```

## Next Steps

### Documentation

- **[CLI Reference](cli-reference.md)** - Complete command options and examples
- [Configuration Guide](configuration.md) - Advanced configuration options
- [Deployment Guide](deployment.md) - Kubernetes and SLURM deployment
- [Batch Build Guide](batch-build.md) - Selective builds for CI/CD
- [Profiling Guide](profiling.md) - Performance analysis
- [Launchers Guide](launchers.md) - Multi-node training frameworks

### Quick Links

- [Main README](../README.md) - Project overview
- [Installation Guide](installation.md) - Setup instructions
- [Contributing Guide](contributing.md) - How to contribute
- [GitHub Issues](https://github.com/ROCm/madengine/issues) - Report issues or get help


# Configuration Guide

Complete guide to configuring madengine for various use cases and environments.

## Configuration Methods

### 1. Inline JSON String

```bash
madengine run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

### 2. JSON Configuration File

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

### 3. YAML Configuration (`--config`)

```bash
madengine run --tags model --config scheduler=slurm --config launcher=torchrun
madengine run --config my_job.yaml
```

> **Mutual exclusion**: `--config` cannot be combined with `--additional-context` or `--additional-context-file`. Using both produces an error.

See [YAML Configuration](#yaml-configuration-config) below for full details.

## YAML Configuration (`--config`)

The `--config` flag provides composable, Hydra-based YAML configuration as an alternative to JSON strings. It is available on both `run` and `build` commands.

### How It Works

1. madengine loads a base `config.yaml` with sensible defaults (AMD hardware, Docker platform, local scheduler)
2. **Config group overrides** (e.g., `scheduler=slurm`) swap in pre-built YAML fragments
3. **Inline overrides** (e.g., `distributed.nnodes=4`) set individual values
4. **User YAML files** (e.g., `my_job.yaml`) merge on top with highest priority

All four can be combined in a single command:

```bash
madengine run --config my_job.yaml \
  --config scheduler=slurm \
  --config launcher=torchrun \
  --config distributed.nnodes=4
```

### Config Groups

madengine ships with pre-built config groups under `src/madengine/configs/`:

#### Default Groups (swapped via `group=option`)

| Group | Default | Options | Description |
|-------|---------|---------|-------------|
| `platform` | `docker` | `docker`, `bare_metal`, `singularity`, `podman` | Execution platform |
| `scheduler` | `local` | `local`, `slurm`, `k8s` | Job scheduler — `slurm` and `k8s` add their respective config sections |
| `hardware` | `amd` | `amd`, `nvidia`, `cpu` | Sets `gpu_vendor`, `guest_os`, runtime device config |
| `launcher` | `none` | `none`, `torchrun`, `deepspeed`, `megatron`, `torchtitan`, `vllm`, `sglang`, `sglang_disagg`, `primus`, `native` | Distributed launcher — sets `distributed.enabled`, `distributed.launcher`, and launcher-specific defaults |

#### Append-Only Groups (added via `+group=option`)

These are not loaded by default. Use the `+` prefix to add them:

| Group | Options | Description |
|-------|---------|-------------|
| `+profile` | `mi300x_8gpu`, `mi300x_single`, `mi250x_4gpu`, `h100_8gpu`, `a100_8gpu` | Hardware profiles — sets GPU type, environment variables, distributed settings |
| `+env` | `nccl_debug`, `nccl_tuned`, `infiniband`, `miopen_defaults` | Environment variable presets |
| `+tools` | `rocprofv3_lightweight`, `rocprofv3_comprehensive`, `power_profiler`, `vram_profiler`, `rocm_trace_lite` | Profiling tool presets |
| `+data` | `local`, `s3`, `minio`, `nas` | Data source configuration |
| `+build` | `default`, `ci`, `multi_arch` | Build presets for CI or multi-arch builds |

### User YAML Files

Create a job-specific YAML file and pass it via `--config`:

```yaml
# my_slurm_job.yaml
model:
  tags: [my_model]
  timeout: 3600

debug: true

env_vars:
  MY_VAR: test_value
  NCCL_DEBUG: INFO

distributed:
  enabled: true
  launcher: torchrun
  nnodes: 2
  nproc_per_node: 4

slurm:
  partition: gpu
  time: "02:00:00"
```

```bash
madengine run --config my_slurm_job.yaml
```

User YAML values merge on top of the base config and any config group selections. You can also combine a user file with overrides:

```bash
madengine run --config my_slurm_job.yaml --config distributed.nnodes=8
```

### Priority Order

1. **Inline overrides** (`key=value`) — highest
2. **User YAML file** — merged on top of composed config
3. **Config group selections** (`scheduler=slurm`)
4. **Base config defaults** — lowest

### Examples

```bash
# Local run with defaults (AMD, Docker, no distribution)
madengine run --tags dummy --config

# SLURM multi-node training
madengine run --tags model \
  --config scheduler=slurm \
  --config launcher=torchrun \
  --config distributed.nnodes=4

# MI300x 8-GPU profile with NCCL debug
madengine run --tags model \
  --config +profile=mi300x_8gpu \
  --config +env=nccl_debug

# NVIDIA hardware
madengine run --tags model --config hardware=nvidia

# Kubernetes with vLLM inference
madengine run --tags model \
  --config scheduler=k8s \
  --config launcher=vllm \
  --config distributed.nnodes=2

# Build with CI preset and multi-arch
madengine build --tags model \
  --config +build=ci \
  --registry docker.io/myorg

# User YAML with profiling
madengine run --config my_job.yaml \
  --config +tools=rocprofv3_lightweight
```

### Metadata from Config

When using `--config`, certain YAML keys are extracted as metadata rather than passed to the internal context:

- `model.tags` — used as `--tags` if not specified on the CLI
- `model.timeout` — used as `--timeout` if not specified
- `model.container_image` — promoted to `MAD_CONTAINER_IMAGE` in context
- `build.registry` — used as `--registry` if not specified
- `build.target_archs` — used as `--target-archs` if not specified
- `platform`, `output`, `summary_output`, `data_config`, `live_output` — extracted to metadata

### Validation

madengine validates the composed config and reports errors for:

- Conflicting scheduler selections (e.g., both `slurm` and `k8s` sections present)
- `distributed.enabled: true` without a `distributed.launcher`
- Invalid `distributed.nnodes` (must be a positive integer)
- Unsupported `platform.type` (currently only `docker` is supported)
- Unknown top-level config keys (catches typos)

## Default Configuration Values

madengine provides sensible defaults for common AMD/Ubuntu workflows:

| Field | Default Value | Customization |
|-------|---------------|---------------|
| `gpu_vendor` | `AMD` | Set to `NVIDIA` for NVIDIA GPUs |
| `guest_os` | `UBUNTU` | Set to `CENTOS` for CentOS containers |

### When Defaults Apply

Defaults are applied during the **build** command when fields are not explicitly provided:

```bash
# Uses defaults: {"gpu_vendor": "AMD", "guest_os": "UBUNTU"}
madengine build --tags model

# Explicit override
madengine build --tags model \
  --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'
```

When defaults are applied, you'll see an informative message:

```
ℹ️  Using default values for build configuration:
   • gpu_vendor: AMD (default)
   • guest_os: UBUNTU (default)

💡 To customize, use --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'
```

### Partial Configuration

You can provide one field and let the other default:

```bash
# Override only gpu_vendor (guest_os defaults to UBUNTU)
madengine build --tags model \
  --additional-context '{"gpu_vendor": "NVIDIA"}'

# Override only guest_os (gpu_vendor defaults to AMD)
madengine build --tags model \
  --additional-context '{"guest_os": "CENTOS"}'
```

### Production Recommendations

For production deployments:
- ✅ **DO** explicitly specify all configuration values
- ✅ **DO** use configuration files for reproducibility
- ⚠️ **AVOID** relying on defaults in automated workflows

### Run Command Behavior

The **run** command does NOT require these values because it can detect GPU vendor at runtime.
Defaults only apply to the **build** command where Dockerfile selection requires them.

## Run phase: log error pattern scan

After a successful container run, madengine may scan the **run log file** for fixed substrings (for example `RuntimeError:`, `OutOfMemoryError`, `Traceback (most recent call last)`). If a match is found, the run can be marked `FAILURE` even when performance metrics exist—intended as a safety net when logs show obvious Python or OOM errors.

Some suites (for example layer unit tests) intentionally print benign `RuntimeError:` text while pytest still passes. In those cases you can **disable** the scan or **narrow** what counts as an error.

Keys can be set in `--additional-context` / `--additional-context-file`, or on the **model** entry in `models.json` (same keys). **Runtime context overrides the model** when both are set.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `log_error_pattern_scan` | bool or string/number (coerced) | `true` | If `false`, skip substring-based log failure detection entirely (rely on exit codes and other signals). |
| `log_error_benign_patterns` | array of strings | `[]` | Extra lines to **exclude** before matching (appended to built-in exclusions such as ROCProf/metrics noise). Model list is merged first, then context list. |
| `log_error_patterns` | array of strings (non-empty) | (built-in list) | If set, **replaces** the default pattern list. Use only when you need a custom allowlist of failure substrings. |

**Example — disable scan for a tag (pytest is authoritative):**

```bash
madengine run --tags my_unit_test_suite \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU", "log_error_pattern_scan": false}'
```

**Example — extra benign substrings (prefer stable strings from real logs):**

```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "log_error_benign_patterns": [
    "expected benign fragment from workload log"
  ]
}
```

Disabling the scan does **not** change performance metric extraction from the log; it only affects the post-hoc grep used to set `has_errors` for status.

## Basic Configuration

**gpu_vendor** (case-insensitive):
- `"AMD"` - AMD ROCm GPUs
- `"NVIDIA"` - NVIDIA CUDA GPUs

**guest_os** (case-insensitive):
- `"UBUNTU"` - Ubuntu Linux
- `"CENTOS"` - CentOS Linux

### ROCm path (run only)

**Host** (where `madengine` runs validation): by default, the ROCm root is **auto-detected** (traditional `/opt/rocm`, [TheRock](https://github.com/ROCm/TheRock) `rocm-sdk` / manifest layout, or `ROCM_PATH`-like env hints). Set `MAD_AUTO_ROCM_PATH=0` to skip auto and use only legacy resolution (`ROCM_PATH` then `/opt/rocm`).

**Overrides** (recommended for CI):

- **Additional context (host):** top-level `"MAD_ROCM_PATH": "/path/to/host/rocm"` — controls where madengine looks for host GPU tools (`rocminfo`, `amd-smi`, etc.).
- **Additional context (container):** `"docker_env_vars": { "MAD_ROCM_PATH": "/path/inside/image" }` — sets the in-container `ROCM_PATH` for Docker runs. If omitted, at `run` time madengine uses the image OCI `Env` (`ROCM_PATH` / `ROCM_HOME`) if present, then an in-container probe, then defaults to `/opt/rocm`. The host-resolved path is **not** mirrored into the container.

These two keys are independent, allowing host and container to use different ROCm installations without confusion.

Precedence (host): top-level `MAD_ROCM_PATH` → auto-detect (unless disabled) → `ROCM_PATH` → `/opt/rocm`.

Precedence (container, **local Docker `run`**, **AMD**): `docker_env_vars.MAD_ROCM_PATH` (maps to `ROCM_PATH` for the workload) or explicit `ROCM_PATH` in `docker_env_vars` → image OCI `Env` (`ROCM_PATH` / `ROCM_HOME`) → in-image probe → default `/opt/rocm` with a warning. Implemented in `ContainerRunner.run_container` after the run image is resolved.

This applies to the run phase; build uses build-only context (no GPU detection) but still honors `MAD_ROCM_PATH` in context when set.

At the start of each container run, a **Run Phase Environment** table is printed showing host vs container installation type (`apt install` or `therock`), ROCm/CUDA root, and version side-by-side. See [Run phase environment table](usage.md#run-phase-environment-table).

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

Automatically applies (see presets under `src/madengine/deployment/presets/k8s/`):
- Namespace: `default`
- Resource limits based on GPU count
- Image pull policy: `Always` (base default)
- Service account: `default`
- GPU vendor detection from context
- `k8s.secrets` defaults (see below)

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
    "ttl_seconds_after_finished": null,
    "allow_privileged_profiling": null,
    "secrets": {
      "strategy": "from_local_credentials",
      "image_pull_secret_names": ["my-registry-secret"],
      "runtime_secret_name": null
    }
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
- `ttl_seconds_after_finished` - Optional Job TTL in seconds (auto-delete finished Job); `null` to omit
- `allow_privileged_profiling` - `null` means enable elevated `securityContext` when tools/profiling are configured; `true`/`false` to force
- `secrets.strategy` - `from_local_credentials` (default): create `Secret` objects from local `credential.json` at deploy time; `existing`: only reference pre-created Secrets; `omit`: no runtime Secret from client
- `secrets.image_pull_secret_names` - Extra pull secret names (strings) merged with any created from `credential.json` when using `from_local_credentials`
- `secrets.runtime_secret_name` - Required for `existing` (pre-created opaque Secret with key `credential.json`); optional for `omit` if you still mount a runtime Secret

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

See [`examples/configs/templates/k8s.yaml`](../examples/configs/templates/k8s.yaml) for the complete annotated YAML template, or [`examples/configs/demo/k8s/`](../examples/configs/demo/k8s/) for ready-to-run examples.

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
    "nodes": 2,
    "nodelist": "node01,node02",
    "time": "24:00:00"
  }
}
```

**Note:** `nodelist` is optional; omit it to let SLURM choose nodes. When set, the job runs only on the listed nodes and node health preflight is skipped.

**SLURM Options:**
- `partition` - SLURM partition name (required)
- `account` - Billing account
- `qos` - Quality of Service
- `gpus_per_node` - GPUs per node (default: 8)
- `nodes` - Number of nodes (default: 1)
- `nodelist` - Comma-separated node names to run on (e.g. `"node01,node02"`); when set, job is restricted to these nodes and automatic node health preflight is skipped
- `exclude` - Comma-separated node names to exclude
- `constraint` - Node feature constraint (e.g., `"infiniband"`)
- `time` - Wall time limit HH:MM:SS (default: `"24:00:00"`)
- `exclusive` - Request exclusive node access (default: `true`)
- `modules` - List of environment modules to load
- `network_interface` - Network interface for NCCL/GLOO (e.g., `"ib0"`)
- `shared_workspace` - Explicit NFS/Lustre shared workspace path

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

See [`examples/configs/templates/slurm.yaml`](../examples/configs/templates/slurm.yaml) for the complete annotated YAML template, or [`examples/configs/demo/slurm/`](../examples/configs/demo/slurm/) for ready-to-run examples.

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

> **YAML config note**: When using `--config`, you must also set `distributed.enabled: true` explicitly. The default config loads `launcher: none` which sets `enabled: false`; setting a launcher alone does not override it.

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


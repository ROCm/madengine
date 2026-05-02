# madengine

<p align="center">
<picture>
  <img src="madengine.png" alt="madengine Logo" />
</picture>
</p>

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/ROCm/madengine/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Version](https://img.shields.io/badge/version-2.0-brightgreen.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **AI model automation and benchmarking platform for local and distributed execution**

madengine is a modern CLI tool for running Large Language Models (LLMs) and Deep Learning models across local and distributed environments. Built for the [MAD (Model Automation and Dashboarding)](https://github.com/ROCm/MAD) ecosystem, it provides seamless execution from single GPUs to multi-node clusters.

## 📖 Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Smoke Testing](#-smoke-testing)
- [Commands](#-commands)
- [Documentation](#-documentation)
- [Architecture](#-architecture)
- [Feature Matrix](#-feature-matrix)
- [Usage Examples](#-usage-examples)
- [Model Discovery](#-model-discovery)
- [Performance Profiling](#-performance-profiling)
- [Reporting and Database](#-reporting-and-database)
- [Installation](#-installation)
- [Tips & Best Practices](#-tips--best-practices)
  - [Log error pattern scan](#log-error-pattern-scan)
  - [Exit codes and CI](#exit-codes-and-ci)
- [Contributing](#-contributing)
- [License](#-license)
- [Links & Resources](#-links--resources)

## ✨ Key Features

- **🚀 Modern CLI** - Rich terminal output with Typer and Rich
- **🎯 Simple Deployment** - Run locally or deploy to Kubernetes/SLURM via configuration
- **🔧 Distributed Launchers** - Full support for torchrun, DeepSpeed, Megatron-LM, TorchTitan, Primus, vLLM, SGLang
- **🐳 Container-Native** - Docker-based execution with GPU support (ROCm, CUDA)
- **📂 ROCm Path** - Auto-detect **host** ROCm root (override with top-level `MAD_ROCM_PATH`); in-container `ROCM_PATH` is set independently via `docker_env_vars.MAD_ROCM_PATH` and resolved at Docker run (image OCI env + in-image probe, not host mirroring) — see [Configuration](docs/configuration.md#rocm-path-run-only)
- **📊 Performance Tools** - Integrated profiling with rocprof/rocprofv3, [rocm-trace-lite](https://github.com/sunway513/rocm-trace-lite) (RTL), rocblas, MIOpen, RCCL tracing
- **🎯 ROCprofv3 Profiles** - 8 pre-configured profiles for compute/memory/communication bottleneck analysis
- **🔍 Environment Validation** - TheRock ROCm detection and validation tools
- **⚙️ Intelligent Defaults** - Minimal K8s configs with automatic preset application
- **📋 Configurable log scan** - Optional `--additional-context` keys to disable or tune post-run log substring checks (see [Log error pattern scan](#log-error-pattern-scan))

## 🚀 Quick Start

```bash
# Install madengine
pip install git+https://github.com/ROCm/madengine.git

# Clone MAD package (required for models)
git clone https://github.com/ROCm/MAD.git && cd MAD

# Discover available models
madengine discover --tags dummy

# Run locally (full workflow: discover/build/run as configured by the model)
madengine run --tags dummy

# Or with explicit configuration
madengine run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

> **Note**: For build operations, `gpu_vendor` defaults to `AMD` and `guest_os` defaults to `UBUNTU` if not specified. For production deployments or non-AMD/Ubuntu environments, explicitly specify these values.

If auto-detection does not find your **host** ROCm root, set top-level `MAD_ROCM_PATH` in `--additional-context`. For a different ROCm root **inside the container**, set `docker_env_vars.MAD_ROCM_PATH` in additional context. If you omit it, madengine derives in-container `ROCM_PATH` when running Docker (from the image's baked-in env, then an in-container probe, then `/opt/rocm` — it does **not** copy the host path). You can also set `ROCM_PATH` / `MAD_AUTO_ROCM_PATH=0` for **host** behavior as documented in [docs/configuration.md](docs/configuration.md):

```bash
# Override host ROCm root:
madengine run --tags dummy --additional-context '{"MAD_ROCM_PATH": "/path/to/rocm"}'
# or: export ROCM_PATH=/path/to/rocm && madengine run --tags dummy
# Override in-container ROCm root independently:
madengine run --tags dummy --additional-context '{"docker_env_vars": {"MAD_ROCM_PATH": "/path/in/container"}}'
```

**Results:** Performance data is written to `perf.csv` (and optionally `perf_entry.csv`). The file is created automatically if missing. Failed runs (including pre-run setup failures) are recorded with status `FAILURE` so every attempted model appears in the table. See [Exit Codes](docs/cli-reference.md#exit-codes) for CI/script usage.

## 🧪 Smoke Testing

Use the prebuilt smoke configs and wrapper script under `examples/` to quickly validate:

- RDMA recommender on SLURM + Kubernetes
- GCM preflight/collector on SLURM (phase-1 scope)

```bash
# SLURM smoke (build + run) + artifact verification
examples/run-smoke.sh slurm MODEL_DIR=/path/to/model MODEL_TAG=your_tag
examples/run-smoke.sh verify-slurm

# Kubernetes smoke (build + run) + artifact verification
examples/run-smoke.sh k8s MODEL_DIR=/path/to/model MODEL_TAG=your_tag
examples/run-smoke.sh verify-k8s
```

Smoke assets:

- `examples/run-smoke.sh`
- `examples/Makefile.smoke`
- `examples/slurm-configs/configs/smoke-rdma-gcm-slurm.json`
- `examples/k8s-configs/configs/smoke-rdma-k8s.json`
- `examples/cluster-smoke-checklist.md`

## 📋 Commands

madengine provides five main commands for model automation and benchmarking:

| Command | Description | Use Case |
|---------|-------------|----------|
| **[discover](#-model-discovery)** | Find available models | Model exploration and validation |
| **[build](#building-images)** | Build Docker images | Create containerized models |
| **[run](#-usage-examples)** | Execute models | Local and distributed execution |
| **[report](docs/cli-reference.md#report---generate-reports)** | Generate HTML reports | Convert CSV to viewable reports |
| **[database](docs/cli-reference.md#database---upload-to-mongodb)** | Upload to MongoDB | Store results in database |

**Quick Start:**

```bash
# Discover models
madengine discover --tags dummy

# Build image (uses AMD/UBUNTU defaults)
madengine build --tags dummy

# Run model
madengine run --tags dummy

# For non-AMD/Ubuntu environments, specify explicitly:
# madengine build --tags dummy --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'

# Generate report
madengine report to-html --csv-file perf_entry.csv

# Upload results
madengine database --csv-file perf_entry.csv --db mydb --collection results
```

For detailed command options, see the **[CLI Command Reference](docs/cli-reference.md)**.

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Complete installation instructions |
| [Usage Guide](docs/usage.md) | Commands, workflows, and examples ([`--skip-model-run`](docs/usage.md#skip-model-run-after-build)) |
| **[CLI Reference](docs/cli-reference.md)** | **Detailed command options and examples** |
| [Deployment](docs/deployment.md) | Kubernetes and SLURM deployment |
| [Configuration](docs/configuration.md) | Advanced options; [run log error pattern scan](docs/configuration.md#run-phase-log-error-pattern-scan) |
| [Batch Build](docs/batch-build.md) | Selective builds for CI/CD |
| [Launchers](docs/launchers.md) | Distributed training frameworks |
| [Profiling](docs/profiling.md) | Performance analysis tools |
| [Contributing](docs/contributing.md) | How to contribute |

## 🏗️ Architecture

```
  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │  madengine CLI v2.0 (Typer + Rich)                                                      │
  │  discover │ build │ run │ report │ database                                             │
  └─────────────────────────────────────────────────────────────────────────────────────────┘
     │           │        │
     │           │        ▼
     │           │  ┌─────────────────────── Orchestration Layer ───────────────────────────┐
     │           │  │  Model Discovery (models.json / scripts/ get_models)                  │
     │           │  │  BuildOrchestrator · RunOrchestrator                                  │
     │           └──│                                                                       |  
     └──────────────┴───────────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────┼──────────────────── Infrastructure Layer ─────────────┐
  │              ▼                  ▼                  ▼                                    │
  │       ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                            │
  │       │ Local        │   │ Kubernetes   │   │ SLURM        │                            │
  │       │ Docker       │   │ Jobs         │   │ Jobs         │                            │
  │       └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                            │
  │              └──────────────────┼──────────────────┘                                    │
  └─────────────────────────────────┼───────────────────────────────────────────────────────┘
                                    ▼
  ┌──────────────────────────────────── Launcher Layer (Distribution) ──────────────────────┐
  │  Train: torchrun · DeepSpeed · Megatron-LM · TorchTitan · Primus                         │
  │  Infer: vLLM · SGLang · SGLang Disagg                                                   │
  └─────────────────────────────────┬───────────────────────────────────────────────────────┘
                                    ▼
                    ┌─────────────────────────────┐
                    │ Performance (CSV/JSON)      │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
       ┌───────────────────┐                  ┌──────────────────┐
       │ report            │                  │ database         │
       │ to-html, to-email │                  │ MongoDB upload   │
       └───────────────────┘                  └──────────────────┘
```

**Component Flow:**

1. **CLI Layer** - User interface with 5 commands (discover, build, run, report, database)
2. **Model Discovery** - Find and validate models from MAD package
3. **Orchestration** - BuildOrchestrator & RunOrchestrator manage workflows
4. **Execution Targets** - Local Docker, Kubernetes Jobs, or SLURM Jobs
5. **Distributed Launchers** - Training (torchrun, DeepSpeed, Megatron-LM, TorchTitan, Primus) and Inference (vLLM, SGLang)
6. **Performance Output** - CSV/JSON results with metrics
7. **Post-Processing** - Report generation (HTML/Email) and database upload (MongoDB)

## 🎯 Feature Matrix

### Supported Launchers & Infrastructure

| Launcher | Local | Kubernetes | SLURM | Type | Key Features |
|----------|-------|-----------|-------|------|--------------|
| **torchrun** | ✅ | ✅ | ✅ | Training | PyTorch DDP/FSDP, elastic training |
| **DeepSpeed** | ✅ | ✅ | ✅ | Training | ZeRO optimization, pipeline parallelism |
| **Megatron-LM** | ✅ | ✅ | ✅ | Training | Tensor+Pipeline parallel, large transformers |
| **TorchTitan** | ✅ | ✅ | ✅ | Training | FSDP2+TP+PP+CP, Llama 3.1 (8B-405B) |
| **Primus** | ✅ | ✅ | ✅ | Training | Megatron / TorchTitan / MaxText via Primus YAML; `distributed.primus` |
| **vLLM** | ✅ | ✅ | ✅ | Inference | v1 engine, PagedAttention, Ray cluster |
| **SGLang** | ✅ | ✅ | ✅ | Inference | RadixAttention, structured generation |
| **SGLang Disagg** | ❌ | ✅ | ✅ | Inference | Disaggregated prefill/decode, Mooncake, 3+ nodes |

**Note:** All launchers support single-GPU, multi-GPU (single node), and multi-node (where infrastructure allows). See [Launchers Guide](docs/launchers.md) for details.

### Parallelism Capabilities

| Launcher | Tensor Parallel (TP) | Pipeline Parallel (PP) | Data Parallel (DP) | Context Parallel (CP) | FSDP/ZeRO | Expert Parallel (EP) | Primary Use Case |
|----------|----------------------|------------------------|--------------------|------------------------|-----------|----------------------|------------------|
| **torchrun** | ❗Manual | ❌No | ❗Manual (DDP) | ❌No | ❗Manual (FSDP) | ❌No | General distributed training |
| **TorchTitan** | ✅Auto | ✅Auto | ✅Auto (FSDP2) | ❗Manual | ✅Auto (FSDP2) | ❌No | Large-scale LLM pre-training |
| **DeepSpeed** | ❗Manual | ❗Manual | ✅Auto (ZeRO) | ❌No | ✅Auto (ZeRO) | ❌No | Memory-efficient training |
| **Megatron-LM** | ✅Auto | ✅Auto | ✅Implicit | ✅Auto | ❌No | ❌No | Large transformer training |
| **Primus** | ❗Manual | ❗Manual | ❗Manual | ❗Manual | ❗Manual | ❌No | Unified pretrain (experiment YAML; backend-specific) |
| **vLLM** | ✅Auto | SLURM: ✅Auto (Multi) / K8s: ❗Disabled | ✅Auto (Replicas) | ❌No | ❌No | ❗Manual | High-throughput inference |
| **SGLang** | ✅Auto | SLURM: ✅Auto (Multi) / K8s: ❗Disabled | ❗Limited | ❌No | ❌No | ❌No | Inference + structured gen |
| **SGLang PD Disagg** | ✅Auto | ❌No | ✅Role-based | ❌No | ❌No | ❌No | Optimized prefill/decode |

**Legend:** ✅Auto = supported and configured by madengine; ❗Manual = supported by launcher but requires user configuration; ❗Limited / ❗Disabled = launcher or platform limitation. See [Launchers Guide](docs/launchers.md) and [Configuration](docs/configuration.md) for details.

### Infrastructure Capabilities

| Feature | Local | Kubernetes | SLURM |
|---------|-------|-----------|-------|
| **Execution** | Docker containers | K8s Jobs | SLURM jobs |
| **Multi-Node** | ❌ | ✅ Indexed Jobs | ✅ Job arrays |
| **Resource Mgmt** | Manual | Declarative (YAML) | Batch scheduler |
| **Monitoring** | Docker logs | kubectl/dashboard | squeue/scontrol |
| **Auto-scaling** | ❌ | ✅ | ❌ |
| **Network** | Host | CNI plugin | InfiniBand/Ethernet |

## 💻 Usage Examples

### Local Execution

```bash
# Single GPU
madengine run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Multi-GPU with torchrun (DDP/FSDP)
madengine run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "docker_gpus": "0,1,2,3",
    "distributed": {
      "launcher": "torchrun",
      "nproc_per_node": 4
    }
  }'

# With DeepSpeed (ZeRO optimization)
madengine run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "docker_gpus": "all",
    "distributed": {
      "launcher": "deepspeed",
      "nproc_per_node": 8
    }
  }'
```

### Kubernetes Deployment

```bash
# Minimal config (auto-defaults applied)
madengine run --tags model \
  --additional-context '{"k8s": {"gpu_count": 2}}'

# Multi-node inference with vLLM
madengine run --tags model \
  --additional-context '{
    "k8s": {
      "namespace": "ml-team",
      "gpu_count": 8
    },
    "distributed": {
      "launcher": "vllm",
      "nnodes": 2,
      "nproc_per_node": 4
    }
  }'

# SGLang with structured generation
madengine run --tags model \
  --additional-context '{
    "k8s": {"gpu_count": 4},
    "distributed": {
      "launcher": "sglang",
      "nproc_per_node": 4
    }
  }'
```

### SLURM Deployment

```bash
# Build phase (local or CI)
madengine build --tags model \
  --registry gcr.io/myproject \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Deploy phase (on SLURM login node)
madengine run --manifest-file build_manifest.json \
  --additional-context '{
    "slurm": {
      "partition": "gpu",
      "nodes": 4,
      "gpus_per_node": 8,
      "time": "24:00:00"
    },
    "distributed": {
      "launcher": "torchtitan",
      "nnodes": 4,
      "nproc_per_node": 8
    }
  }'
```

To run on **specific nodes**, set `nodelist` (comma-separated node names). When set, the job is restricted to those nodes and automatic node health preflight is skipped. Example: `"slurm": { "nodelist": "node01,node02", "nodes": 2, ... }`. See [Configuration](docs/configuration.md#slurm-deployment) and [examples/slurm-configs/basic/03-multi-node-basic-nodelist.json](examples/slurm-configs/basic/03-multi-node-basic-nodelist.json).

### Common Workflows

**Development → Testing → Production:**

```bash
# 1. Develop locally with single GPU
madengine run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# 2. Test multi-GPU locally
madengine run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "docker_gpus": "0,1",
    "distributed": {"launcher": "torchrun", "nproc_per_node": 2}
  }'

# 3. Build and push to registry
madengine build --tags model \
  --registry docker.io/myorg \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# 4. Deploy to Kubernetes
madengine run --manifest-file build_manifest.json
```

**CI/CD Pipeline:**

```bash
# Batch build (selective rebuilds)
madengine build --batch-manifest batch.json \
  --registry docker.io/myorg

# Run tests
madengine run --manifest-file build_manifest.json \
  --additional-context '{"k8s": {"namespace": "ci-test"}}'

# Generate and email reports
madengine report to-email --directory ./results --output ci_report.html

# Upload to database
madengine database --csv-file perf_entry.csv \
  --database-name ci_db --collection-name test_results
```

See [Usage Guide](docs/usage.md), [Configuration Guide](docs/configuration.md), and [CLI Reference](docs/cli-reference.md) for more examples.

### Building Images

```bash
# Build single model
madengine build --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Build with registry (for distributed deployment)
madengine build --tags model1 model2 \
  --registry localhost:5000 \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Build for multiple GPU architectures
madengine build --tags model \
  --target-archs gfx908 gfx90a gfx942 \
  --registry gcr.io/myproject

# Batch build mode (selective builds for CI/CD)
madengine build --batch-manifest examples/build-manifest/batch.json \
  --registry docker.io/myorg

# Clean rebuild (no Docker cache)
madengine build --tags model --clean-docker-cache \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

**Output:** Creates `build_manifest.json` with built image names and configurations.

See [Batch Build Guide](docs/batch-build.md) and examples in [`examples/build-manifest/`](examples/build-manifest/).

## 🔍 Model Discovery

madengine discovers models from the MAD package using three methods:

```bash
# Root models (models.json)
madengine discover --tags pyt_huggingface_bert

# Directory-specific (scripts/{dir}/models.json)
madengine discover --tags dummy2:dummy_2

# Dynamic with parameters (scripts/{dir}/get_models_json.py)
madengine discover --tags dummy3:dummy_3:batch_size=512
```

## 📊 Performance Profiling

madengine includes integrated profiling tools for AMD ROCm:

```bash
# GPU profiling with rocprof
madengine run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{"name": "rocprof"}]
  }'

# ROCprofv3 (ROCm 7.0+) - Advanced profiling with pre-configured profiles
madengine run --tags model \
  --additional-context '{"tools": [{"name": "rocprofv3_compute"}]}'

# Use configuration files for complex setups
madengine run --tags model \
  --additional-context-file examples/profiling-configs/rocprofv3_multi_gpu.json

# Library tracing (rocBLAS, MIOpen, Tensile, RCCL)
madengine run --tags model \
  --additional-context '{"tools": [{"name": "rocblas_trace"}]}'

# rocm-trace-lite — lightweight kernel dispatch trace (SQLite; no rocprofiler-sdk)
# Requires outbound HTTPS to GitHub on first run unless the wheel is baked into the image
# (see docs/profiling.md). Do not combine with rocprof / rocprofv3_* on the same run.
madengine run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU", "tools": [{"name": "rocm_trace_lite"}]}'

# Power and VRAM monitoring
madengine run --tags model \
  --additional-context '{"tools": [
    {"name": "gpu_info_power_profiler"},
    {"name": "gpu_info_vram_profiler"}
  ]}'

# Multiple tools (stackable)
madengine run --tags model \
  --additional-context '{"tools": [
    {"name": "rocprofv3_memory"},
    {"name": "rocblas_trace"},
    {"name": "gpu_info_power_profiler"}
  ]}'
```

**Available Tools:**

| Tool | Purpose | Output |
|------|---------|--------|
| `rocprof` | GPU kernel profiling | Kernel timings, occupancy |
| `rocprofv3_compute` | Compute-bound analysis (ROCm 7.0+) | ALU metrics, wave execution |
| `rocprofv3_memory` | Memory-bound analysis (ROCm 7.0+) | Cache hits, bandwidth |
| `rocprofv3_communication` | Multi-GPU communication (ROCm 7.0+) | RCCL traces, inter-GPU transfers |
| `rocprofv3_lightweight` | Minimal overhead profiling (ROCm 7.0+) | HIP and kernel traces |
| `rocm_trace_lite` | RTL **`lite`** mode — kernel dispatch trace (HSA, SQLite/RPD-style); [`rtl trace --mode lite`](https://sunway513.github.io/rocm-trace-lite/quickstart.html) via `rtl_trace_wrapper.sh` | `rocm_trace_lite_output/trace.db` (and optional `trace.json.gz`, `trace_summary.txt`) |
| `rocm_trace_lite_default` | RTL **`default`** mode — broader dispatch coverage; higher overhead than `lite` (same outputs paths) | Same as `rocm_trace_lite` |
| `rocblas_trace` | rocBLAS library calls | Function calls, arguments |
| `miopen_trace` | MIOpen library calls | Conv/pooling operations |
| `tensile_trace` | Tensile GEMM library | Matrix multiply details |
| `rccl_trace` | RCCL collective ops | Communication patterns |
| `gpu_info_power_profiler` | GPU power consumption | Power usage over time |
| `gpu_info_vram_profiler` | GPU memory usage | VRAM utilization |
| `therock_check` | TheRock ROCm validation | Installation detection |

**ROCprofv3 Profiles** (ROCm 7.0+):

madengine provides 8 pre-configured ROCprofv3 profiles for different bottleneck scenarios:

- `rocprofv3_compute` - Compute-bound workloads (transformers, dense ops)
- `rocprofv3_memory` - Memory-bound workloads (large batches, high-res)
- `rocprofv3_communication` - Multi-GPU distributed training
- `rocprofv3_full` - Comprehensive profiling (all metrics, high overhead)
- `rocprofv3_lightweight` - Minimal overhead (production-friendly)
- `rocprofv3_perfetto` - Perfetto UI compatible traces
- `rocprofv3_api_overhead` - API call timing analysis
- `rocprofv3_pc_sampling` - Kernel hotspot identification

See [`examples/profiling-configs/`](examples/profiling-configs/) for ready-to-use configuration files.

**rocm-trace-lite (`rocm_trace_lite` / `rocm_trace_lite_default`):**

- madengine runs workloads under `scripts/common/tools/rtl_trace_wrapper.sh`, which invokes the `rtl` CLI (or `python3 -m rocm_trace_lite.cli`) with **`RTL_MODE=lite`** or **`RTL_MODE=default`** and writes traces under `rocm_trace_lite_output/`.
- The trace **pre-script** installs the package from a **[GitHub Release wheel](https://github.com/sunway513/rocm-trace-lite/releases)** (not PyPI). By default it uses a **pinned** `linux_x86_64` wheel for reproducible installs. Set **`ROCM_TRACE_LITE_FOLLOW_LATEST=1`** to resolve the latest wheel via the GitHub API, or **`ROCM_TRACE_LITE_WHEEL_URL`** to a direct `.whl` URL for air-gapped installs or non-x86_64 platforms.
- Choose **either** `rocm_trace_lite` **or** rocprof / `rocprofv3_*` for a given run—not both. Details: [Profiling Guide](docs/profiling.md) (section *rocm-trace-lite (RTL)*).

**TheRock Validation:**

```bash
# Validate TheRock installation (AMD's pip-based ROCm)
madengine run --tags dummy_therock \
  --additional-context '{"tools": [{"name": "therock_check"}]}'
```

See [Profiling Guide](docs/profiling.md) for detailed usage and analysis.

## 📊 Reporting and Database

### Generate Reports

Convert performance CSV files to HTML reports:

```bash
# Single CSV to HTML
madengine report to-html --csv-file perf_entry.csv

# Consolidated email report (all CSVs in directory)
madengine report to-email --directory ./results --output summary.html
```

### Upload to Database

Store performance results in MongoDB:

```bash
# Set MongoDB connection
export MONGO_HOST=mongodb.example.com
export MONGO_PORT=27017
export MONGO_USER=myuser
export MONGO_PASSWORD=mypassword

# Upload CSV to MongoDB
madengine database --csv-file perf_entry.csv \
  --database-name performance_db \
  --collection-name model_runs
```

**Use Cases:**
- Track performance over time
- Compare results across different configurations
- Build performance dashboards
- Automated CI/CD reporting

See [CLI Reference](docs/cli-reference.md) for complete options.

## 📦 Installation

```bash
# Basic installation
pip install git+https://github.com/ROCm/madengine.git

# With Kubernetes support
pip install "madengine[kubernetes] @ git+https://github.com/ROCm/madengine.git"

# Development installation
git clone https://github.com/ROCm/madengine.git
cd madengine && pip install -e ".[dev]"
```

See [Installation Guide](docs/installation.md) for detailed instructions.

## 💡 Tips & Best Practices

### General Usage

- **Use configuration files** for complex setups instead of long command lines
- **Test locally first** with single GPU before scaling to multi-node
- **Enable verbose logging** (`--verbose`) when debugging issues
- **Use `--live-output`** for real-time monitoring of long-running operations

### Log error pattern scan

After a local Docker run, madengine can scan the captured **run log** for common failure substrings (for example `RuntimeError:`, `CUDA out of memory`, `Traceback`). That helps catch hard failures when exit codes are ambiguous, but some workloads log benign `RuntimeError:` text while tests still pass.

- **Disable** the scan when another signal is authoritative (e.g. pytest/JUnit inside the image): set `"log_error_pattern_scan": false` in `--additional-context` or in the model entry in `models.json`. See [Configuration — Run phase: log error pattern scan](docs/configuration.md#run-phase-log-error-pattern-scan).
- **Extend exclusions** with `log_error_benign_patterns` (list of strings), or **replace** the default pattern list with `log_error_patterns` (non-empty list of strings) for advanced cases.

### CI / Jenkins

- **Exit codes:** The CLI uses fixed exit codes (`ExitCode` in `madengine.cli.constants`, e.g. `SUCCESS=0`, `RUN_FAILURE=3`, `INVALID_ARGS=4`). Pipelines should treat **non-zero** as failure; no log scraping is required for pass/fail.
- **Streaming:** In Jenkins, avoid redirecting stdout only to a file (`> file`) without `tee` if you want the console to update during the run. Prefer `... 2>&1 | tee madengine.run.log` with `bash -o pipefail` so the step exit code is still from `madengine`.
- **Unbuffered Python:** If output still appears in chunks, set `PYTHONUNBUFFERED=1` (or `python -u`) for the `madengine` process.

### Build & Deployment

- **Separate build and run phases** for distributed deployments
- **Build without executing:** `madengine run --tags … --skip-model-run` skips container execution **after a build in that same invocation** (ignored when using an existing `--manifest-file`). See [Usage — Skip model run after build](docs/usage.md#skip-model-run-after-build).
- **Use registries** for multi-node execution (K8s/SLURM)
- **Use batch build mode** for CI/CD to optimize build times
- **Specify `--target-archs`** when building for multiple GPU architectures

### Performance

- **Start with small timeouts** and increase as needed
- **Use profiling tools** to identify bottlenecks
- **Monitor GPU utilization** with `gpu_info_power_profiler`
- **Profile library calls** with rocBLAS/MIOpen tracing

### Exit codes and CI

madengine uses consistent exit codes for scripts and CI (e.g. Jenkins): `0` = success, `1` = general failure, `2` = build failure, `3` = one or more run failures, `4` = invalid arguments. Failed runs are still written to `perf.csv` with status `FAILURE`. See [CLI Reference — Exit Codes](docs/cli-reference.md#exit-codes) for the full table and examples.

### Troubleshooting

```bash
# Check model is available
madengine discover --tags your_model

# Verbose output for debugging
madengine run --tags model --verbose --live-output

# Keep container alive for inspection
madengine run --tags model --keep-alive

# Clean rebuild if build fails
madengine build --tags model --clean-docker-cache --verbose
```

**ROCm not in /opt/rocm:** Set top-level `MAD_ROCM_PATH` in `--additional-context` for the **host**; for **in-container** paths, set `docker_env_vars.MAD_ROCM_PATH`, or let madengine resolve `ROCM_PATH` at run from the image and probe (see [Configuration](docs/configuration.md#rocm-path-run-only)).

**Common Issues:**
- **False failures with profiling**: If models show FAILURE but have performance metrics, see [Profiling Troubleshooting](docs/profiling.md#false-failure-detection-with-rocprof)
- **False failures from `RuntimeError:` in logs**: If the workload logs expected exception text but tests pass, disable or tune the scan with `log_error_pattern_scan` / `log_error_benign_patterns` — see [Configuration](docs/configuration.md#run-phase-log-error-pattern-scan)
- **ROCProf log errors**: Messages like `E20251230` are informational logs, not errors (fixed in v2.0+)
- **Configuration errors**: Validate JSON with `python -m json.tool your-config.json`

## 🤝 Contributing

We welcome contributions! See [Contributing Guide](docs/contributing.md) for details.

```bash
git clone https://github.com/ROCm/madengine.git
cd madengine
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test module
pytest tests/unit/test_error_handling.py -v

# Run error pattern tests
pytest tests/unit/test_error_handling.py::TestErrorPatternMatching -v
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links & Resources

### Documentation
- **[CLI Reference](docs/cli-reference.md)** - Complete command options
- **[Usage Guide](docs/usage.md)** - Workflows and examples
- **[Deployment Guide](docs/deployment.md)** - Kubernetes/SLURM deployment
- **[Configuration Guide](docs/configuration.md)** - Advanced configuration
- **[All Docs](docs/)** - Complete documentation index

### External Resources
- **MAD Package**: https://github.com/ROCm/MAD
- **Issues & Support**: https://github.com/ROCm/madengine/issues
- **ROCm Documentation**: https://rocm.docs.amd.com/

### Getting Help

**Command Help:**
```bash
madengine --help                    # Main help
madengine <command> --help          # Command-specific help
madengine report --help             # Sub-app help
madengine report to-html --help     # Sub-command help
```

**Quick Checks:**
```bash
# Verify installation
madengine --version

# Discover available models
madengine discover

# Check specific model
madengine discover --tags your_model --verbose
```

**Troubleshooting:**
- Check [CLI Reference](docs/cli-reference.md) for all command options
- Enable `--verbose` flag for detailed error messages
- See [Usage Guide](docs/usage.md) troubleshooting section
- Report issues: https://github.com/ROCm/madengine/issues

---

## ⚠️ Migration Notice (v2.0.0+)

The CLI has been unified! Starting from v2.0.0:
- ✅ Use `madengine` (unified modern CLI with K8s, SLURM, distributed support)
- ❌ Legacy v1.x CLI has been removed

---

**Code Quality**: Clean codebase with no dead code, comprehensive test coverage, and following Python best practices.

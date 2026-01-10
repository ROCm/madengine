# madengine

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
- [Contributing](#-contributing)
- [License](#-license)
- [Links & Resources](#-links--resources)

## ✨ Key Features

- **🚀 Modern CLI** - Rich terminal output with Typer and Rich
- **🎯 Flexible Deployment** - Run locally, Kubernetes, SLURM, or Bare Metal VM with guaranteed isolation
- **🔧 Distributed Launchers** - Full support for torchrun, DeepSpeed, Megatron-LM, TorchTitan, vLLM, SGLang
- **🐳 Container-Native** - Docker-based execution with GPU support (ROCm, CUDA)
- **🖥️ VM Isolation** - Bare metal execution with ephemeral VMs for complete environment cleanup
- **📊 Performance Tools** - Integrated profiling with rocprof, rocblas, MIOpen, RCCL tracing
- **🔍 Environment Validation** - TheRock ROCm detection and validation tools
- **⚙️ Intelligent Defaults** - Minimal configs with automatic preset application

## 🚀 Quick Start

```bash
# Install madengine
pip install git+https://github.com/ROCm/madengine.git

# Clone MAD package (required for models)
git clone https://github.com/ROCm/MAD.git && cd MAD

# Discover available models
madengine discover --tags dummy

# Run locally
madengine run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

**Results saved to `perf_entry.csv`**

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

# Build image
madengine build --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Run model
madengine run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

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
| [Usage Guide](docs/usage.md) | Commands, workflows, and examples |
| **[CLI Reference](docs/cli-reference.md)** | **Detailed command options and examples** |
| [Deployment](docs/deployment.md) | Kubernetes, SLURM, and Bare Metal VM deployment |
| [Bare Metal VM](docs/baremetal-vm.md) | VM-based execution with isolation and cleanup |
| [Configuration](docs/configuration.md) | Advanced configuration options |
| [Batch Build](docs/batch-build.md) | Selective builds for CI/CD |
| [Launchers](docs/launchers.md) | Distributed training frameworks |
| [Profiling](docs/profiling.md) | Performance analysis tools |
| [Contributing](docs/contributing.md) | How to contribute |

## 🏗️ Architecture

```
                    ┌────────────────────────────────────────┐
                    │         madengine CLI v2.0             │
                    │   (Typer + Rich Terminal Interface)    │
                    └────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │             │               │               │             │
   ┌────▼────┐    ┌───▼────┐     ┌────▼────┐    ┌────▼─────┐  ┌─────▼─────┐
   │discover │    │ build  │     │   run   │    │  report  │  │ database  │
   │         │    │        │     │         │    │          │  │           │
   └────┬────┘    └───┬────┘     └────┬────┘    └────┬─────┘  └─────┬─────┘
        │             │               │              │              │
        │             │               │              │              │
        ▼             ▼               ▼              │              │
   ┌────────────────────────────────────┐            │              │
   │    Model Discovery System          │            │              │
   │  • Root models (models.json)       │            │              │
   │  • Directory models (scripts/)     │            │              │
   │  • Dynamic models (get_models.py)  │            │              │
   └────────────────────────────────────┘            │              │
                     │                               │              │
                     ▼                               │              │
        ┌────────────────────────┐                   │              │
        │  Orchestration Layer   │                   │              │
        │  • BuildOrchestrator   │◄───────────────---┘              │
        │  • RunOrchestrator     │                                  │
        └────────┬───────────────┘                                  │
                 │                                                  │
        ┌────────┼───────────────┐                                  │
        │        │        │      │                                  │
   ┌────▼───┐ ┌─▼──────┐ ┌▼─────────┐ ┌▼────────────┐               │
   │ Local  │ │  K8s   │ │  SLURM   │ │ Bare Metal │               │
   │ Docker │ │  Jobs  │ │  Jobs    │ │     VM     │               │
   └────┬───┘ └─┬──────┘ └┬─────────┘ └┬────────────┘               │
        │       │         │            │                            │
        └───────┼─────────┴────────────┘                            │
                │                                                   │
        ┌───────┴─────────┐                                         │
        │   Distributed   │                                         │
        │    Launchers    │                                         │
        └───────┬─────────┘                                         │
                │                                                   │
     ┌──────────┼──────────┐                                        │
     │          │          │                                        │
  ┌──▼───┐  ┌──▼───┐  ┌──▼───┐                                      │
  │Train │  │Train │  │Infer │                                      │
  │      │  │      │  │      │                                      │
  └──┬───┘  └──┬───┘  └──┬───┘                                      │
     │         │         │                                          │
  torchrun  DeepSpeed  vLLM                                         │
  TorchTitan Megatron  SGLang                                       │
             -LM       (Disagg)                                     │
                │                                                   │
                ▼                                                   │
        ┌────────────────┐                                          │
        │ Performance    │                                          │
        │ Output         │                                          │
        │ (CSV/JSON)     │                                          │
        └────┬───────────┘                                          │
             │                                                      │
             └──────────────┬────────────────────────────────────---┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
         ┌────▼─────┐              ┌─────▼──────┐
         │ Reporting│              │  Database  │
         │ • to-html│              │  • MongoDB │
         │ • to-email              │  • Upload  │
         └──────────┘              └────────────┘
```

**Component Flow:**

1. **CLI Layer** - User interface with 5 commands (discover, build, run, report, database)
2. **Model Discovery** - Find and validate models from MAD package
3. **Orchestration** - BuildOrchestrator & RunOrchestrator manage workflows
4. **Execution Targets** - Local Docker, Kubernetes Jobs, SLURM Jobs, or Bare Metal VM
5. **Distributed Launchers** - Training (torchrun, DeepSpeed, TorchTitan, Megatron-LM) and Inference (vLLM, SGLang)
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
| **vLLM** | ✅ | ✅ | ✅ | Inference | v1 engine, PagedAttention, Ray cluster |
| **SGLang** | ✅ | ✅ | ✅ | Inference | RadixAttention, structured generation |
| **SGLang Disagg** | ❌ | ✅ | ✅ | Inference | Disaggregated prefill/decode, Mooncake, 3+ nodes |

**Note:** All launchers support single-GPU, multi-GPU (single node), and multi-node (where infrastructure allows). See [Launchers Guide](docs/launchers.md) for details.

### Parallelism Capabilities

| Launcher | Data Parallel | Tensor Parallel | Pipeline Parallel | Context Parallel | Ray Cluster | Architecture |
|----------|--------------|----------------|-------------------|-----------------|-------------|--------------|
| **torchrun** | ✅ DDP/FSDP | ❌ | ❌ | ❌ | ❌ | Unified |
| **DeepSpeed** | ✅ ZeRO | ❌ | ✅ | ❌ | ❌ | Unified |
| **Megatron-LM** | ✅ | ✅ | ✅ | ❌ | ❌ | Unified |
| **TorchTitan** | ✅ FSDP2 | ✅ | ✅ | ✅ | ❌ | Unified |
| **vLLM** | ❌ | ✅ | ✅ | ❌ | ✅ Multi-node | Unified |
| **SGLang** | ❌ | ✅ | ❌ | ❌ | ✅ Multi-node | Unified |
| **SGLang Disagg** | ❌ | ✅ | ✅ (via disagg) | ❌ | ✅ Multi-node | Disaggregated |

### Infrastructure Capabilities

| Feature | Local | Kubernetes | SLURM | Bare Metal VM |
|---------|-------|-----------|-------|---------------|
| **Execution** | Docker containers | K8s Jobs | SLURM jobs | Ephemeral VMs |
| **Multi-Node** | ❌ | ✅ Indexed Jobs | ✅ Job arrays | ❌ (single-node) |
| **Resource Mgmt** | Manual | Declarative (YAML) | Batch scheduler | VM isolation |
| **Monitoring** | Docker logs | kubectl/dashboard | squeue/scontrol | VM + Docker logs |
| **Auto-scaling** | ❌ | ✅ | ❌ | ❌ |
| **Network** | Host | CNI plugin | InfiniBand/Ethernet | VM networking |
| **GPU Support** | Passthrough | Device plugin | Direct | SR-IOV/VFIO |
| **Cleanup** | Manual | Automatic | Manual | Guaranteed |

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

### Bare Metal VM Execution

```bash
# SSH to bare metal node
ssh admin@baremetal-gpu-node.example.com

# Create config with VM isolation
cat > baremetal-vm-config.json << 'EOF'
{
  "baremetal_vm": {
    "enabled": true,
    "base_image": "/var/lib/libvirt/images/ubuntu-22.04-rocm.qcow2",
    "vcpus": 32,
    "memory": "128G",
    "gpu_passthrough": {
      "mode": "sriov",
      "gpu_vendor": "AMD"
    }
  },
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU"
}
EOF

# Run with VM isolation (guaranteed cleanup)
madengine run --tags model \
  --additional-context-file baremetal-vm-config.json \
  --timeout 3600
```

**Benefits:**
- ✅ Guaranteed clean state after each run
- ✅ Complete environment isolation
- ✅ Near-native GPU performance (95-98%)
- ✅ Works with existing Docker images

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

# Library tracing (rocBLAS, MIOpen, Tensile, RCCL)
madengine run --tags model \
  --additional-context '{"tools": [{"name": "rocblas_trace"}]}'

# Power and VRAM monitoring
madengine run --tags model \
  --additional-context '{"tools": [
    {"name": "gpu_info_power_profiler"},
    {"name": "gpu_info_vram_profiler"}
  ]}'

# Multiple tools (stackable)
madengine run --tags model \
  --additional-context '{"tools": [
    {"name": "rocprof"},
    {"name": "rocblas_trace"},
    {"name": "gpu_info_power_profiler"}
  ]}'
```

**Available Tools:**

| Tool | Purpose | Output |
|------|---------|--------|
| `rocprof` | GPU kernel profiling | Kernel timings, occupancy |
| `rocblas_trace` | rocBLAS library calls | Function calls, arguments |
| `miopen_trace` | MIOpen library calls | Conv/pooling operations |
| `tensile_trace` | Tensile GEMM library | Matrix multiply details |
| `rccl_trace` | RCCL collective ops | Communication patterns |
| `gpu_info_power_profiler` | GPU power consumption | Power usage over time |
| `gpu_info_vram_profiler` | GPU memory usage | VRAM utilization |
| `therock_check` | TheRock ROCm validation | Installation detection |

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

### Build & Deployment

- **Separate build and run phases** for distributed deployments
- **Use registries** for multi-node execution (K8s/SLURM)
- **Use batch build mode** for CI/CD to optimize build times
- **Specify `--target-archs`** when building for multiple GPU architectures

### Performance

- **Start with small timeouts** and increase as needed
- **Use profiling tools** to identify bottlenecks
- **Monitor GPU utilization** with `gpu_info_power_profiler`
- **Profile library calls** with rocBLAS/MIOpen tracing

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

**Common Issues:**
- **False failures with profiling**: If models show FAILURE but have performance metrics, see [Profiling Troubleshooting](docs/profiling.md#false-failure-detection-with-rocprof)
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
- **[Deployment Guide](docs/deployment.md)** - Kubernetes/SLURM/Bare Metal VM deployment
- **[Bare Metal VM Guide](docs/baremetal-vm.md)** - VM-based execution with isolation
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

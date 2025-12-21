# madengine

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/ROCm/madengine/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Version](https://img.shields.io/badge/version-2.0-brightgreen.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **AI model automation and benchmarking platform for local and distributed execution**

madengine is a modern CLI tool for running Large Language Models (LLMs) and Deep Learning models across local and distributed environments. Built for the [MAD (Model Automation and Dashboarding)](https://github.com/ROCm/MAD) ecosystem, it provides seamless execution from single GPUs to multi-node clusters.

## ✨ Key Features

- **🚀 Modern CLI** - Rich terminal output with Typer and Rich
- **🎯 Simple Deployment** - Run locally or deploy to Kubernetes/SLURM via configuration
- **🔧 Distributed Launchers** - Full support for torchrun, DeepSpeed, Megatron-LM, TorchTitan, vLLM, SGLang
- **🐳 Container-Native** - Docker-based execution with GPU support (ROCm, CUDA)
- **📊 Performance Tools** - Integrated profiling with rocprof, rocblas, MIOpen, RCCL tracing
- **⚙️ Intelligent Defaults** - Minimal K8s configs with automatic preset application

## 🚀 Quick Start

```bash
# Install madengine
pip install git+https://github.com/ROCm/madengine.git

# Clone MAD package (required for models)
git clone https://github.com/ROCm/MAD.git && cd MAD

# Discover available models
madengine-cli discover --tags dummy

# Run locally
madengine-cli run --tags dummy \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

**Results saved to `perf_entry.csv`**

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Complete installation instructions |
| [Usage Guide](docs/usage.md) | Commands, workflows, and examples |
| [Deployment](docs/deployment.md) | Kubernetes and SLURM deployment |
| [Configuration](docs/configuration.md) | Advanced configuration options |
| [Launchers](docs/launchers.md) | Distributed training frameworks |
| [Profiling](docs/profiling.md) | Performance analysis tools |
| [Contributing](docs/contributing.md) | How to contribute |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│               madengine-cli                     │
│          (build, run, discover)                 │
└─────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │  Build  │   │   Run   │   │Discover │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │              │
        ▼             ▼              ▼
┌─────────────────────────────────────────────────┐
│           Orchestration Layer                   │
│   (BuildOrchestrator / RunOrchestrator)         │
└─────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │  Local  │   │   K8s   │   │  SLURM  │
   │Container│   │  Deploy │   │  Deploy │
   └─────────┘   └─────────┘   └─────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   torchrun      DeepSpeed        vLLM
   TorchTitan    Megatron-LM      SGLang
```

## 🎯 Feature Matrix

### Supported Launchers & Infrastructure

| Launcher | Local | Kubernetes | SLURM | Type | Key Features |
|----------|-------|-----------|-------|------|--------------|
| **torchrun** | ✅ | ✅ | ✅ | Training | PyTorch DDP/FSDP, elastic training |
| **DeepSpeed** | ✅ | ✅ | ✅ | Training | ZeRO optimization, pipeline parallelism |
| **Megatron-LM** | ✅ | ❌ | ✅ | Training | Tensor+Pipeline parallel, large transformers |
| **TorchTitan** | ✅ | ✅ | ✅ | Training | FSDP2+TP+PP+CP, Llama 3.1 (8B-405B) |
| **vLLM** | ✅ | ✅ | ✅ | Inference | v1 engine, PagedAttention, Ray cluster |
| **SGLang** | ✅ | ✅ | ✅ | Inference | RadixAttention, structured generation |

**Note:** All launchers support single-GPU, multi-GPU (single node), and multi-node (where infrastructure allows). See [Launchers Guide](docs/launchers.md) for details.

### Parallelism Capabilities

| Launcher | Data Parallel | Tensor Parallel | Pipeline Parallel | Context Parallel | Ray Cluster |
|----------|--------------|----------------|-------------------|-----------------|-------------|
| **torchrun** | ✅ DDP/FSDP | ❌ | ❌ | ❌ | ❌ |
| **DeepSpeed** | ✅ ZeRO | ❌ | ✅ | ❌ | ❌ |
| **Megatron-LM** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **TorchTitan** | ✅ FSDP2 | ✅ | ✅ | ✅ | ❌ |
| **vLLM** | ❌ | ✅ | ✅ | ❌ | ✅ Multi-node |
| **SGLang** | ❌ | ✅ | ❌ | ❌ | ✅ Multi-node |

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
madengine-cli run --tags model \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Multi-GPU with torchrun
madengine-cli run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "docker_gpus": "0,1,2,3",
    "distributed": {
      "launcher": "torchrun",
      "nproc_per_node": 4
    }
  }'
```

### Kubernetes Deployment

```bash
# Minimal config (auto-defaults)
madengine-cli run --tags model \
  --additional-context '{"k8s": {"gpu_count": 2}}'

# Multi-node with vLLM
madengine-cli run --tags model \
  --additional-context '{
    "k8s": {"gpu_count": 8},
    "distributed": {
      "launcher": "vllm",
      "nnodes": 2,
      "nproc_per_node": 4
    }
  }'
```

### SLURM Deployment

```bash
# Multi-node with TorchTitan
madengine-cli run --tags model \
  --additional-context '{
    "slurm": {
      "partition": "gpu",
      "nodes": 4,
      "gpus_per_node": 8
    },
    "distributed": {
      "launcher": "torchtitan",
      "nnodes": 4,
      "nproc_per_node": 8
    }
  }'
```

See [Usage Guide](docs/usage.md) and [Configuration Guide](docs/configuration.md) for more examples.

### Building Images

```bash
# Build with tags
madengine-cli build --tags model1 model2 \
  --registry localhost:5000 \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Batch build mode (selective builds for CI/CD)
madengine-cli build --batch-manifest examples/build-manifest/batch.json \
  --registry docker.io/myorg
```

See [Batch Build Guide](docs/batch-build.md) and examples in [`examples/build-manifest/`](examples/build-manifest/).

## 🔍 Model Discovery

madengine discovers models from the MAD package using three methods:

```bash
# Root models (models.json)
madengine-cli discover --tags pyt_huggingface_bert

# Directory-specific (scripts/{dir}/models.json)
madengine-cli discover --tags dummy2:dummy_2

# Dynamic with parameters (scripts/{dir}/get_models_json.py)
madengine-cli discover --tags dummy3:dummy_3:batch_size=512
```

## 📊 Performance Profiling

```bash
# GPU profiling
madengine-cli run --tags model \
  --additional-context '{"tools": [{"name": "rocprof"}]}'

# Library tracing (rocBLAS, MIOpen, Tensile, RCCL)
madengine-cli run --tags model \
  --additional-context '{"tools": [{"name": "rocblas_trace"}]}'

# Power and VRAM monitoring
madengine-cli run --tags model \
  --additional-context '{"tools": [{"name": "gpu_info_power_profiler"}]}'
```

**Available Tools:** rocprof, rocblas_trace, miopen_trace, tensile_trace, rccl_trace, gpu_info_power_profiler, gpu_info_vram_profiler

See [Profiling Guide](docs/profiling.md) for details.

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

## 🤝 Contributing

We welcome contributions! See [Contributing Guide](docs/contributing.md) for details.

```bash
git clone https://github.com/ROCm/madengine.git
cd madengine
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **MAD Package**: https://github.com/ROCm/MAD
- **Issues**: https://github.com/ROCm/madengine/issues
- **ROCm**: https://rocm.docs.amd.com/

---

**Note:** For legacy `madengine` CLI (v1.x), see [Legacy CLI Guide](docs/legacy-cli.md). New projects should use `madengine-cli`.

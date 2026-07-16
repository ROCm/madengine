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

madengine is a modern CLI tool for running Large Language Models (LLMs) and Deep Learning models across local and distributed environments. Built for the [MAD (Model Automation and Dashboarding)](https://github.com/ROCm/MAD) ecosystem, it provides seamless execution from single GPUs to multi-node clusters — with the same command working locally, on Kubernetes, and on SLURM.

## ✨ Key Features

- **🚀 Modern CLI** — Rich terminal output with Typer and Rich
- **🎯 Simple Deployment** — Run locally or deploy to Kubernetes/SLURM by adding a config key; no code changes
- **🔧 Distributed Launchers** — torchrun, DeepSpeed, Megatron-LM, TorchTitan, Primus, vLLM, SGLang
- **🐳 Container-Native** — Docker-based execution with GPU support (ROCm, CUDA)
- **📊 Performance Tools** — Integrated profiling with rocprof/rocprofv3, [rocm-trace-lite](https://github.com/sunway513/rocm-trace-lite), rocBLAS/MIOpen/RCCL tracing → see [Profiling](docs/profiling.md)
- **⚙️ Intelligent Defaults** — Minimal configs auto-merged with presets; host/in-container ROCm path auto-detected → see [Configuration](docs/configuration.md#rocm-path-run-only)

## 🚀 Quick Start

```bash
# Install madengine
pip install git+https://github.com/ROCm/madengine.git

# Clone the MAD package (required for models)
git clone https://github.com/ROCm/MAD.git && cd MAD

# Discover available models
madengine discover --tags dummy

# Run locally (discover → build → run, as configured by the model)
madengine run --tags dummy
```

> **Note:** For build operations `gpu_vendor` defaults to `AMD` and `guest_os` to `UBUNTU`. For non-AMD/Ubuntu environments, set them explicitly, e.g. `--additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'`.

**Results:** Performance data is written to `perf.csv` (and optionally `perf_entry.csv`), created automatically if missing. Failed runs are recorded with status `FAILURE` so every attempted model appears. See [Exit Codes](docs/cli-reference.md#exit-codes) for CI usage. If ROCm isn't auto-detected, set `MAD_ROCM_PATH` — see [Configuration](docs/configuration.md#rocm-path-run-only).

## 🏗️ Architecture

madengine is organized in layers: the CLI drives orchestrators that discover and build models, then hand off to a local or distributed execution target, which runs the model under the appropriate launcher and emits performance data for reporting.

```mermaid
flowchart TB
    subgraph CLI["CLI Layer — Typer + Rich"]
        C1[discover]
        C2[build]
        C3[run]
        C4[report]
        C5[database]
    end

    subgraph ORC["Orchestration Layer"]
        O1[DiscoverModels]
        O2[BuildOrchestrator]
        O3[RunOrchestrator]
        MAN[(build_manifest.json)]
    end

    subgraph EXEC["Execution / Deployment Layer"]
        E1[ContainerRunner<br/>local Docker]
        E2[DeploymentFactory]
        K8S[Kubernetes Jobs]
        SLURM[SLURM Jobs]
    end

    subgraph LAUNCH["Launcher Layer"]
        T[Train: torchrun · DeepSpeed<br/>Megatron-LM · TorchTitan · Primus]
        I[Infer: vLLM · SGLang · SGLang Disagg]
    end

    OUT[(perf.csv / JSON)]

    C1 --> O1
    C2 --> O2
    C3 --> O3
    O2 --> MAN --> O3
    O1 --> O2
    O3 --> E1
    O3 --> E2
    E2 --> K8S
    E2 --> SLURM
    E1 --> LAUNCH
    K8S --> LAUNCH
    SLURM --> LAUNCH
    LAUNCH --> OUT
    OUT --> C4
    OUT --> C5
```

1. **CLI Layer** — five commands: `discover`, `build`, `run`, `report`, `database`
2. **Orchestration** — `DiscoverModels` finds models; `BuildOrchestrator` builds images and writes `build_manifest.json`; `RunOrchestrator` reads/triggers the build and infers the target
3. **Execution / Deployment** — local `ContainerRunner`, or `DeploymentFactory` → Kubernetes / SLURM
4. **Launchers** — distributed training and inference frameworks
5. **Output & Post-Processing** — `perf.csv`/JSON results → `report` (HTML/email) and `database` (MongoDB)

## 🔄 Workflow

The core pipeline is the same everywhere: discover models, build images once, run them against a target, then report. Build and run can be separated so images are built once (e.g. in CI) and reused across nodes.

```mermaid
flowchart LR
    D[discover<br/>find models by tag] --> B[build<br/>Docker images]
    B --> M[(build_manifest.json)]
    M --> R[run<br/>infer target + execute]
    R --> P[(perf.csv)]
    P --> RP[report<br/>HTML / email]
    P --> DB[database<br/>MongoDB]
```

**Deployment target is inferred from the config** (Convention over Configuration) — no `deploy` flag needed:

```mermaid
flowchart TD
    A[additional_context] --> Q{which key?}
    Q -->|k8s / kubernetes| K[Kubernetes deployment]
    Q -->|slurm| S[SLURM deployment]
    Q -->|neither| L[Local Docker execution]
```

## 📋 Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| **[discover](docs/usage.md#model-discovery)** | Find available models | Model exploration and validation |
| **[build](docs/usage.md#build-workflow)** | Build Docker images | Create containerized models |
| **[run](docs/usage.md#run-workflow)** | Execute models | Local and distributed execution |
| **[report](docs/cli-reference.md#report---generate-reports)** | Generate HTML/email reports | Convert CSV to viewable reports |
| **[database](docs/cli-reference.md#database---upload-to-mongodb)** | Upload to MongoDB | Store results in a database |

```bash
madengine discover --tags dummy          # Find models
madengine build --tags dummy             # Build image (AMD/UBUNTU defaults)
madengine run --tags dummy               # Run model
madengine report to-html --csv-file perf_entry.csv          # Report
madengine database --csv-file perf_entry.csv --db mydb --collection results  # Upload
```

For all options and examples, see the **[CLI Reference](docs/cli-reference.md)**.

## 💻 Usage Examples

```bash
# Local, multi-GPU with torchrun (DDP/FSDP)
madengine run --tags model \
  --additional-context '{"docker_gpus": "0,1,2,3",
    "distributed": {"launcher": "torchrun", "nproc_per_node": 4}}'

# Kubernetes (minimal config, presets auto-applied)
madengine run --tags model \
  --additional-context '{"k8s": {"gpu_count": 2}}'

# SLURM (build once, then deploy)
madengine build --tags model --registry gcr.io/myproject
madengine run --manifest-file build_manifest.json \
  --additional-context '{"slurm": {"partition": "gpu", "nodes": 4, "gpus_per_node": 8},
    "distributed": {"launcher": "torchtitan", "nnodes": 4, "nproc_per_node": 8}}'
```

More local/K8s/SLURM/CI recipes: [Usage Guide](docs/usage.md) · [Configuration](docs/configuration.md) · [CLI Reference](docs/cli-reference.md).

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Complete installation instructions |
| [Usage Guide](docs/usage.md) | Commands, workflows, and examples |
| **[CLI Reference](docs/cli-reference.md)** | **Detailed command options and examples** |
| [Configuration](docs/configuration.md) | Advanced options, ROCm path, log error scan |
| [Deployment](docs/deployment.md) | Kubernetes and SLURM deployment |
| [Batch Build](docs/batch-build.md) | Selective builds for CI/CD |
| [Launchers](docs/launchers.md) | Distributed frameworks + capability matrices |
| [Profiling](docs/profiling.md) | Performance analysis tools |
| [Contributing](docs/contributing.md) | How to contribute |

## 🎯 Supported Launchers

| Launcher | Local | Kubernetes | SLURM | Type | Key Features |
|----------|-------|-----------|-------|------|--------------|
| **torchrun** | ✅ | ✅ | ✅ | Training | PyTorch DDP/FSDP, elastic training |
| **DeepSpeed** | ✅ | ✅ | ✅ | Training | ZeRO optimization, pipeline parallelism |
| **Megatron-LM** | ✅ | ✅ | ✅ | Training | Tensor+Pipeline parallel, large transformers |
| **TorchTitan** | ✅ | ✅ | ✅ | Training | FSDP2+TP+PP+CP, Llama 3.1 (8B–405B) |
| **Primus** | ✅ | ✅ | ✅ | Training | Megatron / TorchTitan / MaxText via Primus YAML |
| **vLLM** | ✅ | ✅ | ✅ | Inference | v1 engine, PagedAttention, Ray cluster |
| **SGLang** | ✅ | ✅ | ✅ | Inference | RadixAttention, structured generation |
| **SGLang Disagg** | ❌ | ✅ | ✅ | Inference | Disaggregated prefill/decode, Mooncake, 3+ nodes |

All launchers support single-GPU, multi-GPU, and multi-node (where infrastructure allows). See the [Launchers Guide](docs/launchers.md) for the full **parallelism** and **infrastructure** capability matrices.

## 📊 Profiling

madengine ships integrated profiling for AMD ROCm — `rocprof`, eight pre-configured `rocprofv3` profiles (ROCm 7.0+), `rocm-trace-lite`, library tracing (rocBLAS/MIOpen/Tensile/RCCL), and power/VRAM monitors. Tools are stackable via `--additional-context '{"tools": [...]}'`.

```bash
madengine run --tags model --additional-context '{"tools": [{"name": "rocprofv3_compute"}]}'
```

See the [Profiling Guide](docs/profiling.md) and ready-to-use configs in [`examples/profiling-configs/`](examples/profiling-configs/).

## 📦 Installation

```bash
# Install (all dependencies, including Kubernetes support, included)
pip install git+https://github.com/ROCm/madengine.git

# Development installation
git clone https://github.com/ROCm/madengine.git
cd madengine && pip install -e .
```

See the [Installation Guide](docs/installation.md) for details.

## 💡 Tips & Troubleshooting

- **Test locally first** with a single GPU before scaling to multi-node; use config files for complex setups.
- **Debugging:** add `--verbose --live-output`; keep a container for inspection with `--keep-alive`.
- **CI:** the CLI uses fixed [exit codes](docs/cli-reference.md#exit-codes) (`0` success, `2` build failure, `3` run failure, `4` invalid args) — no log scraping needed.
- **Log error scan** can flag benign `RuntimeError:` text; disable or tune it via `log_error_pattern_scan` / `log_error_benign_patterns` — see [Configuration](docs/configuration.md#run-phase-log-error-pattern-scan).

More: [Usage — Troubleshooting](docs/usage.md#troubleshooting) · [Profiling — False failures](docs/profiling.md#false-failure-detection-with-rocprof).

## 🤝 Contributing

Contributions are welcome! See the [Contributing Guide](docs/contributing.md).

```bash
git clone https://github.com/ROCm/madengine.git
cd madengine && python3 -m venv venv && source venv/bin/activate
pip install -e .
pytest
```

## 📄 License

MIT License — see [LICENSE](LICENSE).

## 🔗 Links & Resources

- **MAD Package:** https://github.com/ROCm/MAD
- **Issues & Support:** https://github.com/ROCm/madengine/issues
- **ROCm Documentation:** https://rocm.docs.amd.com/
- **Command help:** `madengine --help` · `madengine <command> --help`

---

## ⚠️ Migration Notice (v2.0.0+)

The CLI has been unified. Starting from v2.0.0, use `madengine` (with K8s, SLURM, and distributed support); the legacy v1.x CLI has been removed.

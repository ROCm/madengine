# MADEngine Framework - Complete Architecture & Flow Documentation

> **Purpose**: Comprehensive architecture documentation for refactoring the madengine framework

**Document Version**: 1.0  
**Last Updated**: November 28, 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Directory Structure](#3-directory-structure)
4. [CLI Entry Points](#4-cli-entry-points)
5. [Core Component Flows](#5-core-component-flows)
6. [Distributed Orchestrator Flow](#6-distributed-orchestrator-flow)
7. [Distributed Runner Flows](#7-distributed-runner-flows)
8. [Complete Command Flow Examples](#8-complete-command-flow-examples)
9. [Key Data Structures](#9-key-data-structures)
10. [Refactoring Recommendations](#10-refactoring-recommendations)
11. [Execution Flow Diagrams](#11-execution-flow-diagrams)

---

## 1. PROJECT OVERVIEW

**madengine** is an enterprise-grade AI model automation and distributed benchmarking platform designed to:
- Build and run AI models (LLMs, Deep Learning) in Docker containers
- Support both local single-node and distributed multi-node execution
- Integrate with MAD (Model Automation and Dashboarding) ecosystem
- Provide split build/run architecture for optimal resource utilization

### Key Philosophy

**Separate Docker image building (CPU-intensive) from model execution (GPU-intensive)** for distributed scenarios.

### Core Capabilities

- **Dual CLI Interface**: Legacy (argparse) + Modern (Typer+Rich)
- **Model Discovery**: Static, directory-specific, and dynamic Python-based discovery
- **Docker Integration**: Full containerization with GPU support (ROCm, CUDA, Intel)
- **Distributed Execution**: SSH, Ansible, Kubernetes, and SLURM runners
- **Split Architecture**: Separate build/run phases optimized for different infrastructure

---

## 2. HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MADEngine Framework                        │
│                                                                     │
│  ┌───────────────────┐              ┌───────────────────┐         │
│  │   Legacy CLI      │              │   Modern CLI      │         │
│  │   (mad.py)        │              │   (mad_cli.py)    │         │
│  │   - argparse      │              │   - Typer + Rich  │         │
│  │   - simple cmds   │              │   - distributed   │         │
│  └─────────┬─────────┘              └─────────┬─────────┘         │
│            │                                   │                    │
│            └───────────────┬───────────────────┘                    │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Core Components Layer                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│  │  │ Context      │  │ Console      │  │ DataProvider │     │  │
│  │  │ - GPU detect │  │ - Output     │  │ - Data mgmt  │     │  │
│  │  │ - Env vars   │  │ - Logging    │  │ - Credentials│     │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Tools/Orchestration Layer                      │  │
│  │  ┌───────────────────┐  ┌────────────────────┐             │  │
│  │  │ DiscoverModels    │  │ DockerBuilder      │             │  │
│  │  │ - Find models     │  │ - Build images     │             │  │
│  │  │ - Parse tags      │  │ - Push to registry │             │  │
│  │  └───────────────────┘  └────────────────────┘             │  │
│  │  ┌───────────────────┐  ┌────────────────────┐             │  │
│  │  │ ContainerRunner   │  │ Distributed        │             │  │
│  │  │ - Run containers  │  │ Orchestrator       │             │  │
│  │  │ - Collect metrics │  │ - Build/Run phases │             │  │
│  │  └───────────────────┘  └────────────────────┘             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Distributed Runners Layer                      │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐            │  │
│  │  │  SSH   │  │Ansible │  │  K8s   │  │ SLURM  │            │  │
│  │  │ Runner │  │ Runner │  │ Runner │  │ Runner │            │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘            │  │
│  │              (RunnerFactory manages all)                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. DIRECTORY STRUCTURE

```
madengine/
├── src/madengine/
│   ├── __init__.py                 # Package initialization
│   ├── mad.py                      # Legacy CLI entry point (argparse)
│   ├── mad_cli.py                  # Modern CLI entry point (Typer+Rich)
│   │
│   ├── core/                       # Core system components
│   │   ├── console.py              # Output and logging management
│   │   ├── context.py              # GPU/OS detection, env management
│   │   ├── constants.py            # System constants
│   │   ├── dataprovider.py         # Data source management
│   │   ├── docker.py               # Docker client wrapper
│   │   ├── errors.py               # Error handling framework
│   │   └── timeout.py              # Timeout management
│   │
│   ├── tools/                      # CLI tool implementations
│   │   ├── discover_models.py      # Model discovery engine
│   │   ├── docker_builder.py       # Docker image builder
│   │   ├── container_runner.py     # Container execution engine
│   │   ├── distributed_orchestrator.py  # Build/run orchestration
│   │   ├── run_models.py           # Legacy run command
│   │   ├── csv_to_html.py          # Report generation
│   │   ├── csv_to_email.py         # Email reporting
│   │   ├── update_perf_csv.py      # Performance metrics
│   │   └── *_db.py                 # Database operations
│   │
│   ├── runners/                    # Distributed execution runners
│   │   ├── base.py                 # Abstract base runner
│   │   ├── factory.py              # Runner factory pattern
│   │   ├── ssh_runner.py           # SSH-based execution
│   │   ├── ansible_runner.py       # Ansible orchestration
│   │   ├── k8s_runner.py           # Kubernetes execution
│   │   ├── slurm_runner.py         # HPC/SLURM execution
│   │   ├── orchestrator_generation.py  # Config generators
│   │   ├── template_generator.py   # Template engine
│   │   └── templates/              # Jinja2 templates
│   │
│   ├── utils/                      # Utility functions
│   │   ├── gpu_validator.py        # GPU detection/validation
│   │   ├── ops.py                  # Common operations
│   │   └── log_formatting.py       # Log formatting
│   │
│   └── db/                         # Database layer
│       ├── database.py             # Database connection
│       ├── database_functions.py   # DB operations
│       └── upload_csv_to_db.py     # CSV upload
│
├── tests/                          # Test suite (95%+ coverage)
├── docs/                           # Documentation
├── pyproject.toml                  # Modern Python packaging
├── README.md                       # Comprehensive documentation
└── DEVELOPER_GUIDE.md              # Development guidelines
```

---

## 4. CLI ENTRY POINTS

### 4.1 Legacy CLI: `madengine` (mad.py)

**Purpose**: Backward-compatible interface for simple local workflows

**Main Commands**:
```bash
madengine run --tags <models>      # Run models locally
madengine discover --tags <models> # Discover available models
madengine report to-html           # Generate HTML report
madengine database create-table    # Database operations
madengine validate-gpu             # Validate GPU installation
```

**Flow**:
```
User Command
    ↓
mad.py (argparse parser)
    ↓
Command Router Functions (run_models, discover_models, etc.)
    ↓
Tool Classes (RunModels, DiscoverModels, etc.)
    ↓
Core Components (Context, Console, Docker)
```

**Key Components**:
- `main()`: Entry point with argparse setup
- Command routers: `run_models()`, `discover_models()`, etc.
- Direct integration with tool classes

### 4.2 Modern CLI: `madengine-cli` (mad_cli.py)

**Purpose**: Production-ready interface with distributed execution support

**Main Commands**:
```bash
# Build Phase
madengine-cli build --tags <models> --registry <url>

# Run Phase  
madengine-cli run --tags <models> --timeout <seconds>
madengine-cli run --manifest-file build_manifest.json

# Distributed Runners
madengine-cli runner ssh --inventory inventory.yml
madengine-cli runner ansible --inventory cluster.yml
madengine-cli runner k8s --inventory k8s.yml
madengine-cli runner slurm --inventory slurm.yml

# Configuration Generators
madengine-cli generate ansible --manifest-file manifest.json
madengine-cli generate k8s --manifest-file manifest.json
madengine-cli generate slurm --manifest-file manifest.json
```

**Flow**:
```
User Command
    ↓
mad_cli.py (Typer app with Rich formatting)
    ↓
Command Handlers (build_command, run_command, runner commands)
    ↓
DistributedOrchestrator
    ↓
Core Tools (DiscoverModels, DockerBuilder, ContainerRunner)
    ↓
Distributed Runners (via RunnerFactory)
```

**Key Features**:
- Typer for modern CLI with type hints
- Rich for beautiful terminal output
- Sub-applications: `generate`, `runner`
- Unified error handling with ErrorHandler

---

## 5. CORE COMPONENT FLOWS

### 5.1 Context Component (core/context.py)

**Purpose**: Manages system context (GPU vendor, OS, environment)

**Initialization Flow**:
```
Context.__init__(additional_context, build_only_mode)
    ↓
├─ Parse additional_context (JSON string or file)
├─ Read MAD_SECRETS environment variables
├─ Determine mode:
│   ├─ build_only_mode=True  → init_build_context()
│   └─ build_only_mode=False → init_runtime_context()
    ↓
init_runtime_context()
    ├─ get_host_os() → UBUNTU/CENTOS/ROCKY
    ├─ get_gpu_vendor() → AMD/NVIDIA/INTEL
    ├─ get_system_gpu_architecture() → gfx908/gfx90a/etc
    ├─ get_system_ngpus() → Number of GPUs
    ├─ get_docker_gpus() → GPU device mapping
    └─ Populate ctx dict:
        ├─ docker_build_arg: {}
        └─ docker_env_vars: {}
```

**Key Methods**:

| Method | Purpose | Return Type |
|--------|---------|-------------|
| `get_gpu_vendor()` | Detects AMD (rocm-smi), NVIDIA (nvidia-smi), INTEL | str |
| `get_system_gpu_architecture()` | Extracts GPU arch (e.g., gfx90a) | str |
| `get_host_os()` | Detects OS (UBUNTU/CENTOS/ROCKY) | str |
| `get_system_ngpus()` | Counts available GPUs | int |
| `get_docker_gpus()` | Maps GPU devices for Docker | str |
| `filter()` | Replaces placeholders in strings | str |
| `init_build_context()` | Initialize build-only context | None |
| `init_runtime_context()` | Initialize full runtime context | None |
| `ensure_runtime_context()` | Lazy initialization of runtime | None |

**Context Dictionary Structure**:
```python
ctx = {
    "host_os": "UBUNTU",
    "gpu_vendor": "AMD",
    "docker_build_arg": {
        "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a",
        "BASE_DOCKER": "rocm/pytorch:latest"
    },
    "docker_env_vars": {
        "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a",
        "ROCR_VISIBLE_DEVICES": "0,1,2,3"
    },
    "numa_balancing": "enabled",
    "n_gpus": 4
}
```

---

### 5.2 Model Discovery (tools/discover_models.py)

**Purpose**: Finds and parses model definitions from MAD package

**Discovery Flow**:
```
DiscoverModels.run()
    ↓
1. discover_models()
   ├─ Read models.json (root level)
   ├─ Walk scripts/ directory
   │   ├─ Find models.json in subdirs → Add to models list
   │   └─ Find get_models_json.py → Import and execute
   └─ Populate self.models list
    ↓
2. discover_custom_models()
   ├─ Import get_models_json.py as module
   ├─ Call get_models_json(params) function
   └─ Return CustomModel instances
    ↓
3. filter_models()
   ├─ Parse --tags argument
   │   ├─ Simple tag: "dummy"
   │   ├─ Directory tag: "dummy2:dummy_2"
   │   └─ Parameterized: "dummy3:model:batch_size=512"
   ├─ Match against discovered models
   └─ Return filtered list
    ↓
4. Return selected_models
```

**Tag System**:
```
Format: [directory]:[model_name]:[param1=value1]:[param2=value2]

Examples:
  dummy              → Root level model named "dummy"
  dummy2:dummy_2     → Model "dummy_2" in scripts/dummy2/
  dummy3:model:bs=32 → Model with batch_size=32 parameter
```

**Discovery Methods**:

| Method | Purpose | File Source |
|--------|---------|-------------|
| Root models | Static definitions at package root | `models.json` |
| Directory-specific | Organized models in subdirs | `scripts/{dir}/models.json` |
| Dynamic discovery | Python-generated configs | `scripts/{dir}/get_models_json.py` |

**Model Definition Structure**:
```python
{
    "name": "dummy",
    "dockerfile": "scripts/dummy/Dockerfile",
    "dockercontext": "./docker",
    "scripts": "scripts/dummy",
    "n_gpus": "1",
    "timeout": 3600,
    "tags": ["dummy", "test"],
    "args": "--batch-size 32",
    "cred": "AMD_GITHUB",
    "data": "model_data"
}
```

---

### 5.3 Docker Builder (tools/docker_builder.py)

**Purpose**: Builds Docker images for discovered models

**Build Flow**:
```
DockerBuilder.build_all_models(models, credentials, registry)
    ↓
For each model:
    ↓
    build_image(model_info, dockerfile, credentials)
        ↓
        1. Generate image name: ci-<model>_<dockerfile>
        2. Get docker context path
        3. Prepare build args:
           ├─ From context.ctx["docker_build_arg"]
           ├─ From credentials (if model requires)
           └─ Additional GPU arch args
        4. Build command:
           docker build [--no-cache] --network=host \
             -t <image_name> --pull -f <dockerfile> \
             <build_args> <context_path>
        5. Execute with live output to log file
        6. Get docker SHA: docker inspect --format='{{.Id}}'
        7. Return build_info dict:
           {
             "docker_image": "ci-model_dockerfile",
             "docker_sha": "sha256:...",
             "dockerfile": "path/to/Dockerfile",
             "build_duration": 123.45,
             "base_docker": "rocm/pytorch:latest"
           }
    ↓
    tag_and_push_image(docker_image, registry)
        ↓
        1. docker tag <local_image> <registry>/<image>
        2. docker push <registry>/<image>
        3. Return registry_image path
    ↓
Save build_manifest.json:
{
  "registry": "docker.io",
  "built_images": {
    "model_name": {
      "docker_image": "...",
      "docker_sha": "...",
      "registry_image": "docker.io/org/image:tag",
      "build_duration": 123.45
    }
  }
}
```

**Key Methods**:

| Method | Purpose | Output |
|--------|---------|--------|
| `build_image()` | Build Docker image for model | build_info dict |
| `tag_and_push_image()` | Tag and push to registry | registry_image path |
| `build_all_models()` | Build multiple models | Summary dict |
| `get_build_arg()` | Prepare Docker build args | Build arg string |
| `get_context_path()` | Get Docker build context | Context path |

**GPU Architecture Variables**:
The builder handles multiple GPU architecture variables used in MAD/DLM Dockerfiles:
- `MAD_SYSTEM_GPU_ARCHITECTURE`
- `PYTORCH_ROCM_ARCH`
- `GPU_TARGETS`
- `GFX_COMPILATION_ARCH`
- `GPU_ARCHS`

---

### 5.4 Container Runner (tools/container_runner.py)

**Purpose**: Executes Docker containers and collects performance metrics

**Execution Flow**:
```
ContainerRunner.run_models_from_manifest(manifest_file)
    ↓
1. load_build_manifest(manifest_file)
   ├─ Read build_manifest.json
   └─ Extract built_images dict
    ↓
2. login_to_registry(registry, credentials)
   ├─ docker login <registry>
   └─ Use credentials from credential.json or env vars
    ↓
3. For each model in manifest:
    ↓
    pull_image(registry_image)
        ├─ docker pull <registry_image>
        └─ Verify image exists locally
    ↓
    run_single_model(model_info, build_info)
        ↓
        a) Prepare Docker run command:
           docker run --rm \
             --device=/dev/kfd --device=/dev/dri \
             --group-add video \
             -v <model_dir>:/workspace \
             -e MAD_SYSTEM_GPU_ARCHITECTURE=<arch> \
             -e ROCR_VISIBLE_DEVICES=<devices> \
             <docker_image> \
             bash -c "cd /workspace && ./run.sh"
        
        b) Execute container with timeout
           ├─ Redirect stdout/stderr to log file
           ├─ Monitor execution
           └─ Capture exit code
        
        c) Parse performance output:
           ├─ Look for "Performance:" in stdout
           ├─ Extract metric value
           └─ Parse multiple_results if configured
        
        d) Create run_details dict:
           {
             "model": "model_name",
             "status": "SUCCESS/FAILURE",
             "performance": "123.45",
             "metric": "tokens/sec",
             "test_duration": 45.67,
             "gpu_architecture": "gfx90a",
             ...
           }
    ↓
4. Update perf.csv with results
   ├─ Call update_perf_csv()
   └─ Append row to performance CSV
    ↓
5. Return execution summary
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `load_build_manifest()` | Load manifest from JSON file |
| `login_to_registry()` | Authenticate with Docker registry |
| `pull_image()` | Pull Docker image from registry |
| `run_single_model()` | Execute single model container |
| `run_models_from_manifest()` | Execute all models from manifest |
| `create_run_details_dict()` | Create performance record |
| `ensure_perf_csv_exists()` | Initialize CSV with headers |

---

## 6. DISTRIBUTED ORCHESTRATOR FLOW

### 6.1 Build-Only Phase

**Command**: `madengine-cli build --tags dummy --registry docker.io`

```
DistributedOrchestrator(build_only_mode=True)
    ↓
build_phase()
    ↓
    1. Initialize Context (build_only_mode=True)
       ├─ Skip GPU detection
       └─ Use provided docker_build_arg
    ↓
    2. Discover Models
       ├─ DiscoverModels.run()
       └─ Get list of models to build
    ↓
    3. Build All Images
       ├─ DockerBuilder.build_all_models()
       ├─ For each model: build + tag + push
       └─ Track built_images
    ↓
    4. Generate build_manifest.json
       {
         "registry": "docker.io",
         "built_images": {...},
         "build_context": {...}
       }
    ↓
    5. Return build summary
```

**Use Case**: Build Docker images on CPU-only nodes without GPU requirements.

---

### 6.2 Run-Only Phase

**Command**: `madengine-cli run --manifest-file build_manifest.json`

```
DistributedOrchestrator(build_only_mode=False)
    ↓
run_phase(manifest_file)
    ↓
    1. Initialize Context (runtime mode)
       ├─ Detect GPU vendor and architecture
       └─ Setup docker_env_vars
    ↓
    2. Load build_manifest.json
       └─ Extract built_images and registry
    ↓
    3. Login to Registry
       └─ docker login <registry>
    ↓
    4. Run All Models
       ├─ ContainerRunner.run_models_from_manifest()
       ├─ Pull each image
       ├─ Execute containers
       └─ Collect performance metrics
    ↓
    5. Generate perf.csv
    ↓
    6. Return execution summary
```

**Use Case**: Execute pre-built images on GPU nodes.

---

### 6.3 Full Workflow (Build + Run)

**Command**: `madengine-cli run --tags dummy --registry localhost:5000`

```
Intelligent Workflow Detection:
    ├─ No manifest_file provided
    ├─ Tags provided
    └─ Decision: Execute full workflow
    ↓
full_workflow()
    ↓
    1. Execute build_phase()
       ├─ Build all images
       ├─ Push to registry
       └─ Generate manifest
    ↓
    2. Execute run_phase(generated_manifest)
       ├─ Pull images
       ├─ Run containers
       └─ Collect metrics
    ↓
    3. Return combined summary
```

**Use Case**: Local development or single-node deployment.

---

## 7. DISTRIBUTED RUNNER FLOWS

### 7.1 Runner Factory Pattern

```
RunnerFactory.create_runner(runner_type, **kwargs)
    ↓
Registered Runners:
    ├─ "ssh"       → SSHDistributedRunner
    ├─ "ansible"   → AnsibleDistributedRunner
    ├─ "k8s"       → KubernetesDistributedRunner
    └─ "slurm"     → SlurmDistributedRunner
    ↓
Return: BaseDistributedRunner instance
```

**Registration Process**:
- `register_default_runners()` called on module import
- Each runner imports conditionally (graceful degradation)
- Factory provides `get_available_runners()` for discovery

---

### 7.2 SSH Runner Flow

**Command**: `madengine-cli runner ssh --inventory inventory.yml`

```
SSHDistributedRunner.__init__(inventory.yml)
    ↓
    1. Load inventory
       ├─ Parse YAML/JSON
       └─ Create NodeConfig objects
    ↓
    2. setup_infrastructure()
       ├─ For each node:
       │   ├─ SSH connect
       │   ├─ Clone MAD repository
       │   ├─ Setup virtual environment
       │   ├─ Install madengine
       │   ├─ Copy credential.json
       │   ├─ Copy data.json
       │   └─ Copy build_manifest.json
    ↓
    3. execute_workload()
       ├─ For each node (in parallel):
       │   ├─ SSH execute: madengine-cli run --manifest-file ...
       │   ├─ Monitor execution
       │   └─ Collect results
    ↓
    4. cleanup_infrastructure()
       └─ Collect perf.csv from each node
    ↓
    5. generate_report(runner_report.json)
```

**Key Features**:
- Direct SSH connections via paramiko
- Parallel execution across nodes
- SCP file transfer for configs and results

---

### 7.3 Ansible Runner Flow

**Command**: `madengine-cli runner ansible --inventory cluster.yml`

```
AnsibleDistributedRunner.__init__(cluster.yml)
    ↓
    1. Load Ansible inventory
    ↓
    2. setup_infrastructure()
       ├─ Generate Ansible playbook (if not provided)
       └─ Validate playbook
    ↓
    3. execute_workload()
       ├─ ansible-playbook -i inventory.yml playbook.yml
       │   Playbook tasks:
       │   ├─ Clone MAD repo on all nodes
       │   ├─ Setup Python venv
       │   ├─ Install madengine
       │   ├─ Copy configurations
       │   ├─ Execute: madengine-cli run
       │   └─ Fetch results
    ↓
    4. cleanup_infrastructure()
       └─ Aggregate results from all nodes
    ↓
    5. generate_report(ansible_results.json)
```

**Key Features**:
- Orchestrated deployment via Ansible
- Inventory management
- Rich error reporting from ansible-runner

---

### 7.4 Kubernetes Runner Flow

**Command**: `madengine-cli runner k8s --inventory k8s.yml`

```
KubernetesDistributedRunner.__init__(k8s.yml)
    ↓
    1. Load K8s inventory
       └─ Parse pod configurations
    ↓
    2. setup_infrastructure()
       ├─ Connect to K8s cluster
       ├─ Create namespace (if not exists)
       ├─ Create ConfigMaps:
       │   ├─ credential.json
       │   ├─ data.json
       │   └─ build_manifest.json
       └─ Generate Job manifests
    ↓
    3. execute_workload()
       ├─ For each model:
       │   ├─ Create K8s Job:
       │   │   spec:
       │   │     containers:
       │   │       - image: madengine-executor
       │   │         command: ["bash", "-c", "git clone MAD && ..."]
       │   │         volumeMounts:
       │   │           - name: config
       │   │             mountPath: /config
       │   ├─ kubectl apply -f job.yaml
       │   ├─ Monitor job status
       │   └─ kubectl logs job/<name>
    ↓
    4. cleanup_infrastructure()
       ├─ Collect logs from all pods
       └─ Delete jobs (optional)
    ↓
    5. generate_report(k8s_results.json)
```

**Key Features**:
- Cloud-native execution
- Dynamic Job creation
- ConfigMap management
- Namespace isolation

---

### 7.5 SLURM Runner Flow

**Step 1: Generate SLURM Configuration**

**Command**: `madengine-cli generate slurm --manifest-file manifest.json`

```
generate_slurm_setup()
    ├─ Create slurm-setup/ directory
    ├─ Generate job array script:
    │   #!/bin/bash
    │   #SBATCH --job-name=madengine
    │   #SBATCH --partition=gpu
    │   #SBATCH --gres=gpu:1
    │   #SBATCH --array=0-N  # N = number of models
    │   
    │   # Setup MAD environment
    │   git clone MAD && cd MAD
    │   python3 -m venv venv && source venv/bin/activate
    │   pip install madengine
    │   
    │   # Get model from array
    │   MODEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)
    │   
    │   # Execute
    │   madengine-cli run --manifest-file build_manifest.json \
    │     --tags $MODEL
    └─ Save job_script.sh
```

**Step 2: Submit and Monitor Jobs**

**Command**: `madengine-cli runner slurm --inventory slurm.yml`

```
SlurmDistributedRunner.__init__(slurm.yml)
    ↓
    1. Load SLURM inventory
       └─ Get login_node, partitions
    ↓
    2. setup_infrastructure()
       ├─ SSH to login node
       ├─ Copy job scripts and configs
       └─ Verify SLURM availability
    ↓
    3. execute_workload()
       ├─ sbatch job_script.sh
       ├─ Monitor: squeue -u $USER
       └─ Wait for completion
    ↓
    4. cleanup_infrastructure()
       ├─ Collect slurm-*.out logs
       └─ Aggregate results
    ↓
    5. generate_report(slurm_results.json)
```

**Key Features**:
- HPC cluster execution
- Job arrays for parallel models
- Resource management via SLURM
- Module system integration

---

## 8. COMPLETE COMMAND FLOW EXAMPLES

### 8.1 Local Single-Node Execution

**Command**: `madengine-cli run --tags dummy --timeout 3600`

**Complete Flow**:
```
1. mad_cli.py → run_command()
    ↓
2. Create DistributedOrchestrator(build_only_mode=False)
    ↓
3. Detect: No manifest provided + Tags provided
   → Execute full_workflow()
    ↓
4. Build Phase:
   a. DiscoverModels.run() → Find "dummy" model
   b. DockerBuilder.build_image() → Build Docker image
   c. DockerBuilder.tag_and_push_image() → Push to registry (optional)
   d. Generate build_manifest.json
    ↓
5. Run Phase:
   a. ContainerRunner.load_build_manifest()
   b. ContainerRunner.run_single_model()
   c. Execute Docker container with model
   d. Parse performance output
   e. Update perf.csv
    ↓
6. Display summary with Rich formatting
```

---

### 8.2 Distributed Build on CPU Node

**Command**: 
```bash
madengine-cli build --tags production_models \
  --registry docker.io \
  --additional-context '{"gpu_vendor":"AMD","guest_os":"UBUNTU"}'
```

**Complete Flow**:
```
1. mad_cli.py → build_command()
    ↓
2. Create DistributedOrchestrator(build_only_mode=True)
    ↓
3. Context initialization:
   - Skip GPU detection
   - Use provided gpu_vendor/guest_os
   - Set docker_build_arg from context
    ↓
4. DiscoverModels.run() → Find all models with "production_models" tag
    ↓
5. For each model:
   a. DockerBuilder.build_image()
   b. docker build with MAD_SYSTEM_GPU_ARCHITECTURE (if provided)
   c. Tag: docker tag ci-model docker.io/org/model:latest
   d. Push: docker push docker.io/org/model:latest
    ↓
6. Generate build_manifest.json:
   {
     "registry": "docker.io",
     "built_images": {
       "model1": {"registry_image": "docker.io/org/model1:latest", ...},
       "model2": {"registry_image": "docker.io/org/model2:latest", ...}
     }
   }
    ↓
7. Output: build_manifest.json ready for distribution
```

---

### 8.3 Distributed Execution via Ansible

**Command**: 
```bash
madengine-cli runner ansible \
  --inventory cluster.yml \
  --playbook deployment.yml
```

**Complete Flow**:
```
1. mad_cli.py → runner_ansible_command()
    ↓
2. RunnerFactory.create_runner("ansible")
    ↓
3. AnsibleDistributedRunner.__init__()
   a. Load cluster.yml:
      nodes:
        - hostname: gpu-node-1
          address: 192.168.1.101
          gpu_vendor: AMD
        - hostname: gpu-node-2
          address: 192.168.1.102
          gpu_vendor: AMD
    ↓
4. setup_infrastructure():
   a. Generate/validate Ansible playbook
   b. Prepare inventory for ansible-playbook
    ↓
5. execute_workload():
   a. Run: ansible-playbook -i cluster.yml deployment.yml
   b. Playbook executes on all nodes:
      - Clone MAD repo
      - Install madengine
      - Copy build_manifest.json
      - Execute: madengine-cli run --manifest-file build_manifest.json
      - Collect perf.csv
    ↓
6. cleanup_infrastructure():
   a. Fetch all perf.csv files from nodes
   b. Aggregate results
    ↓
7. generate_report():
   a. Create ansible_results.json with:
      - Total nodes: 2
      - Successful: 2
      - Failed: 0
      - Per-node results and metrics
```

---

## 9. KEY DATA STRUCTURES

### 9.1 Model Definition (models.json)

```json
{
  "name": "dummy",
  "dockerfile": "scripts/dummy/Dockerfile",
  "dockercontext": "./docker",
  "scripts": "scripts/dummy",
  "n_gpus": "1",
  "timeout": 3600,
  "tags": ["dummy", "test"],
  "args": "--batch-size 32",
  "cred": "AMD_GITHUB",
  "data": "model_data",
  "training_precision": "fp16",
  "owner": "team-name",
  "url": "https://github.com/...",
  "skip_gpu_arch": "false",
  "multiple_results": "",
  "additional_docker_run_options": ""
}
```

**Field Descriptions**:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique model identifier |
| `dockerfile` | Yes | Path to Dockerfile |
| `dockercontext` | No | Docker build context path |
| `scripts` | Yes | Path to model scripts |
| `n_gpus` | No | Number of GPUs required |
| `timeout` | No | Execution timeout in seconds |
| `tags` | Yes | List of tags for filtering |
| `args` | No | Command-line arguments |
| `cred` | No | Credential key from credential.json |
| `data` | No | Data provider key from data.json |

---

### 9.2 Build Manifest (build_manifest.json)

```json
{
  "registry": "docker.io",
  "build_context": {
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "docker_build_arg": {
      "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a"
    }
  },
  "built_images": {
    "dummy": {
      "docker_image": "ci-dummy_dockerfile",
      "docker_sha": "sha256:abc123...",
      "registry_image": "docker.io/org/dummy:latest",
      "dockerfile": "scripts/dummy/Dockerfile",
      "build_duration": 123.45,
      "base_docker": "rocm/pytorch:latest",
      "build_timestamp": "2025-11-28T10:30:00Z"
    }
  },
  "summary": {
    "total_models": 1,
    "successful_builds": 1,
    "failed_builds": 0,
    "total_duration": 150.0
  }
}
```

---

### 9.3 Performance CSV (perf.csv)

```csv
model,n_gpus,training_precision,pipeline,args,tags,docker_file,base_docker,docker_sha,docker_image,git_commit,machine_name,gpu_architecture,performance,metric,relative_change,status,build_duration,test_duration,dataname,data_provider_type,data_size,data_download_duration,build_number,additional_docker_run_options
dummy,1,fp16,ci,--batch-size 32,"dummy,test",scripts/dummy/Dockerfile,rocm/pytorch:latest,sha256:abc,ci-dummy,abcd1234,gpu-node-1,gfx90a,245.67,tokens/sec,0.0,SUCCESS,123.45,45.67,imagenet,nas,100GB,30.5,1,
```

**CSV Fields**:

| Category | Fields |
|----------|--------|
| Model Info | model, n_gpus, training_precision, args, tags |
| Docker Info | docker_file, base_docker, docker_sha, docker_image |
| System Info | machine_name, gpu_architecture, git_commit |
| Performance | performance, metric, relative_change, status |
| Timing | build_duration, test_duration |
| Data | dataname, data_provider_type, data_size, data_download_duration |
| Metadata | pipeline, build_number |

---

### 9.4 Runner Inventory Formats

#### SSH/Ansible Inventory (inventory.yml)

```yaml
nodes:
  - hostname: "gpu-node-1"
    address: "192.168.1.101"
    port: 22
    username: "madengine"
    ssh_key_path: "~/.ssh/id_rsa"
    gpu_count: 4
    gpu_vendor: "AMD"
    labels:
      env: "production"
      tier: "gpu-high"
    environment:
      ROCR_VISIBLE_DEVICES: "0,1,2,3"
      
  - hostname: "gpu-node-2"
    address: "192.168.1.102"
    port: 22
    username: "madengine"
    ssh_key_path: "~/.ssh/id_rsa"
    gpu_count: 8
    gpu_vendor: "AMD"
    labels:
      env: "production"
      tier: "gpu-premium"
    environment:
      ROCR_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

#### Kubernetes Inventory (k8s_inventory.yml)

```yaml
pods:
  - name: "madengine-pod-1"
    node_selector:
      gpu-type: "amd"
      tier: "high-memory"
    resources:
      requests:
        amd.com/gpu: "2"
        memory: "32Gi"
        cpu: "8"
      limits:
        amd.com/gpu: "2"
        memory: "64Gi"
        cpu: "16"
    gpu_vendor: "AMD"
    labels:
      app: "madengine"
      env: "production"
    
  - name: "madengine-pod-2"
    node_selector:
      gpu-type: "amd"
    resources:
      requests:
        amd.com/gpu: "4"
        memory: "64Gi"
        cpu: "16"
      limits:
        amd.com/gpu: "4"
        memory: "128Gi"
        cpu: "32"
    gpu_vendor: "AMD"
```

#### SLURM Inventory (slurm_inventory.yml)

```yaml
slurm_cluster:
  login_node:
    hostname: "hpc-login01"
    address: "hpc-login01.example.com"
    port: 22
    username: "madengine"
    ssh_key_path: "~/.ssh/id_rsa"
    
  partitions:
    - name: "gpu"
      max_time: "24:00:00"
      nodes: 32
      gpu_types: ["MI250X", "MI210"]
      gpu_vendor: "AMD"
      qos: "normal"
      
    - name: "gpu-priority"
      max_time: "48:00:00"
      nodes: 8
      gpu_types: ["MI250X"]
      gpu_vendor: "AMD"
      qos: "high"
      
  modules:
    - "rocm/5.7.0"
    - "python/3.10"
    - "git/2.40"
```

---

### 9.5 Credential Configuration (credential.json)

```json
{
  "dockerhub": {
    "username": "dockerhub_username",
    "password": "dockerhub_token",
    "repository": "my-org"
  },
  "AMD_GITHUB": {
    "username": "github_username",
    "password": "github_personal_access_token"
  },
  "MAD_AWS_S3": {
    "username": "aws_access_key_id",
    "password": "aws_secret_access_key",
    "region": "us-west-2"
  },
  "private_registry": {
    "username": "registry_user",
    "password": "registry_token",
    "repository": "company.registry.com/ml-models"
  }
}
```

**Environment Variable Override**:
```bash
export MAD_DOCKERHUB_USER=my_username
export MAD_DOCKERHUB_PASSWORD=my_token
export MAD_DOCKERHUB_REPO=my_org
```

---

### 9.6 Data Provider Configuration (data.json)

```json
{
  "data_sources": {
    "model_data": {
      "nas": {
        "path": "/mnt/nas/datasets/model_data",
        "mount_point": "/data"
      },
      "minio": {
        "path": "s3://minio-server/datasets/model_data",
        "endpoint": "http://minio.local:9000"
      },
      "aws": {
        "path": "s3://my-bucket/datasets/model_data",
        "region": "us-west-2"
      }
    },
    "imagenet": {
      "nas": {
        "path": "/mnt/nas/datasets/imagenet"
      },
      "aws": {
        "path": "s3://public-datasets/imagenet"
      }
    }
  },
  "mirrorlocal": "/tmp/local_data_mirror",
  "default_provider": "nas"
}
```

---

## 10. REFACTORING RECOMMENDATIONS

### 10.1 CLI Consolidation

**Current Issue**: Dual CLI (mad.py + mad_cli.py) creates maintenance overhead

**Recommendation**: 
```
Phase 1: Feature Parity
├─ Ensure mad_cli.py has all mad.py functionality
├─ Add legacy command aliases in mad_cli.py
└─ Update tests to cover both interfaces

Phase 2: Deprecation
├─ Add deprecation warnings to mad.py
├─ Update documentation to favor mad_cli.py
└─ Provide migration guide

Phase 3: Removal
├─ Remove mad.py after 2-3 releases
├─ Keep mad entry point as alias to madengine-cli
└─ Update all examples and documentation
```

**Implementation**:
```python
# mad_cli.py - Add legacy compatibility
@app.command(name="run", hidden=False)
def run_legacy_command(
    tags: List[str] = typer.Option(...),
    live_output: bool = typer.Option(False, "--live-output", "-l")
):
    """Legacy run command (deprecated, use: madengine-cli run)"""
    console.print("[yellow]Warning: Legacy command style. "
                  "Please use 'madengine-cli run' instead.[/yellow]")
    # Delegate to new implementation
    return run_command(tags=tags, live_output=live_output)
```

---

### 10.2 Orchestrator Simplification

**Current Issue**: `DistributedOrchestrator` has complex workflow detection logic

**Recommendation**: Split into specialized orchestrators

**Proposed Structure**:
```python
# New structure
class BuildOrchestrator:
    """Handles Docker image building only"""
    def execute(self, models, registry, clean_cache):
        # Build logic only
        pass

class RunOrchestrator:
    """Handles container execution only"""
    def execute(self, manifest_file, timeout):
        # Run logic only
        pass

class FullWorkflowOrchestrator:
    """Composes build + run orchestrators"""
    def __init__(self):
        self.build_orch = BuildOrchestrator()
        self.run_orch = RunOrchestrator()
    
    def execute(self, models, registry):
        manifest = self.build_orch.execute(models, registry)
        results = self.run_orch.execute(manifest)
        return results

# Factory pattern for creation
class OrchestratorFactory:
    @staticmethod
    def create(mode: str, **kwargs):
        if mode == "build":
            return BuildOrchestrator(**kwargs)
        elif mode == "run":
            return RunOrchestrator(**kwargs)
        elif mode == "full":
            return FullWorkflowOrchestrator(**kwargs)
```

**Benefits**:
- Clear separation of concerns
- Easier testing (mock each orchestrator independently)
- Explicit workflow selection
- Simpler code paths

---

### 10.3 Context Initialization Refactoring

**Current Issue**: Context class mixes build-time and runtime concerns

**Recommendation**: Create specialized context classes

**Proposed Structure**:
```python
# Base context class
class BaseContext(ABC):
    """Abstract base for all contexts"""
    def __init__(self, additional_context=None):
        self.ctx = {}
        self._load_additional_context(additional_context)
    
    @abstractmethod
    def initialize(self):
        """Initialize context-specific data"""
        pass

# Build context (no GPU detection)
class BuildContext(BaseContext):
    """Context for build-only operations"""
    def initialize(self):
        self.ctx["host_os"] = self._get_host_os()
        # Only build-related context
        # No GPU detection
        return self

# Runtime context (with GPU detection)
class RuntimeContext(BaseContext):
    """Context for runtime operations"""
    def initialize(self):
        self.ctx["host_os"] = self._get_host_os()
        self.ctx["gpu_vendor"] = self._get_gpu_vendor()
        self.ctx["gpu_architecture"] = self._get_gpu_architecture()
        self.ctx["n_gpus"] = self._get_system_ngpus()
        return self

# Factory for context creation
class ContextFactory:
    @staticmethod
    def create(mode: str, **kwargs):
        if mode == "build":
            return BuildContext(**kwargs).initialize()
        elif mode == "runtime":
            return RuntimeContext(**kwargs).initialize()
        else:
            raise ValueError(f"Unknown context mode: {mode}")

# Usage
build_ctx = ContextFactory.create("build", additional_context=ctx_json)
runtime_ctx = ContextFactory.create("runtime")
```

**Benefits**:
- Clear separation between build and runtime
- No conditional logic based on mode flags
- Type safety (different classes for different purposes)
- Easier to add new context types

---

### 10.4 Error Handling Standardization

**Current Issue**: Mix of exceptions, error returns, and console.print errors

**Recommendation**: Consistent error handling framework

**Proposed Structure**:
```python
# Custom exception hierarchy
class MADEngineError(Exception):
    """Base exception for all madengine errors"""
    def __init__(self, message, context=None, suggestions=None):
        self.message = message
        self.context = context or {}
        self.suggestions = suggestions or []
        super().__init__(message)

class ModelDiscoveryError(MADEngineError):
    """Errors during model discovery"""
    pass

class DockerBuildError(MADEngineError):
    """Errors during Docker builds"""
    pass

class ContainerExecutionError(MADEngineError):
    """Errors during container execution"""
    pass

class DistributedExecutionError(MADEngineError):
    """Errors during distributed execution"""
    pass

# Centralized error handler
class ErrorHandler:
    def __init__(self, console, verbose=False):
        self.console = console
        self.verbose = verbose
    
    def handle(self, error: MADEngineError):
        """Handle error with rich formatting"""
        self.console.print_error(f"[red]Error:[/red] {error.message}")
        
        if error.context and self.verbose:
            self.console.print("[dim]Context:[/dim]")
            for key, value in error.context.items():
                self.console.print(f"  {key}: {value}")
        
        if error.suggestions:
            self.console.print("[yellow]Suggestions:[/yellow]")
            for suggestion in error.suggestions:
                self.console.print(f"  • {suggestion}")

# Usage throughout codebase
try:
    models = discover_models()
except FileNotFoundError as e:
    raise ModelDiscoveryError(
        "models.json file not found",
        context={
            "cwd": os.getcwd(),
            "expected_path": "models.json"
        },
        suggestions=[
            "Ensure you're running from within a MAD package directory",
            "Check that models.json exists in the current directory",
            "Clone the MAD repository: git clone https://github.com/ROCm/MAD.git"
        ]
    ) from e
```

**Benefits**:
- Consistent error messages across the framework
- Better user experience with actionable suggestions
- Easier debugging with context information
- Centralized formatting logic

---

### 10.5 Runner Interface Consistency

**Current Issue**: Runners have slightly different initialization patterns

**Recommendation**: Enforce strict interface contract

**Proposed Changes**:
```python
# Strengthen BaseDistributedRunner contract
class BaseDistributedRunner(ABC):
    """Abstract base class for distributed runners"""
    
    # Required class attributes
    RUNNER_TYPE: str  # e.g., "ssh", "ansible", "k8s"
    REQUIRED_DEPENDENCIES: List[str]  # e.g., ["paramiko", "scp"]
    
    def __init__(self, inventory_path: str, console=None, verbose=False):
        """Standardized initialization"""
        self._validate_dependencies()
        self.inventory_path = inventory_path
        self.console = console or Console()
        self.verbose = verbose
        self.nodes = self._load_inventory(inventory_path)
    
    @classmethod
    def _validate_dependencies(cls):
        """Check if required dependencies are installed"""
        missing = []
        for dep in cls.REQUIRED_DEPENDENCIES:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            raise ImportError(
                f"{cls.RUNNER_TYPE} runner requires: {', '.join(missing)}\n"
                f"Install with: pip install madengine[{cls.RUNNER_TYPE}]"
            )
    
    @abstractmethod
    def _parse_inventory_format(self, data: Dict) -> List[NodeConfig]:
        """Parse runner-specific inventory format"""
        pass
    
    # Standard workflow methods (already exist)
    @abstractmethod
    def setup_infrastructure(self, workload: WorkloadSpec) -> bool:
        pass
    
    @abstractmethod
    def execute_workload(self, workload: WorkloadSpec) -> DistributedResult:
        pass
    
    @abstractmethod
    def cleanup_infrastructure(self, workload: WorkloadSpec) -> bool:
        pass

# Each runner implements consistently
class SSHDistributedRunner(BaseDistributedRunner):
    RUNNER_TYPE = "ssh"
    REQUIRED_DEPENDENCIES = ["paramiko", "scp"]
    
    def _parse_inventory_format(self, data: Dict) -> List[NodeConfig]:
        # SSH-specific parsing
        pass

class AnsibleDistributedRunner(BaseDistributedRunner):
    RUNNER_TYPE = "ansible"
    REQUIRED_DEPENDENCIES = ["ansible", "ansible_runner"]
    
    def _parse_inventory_format(self, data: Dict) -> List[NodeConfig]:
        # Ansible-specific parsing
        pass
```

**Benefits**:
- Clear dependency requirements
- Consistent initialization across all runners
- Better error messages for missing dependencies
- Easier to add new runners

---

### 10.6 Configuration Management Consolidation

**Current Issue**: Multiple config files (credential.json, data.json, tools.json, etc.)

**Recommendation**: Unified configuration system

**Proposed Structure**:
```yaml
# madengine.yaml (single config file)
madengine:
  version: "1.0"
  
  # Registry settings
  registry:
    default: "docker.io"
    credentials:
      dockerhub:
        username: "${DOCKERHUB_USER}"
        password: "${DOCKERHUB_TOKEN}"
        repository: "my-org"
      private:
        url: "registry.company.com"
        username: "${PRIVATE_REGISTRY_USER}"
        password: "${PRIVATE_REGISTRY_TOKEN}"
  
  # Data providers
  data:
    default_provider: "nas"
    mirror_local: "/tmp/mad_data"
    sources:
      model_data:
        nas:
          path: "/mnt/nas/datasets"
          mount_point: "/data"
        s3:
          bucket: "my-datasets"
          region: "us-west-2"
          credentials: "${AWS_CREDENTIALS}"
  
  # Build settings
  build:
    default_context: "./docker"
    cache_enabled: true
    parallel_builds: 4
    
  # Runtime settings
  runtime:
    default_timeout: 3600
    keep_containers: false
    live_output: true
    
  # Distributed execution
  distributed:
    mad_repo: "https://github.com/ROCm/MAD.git"
    setup_timeout: 600
    default_runner: "ssh"

# Python code to load config
class Config:
    def __init__(self, config_file="madengine.yaml"):
        with open(config_file) as f:
            self._data = yaml.safe_load(f)
        self._resolve_env_vars()
    
    def _resolve_env_vars(self):
        """Replace ${VAR} with environment variables"""
        # Recursive resolution logic
        pass
    
    def get(self, path: str, default=None):
        """Get config value by dot-separated path"""
        # e.g., config.get("registry.credentials.dockerhub.username")
        pass

# Usage
config = Config()
username = config.get("registry.credentials.dockerhub.username")
```

**Migration Strategy**:
1. Support both old (credential.json) and new (madengine.yaml) formats
2. Add converter tool: `madengine-cli config migrate`
3. Deprecate old format after 2 releases
4. Remove old format support

**Benefits**:
- Single source of truth for configuration
- Environment variable support
- Better validation with schema
- Easier to version control

---

### 10.7 Testing Strategy Enhancement

**Current Issue**: Some integration tests require actual GPU hardware

**Recommendation**: Comprehensive mocking strategy

**Proposed Structure**:
```python
# tests/fixtures/mock_gpu.py
class MockGPUDetector:
    """Mock GPU detection for testing"""
    def __init__(self, vendor="AMD", arch="gfx90a", count=4):
        self.vendor = vendor
        self.arch = arch
        self.count = count
    
    def get_gpu_vendor(self):
        return self.vendor
    
    def get_system_gpu_architecture(self):
        return self.arch
    
    def get_system_ngpus(self):
        return self.count

# tests/fixtures/mock_docker.py
class MockDockerClient:
    """Mock Docker client for testing"""
    def __init__(self):
        self.built_images = []
        self.pushed_images = []
        self.run_containers = []
    
    def build(self, path, tag, **kwargs):
        self.built_images.append(tag)
        return {"Id": f"sha256:mock_{tag}"}
    
    def push(self, image):
        self.pushed_images.append(image)
        return True
    
    def run(self, image, command, **kwargs):
        self.run_containers.append((image, command))
        return "mock_output"

# tests/test_orchestrator.py
@pytest.fixture
def mock_context(monkeypatch):
    """Fixture providing mocked context"""
    mock_gpu = MockGPUDetector()
    monkeypatch.setattr("madengine.core.context.get_gpu_vendor", 
                       mock_gpu.get_gpu_vendor)
    monkeypatch.setattr("madengine.core.context.get_system_gpu_architecture",
                       mock_gpu.get_system_gpu_architecture)
    return mock_gpu

@pytest.fixture
def mock_docker(monkeypatch):
    """Fixture providing mocked Docker"""
    mock_client = MockDockerClient()
    monkeypatch.setattr("madengine.core.docker.Docker", 
                       lambda: mock_client)
    return mock_client

def test_build_orchestrator(mock_context, mock_docker):
    """Test build orchestrator without real GPU/Docker"""
    orch = BuildOrchestrator(build_only_mode=True)
    result = orch.execute(models=[...], registry="mock.registry")
    
    assert len(mock_docker.built_images) == 1
    assert mock_docker.built_images[0] == "ci-dummy_dockerfile"
    assert result["successful_builds"] == 1

# Separate test markers
# pytest -m unit          # Fast unit tests with mocks
# pytest -m integration   # Integration tests (may require Docker)
# pytest -m gpu           # GPU-required tests
# pytest -m slow          # Slow tests
```

**Test Organization**:
```
tests/
├── unit/                    # Fast unit tests with mocks
│   ├── test_context.py
│   ├── test_discover.py
│   └── test_orchestrator.py
├── integration/             # Integration tests (Docker required)
│   ├── test_docker_build.py
│   └── test_container_run.py
├── distributed/             # Distributed runner tests
│   ├── test_ssh_runner.py
│   └── test_ansible_runner.py
├── gpu/                     # GPU-required tests
│   └── test_gpu_execution.py
└── fixtures/                # Shared fixtures
    ├── mock_gpu.py
    ├── mock_docker.py
    └── sample_data.py
```

**Benefits**:
- Tests run quickly without GPU/Docker
- Clear separation of test types
- Easy to run subsets of tests
- Better CI/CD integration

---

### 10.8 Additional Refactoring Opportunities

#### **A. Logging Standardization**

**Current**: Mix of `print()`, `logging`, and `rich.console.print()`

**Recommendation**: Unified logging interface
```python
class MADLogger:
    """Unified logging for madengine"""
    def __init__(self, name, use_rich=True):
        self.logger = logging.getLogger(name)
        self.console = Console() if use_rich else None
    
    def info(self, message, rich=True):
        self.logger.info(message)
        if rich and self.console:
            self.console.print(f"[blue]ℹ[/blue] {message}")
    
    def success(self, message):
        self.logger.info(message)
        if self.console:
            self.console.print(f"[green]✓[/green] {message}")
    
    def warning(self, message):
        self.logger.warning(message)
        if self.console:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
    
    def error(self, message):
        self.logger.error(message)
        if self.console:
            self.console.print(f"[red]✗[/red] {message}")
```

#### **B. Model Discovery Caching**

**Recommendation**: Cache discovered models to speed up repeated operations
```python
class DiscoverModels:
    _cache = {}  # Class-level cache
    
    def run(self, use_cache=True):
        cache_key = self._get_cache_key()
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        models = self._discover_models()
        self._cache[cache_key] = models
        return models
```

#### **C. Performance Metrics Standardization**

**Recommendation**: Structured performance data
```python
@dataclass
class PerformanceMetrics:
    model_name: str
    performance_value: float
    metric_unit: str
    gpu_architecture: str
    build_duration: float
    test_duration: float
    status: str
    timestamp: datetime
    
    def to_csv_row(self) -> dict:
        """Convert to CSV format"""
        pass
    
    def to_json(self) -> dict:
        """Convert to JSON format"""
        pass
```

---

## 11. EXECUTION FLOW DIAGRAMS

### 11.1 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                   (CLI: mad.py or mad_cli.py)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Command Processing                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Validate Args│  │ Parse Context│  │ Setup Logging│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                            │
│              (DistributedOrchestrator)                          │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Workflow Decision:                              │          │
│  │  • Build-only mode?  → build_phase()             │          │
│  │  • Run-only mode?    → run_phase()               │          │
│  │  • Full workflow?    → full_workflow()           │          │
│  └──────────────────────────────────────────────────┘          │
└─────────┬──────────────────────────────────┬─────────────┬──────┘
          │                                  │             │
          ▼                                  ▼             ▼
┌──────────────────┐        ┌──────────────────┐  ┌──────────────┐
│  DiscoverModels  │        │  DockerBuilder   │  │ContainerRunner│
│                  │        │                  │  │              │
│ • Load models.json│       │ • Build images   │  │• Pull images │
│ • Parse tags     │        │ • Push to registry│ │• Run containers│
│ • Filter models  │        │ • Generate SHA   │  │• Collect metrics│
└──────────────────┘        └──────────────────┘  └──────────────┘
          │                          │                    │
          └──────────────────────────┴────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Services                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Context    │  │    Docker    │  │ DataProvider │         │
│  │              │  │              │  │              │         │
│  │ GPU detection│  │ Build/Run ops│  │ Data sources │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
          │                          │                    │
          └──────────────────────────┴────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Generation                          │
│  • build_manifest.json (for distribution)                       │
│  • perf.csv (performance metrics)                               │
│  • execution logs (detailed output)                             │
│  • Summary reports (JSON/HTML)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 11.2 Distributed Execution Flow

```
┌──────────────┐
│ Build Node   │  (CPU-only, no GPU required)
│ (Central)    │
└───────┬──────┘
        │
        │ 1. madengine-cli build --tags models --registry docker.io
        │
        ▼
┌────────────────────────────────┐
│ Discover & Build Docker Images │
│  • Find all models             │
│  • Build with provided context │
│  • Push to Docker registry     │
└───────┬────────────────────────┘
        │
        │ 2. Generate build_manifest.json
        │
        ▼
┌────────────────────────────────┐
│  build_manifest.json           │
│  • Registry location           │
│  • Built image details         │
│  • Build context               │
└───────┬───────────┬────────────┘
        │           │
        │           │ 3. Distribute manifest
        │           │
        ▼           ▼
┌──────────────┐  ┌──────────────┐
│  GPU Node 1  │  │  GPU Node 2  │
│              │  │              │
└──────┬───────┘  └──────┬───────┘
       │                 │
       │ 4. Pull images from registry
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Docker Pull  │  │ Docker Pull  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       │ 5. Run containers with models
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│Execute Models│  │Execute Models│
│              │  │              │
│• Run.sh      │  │• Run.sh      │
│• Collect perf│  │• Collect perf│
└──────┬───────┘  └──────┬───────┘
       │                 │
       │ 6. Generate results
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  perf.csv    │  │  perf.csv    │
│  logs        │  │  logs        │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                │ 7. Aggregate results
                │
                ▼
        ┌──────────────┐
        │ Final Report │
        │  • Combined  │
        │    metrics   │
        │  • Status    │
        └──────────────┘
```

---

### 11.3 Model Discovery Flow

```
Start
  │
  ▼
┌────────────────────────────────┐
│ DiscoverModels.run()           │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ 1. Check for models.json       │
│    in current directory        │
└────────┬───────────────────────┘
         │
         ▼
    ┌────────┐
    │ Found? │
    └───┬─┬──┘
      No│ │Yes
        │ │
        │ └─────────────────────┐
        │                       ▼
        │           ┌────────────────────┐
        │           │ Load root models   │
        │           └────────┬───────────┘
        │                    │
        ▼                    ▼
┌────────────────┐  ┌─────────────────────┐
│ Raise Error    │  │ 2. Walk scripts/ dir│
└────────────────┘  └────────┬────────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │ For each subdir:   │
                    │ • Check for        │
                    │   models.json      │
                    │ • Check for        │
                    │   get_models_json.py│
                    └────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ models.json  │ │get_models_   │ │ Neither      │
    │   found      │ │json.py found │ │ (skip dir)   │
    └──────┬───────┘ └──────┬───────┘ └──────────────┘
           │                │
           ▼                ▼
    ┌──────────────┐ ┌──────────────┐
    │ Load static  │ │ Import & exec│
    │ definitions  │ │ dynamic code │
    └──────┬───────┘ └──────┬───────┘
           │                │
           │                ▼
           │         ┌──────────────┐
           │         │Call function │
           │         │with params   │
           │         └──────┬───────┘
           │                │
           └────────┬───────┘
                    │
                    ▼
           ┌────────────────────┐
           │ Accumulate all     │
           │ discovered models  │
           └────────┬───────────┘
                    │
                    ▼
           ┌────────────────────┐
           │ 3. Filter by tags  │
           │ Parse tag format:  │
           │ dir:model:params   │
           └────────┬───────────┘
                    │
                    ▼
           ┌────────────────────┐
           │ 4. Return filtered │
           │    model list      │
           └────────────────────┘
```

---

### 11.4 Container Execution Flow

```
Start
  │
  ▼
┌────────────────────────────────┐
│ ContainerRunner.               │
│ run_models_from_manifest()     │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Load build_manifest.json       │
│ • Extract registry             │
│ • Extract built_images         │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Login to Docker registry       │
│ • Use credentials from         │
│   credential.json or env       │
└────────┬───────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ For each model in manifest:    │
└────────┬───────────────────────┘
         │
         ▼
    ┌────────────────────────────┐
    │ Pull image from registry   │
    │ docker pull <image>        │
    └────────┬───────────────────┘
             │
             ▼
    ┌────────────────────────────┐
    │ Prepare Docker run command:│
    │ • Mount volumes            │
    │ • Set GPU devices          │
    │ • Set environment vars     │
    │ • Add runtime options      │
    └────────┬───────────────────┘
             │
             ▼
    ┌────────────────────────────┐
    │ Execute container:         │
    │ docker run ... <image>     │
    │   bash -c "./run.sh"       │
    └────────┬───────────────────┘
             │
             ├─────────────────────┐
             │                     │
             ▼                     ▼
    ┌────────────────┐    ┌────────────────┐
    │ stdout → log   │    │ Apply timeout  │
    │ stderr → log   │    │ monitoring     │
    └────────┬───────┘    └────────┬───────┘
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
               ┌────────────────────┐
               │ Parse output:      │
               │ • Look for         │
               │   "Performance:"   │
               │ • Extract metrics  │
               │ • Check status     │
               └────────┬───────────┘
                        │
                        ▼
               ┌────────────────────┐
               │ Create run_details:│
               │ • model name       │
               │ • performance      │
               │ • status           │
               │ • duration         │
               │ • GPU info         │
               └────────┬───────────┘
                        │
                        ▼
               ┌────────────────────┐
               │ Append to perf.csv │
               └────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Return execution summary       │
│ • Total models                 │
│ • Successful runs              │
│ • Failed runs                  │
│ • Aggregate metrics            │
└────────────────────────────────┘
```

---

## 12. SUMMARY & NEXT STEPS

### Key Takeaways

1. **madengine is well-architected** with clear separation between:
   - CLI interfaces (legacy + modern)
   - Core components (context, docker, data)
   - Orchestration layer (build/run workflows)
   - Distributed runners (SSH, Ansible, K8s, SLURM)

2. **Main strengths**:
   - Split architecture enables efficient resource utilization
   - Rich distributed execution support
   - Comprehensive error handling framework
   - High test coverage (95%+)

3. **Primary refactoring opportunities**:
   - CLI consolidation (deprecate legacy CLI)
   - Orchestrator simplification (split into specialized classes)
   - Context initialization (separate BuildContext/RuntimeContext)
   - Configuration management (unified madengine.yaml)

### Recommended Refactoring Priority

**Phase 1: Foundation** (Weeks 1-2)
- [ ] Implement unified configuration system (madengine.yaml)
- [ ] Create specialized context classes (BuildContext, RuntimeContext)
- [ ] Standardize error handling across all components
- [ ] Enhance testing with comprehensive mocks

**Phase 2: Orchestration** (Weeks 3-4)
- [ ] Split DistributedOrchestrator into specialized classes
- [ ] Implement OrchestratorFactory pattern
- [ ] Refactor workflow detection logic
- [ ] Add integration tests for all workflow types

**Phase 3: CLI & Runners** (Weeks 5-6)
- [ ] Add legacy command support to mad_cli.py
- [ ] Deprecate mad.py with warnings
- [ ] Strengthen BaseDistributedRunner interface
- [ ] Standardize runner inventory formats

**Phase 4: Polish** (Weeks 7-8)
- [ ] Complete documentation updates
- [ ] Migration guides for users
- [ ] Performance optimization
- [ ] Final testing and validation

### Success Metrics

- [ ] Reduced code duplication (<10% duplicated code)
- [ ] Improved test execution time (<5 minutes for unit tests)
- [ ] Better error messages (user surveys)
- [ ] Easier onboarding (documentation feedback)
- [ ] Maintained backward compatibility (zero breaking changes)

---

**End of Architecture Flow Documentation**

This document provides a comprehensive view of the madengine framework for refactoring purposes. Use it as a reference during the refactoring process to ensure all components and flows are properly understood and maintained.


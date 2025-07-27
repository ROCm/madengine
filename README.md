# madengine

A comprehensive AI model automation and benchmarking toolkit designed to work seamlessly with the [MAD (Model Automation and Dashboarding)](https://github.com/ROCm/MAD) package ecosystem.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://docker.com)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [MAD Model Discovery](#mad-model-discovery)
- [Command Line Interface](#command-line-interface)
- [Distributed Execution](#distributed-execution)
  - [Distributed Runner System](#distributed-runner-system)
  - [Runner Types](#runner-types)
  - [Inventory Configuration](#inventory-configuration)
  - [Examples](#examples)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Deployment Scenarios](#deployment-scenarios)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Overview

madengine is an enterprise-grade AI model automation and dashboarding command-line tool designed to run Large Language Models (LLMs) and Deep Learning models locally or in distributed environments. It provides a modern, production-ready solution for AI model benchmarking with comprehensive CI/CD integration capabilities.

### Key Capabilities

- **Reliable Model Execution**: Run AI models reliably across supported platforms with quality assurance
- **Distributed Architecture**: Split build and execution phases for optimal resource utilization
- **Comprehensive Automation**: Minimalistic, out-of-the-box solution for hardware and software stack validation
- **Real-time Metrics**: Audience-relevant AI model performance tracking with intuitive presentation
- **Enterprise Integration**: Best practices for internal projects and external open-source model handling
- **MAD Ecosystem Integration**: Seamless integration with the MAD package for model discovery and management

### MAD Package Integration

madengine is designed to work within the **MAD (Model Automation and Dashboarding) package**, which serves as a comprehensive model hub containing:

- Docker configurations and container definitions
- Model scripts and automation workflows  
- Adopted AI models with standardized interfaces
- Data providers and credential management
- Build tools and environment configurations

## Features

🚀 **Modern CLI Interface**: Built with Typer and Rich for excellent user experience  
📊 **Rich Terminal Output**: Progress bars, tables, panels with syntax highlighting  
🎯 **Intelligent Workflows**: Automatic detection of build-only vs. full workflow operations  
🔄 **Distributed Execution**: Separate build and run phases for scalable deployments  
🐳 **Docker Integration**: Containerized model execution with GPU support  
📋 **Model Discovery**: Automatic discovery from MAD package structure  
🏷️ **Flexible Tagging**: Hierarchical model selection with parameterization  
⚡ **Performance Optimized**: Built for speed and resource efficiency  
🔐 **Credential Management**: Centralized authentication for repositories and registries  
📈 **Monitoring & Reporting**: Comprehensive metrics collection and analysis  
🌐 **Multi-Platform**: Support for AMD ROCm, NVIDIA CUDA, and Intel architectures  
🔧 **Extensible**: Plugin architecture for custom tools and integrations  
📦 **Batch Processing**: Support for batch manifest files with selective building  
🏃 **Streamlined Runners**: Simplified distributed execution interface with comprehensive reporting

## Architecture

![madengine Architecture Overview](docs/img/architecture_overview.png)

### Traditional vs. Modern Approach

**Legacy Monolithic Workflow:**
```
Model Discovery → Docker Build → Container Run → Performance Collection
```

**Modern Split Architecture:**
```
BUILD PHASE (Central/CI Server):
  Model Discovery → Docker Build → Push to Registry → Export Manifest

RUN PHASE (GPU Nodes):
  Load Manifest → Pull Images → Container Run → Performance Collection
```

### Benefits of Split Architecture

- **Resource Efficiency**: Build on CPU-optimized instances, run on GPU-optimized nodes
- **Parallel Execution**: Multiple nodes can execute different models simultaneously  
- **Reproducibility**: Consistent Docker images ensure identical results across environments
- **Scalability**: Easy horizontal scaling by adding execution nodes
- **Cost Optimization**: Use appropriate instance types for each workflow phase
- **CI/CD Integration**: Seamless integration with existing DevOps pipelines

## Installation

madengine is designed to work within the [MAD (Model Automation and Dashboarding)](https://github.com/ROCm/MAD) package ecosystem. Follow these steps for proper installation and setup.

### Prerequisites

- **Python 3.8 or higher**
- **Git** for repository management
- **Docker** with GPU support (ROCm for AMD, CUDA for NVIDIA)
- **MAD package** cloned and available locally

### Development Installation

```bash
# Clone MAD package first
git clone git@github.com:ROCm/MAD.git
cd MAD

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Clone madengine into MAD directory or install as dependency
git clone git@github.com:ROCm/madengine.git
cd madengine

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks (recommended for contributors)
pre-commit install
```

### Production Installation

```bash
# Navigate to MAD package directory
cd /path/to/MAD

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install madengine
pip install git+https://github.com/ROCm/madengine.git@main

# Or install from local source
git clone git@github.com:ROCm/madengine.git
cd madengine
pip install .
```

### Distributed Runner Dependencies

Install dependencies for specific runner types:

```bash
# SSH Runner
pip install madengine[ssh]

# Ansible Runner
pip install madengine[ansible]

# Kubernetes Runner
pip install madengine[kubernetes]

# All runners
pip install madengine[runners]

# Development environment
pip install madengine[all]
```

### Manual Dependencies

If you prefer to install dependencies manually:

```bash
# SSH Runner
pip install paramiko>=2.7.0 scp>=0.14.0

# Ansible Runner
pip install ansible-runner>=2.0.0 PyYAML>=5.4.0

# Kubernetes Runner
pip install kubernetes>=20.0.0 PyYAML>=5.4.0
```

### Docker Environment Setup

For GPU-accelerated model execution:

```bash
# AMD ROCm support
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video

# NVIDIA CUDA support  
docker run --rm --gpus all

# Verify GPU access in container
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi
```

### Development Environment

For contributors and developers:

```bash
# Install with all development tools
pip install -e ".[dev]"

# Development workflow
pytest              # Run tests
black src/ tests/   # Format code  
isort src/ tests/   # Sort imports
flake8 src/ tests/  # Lint code
mypy src/madengine  # Type checking
```

### Modern Package Management

This project uses modern Python packaging standards:
- **`pyproject.toml`**: Single source of truth for dependencies and configuration
- **Hatchling build backend**: Modern, efficient build system
- **No requirements.txt**: All dependencies managed in pyproject.toml
- **pip ≥ 21.3**: Full pyproject.toml support required

## Quick Start

![Distributed Workflow](docs/img/distributed_workflow.png)

### Single-Node Workflow

Perfect for development, testing, or single-workstation deployments:

```bash
# Navigate to MAD package directory
cd /path/to/MAD

# Run complete workflow (build + execute)
madengine-cli run --tags dummy --registry localhost:5000 --timeout 3600

# Run with live output and detailed logging
madengine-cli run --tags dummy --live-output --verbose \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

### Split Build/Run Workflow

For distributed deployments and production environments:

```bash
# Build Phase (on build server)
cd /path/to/MAD
madengine-cli build --tags dummy resnet --registry docker.io \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}' \
  --clean-docker-cache

# Alternative: Batch build mode
madengine-cli build --batch-manifest batch.json \
  --registry docker.io \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Run Phase (on GPU nodes)
madengine-cli run --manifest-file build_manifest.json --timeout 1800
```

### Multi-Node Production Deployment

```bash
# Build on central server
madengine-cli build --tags production_models --registry prod.registry.com \
  --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "UBUNTU"}' \
  --summary-output build_report.json

# Transfer manifest to GPU cluster
scp build_manifest.json user@gpu-cluster:/path/to/madengine/

# Execute on GPU nodes (registry auto-detected from manifest)
madengine-cli run --manifest-file build_manifest.json \
  --summary-output execution_report.json
```

## MAD Model Discovery

madengine automatically discovers available models from the MAD package structure, supporting multiple discovery methods for maximum flexibility.

### Discovery Sources

#### 1. Root Models Configuration (`models.json`)
Traditional static model definitions at the MAD package root:

```bash
# Discover and run models from root configuration
madengine-cli run --tags dummy                    # Single model
madengine-cli run --tags dummy pyt_huggingface_bert  # Multiple models
madengine discover --tags dummy                   # List available models
```

#### 2. Directory-Specific Models (`scripts/{model_dir}/models.json`)
Organized model definitions in subdirectories:

```bash
# Run models from specific directories
madengine-cli run --tags dummy2:dummy_2
madengine discover --tags dummy2:dummy_2
```

#### 3. Dynamic Model Discovery (`scripts/{model_dir}/get_models_json.py`)
Python scripts that generate model configurations dynamically:

```bash
# Run dynamic models with parameters
madengine-cli run --tags dummy3:dummy_3
madengine-cli run --tags dummy3:dummy_3:batch_size=512:in=32:out=16
```

### Required MAD Structure

For proper model discovery, ensure your MAD package follows this structure:

```
MAD/
├── models.json                          # Root model definitions
├── scripts/
│   ├── dummy2/
│   │   ├── models.json                  # Static model configs
│   │   └── run.sh
│   ├── dummy3/
│   │   ├── get_models_json.py          # Dynamic model discovery  
│   │   └── run.sh
│   └── common/
│       └── tools.json                   # Build tools configuration
├── data.json                            # Data provider configurations
├── credential.json                      # Authentication credentials
└── pyproject.toml                       # madengine package config
```

### Tag System Examples

**Simple Tags:**
```bash
madengine-cli run --tags dummy                    # From root models.json
madengine-cli run --tags pyt_huggingface_bert     # Standard model
```

**Directory Tags:**
```bash
madengine-cli run --tags dummy2:dummy_2           # Directory-specific model
```

**Parameterized Tags:**
```bash
madengine-cli run --tags dummy3:dummy_3:batch_size=512              # Single parameter
madengine-cli run --tags dummy3:dummy_3:batch_size=512:in=32:out=16 # Multiple parameters
```

### Discovery Validation

```bash
# List all discoverable models
madengine discover

# Discover specific models
madengine discover --tags dummy
madengine discover --tags dummy2:dummy_2  
madengine discover --tags dummy3:dummy_3:batch_size=256
```

### Batch Build Mode

The CLI supports batch building mode using a batch manifest file that specifies which models to build and their configurations:

#### Batch Manifest Format (batch.json)

```json
[
  {
    "model_name": "dummy",
    "build_new": true,
    "registry": "docker.io",
    "registry_image": "my-org/dummy:latest"
  },
  {
    "model_name": "resnet",
    "build_new": false,
    "registry_image": "existing-registry/resnet:v1.0"
  },
  {
    "model_name": "bert",
    "build_new": true,
    "registry": "localhost:5000"
  }
]
```

#### Batch Build Usage

```bash
# Build only models marked with build_new=true
madengine-cli build --batch-manifest batch.json \
  --registry docker.io \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Note: Cannot use both --batch-manifest and --tags together
```

**Batch Manifest Features:**
- **Selective Building**: Only models with `build_new=true` are built
- **Registry Override**: Per-model registry configuration
- **Image Tracking**: Tracks both built and pre-existing images
- **Manifest Integration**: All models (built and existing) are included in final build manifest

## Command Line Interface

madengine provides two CLI interfaces: the traditional `madengine` command and the modern `madengine-cli` for distributed workflows.

### Traditional CLI (`madengine`)

Basic model execution and discovery:

```bash
# Run models locally
madengine run --tags pyt_huggingface_bert --live-output \
  --additional-context '{"guest_os": "UBUNTU"}'

# Discover available models
madengine discover --tags dummy

# Generate reports
madengine report to-html --csv-file-path perf.csv

# Database operations
madengine database create-table
```

### Modern Distributed CLI (`madengine-cli`)

Advanced distributed workflows with rich terminal output:

#### Build Command
```bash
madengine-cli build [OPTIONS]
```

Create Docker images and build manifests for distributed execution:

```bash
# Basic build with registry
madengine-cli build --tags dummy --registry localhost:5000

# Build with comprehensive configuration
madengine-cli build --tags production_models \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}' \
  --clean-docker-cache \
  --summary-output build_summary.json

# Batch build mode using batch manifest file
madengine-cli build --batch-manifest batch.json \
  --registry docker.io \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

#### Run Command
```bash
madengine-cli run [OPTIONS]
```

Intelligent execution with automatic workflow detection:

```bash
# Execution-only (when manifest exists)
madengine-cli run --manifest-file build_manifest.json --timeout 1800

# Complete workflow (when no manifest)
madengine-cli run --tags dummy --registry localhost:5000 --timeout 3600

# Advanced execution with monitoring
madengine-cli run --tags models --live-output --verbose --keep-alive
```

#### Distributed Runner Commands
```bash
madengine-cli runner <runner_type> [OPTIONS]
```

Execute models across multiple nodes with different infrastructure types:

```bash
# SSH Runner - Direct SSH connections to remote nodes
madengine-cli runner ssh \
    --inventory inventory.yml \
    --manifest-file build_manifest.json \
    --report-output ssh_execution_report.json \
    --verbose

# Ansible Runner - Orchestrated deployment using playbooks
madengine-cli runner ansible \
    --inventory cluster.yml \
    --playbook madengine_distributed.yml \
    --report-output ansible_execution_report.json \
    --verbose

# Kubernetes Runner - Cloud-native execution in K8s clusters
madengine-cli runner k8s \
    --inventory k8s_inventory.yml \
    --manifests-dir k8s-setup \
    --report-output k8s_execution_report.json \
    --verbose
```

#### Generate Commands
```bash
# Generate Ansible playbook for cluster deployment
madengine-cli generate ansible \
    --manifest-file build_manifest.json \
    --output cluster-deployment.yml

# Generate Kubernetes manifests
madengine-cli generate k8s \
    --manifest-file build_manifest.json \
    --namespace madengine-prod
```

### Command Options

**Global Options:**
- `--verbose, -v`: Enable detailed logging with rich output
- `--version`: Show version information

**Core Options:**
- `--tags, -t`: Model tags to process (multiple allowed)
- `--registry, -r`: Docker registry URL
- `--additional-context, -c`: Runtime context as JSON string
- `--additional-context-file, -f`: Runtime context from file
- `--timeout`: Execution timeout in seconds
- `--live-output, -l`: Real-time output streaming

**Build Configuration:**
- `--clean-docker-cache`: Rebuild without cache
- `--manifest-output, -m`: Build manifest output file
- `--summary-output, -s`: Summary report output file
- `--batch-manifest`: Input batch.json file for batch build mode

**Advanced Configuration:**
- `--data-config`: Custom data configuration file
- `--tools-config`: Custom tools configuration
- `--force-mirror-local`: Local data mirroring path
- `--disable-skip-gpu-arch`: Disable GPU architecture filtering
- `--sys-env-details`: Generate system config env details

## Distributed Execution

madengine supports sophisticated distributed execution scenarios, enabling separation of build and runtime environments for optimal resource utilization and scalability.

### Distributed Runner System

The MADEngine distributed runner system provides a unified interface for orchestrating workloads across multiple nodes and clusters using different infrastructure types (SSH, Ansible, Kubernetes).

#### Key Features

- **Modular Architecture**: Pluggable runner implementations for different infrastructure types
- **Unified Interface**: Consistent CLI and API across all runner types
- **Flexible Inventory**: Support for JSON and YAML inventory formats
- **Rich Reporting**: Detailed execution reports with performance metrics saved to specified output files
- **Error Handling**: Comprehensive error handling and recovery mechanisms
- **Parallel Execution**: Automatic parallel execution based on inventory configuration
- **Automated Setup**: Automatically clones ROCm/MAD repository and installs madengine on each node/pod
- **Environment Management**: Runs madengine from the MAD directory using default MODEL_DIR
- **Simplified Interface**: Streamlined command interface focusing on essential options (inventory, manifest/playbook files, and reporting)

#### Runner Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MADEngine CLI                               │
│                (madengine-cli runner)                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Runner Factory                               │
│              (RunnerFactory.create_runner)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Base Distributed Runner                         │
│                (BaseDistributedRunner)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   SSH Runner    │  │ Ansible Runner  │  │ Kubernetes      │
│                 │  │                 │  │ Runner          │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Container Runner                               │
│              (existing ContainerRunner)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Use Cases

#### 1. Single GPU Node (Development & Testing)
- Individual developers with dedicated GPU workstations
- Simplified workflow maintaining production patterns
- Local model development and validation

#### 2. Multi-Node GPU Clusters (Production)
- Enterprise environments with multiple GPU servers
- Parallel execution and resource sharing
- Centralized build with distributed execution

#### 3. Cloud-Native Deployments (Kubernetes)
- Modern cloud infrastructure with container orchestration
- Auto-scaling and resource management
- Integration with cloud services

#### 4. Hybrid Infrastructure (On-Premise + Cloud)
- Mixed on-premise and cloud resources
- Workload distribution and cost optimization
- Compliance and data locality requirements

#### 5. CI/CD Pipeline Integration
- Continuous integration for ML model validation
- Automated testing and quality gates
- Reproducible benchmarking workflows

### Runner Types

#### Node/Pod Preparation Process

Before executing any workload, all runners perform the following preparation steps on each node or pod:

1. **Clone ROCm/MAD Repository**: If the MAD directory doesn't exist, it clones the repository from `https://github.com/ROCm/MAD.git`. If it exists, it pulls the latest changes.

2. **Setup Virtual Environment**: Creates a Python virtual environment in the MAD directory (`MAD/venv/`).

3. **Install MADEngine**: Installs madengine and all dependencies using `pip install -r requirements.txt` from the MAD repository.

4. **Install Dependencies**: Installs all dependencies from the MAD repository's `requirements.txt` file, plus additional runner-specific dependencies (paramiko, scp, ansible-runner, kubernetes, PyYAML).

5. **Copy Supporting Files**: Copies essential files like:
   - `credential.json` - Authentication credentials
   - `data.json` - Data configuration
   - `models.json` - Model definitions
   - `build_manifest.json` - Build manifest from the build phase
   - `scripts/` directory - Supporting scripts

6. **Verify Installation**: Validates that `madengine-cli` is accessible and working properly.

7. **Execute from MAD Directory**: All madengine commands are executed from the MAD directory with the virtual environment activated, ensuring the default MODEL_DIR is used.

This preparation ensures that each node/pod has a complete, isolated MADEngine environment ready for container execution.

#### 1. SSH Runner

Executes models on remote nodes via SSH connections with automatic environment setup.

**Use Cases:**
- Individual GPU workstations
- Small to medium clusters
- Development and testing
- Simple deployment scenarios

**Features:**
- Direct SSH connections using paramiko
- Secure file transfer with SCP
- Parallel execution across nodes
- Real-time command output capture
- Automatic MAD repository cloning and setup
- Virtual environment management per node

**Installation:**
```bash
# SSH Runner dependencies
pip install madengine[ssh]
# Or manually: pip install paramiko>=2.7.0 scp>=0.14.0
```

**Example:**
```bash
madengine-cli runner ssh \
    --inventory inventory.yml \
    --manifest-file build_manifest.json \
    --report-output ssh_execution_report.json \
    --verbose
```

#### 2. Ansible Runner

Executes models using Ansible playbooks for orchestrated deployment with automated environment setup.

**Use Cases:**
- Large-scale clusters
- Complex deployment scenarios
- Configuration management
- Automated infrastructure setup

**Features:**
- Ansible playbook generation
- Inventory management
- Parallel execution with Ansible
- Rich error reporting and recovery
- Automated MAD repository setup across all nodes
- Consistent environment configuration

**Installation:**
```bash
# Ansible Runner dependencies
pip install madengine[ansible]
# Or manually: pip install ansible-runner>=2.0.0 PyYAML>=5.4.0
```

**Example:**
```bash
madengine-cli runner ansible \
    --inventory cluster.yml \
    --playbook madengine_distributed.yml \
    --report-output ansible_execution_report.json \
    --verbose
```

#### 3. Kubernetes Runner

Executes models as Kubernetes Jobs in a cluster with containerized MAD environment setup.

**Use Cases:**
- Cloud-native deployments
- Container orchestration
- Auto-scaling scenarios
- Enterprise Kubernetes clusters

**Features:**
- Dynamic Job creation
- ConfigMap management
- Resource management
- Namespace isolation
- Containerized MAD environment setup
- Automatic git repository cloning in pods

**Installation:**
```bash
# Kubernetes Runner dependencies
pip install madengine[kubernetes]
# Or manually: pip install kubernetes>=20.0.0 PyYAML>=5.4.0
```

**Example:**
```bash
madengine-cli runner k8s \
    --inventory k8s_inventory.yml \
    --manifests-dir k8s-setup \
    --report-output k8s_execution_report.json \
    --verbose
```

### Inventory Configuration

#### SSH/Ansible Inventory (inventory.yml)

```yaml
# Simple format
nodes:
  - hostname: "gpu-node-1"
    address: "192.168.1.101"
    port: 22
    username: "root"
    ssh_key_path: "~/.ssh/id_rsa"
    gpu_count: 4
    gpu_vendor: "AMD"
    labels:
      gpu_architecture: "gfx908"
      datacenter: "dc1"
    environment:
      ROCR_VISIBLE_DEVICES: "0,1,2,3"

# Ansible-style format
gpu_nodes:
  - hostname: "gpu-node-2"
    address: "192.168.1.102"
    port: 22
    username: "madengine"
    ssh_key_path: "/opt/keys/madengine_key"
    gpu_count: 8
    gpu_vendor: "NVIDIA"
    labels:
      gpu_architecture: "V100"
      datacenter: "dc2"
    environment:
      CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

#### Kubernetes Inventory (k8s_inventory.yml)

```yaml
# Pod specifications
pods:
  - name: "madengine-pod-1"
    node_selector:
      gpu-type: "amd"
      gpu-architecture: "gfx908"
    resources:
      requests:
        amd.com/gpu: "2"
      limits:
        amd.com/gpu: "2"
    gpu_count: 2
    gpu_vendor: "AMD"
    environment:
      ROCR_VISIBLE_DEVICES: "0,1"
      MAD_GPU_ARCH: "gfx908"

# Node selectors
node_selectors:
  - labels:
      gpu-type: "nvidia"
      instance-type: "gpu-xlarge"
    gpu_count: 8
    gpu_vendor: "NVIDIA"
    environment:
      CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

#### Node Selector Examples

Filter nodes based on criteria:

```bash
# GPU vendor filtering
--node-selector '{"gpu_vendor": "AMD"}'

# Label-based filtering
--node-selector '{"datacenter": "dc1", "gpu_architecture": "gfx908"}'

# Multiple criteria
--node-selector '{"gpu_vendor": "NVIDIA", "instance-type": "gpu-large"}'
```

#### Additional Context Examples

Pass runtime configuration:

```bash
# Basic context
--additional-context '{"timeout_multiplier": 2.0}'

# GPU configuration
--additional-context '{"tools": [{"name": "rocprof"}], "gpu_vendor": "AMD"}'

# Complex context
--additional-context '{"docker_env_vars": {"ROCR_VISIBLE_DEVICES": "0,1"}, "timeout_multiplier": 1.5}'
```

### Examples

#### Example 1: Development Testing

Test a model on a single GPU workstation:

```bash
# SSH to single node
madengine-cli runner ssh \
    --inventory dev_inventory.yml \
    --manifest-file build_manifest.json \
    --tags dummy \
    --timeout 1800 \
    --verbose
```

#### Example 2: Multi-Node Cluster

Run models across multiple nodes in parallel:

```bash
# Ansible orchestration
madengine-cli runner ansible \
    --inventory cluster_inventory.yml \
    --manifest-file build_manifest.json \
    --tags dummy resnet bert \
    --parallelism 4 \
    --registry production.registry.com \
    --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}' \
    --report-output cluster_results.json
```

#### Example 3: Cloud Kubernetes Deployment

Deploy to cloud Kubernetes cluster:

```bash
# Generate manifests first
madengine-cli generate k8s \
    --manifest-file build_manifest.json \
    --namespace madengine-prod

# Run using the generated manifests
madengine-cli runner k8s \
    --inventory k8s_prod_inventory.yml \
    --manifests-dir k8s-manifests \
    --kubeconfig ~/.kube/prod_config

# Manifests are automatically applied by the runner
```

#### Example 4: AMD GPU Cluster

Specific configuration for AMD GPU cluster:

```bash
madengine-cli runner ansible \
    --inventory amd_cluster.yml \
    --manifest-file build_manifest.json \
    --tags pytorch_models \
    --node-selector '{"gpu_vendor": "AMD"}' \
    --additional-context '{"tools": [{"name": "rocprof"}], "gpu_vendor": "AMD", "guest_os": "UBUNTU"}' \
    --timeout 7200 \
    --parallelism 2 \
    --verbose
```

### Registry Integration

#### Automatic Registry Detection
The CLI automatically handles registry information:

```bash
# Build phase stores registry info in manifest
madengine-cli build --tags models --registry docker.io

# Run phase auto-detects registry from manifest
madengine-cli run --manifest-file build_manifest.json
```

#### Registry Credentials

Configure registry access in `credential.json`:

```json
{
  "dockerhub": {
    "repository": "your-repository",
    "username": "your-dockerhub-username",
    "password": "your-dockerhub-token"
  },
  "localhost:5000": {
    "repository": "local-repository",    
    "username": "local-registry-user", 
    "password": "local-registry-pass"
  },
  "my-registry.com": {
    "repository": "custon-repository",    
    "username": "custom-registry-user",
    "password": "custom-registry-token"
  }
}
```

**Registry Mapping:**
- `docker.io` or empty → uses `dockerhub` credentials
- `localhost:5000` → uses `localhost:5000` credentials
- Custom registries → uses registry URL as credential key

### Orchestration Integration

#### Ansible Deployment

```bash
# Generate Ansible playbook
madengine-cli generate ansible \
  --manifest-file build_manifest.json \
  --output cluster-deployment.yml

# Create inventory for GPU cluster
cat > gpu_inventory << EOF
[gpu_nodes]
gpu-01 ansible_host=192.168.1.101
gpu-02 ansible_host=192.168.1.102
gpu-03 ansible_host=192.168.1.103

[gpu_nodes:vars]
madengine_path=/opt/madengine
registry_url=production.registry.com
EOF

# Deploy to cluster
ansible-playbook -i gpu_inventory cluster-deployment.yml
```

#### Kubernetes Deployment

```bash
# Generate Kubernetes manifests
madengine-cli generate k8s \
  --manifest-file build_manifest.json \
  --namespace madengine-prod

# Deploy to cluster
kubectl create namespace madengine-prod
kubectl apply -f k8s-madengine-configmap.yaml
kubectl apply -f k8s-madengine-job.yaml

# Monitor execution
kubectl get jobs -n madengine-prod
kubectl logs -n madengine-prod job/madengine-job -f
```

## Configuration

### Context System

Contexts are runtime parameters that control model execution behavior:

```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU", 
  "timeout_multiplier": 2.0,
  "tools": [{"name": "rocprof"}]
}
```

**Required Fields for Build Operations:**
- `gpu_vendor`: AMD, NVIDIA, INTEL
- `guest_os`: UBUNTU, CENTOS, ROCKY

### Credential Management

Centralized authentication in `credential.json`:

```json
{
  "AMD_GITHUB": {
    "username": "github_username",
    "password": "github_token"
  },
  "dockerhub": {
    "username": "dockerhub_username", 
    "password": "dockerhub_token"
  },
  "MAD_AWS_S3": {
    "username": "aws_access_key",
    "password": "aws_secret_key"
  }
}
```

### Data Provider Configuration

Configure data sources in `data.json`:

```json
{
  "data_sources": {
    "model_data": {
        "nas": {
            "path": "/home/datum"
        },
        "minio": {
            "path": "s3://datasets/datum"
        },
        "aws": {
            "path": "s3://datasets/datum"
        }
    }
  }
}
```

### Tools Configuration

Customize build tools in `scripts/common/tools.json`:

```json
{
  "tools": {
    "rocprof": {
      "cmd": "rocprof",
      "env_vars": {...}
    },
    "nvprof": {
      "cmd": "nvprof",
      "env_vars": {...}
    }    
  }
}
```

### Environment Variables

madengine supports various environment variables for configuration and behavior control:

| Variable | Type | Description |
|----------|------|-------------|
| `MAD_VERBOSE_CONFIG` | boolean | Set to "true" to enable verbose configuration logging |
| `MAD_SETUP_MODEL_DIR` | boolean | Set to "true" to enable automatic MODEL_DIR setup during import |
| `MODEL_DIR` | string | Path to model directory to copy to current working directory |
| `MAD_MINIO` | JSON string | MinIO configuration for distributed storage |
| `MAD_AWS_S3` | JSON string | AWS S3 configuration for cloud storage |
| `NAS_NODES` | JSON string | NAS nodes configuration for network storage |
| `PUBLIC_GITHUB_ROCM_KEY` | JSON string | GitHub token configuration for ROCm access |

**Configuration Priority:**
1. Environment variables (as JSON strings)
2. `credential.json` file
3. Built-in defaults

**Example Usage:**
```bash
# Enable verbose logging
export MAD_VERBOSE_CONFIG=true

# Configure AWS S3 access
export MAD_AWS_S3='{"username": "aws_access_key", "password": "aws_secret_key"}'

# Set model directory
export MODEL_DIR=/path/to/models
```

## Advanced Usage

### Custom Timeouts

```bash
# Model-specific timeout in models.json
{"timeout": 3600}

# Command-line timeout override  
madengine-cli run --tags models --timeout 7200

# No timeout (run indefinitely)
madengine-cli run --tags models --timeout 0
```

### Performance Profiling

```bash
# Enable GPU profiling
madengine run --tags pyt_huggingface_bert \
  --additional-context '{"tools": [{"name":"rocprof"}]}'

# Memory and performance monitoring  
madengine-cli run --tags models --live-output --verbose \
  --summary-output detailed_metrics.json
```

### Local Data Mirroring

```bash
# Force local mirroring for all workloads
madengine-cli run --tags models --force-mirror-local /tmp/mirror

# Configure per-model in data.json
{
  "mirrorlocal": "/path/to/local/mirror"
}
```

### Development and Debugging

```bash
# Keep containers alive for debugging
madengine-cli run --tags models --keep-alive --keep-model-dir

# Skip model execution (build/setup only)
madengine-cli run --tags models --skip-model-run

# Detailed logging with stack traces
madengine-cli run --tags models --verbose
```

## Deployment Scenarios

### Scenario 1: AI Research Lab

**Setup**: Multiple GPU workstations, shared storage, local registry  
**Goal**: Compare models across different GPU types

```bash
# Central build server
madengine-cli build --tags research_models --registry lab-registry:5000 \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# Distribute via shared storage
cp build_manifest.json /shared/nfs/madengine/

# Execute on researcher workstations
madengine-cli run --manifest-file /shared/nfs/madengine/build_manifest.json \
  --live-output --timeout 7200
```

### Scenario 2: Cloud Service Provider

**Setup**: Kubernetes cluster, CI/CD pipeline, cloud registry  
**Goal**: ML benchmarking as a service

```bash
# CI/CD build pipeline
madengine-cli build --tags customer_models --registry gcr.io/ml-bench \
  --additional-context-file customer_context.json

# Alternative: Use batch manifest for selective builds
madengine-cli build --batch-manifest customer_models.json \
  --registry gcr.io/ml-bench \
  --additional-context-file customer_context.json

# Generate K8s deployment
madengine-cli generate k8s \
  --manifest-file build_manifest.json \
  --namespace customer-bench-${CUSTOMER_ID}

# Auto-scaling deployment  
kubectl apply -f k8s-manifests/ --namespace customer-bench-${CUSTOMER_ID}
```
### Scenario 3: Data Center

**Setup**: Large-scale on-premise data center with heterogeneous GPU nodes  
**Goal**: Centralized model benchmarking and resource utilization optimization

```bash
# Centralized build on dedicated build server
madengine-cli build --tags datacenter_models --registry dc-registry.local \
  --additional-context '{"gpu_vendor": "NVIDIA", "guest_os": "UBUNTU"}' \
  --summary-output datacenter_build_$(date +%Y%m%d).json

# Distribute manifest to compute nodes via shared storage or automation
cp datacenter_build_$(date +%Y%m%d).json /mnt/shared/madengine/

# Execute distributed runs across GPU nodes using Ansible
madengine-cli runner ansible \
  --inventory datacenter_inventory.yml \
  --manifest-file /mnt/shared/madengine/datacenter_build_$(date +%Y%m%d).json \
  --tags datacenter_models \
  --parallelism 8 \
  --report-output datacenter_results.json \
  --verbose
```

## Best Practices

### 1. Inventory Management

- **Version Control**: Store inventory files in version control
- **Environment Separation**: Use different inventories for dev/test/prod
- **Documentation**: Document node purposes and configurations
- **Validation**: Validate inventory files before use

### 2. Security

- **SSH Keys**: Use SSH keys instead of passwords
- **Least Privilege**: Use dedicated user accounts with minimal permissions
- **Network Security**: Restrict network access to necessary ports
- **Credential Management**: Store credentials securely

### 3. Performance Optimization

- **Parallelism**: Tune parallelism based on cluster size and network capacity
- **Resource Allocation**: Match resource requests to actual needs
- **Timeout Management**: Set appropriate timeouts for different model types
- **Registry Optimization**: Use local or nearby registries for faster pulls

### 4. Error Handling

- **Retry Logic**: Implement retry logic for transient failures
- **Monitoring**: Monitor execution progress and resource usage
- **Logging**: Enable verbose logging for troubleshooting
- **Cleanup**: Ensure proper cleanup of resources on failure

### 5. Scalability

- **Horizontal Scaling**: Add more nodes rather than larger nodes
- **Load Balancing**: Distribute workloads evenly across nodes
- **Resource Monitoring**: Monitor cluster resource usage
- **Auto-scaling**: Use Kubernetes HPA for dynamic scaling

## Troubleshooting

### Common Issues

#### 1. SSH Connection Failures

**Problem**: Cannot connect to nodes via SSH

**Solutions:**
- Check network connectivity: `ping <node_address>`
- Verify SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
- Test manual SSH: `ssh -i ~/.ssh/id_rsa user@node`
- Check SSH service: `systemctl status sshd`

#### 2. Ansible Playbook Errors

**Problem**: Ansible playbook execution fails

**Solutions:**
- Test Ansible connectivity: `ansible all -i inventory.yml -m ping`
- Check Python installation on nodes: `ansible all -i inventory.yml -m setup`
- Verify inventory format: `ansible-inventory -i inventory.yml --list`
- Run with increased verbosity: `--verbose`

#### 3. Kubernetes Job Failures

**Problem**: Kubernetes Jobs fail to start or complete

**Solutions:**
- Check cluster status: `kubectl get nodes`
- Verify namespace: `kubectl get namespaces`
- Check resource quotas: `kubectl describe quota -n madengine`
- Inspect job logs: `kubectl logs job/madengine-job -n madengine`

#### 4. Docker Image Pull Failures

**Problem**: Cannot pull Docker images on nodes

**Solutions:**
- Test registry connectivity: `docker pull <registry>/<image>`
- Check registry credentials: `docker login <registry>`
- Verify image exists: `docker images`
- Check network access to registry

#### 5. GPU Resource Issues

**Problem**: GPU not detected or allocated

**Solutions:**
- Check GPU drivers: `nvidia-smi` or `rocm-smi`
- Verify GPU resource labels: `kubectl describe nodes`
- Check device plugin status: `kubectl get pods -n kube-system`
- Validate GPU configuration in inventory

#### 6. MAD Environment Setup Issues

**Problem**: MAD repository cloning or madengine installation fails

**Solutions:**
- Check network connectivity to GitHub: `ping github.com`
- Verify git is installed: `git --version`
- Check Python version: `python3 --version`
- Verify pip is available: `pip --version`
- Check disk space: `df -h`
- Manually test git clone: `git clone https://github.com/ROCm/MAD.git`

#### 7. Virtual Environment Issues

**Problem**: Virtual environment creation or activation fails

**Solutions:**
- Check python3-venv package: `apt install python3-venv` (Ubuntu/Debian)
- Verify Python path: `which python3`
- Check permissions in working directory
- Manually test venv creation: `python3 -m venv test_venv`

### Debugging Tips

1. **Enable Verbose Logging**: Always use `--verbose` for troubleshooting
2. **Check Resource Usage**: Monitor CPU, memory, and GPU usage
3. **Validate Inventory**: Test inventory files with small workloads first
4. **Test Network Connectivity**: Ensure all nodes can communicate
5. **Review Logs**: Check logs on all nodes for error messages

### Performance Optimization

1. **Network Optimization**:
   - Use fast network connections (10GbE or better)
   - Minimize network latency between nodes
   - Use local registries when possible

2. **Resource Allocation**:
   - Match CPU and memory requests to actual needs
   - Avoid resource over-subscription
   - Use appropriate GPU counts per node

3. **Parallelism Tuning**:
   - Start with low parallelism and increase gradually
   - Monitor resource usage during execution
   - Consider network bandwidth limitations

4. **Storage Optimization**:
   - Use fast storage (NVMe SSD) for temporary files
   - Implement proper cleanup of temporary files
   - Consider using shared storage for large datasets

## API Reference

### Command Line Interface

```bash
# Build Command
madengine-cli build [OPTIONS]

# Run Command  
madengine-cli run [OPTIONS]

# Generate Commands
madengine-cli generate <ansible|k8s> [OPTIONS]

# Runner Commands
madengine-cli runner <ssh|ansible|k8s> [OPTIONS]
```

### Build Command Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--tags` | `-t` | Model tags to build (can specify multiple) | `[]` |
| `--registry` | `-r` | Docker registry to push images to | `None` |
| `--batch-manifest` | | Input batch.json file for batch build mode | `None` |
| `--additional-context` | `-c` | Additional context as JSON string | `"{}"` |
| `--additional-context-file` | `-f` | File containing additional context JSON | `None` |
| `--clean-docker-cache` | | Rebuild images without using cache | `false` |
| `--manifest-output` | `-m` | Output file for build manifest | `build_manifest.json` |
| `--summary-output` | `-s` | Output file for build summary JSON | `None` |
| `--live-output` | `-l` | Print output in real-time | `false` |
| `--verbose` | `-v` | Enable verbose logging | `false` |

### Run Command Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--tags` | `-t` | Model tags to run (can specify multiple) | `[]` |
| `--manifest-file` | `-m` | Build manifest file path | `""` |
| `--registry` | `-r` | Docker registry URL | `None` |
| `--timeout` | | Timeout for model run in seconds | `-1` |
| `--additional-context` | `-c` | Additional context as JSON string | `"{}"` |
| `--additional-context-file` | `-f` | File containing additional context JSON | `None` |
| `--keep-alive` | | Keep Docker containers alive after run | `false` |
| `--keep-model-dir` | | Keep model directory after run | `false` |
| `--skip-model-run` | | Skip running the model | `false` |
| `--live-output` | `-l` | Print output in real-time | `false` |
| `--verbose` | `-v` | Enable verbose logging | `false` |

### Runner Types

- `ssh`: SSH-based distributed runner
- `ansible`: Ansible-based distributed runner  
- `k8s`: Kubernetes-based distributed runner

### Build Modes

- **Tag-based builds**: `--tags dummy resnet` - Build specific models by tags
- **Batch builds**: `--batch-manifest batch.json` - Build from batch manifest file with selective building

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--inventory, -i` | Path to inventory file | `inventory.yml` |
| `--manifest-file, -m` | Build manifest file | `build_manifest.json` |
| `--report-output` | Report output file | `runner_report.json` |
| `--verbose, -v` | Enable verbose logging | `false` |

### Runner-Specific Options

#### SSH Runner

| Option | Description | Default |
|--------|-------------|---------|
| `--inventory, -i` | Path to inventory file (YAML or JSON format) | `inventory.yml` |
| `--manifest-file, -m` | Build manifest file (generated by 'madengine-cli build') | `build_manifest.json` |
| `--report-output` | Output file for execution report | `runner_report.json` |

#### Ansible Runner

| Option | Description | Default |
|--------|-------------|---------|
| `--inventory, -i` | Path to inventory file (YAML or JSON format) | `inventory.yml` |
| `--playbook` | Path to Ansible playbook file (generated by 'madengine-cli generate ansible') | `madengine_distributed.yml` |
| `--report-output` | Output file for execution report | `runner_report.json` |

#### Kubernetes Runner

| Option | Description | Default |
|--------|-------------|---------|
| `--inventory, -i` | Path to inventory file (YAML or JSON format) | `inventory.yml` |
| `--manifests-dir, -d` | Directory containing Kubernetes manifests (generated by 'madengine-cli generate k8s') | `k8s-setup` |
| `--kubeconfig` | Path to kubeconfig file | Auto-detected |
| `--report-output` | Output file for execution report | `runner_report.json` |

### Exit Codes

- `0`: Success
- `1`: General failure
- `2`: Build failure
- `3`: Run failure
- `4`: Invalid arguments

## Contributing

We welcome contributions to madengine! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone git@github.com:yourusername/madengine.git
cd madengine

# Install development dependencies
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest

# Code formatting and linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/madengine
```

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive tests
- Update documentation for new features
- Use semantic commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Legacy Commands Reference

For compatibility with existing workflows, the traditional CLI commands remain available:

### Model Execution
```bash
madengine run --tags pyt_huggingface_bert --live-output \
  --additional-context '{"guest_os": "UBUNTU"}'
```

### Model Discovery  
```bash
madengine discover --tags dummy
madengine discover --tags dummy2:dummy_2
madengine discover --tags dummy3:dummy_3:batch_size=512
```

### Report Generation
```bash
madengine report to-html --csv-file-path perf.csv
madengine report to-email --csv-file-path perf.csv
madengine report update-perf --perf-csv perf.csv
```

### Database Operations
```bash
madengine database create-table
madengine database update-table --csv-file-path perf.csv
madengine database upload-mongodb --type model --file-path data.json
```

### GPU Tools Integration
```bash
# GPU profiling with ROCm
madengine run --tags models \
  --additional-context '{"tools": [{"name":"rocprof"}]}'

# Library tracing
madengine run --tags models \
  --additional-context '{"tools": [{"name":"trace"}]}'
```

---

**Note**: You cannot use backslash '/' or colon ':' characters in model names or tags within `models.json` or `get_models_json.py` scripts, as these are reserved for the hierarchical tag system.

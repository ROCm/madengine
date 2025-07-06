# madengine Distributed Execution Solution

## Overview

This solution splits the madengine `run_models.py` workflow into separate **build** and **run** phases to enable distributed execution scenarios such as:

- **Ansible**: Build images on a central host, distribute and run on multiple GPU nodes
- **Kubernetes**: Build images in CI/CD, deploy as jobs across GPU clusters
- **Multi-node setups**: Build once, run on multiple remote nodes with different GPU configurations

## Architecture

### Original Flow Problem
The original `run_models.py` has a tightly coupled flow:
```
Model Discovery → Docker Build → Container Run → Performance Collection
```

### New Split Architecture
```
BUILD PHASE (Central Host):
  Model Discovery → Docker Build → Push to Registry → Export Manifest

RUN PHASE (Remote Nodes): 
  Load Manifest → Pull Images → Container Run → Performance Collection
```

## Components

### 1. DockerBuilder (`docker_builder.py`)
Handles the Docker image building phase:
- Builds images for all discovered models
- Pushes images to a registry (optional)
- Exports a build manifest with image metadata
- Supports credential handling and build arguments

### 2. ContainerRunner (`container_runner.py`)
Handles the container execution phase:
- Loads build manifest from build phase
- Pulls images from registry if needed
- Runs containers with proper GPU, mount, and environment configurations
- Collects performance metrics and results

### 3. DistributedOrchestrator (`distributed_orchestrator.py`)
Coordinates the distributed workflow:
- Manages both build and run phases
- Supports complete workflows or individual phases
- Generates deployment configurations for external tools
- Handles credential and context management

### 4. Distributed CLI (`distributed_cli.py`)
Command-line interface for distributed operations:
- `build` - Build images and create manifest
- `run` - Smart command that either runs execution-only (if manifest exists) or complete workflow (build + run)
- `export-config` - Export execution configuration for external tools
- `generate ansible` - Create Ansible playbooks
- `generate k8s` - Create Kubernetes manifests

## Usage Examples

### 1. Basic Split Workflow

**Build Phase (on CI/Build server):**
```bash
# Build all models and push to registry
python -m madengine.tools.distributed_cli build \
    --registry localhost:5000 \
    --clean-docker-cache \
    --manifest-output build_manifest.json

# This creates:
# - build_manifest.json (contains image info, model info, build metadata)
# - Images pushed to localhost:5000 registry
```

**Run Phase (on GPU nodes):**
```bash
# Copy build_manifest.json to GPU nodes, then:
python -m madengine.tools.distributed_cli run \
    --manifest-file build_manifest.json \
    --timeout 3600

# Registry information is automatically detected from the manifest
# No need to specify --registry parameter unless you want to override
```

### 2. Smart Run Command (Complete Workflow)

The `run` command is smart and can automatically detect whether to perform execution-only or complete workflow:

**Complete Workflow (when no manifest exists):**
```bash
# Automatically runs build + run phases
python -m madengine.tools.distributed_cli run \
    --registry localhost:5000 \
    --timeout 3600 \
    --clean-docker-cache
```

### 3. Ansible Deployment

**Export execution configuration:**
```bash
# Export execution configuration for external tools
python -m madengine.tools.distributed_cli export-config \
    --output execution_config.json
```

**Generate Ansible playbook:**
```bash
# Generate Ansible playbook using the manifest and config
python -m madengine.tools.distributed_cli generate ansible \
    --manifest-file build_manifest.json \
    --execution-config execution_config.json \
    --output madengine_distributed.yml
```

**Run with Ansible:**
```bash
# Deploy to GPU cluster
ansible-playbook -i gpu_inventory madengine_distributed.yml
```

### 4. Kubernetes Deployment

**Export execution configuration:**
```bash
# Export execution configuration for external tools
python -m madengine.tools.distributed_cli export-config \
    --output execution_config.json
```

**Generate K8s manifests:**
```bash
python -m madengine.tools.distributed_cli generate k8s \
    --manifest-file build_manifest.json \
    --execution-config execution_config.json \
    --namespace madengine-prod
```

**Deploy to Kubernetes:**
```bash
kubectl apply -f k8s-madengine-configmap.yaml
kubectl apply -f k8s-madengine-job.yaml
```

**Note**: The generated Kubernetes manifests are templates that should be customized for your environment:
- Update the `nodeSelector` to match your GPU node labels
- Adjust resource requests/limits based on model requirements  
- Modify the container image to use your actual distributed runner image
- Update GPU resource types (nvidia.com/gpu vs amd.com/gpu) based on your hardware
- Update the command to use the correct distributed CLI: `python3 -m madengine.tools.distributed_cli run --manifest-file=/config/manifest.json`

### 5. Configuration Export

The `export-config` command allows you to export execution configurations that can be used by external orchestration tools:

```bash
# Export configuration with specific tags
python -m madengine.tools.distributed_cli export-config \
    --tags llama bert \
    --output execution_config.json

# Export configuration for all discovered models
python -m madengine.tools.distributed_cli export-config \
    --output execution_config.json
```

The exported configuration includes:
- Model discovery information
- Required credentials
- Docker environment variables and mounts
- GPU configuration details

This is useful for integrating madengine with external tools like CI/CD pipelines, monitoring systems, or custom orchestration frameworks.

### 6. Smart Run Command Behavior

The `run` command in the distributed CLI is intelligent and automatically detects the appropriate workflow based on the arguments provided:

#### Execution-Only Mode
When a `--manifest-file` is provided **and** the file exists:
```bash
# Only runs the execution phase using existing manifest
# Registry is automatically detected from the manifest
python -m madengine.tools.distributed_cli run \
    --manifest-file build_manifest.json \
    --timeout 3600

# Optional: Override registry from manifest
python -m madengine.tools.distributed_cli run \
    --manifest-file build_manifest.json \
    --registry custom-registry.com \
    --timeout 3600

# Note: No --tags parameter needed when using manifest file
# The manifest contains both built images and model information
# ensuring exact reproduction of the build configuration
```

#### Complete Workflow Mode  
When **no** `--manifest-file` is provided **or** the manifest file doesn't exist:
```bash
# Runs both build and execution phases
python -m madengine.tools.distributed_cli run \
    --tags resnet \
    --registry localhost:5000 \
    --clean-docker-cache \
    --timeout 3600
```

This smart behavior eliminates the need for a separate `full` command and makes the CLI more intuitive to use.

### 7. CLI Examples Summary

Here are some comprehensive examples of using the distributed CLI:

```bash
# Build models with specific tags and push to registry
python -m madengine.tools.distributed_cli build \
    --tags llama bert resnet \
    --registry localhost:5000 --clean-docker-cache

# Run models using pre-built manifest with auto-detected registry (execution-only)
# No --registry needed - registry is auto-detected from the manifest
python -m madengine.tools.distributed_cli run \
    --manifest-file build_manifest.json --timeout 3600

# Complete workflow with specific tags and registry (build + run)
python -m madengine.tools.distributed_cli run \
    --tags resnet --registry localhost:5000 --timeout 3600 --live-output

# Export configuration for external orchestration tools
python -m madengine.tools.distributed_cli export-config \
    --tags llama --output execution_config.json

# Generate Ansible playbook for distributed execution
python -m madengine.tools.distributed_cli generate ansible \
    --manifest-file build_manifest.json \
    --execution-config execution_config.json \
    --output madengine.yml

# Generate Kubernetes manifests with custom namespace
python -m madengine.tools.distributed_cli generate k8s \
    --namespace madengine-prod --tags llama
```

### 8. Advanced CLI Usage

The distributed CLI supports all standard madengine arguments for model filtering and execution control:

#### Model Selection and Filtering
```bash
# Build specific models by tags
python -m madengine.tools.distributed_cli build \
    --tags llama bert resnet \
    --registry localhost:5000

# Build with additional context for custom base images
python -m madengine.tools.distributed_cli build \
    --additional-context "{'docker_build_arg':{'BASE_DOCKER':'custom:latest'}}" \
    --registry localhost:5000

# Build with context file
python -m madengine.tools.distributed_cli build \
    --additional-context-file context.json \
    --registry localhost:5000
```

#### Execution Control
```bash
# Run with custom timeout and keep containers alive for debugging
# Registry auto-detected from manifest
python -m madengine.tools.distributed_cli run \
    --manifest-file build_manifest.json \
    --timeout 7200 \
    --keep-alive \
    --live-output

# Override registry if needed (fallback mode)
python -m madengine.tools.distributed_cli run \
    --manifest-file build_manifest.json \
    --registry custom-registry.com \
    --tags llama \
    --timeout 3600
```

#### Data Configuration
```bash
# Use custom data configuration
python -m madengine.tools.distributed_cli full \
    --data-config-file-name custom_data.json \
    --force-mirror-local /shared/data \
    --registry localhost:5000
```

#### Build Optimization
```bash
# Clean build without cache for reproducible images
python -m madengine.tools.distributed_cli build \
    --clean-docker-cache \
    --registry localhost:5000

# Save detailed build and execution summaries
python -m madengine.tools.distributed_cli full \
    --registry localhost:5000 \
    --summary-output full_workflow_summary.json
```

## Integration with Existing madengine

### Minimal Changes Required

The solution maintains compatibility with existing madengine components:

1. **Context System**: Uses existing `Context` class for configuration
2. **Data Provider**: Integrates with existing `Data` class for data management  
3. **Docker Integration**: Uses existing `Docker` class for container management
4. **Model Discovery**: Uses existing `DiscoverModels` for finding models

### Migration Path

1. **Immediate**: Use new distributed CLI for split workflows
2. **Gradual**: Migrate existing workflows to use distributed orchestrator
3. **Full Integration**: Replace `run_models.py` with distributed orchestrator

## Step-by-Step: Building and Running a Single Model

This section provides a complete walkthrough for building and running a single model (`dummy`) in a distributed scenario, from initial setup to deployment on GPU nodes.

### Prerequisites

1. **Docker Registry**: A accessible Docker registry (local or remote)
2. **GPU Node(s)**: Target machines with GPU drivers and Docker installed
3. **Network Access**: GPU nodes can access the Docker registry
4. **madengine**: Installed on build machine and GPU nodes

### Phase 1: Build and Prepare (Central Build Machine)

#### Step 1: Navigate to madengine Directory
```bash
cd /path/to/madengine
```

#### Step 2: Build the Dummy Model
```bash
# Build just the dummy model and push to registry
python -m madengine.tools.distributed_cli build \
    --tags dummy \
    --registry localhost:5000 \
    --manifest-output dummy_build_manifest.json \
    --summary-output dummy_build_summary.json
```

This will:
- Discover models with the "dummy" tag
- Build Docker images for the dummy model variants
- Push images to the registry at `localhost:5000`
- Create `dummy_build_manifest.json` with build metadata
- Generate `dummy_build_summary.json` with build status

#### Step 3: Verify Build Results
```bash
# Check build summary for any failures
cat dummy_build_summary.json

# Example successful output:
{
  "successful_builds": [
    {
      "model_name": "dummy",
      "image_tag": "localhost:5000/madengine/dummy:latest",
      "build_time": "2024-01-15T10:30:00Z",
      "image_size": "1.2GB"
    }
  ],
  "failed_builds": [],
  "total_build_time": 180.5,
  "registry_url": "localhost:5000"
}
```

#### Step 4: Export Execution Configuration (Optional)
```bash
# Export configuration for external orchestration tools
python -m madengine.tools.distributed_cli export-config \
    --tags dummy \
    --output dummy_execution_config.json
```

### Phase 2: Manual Deployment to GPU Node

#### Step 5: Transfer Manifest to GPU Node
```bash
# Copy manifest to GPU node (replace gpu-node-01 with actual hostname/IP)
scp dummy_build_manifest.json user@gpu-node-01:/home/user/madengine/
```

#### Step 6: Run on GPU Node
```bash
# SSH to GPU node
ssh user@gpu-node-01

# Navigate to madengine directory on GPU node
cd /home/user/madengine

# Run the dummy model using the manifest
# Registry is automatically detected from the manifest
python -m madengine.tools.distributed_cli run \
    --manifest-file dummy_build_manifest.json \
    --timeout 1800 \
    --live-output \
    --summary-output dummy_execution_summary.json
```

#### Step 7: Verify Execution Results
```bash
# Check execution summary
cat dummy_execution_summary.json

# Example successful output:
{
  "successful_runs": [
    {
      "model_name": "dummy",
      "execution_time": 45.2,
      "gpu_used": "GPU-0",
      "peak_gpu_memory": "2.1GB",
      "exit_code": 0,
      "output_file": "perf.csv"
    }
  ],
  "failed_runs": [],
  "total_execution_time": 45.2,
  "gpu_node": "gpu-node-01"
}

# Check performance results
head perf.csv
```

### Phase 3: Automated Deployment with Ansible

#### Step 8: Generate Ansible Playbook
```bash
# Back on build machine - generate Ansible playbook
python -m madengine.tools.distributed_cli generate ansible \
    --manifest-file dummy_build_manifest.json \
    --execution-config dummy_execution_config.json \
    --output dummy_ansible_playbook.yml
```

#### Step 9: Create Ansible Inventory
```bash
# Create inventory file for your GPU nodes
cat > gpu_inventory << EOF
[gpu_nodes]
gpu-node-01 ansible_host=192.168.1.101 ansible_user=madengine
gpu-node-02 ansible_host=192.168.1.102 ansible_user=madengine

[gpu_nodes:vars]
madengine_path=/home/madengine/madengine
registry_url=localhost:5000
EOF
```

#### Step 10: Deploy with Ansible
```bash
# Run Ansible playbook to deploy to all GPU nodes
ansible-playbook -i gpu_inventory dummy_ansible_playbook.yml

# Check results on all nodes
ansible gpu_nodes -i gpu_inventory -m shell -a "cat /home/madengine/madengine/perf.csv | head -5"
```

### Phase 4: Kubernetes Deployment

#### Step 11: Generate Kubernetes Manifests
```bash
# Generate K8s manifests for the dummy model
python -m madengine.tools.distributed_cli generate k8s \
    --manifest-file dummy_build_manifest.json \
    --execution-config dummy_execution_config.json \
    --namespace madengine-dummy
```

#### Step 12: Customize Kubernetes Manifests
```bash
# Edit the generated manifests to match your cluster
# Update k8s-madengine-job.yaml:
# - nodeSelector for GPU nodes
# - Resource requests/limits  
# - GPU resource type (nvidia.com/gpu or amd.com/gpu)
# - Image registry URLs

vim k8s-madengine-job.yaml
```

#### Step 13: Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace madengine-dummy

# Apply manifests
kubectl apply -f k8s-madengine-configmap.yaml
kubectl apply -f k8s-madengine-job.yaml

# Monitor job progress
kubectl get jobs -n madengine-dummy
kubectl get pods -n madengine-dummy
kubectl logs -n madengine-dummy job/madengine-dummy-job

# Get results
kubectl get configmap madengine-results -n madengine-dummy -o yaml
```

### Key Benefits of This Workflow

1. **Separation of Concerns**: Build once on a central machine, run anywhere
2. **Resource Efficiency**: GPU nodes don't need build dependencies  
3. **Scalability**: Easy to run on multiple nodes simultaneously
4. **Reproducibility**: Same Docker images ensure consistent results
5. **Integration**: Works with existing orchestration tools (Ansible, K8s)

### Troubleshooting Single Model Deployment

#### Common Issues and Solutions

**Build Phase Issues:**
```bash
# Check Docker registry connectivity
docker login localhost:5000
docker images | grep dummy

# Verify model discovery
python -m madengine.tools.discover_models --tags dummy
```

**Run Phase Issues:**
```bash
# Check image pull from registry
docker pull localhost:5000/madengine/dummy:latest

# Verify GPU availability
nvidia-smi  # or rocm-smi for AMD GPUs

# Check Docker GPU runtime
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

**Network Issues:**
```bash
# Test registry connectivity from GPU node
curl -v http://localhost:5000/v2/_catalog

# Check firewall rules for registry port
sudo ufw status | grep 5000
```

### Performance Considerations for Single Model

1. **Image Size**: The dummy model image is relatively small (~1.2GB), making it ideal for testing
2. **Runtime**: Typical execution time is 30-60 seconds
3. **Memory**: Requires ~2GB GPU memory
4. **Network**: Image transfer time depends on registry bandwidth

This single-model workflow serves as a foundation for scaling up to multi-model, multi-node distributed execution scenarios.

## Quick Reference: Minimal Single-Model Workflow

For quick deployment of a single model in a distributed scenario, here's the minimal command sequence:

### Manual Deployment (Build Machine → GPU Node)

**Build Phase:**
```bash
# 1. Build and push model
python -m madengine.tools.distributed_cli build --tags dummy --registry localhost:5000

# 2. Transfer manifest
scp build_manifest.json user@gpu-node:/path/to/madengine/
```

**Run Phase (on GPU node):**
```bash
# 3. Run model (registry auto-detected from manifest)
python -m madengine.tools.distributed_cli run --manifest-file build_manifest.json
```

### Ansible Deployment (Build Machine → Multiple GPU Nodes)

```bash
# 1. Build and export config
python -m madengine.tools.distributed_cli build --tags dummy --registry localhost:5000
python -m madengine.tools.distributed_cli export-config --tags dummy

# 2. Generate and run Ansible playbook
python -m madengine.tools.distributed_cli generate ansible
ansible-playbook -i gpu_inventory madengine_distributed.yml
```

### Kubernetes Deployment (CI/CD → K8s Cluster)

```bash
# 1. Build and export config (in CI/CD)
python -m madengine.tools.distributed_cli build --tags dummy --registry my-registry.com
python -m madengine.tools.distributed_cli export-config --tags dummy

# 2. Generate and deploy K8s manifests
python -m madengine.tools.distributed_cli generate k8s --namespace madengine-prod
kubectl apply -f k8s-madengine-configmap.yaml
kubectl apply -f k8s-madengine-job.yaml
```

**Key Files Generated:**
- `build_manifest.json` - Contains built image metadata and execution info
- `execution_config.json` - Runtime configuration for external tools  
- `*_summary.json` - Build/execution status and metrics
- `madengine_distributed.yml` - Ansible playbook
- `k8s-madengine-*.yaml` - Kubernetes manifests

**Next Steps:**
- Scale to multiple models by using different `--tags` filters
- Integrate with your existing CI/CD pipeline using the `export-config` command
- Monitor execution using the summary JSON files for automated reporting
- Customize Ansible/K8s templates for your infrastructure requirements

### 9. Build Manifest Format

The build manifest has been enhanced to ensure reliable execution across distributed environments:

#### Enhanced Manifest Structure
```json
{
  "built_images": {
    "ci-dummy_ubuntu_amd": {
      "docker_image": "ci-dummy_ubuntu_amd",
      "dockerfile": "/path/to/dummy.ubuntu.amd.Dockerfile",
      "base_docker": "ubuntu:22.04",
      "build_duration": 45.2,
      "registry_image": "localhost:5000/ci-dummy_ubuntu_amd"
    }
  },
  "built_models": {
    "ci-dummy_ubuntu_amd": {
      "name": "dummy",
      "path": "/scripts/dummy",
      "tags": ["dummy", "test"],
      "dockerfile": "/path/to/dummy.ubuntu.amd.Dockerfile"
    }
  },
  "registry": "localhost:5000",
  "context": {
    "docker_env_vars": {},
    "docker_mounts": {},
    "docker_build_arg": {}
  }
}
```

#### Key Improvements

1. **Model Information Storage**: The manifest now includes `built_models` that maps each built image to its corresponding model information
2. **Registry Auto-Detection**: The manifest includes top-level `registry` field for automatic registry detection during execution
3. **Exact Reproduction**: No need to specify `--tags` or `--registry` during execution when using a manifest file
4. **Backward Compatibility**: Falls back to name-based matching for older manifest files
5. **Reliable Matching**: Direct image-to-model mapping eliminates matching errors

#### Execution Behavior

**With Enhanced Manifest (Recommended):**
```bash
# Build phase creates enhanced manifest with registry information
python -m madengine.tools.distributed_cli build --tags dummy --registry localhost:5000

# Run phase uses stored model and registry information - no additional parameters needed
python -m madengine.tools.distributed_cli run --manifest-file build_manifest.json
```

**Fallback Mode (Legacy Manifests):**
```bash
# For older manifests without built_models, uses name-based matching
python -m madengine.tools.distributed_cli run \
    --manifest-file legacy_manifest.json \
    --tags dummy  # May need tags for discovery
```

This improvement addresses the common issue where models discovered during execution don't match the built images, ensuring consistent and reliable distributed execution.

# MADEngine CLI Refactoring Plan - Production Ready

> **Version**: 2.0  
> **Last Updated**: November 28, 2025  
> **Status**: Draft for Review

---

## Executive Summary

This document outlines a comprehensive refactoring plan for madengine-cli to simplify distributed execution while maintaining backward compatibility with the legacy madengine. The refactoring focuses on three deployment scenarios: **Local**, **SLURM**, and **Kubernetes**, eliminating the complex SSH/Ansible runner infrastructure.

### Key Objectives

1. **Simplify deployment model** - Three clear scenarios instead of complex runner abstraction
2. **Leverage existing strengths** - Keep proven build/run phase implementations
3. **Clarify terminology** - Separate infrastructure (Docker/K8s/SLURM) from execution methods (torchrun/deepspeed)
4. **Maintain compatibility** - Zero breaking changes to legacy madengine
5. **Production ready** - Template-based, testable, and maintainable solution

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Architecture Clarification](#2-architecture-clarification)
3. [Proposed Solution](#3-proposed-solution)
4. [Implementation Plan](#4-implementation-plan)
5. [Migration Strategy](#5-migration-strategy)
6. [Testing Strategy](#6-testing-strategy)
7. [Timeline & Milestones](#7-timeline--milestones)

---

## 1. PROBLEM ANALYSIS

### 1.1 Current Issues

**Terminology Confusion**:
- Current "runners" (SSH/Ansible/K8s/SLURM) distribute **madengine execution itself**
- But users need to distribute **model workload execution** (using torchrun, deepspeed, etc.)
- This creates confusion between "infrastructure" and "execution method"

**Complexity**:
- Four runner types (SSH, Ansible, K8s, SLURM) with different abstractions
- Complex setup process (clone MAD, setup venv, install madengine on each node)
- Not aligned with how K8s and SLURM are actually used in practice

**K8s/SLURM Usage Gap**:
- **K8s Reality**: Users deploy pods with model containers directly, not madengine containers
- **SLURM Reality**: Users submit sbatch scripts that run models, not madengine setup scripts
- Current implementation adds unnecessary indirection

### 1.2 What Works Well (Keep These)

✅ **Build Phase** (`DockerBuilder`):
- Model discovery via tags
- Docker image building with GPU architecture support
- Registry push/pull
- Manifest generation

✅ **Run Phase** (`ContainerRunner`):
- Local Docker container execution
- GPU device mapping
- Performance metric collection
- Timeout management

✅ **Core Components**:
- Context (GPU detection, environment)
- DataProvider (data source management)
- Model discovery system
- Error handling framework

---

## 2. ARCHITECTURE CLARIFICATION

### 2.1 Terminology Alignment

**Infrastructure Layer** (Where workload runs):
```
┌─────────────────────────────────────────────────────┐
│ Infrastructure Targets                              │
├─────────────────────────────────────────────────────┤
│ • Local:      Docker on current node               │
│ • SLURM:      HPC cluster with job scheduler        │
│ • Kubernetes: Container orchestration platform      │
└─────────────────────────────────────────────────────┘
```

**Execution Methods** (How model runs within container):
```
┌─────────────────────────────────────────────────────┐
│ Execution Launchers (Inside Container)             │
├─────────────────────────────────────────────────────┤
│ • Single GPU:    python train.py                   │
│ • Multi GPU:     torchrun --nproc_per_node=8       │
│ • Distributed:   torchrun --nnodes=4               │
│ • DeepSpeed:     deepspeed --hostfile=...          │
│ • Megatron:      Megatron-LM launcher              │
└─────────────────────────────────────────────────────┘
```

**madengine's Scope**:
- ✅ Handles **infrastructure layer** (where to run)
- ✅ Builds Docker images with model code
- ❌ Does NOT implement execution methods (models handle this)

### 2.2 Correct Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        User Commands                           │
│  madengine-cli build    # Build Docker images                  │
│  madengine-cli run      # Run locally OR deploy to infra       │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                   Build Phase (Keep As-Is)                     │
│  • DiscoverModels                                              │
│  • DockerBuilder                                               │
│  • Generate build_manifest.json                                │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────┴────────────────┬───────────────────┐
        │                                 │                   │
        ▼                                 ▼                   ▼
┌──────────────┐              ┌──────────────┐    ┌──────────────┐
│ Local Run    │              │ SLURM Deploy │    │ K8s Deploy   │
│ (Existing)   │              │ (New)        │    │ (New)        │
├──────────────┤              ├──────────────┤    ├──────────────┤
│• Pull image  │              │• Gen sbatch  │    │• Gen pod.yaml│
│• Run container│             │• Submit job  │    │• kubectl apply│
│• Collect perf│              │• Monitor     │    │• Monitor     │
└──────────────┘              └──────────────┘    └──────────────┘
```

### 2.3 Reference Projects Analysis

**K8s Demo (`/home/ysha/amd/k8s-demo`)**:
- Pattern: Generate pod.yaml → `kubectl apply -f pod.yaml`
- Pod runs model container directly (not madengine)
- Simple, straightforward deployment

**SGLang Disagg (`/home/ysha/playground/MAD-private/scripts/sglang_disagg`)**:
- Pattern: Generate sbatch script → `sbatch job.sh`
- Script runs model directly (not madengine setup)
- Uses SLURM for resource allocation

**Primus Project** (https://github.com/AMD-AGI/Primus):
- Supports multiple backends (Megatron-LM, TorchTitan, JAX MaxText)
- Infrastructure-agnostic (can run on SLURM, K8s, etc.)
- madengine should orchestrate infrastructure, Primus handles execution

---

## 3. PROPOSED SOLUTION

### 3.1 Simplified Command Structure

**Three Deployment Modes** (specified via `--additional-context`):

```bash
# Mode 1: Local (Default - existing behavior)
madengine-cli run --tags dummy

# Mode 2: SLURM Deployment
madengine-cli run --tags dummy \
  --additional-context '{"deploy": "slurm", "slurm": {...config...}}'

# Mode 3: Kubernetes Deployment
madengine-cli run --tags dummy \
  --additional-context '{"deploy": "k8s", "k8s": {...config...}}'
```

**Remove These Commands**:
- ❌ `madengine-cli generate` (replaced by automatic template generation)
- ❌ `madengine-cli runner ssh/ansible/k8s/slurm` (replaced by unified `run` with deploy mode)

### 3.2 Enhanced build_manifest.json

Add deployment configuration to manifest:

```json
{
  "registry": "docker.io",
  "deployment": {
    "target": "local|slurm|k8s",
    "config": {
      // Target-specific configuration
    }
  },
  "built_images": {
    "model_name": {
      "docker_image": "ci-model_dockerfile",
      "registry_image": "docker.io/org/model:tag",
      // Existing fields...
      
      // New: Execution configuration
      "execution": {
        "launcher": "torchrun",  // or "python", "deepspeed"
        "nnodes": 4,
        "nproc_per_node": 8,
        "master_addr": "auto",   // Auto-configured by infra
        "master_port": 29500
      }
    }
  }
}
```

### 3.3 New Directory Structure

```
src/madengine/
├── mad.py                      # Legacy CLI (keep, deprecate gradually)
├── mad_cli.py                  # Modern CLI (refactor)
│
├── core/                       # Keep as-is (stable foundation)
│   ├── context.py
│   ├── docker.py
│   ├── dataprovider.py
│   └── ...
│
├── tools/                      # Keep existing tools
│   ├── discover_models.py     # Keep
│   ├── docker_builder.py      # Keep
│   ├── container_runner.py    # Keep + enhance
│   ├── distributed_orchestrator.py  # Refactor → deployment_orchestrator.py
│   └── ...
│
├── deployment/                 # NEW: Deployment infrastructure
│   ├── __init__.py
│   ├── base.py                # BaseDeployment abstract class
│   ├── local.py               # LocalDeployment (wraps existing)
│   ├── slurm.py               # SlurmDeployment (new)
│   ├── kubernetes.py          # KubernetesDeployment (new)
│   ├── factory.py             # DeploymentFactory
│   └── templates/             # Jinja2 templates
│       ├── slurm/
│       │   ├── job.sh.j2
│       │   └── job_array.sh.j2
│       └── kubernetes/
│           ├── pod.yaml.j2
│           ├── job.yaml.j2
│           └── deployment.yaml.j2
│
└── runners/                    # DEPRECATED (to be removed)
    └── ... (keep for now, mark deprecated)
```

---

## 4. IMPLEMENTATION PLAN

### 4.1 Phase 1: Foundation (Week 1-2)

#### 4.1.1 Create Deployment Abstraction

**File**: `src/madengine/deployment/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class DeploymentConfig:
    """Base configuration for deployments"""
    target: str  # "local", "slurm", "k8s"
    manifest_file: str
    timeout: int = 3600
    namespace: Optional[str] = None  # For K8s
    partition: Optional[str] = None  # For SLURM
    
    # Common execution settings
    launcher: str = "python"  # "python", "torchrun", "deepspeed"
    nnodes: int = 1
    nproc_per_node: int = 1
    
    # Additional context
    context: Dict[str, Any] = None


@dataclass
class DeploymentResult:
    """Result of deployment operation"""
    status: str  # "success", "failed", "pending"
    deployment_id: str
    message: str
    metrics: Dict[str, Any] = None
    logs_path: Optional[str] = None


class BaseDeployment(ABC):
    """Abstract base class for all deployment targets"""
    
    DEPLOYMENT_TYPE: str = "base"
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.manifest = self._load_manifest(config.manifest_file)
    
    def _load_manifest(self, manifest_file: str) -> Dict:
        """Load build manifest"""
        import json
        with open(manifest_file) as f:
            return json.load(f)
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate deployment configuration and requirements"""
        pass
    
    @abstractmethod
    def prepare(self) -> bool:
        """Prepare deployment (generate configs, check resources)"""
        pass
    
    @abstractmethod
    def deploy(self) -> DeploymentResult:
        """Execute deployment"""
        pass
    
    @abstractmethod
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Monitor deployment status"""
        pass
    
    @abstractmethod
    def collect_results(self, deployment_id: str) -> Dict:
        """Collect execution results and metrics"""
        pass
    
    @abstractmethod
    def cleanup(self, deployment_id: str) -> bool:
        """Cleanup deployment resources"""
        pass
    
    def execute(self) -> DeploymentResult:
        """Full deployment workflow"""
        if not self.validate():
            return DeploymentResult(
                status="failed",
                deployment_id="",
                message="Validation failed"
            )
        
        if not self.prepare():
            return DeploymentResult(
                status="failed",
                deployment_id="",
                message="Preparation failed"
            )
        
        result = self.deploy()
        
        if result.status == "success":
            # Monitor until completion
            while True:
                status = self.monitor(result.deployment_id)
                if status.status in ["success", "failed"]:
                    break
            
            # Collect results
            results = self.collect_results(result.deployment_id)
            result.metrics = results
        
        return result
```

#### 4.1.2 Implement LocalDeployment

**File**: `src/madengine/deployment/local.py`

```python
from .base import BaseDeployment, DeploymentConfig, DeploymentResult
from madengine.tools.container_runner import ContainerRunner


class LocalDeployment(BaseDeployment):
    """Local deployment using existing ContainerRunner"""
    
    DEPLOYMENT_TYPE = "local"
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.runner = ContainerRunner(
            context=self._get_context(),
            live_output=config.context.get("live_output", False)
        )
    
    def validate(self) -> bool:
        """Validate local deployment requirements"""
        # Check Docker is available
        # Check GPU if required
        return True
    
    def prepare(self) -> bool:
        """Prepare local deployment"""
        # Existing ContainerRunner handles this
        return True
    
    def deploy(self) -> DeploymentResult:
        """Execute local deployment using ContainerRunner"""
        try:
            # Use existing run_models_from_manifest
            summary = self.runner.run_models_from_manifest(
                manifest_file=self.config.manifest_file,
                timeout=self.config.timeout
            )
            
            return DeploymentResult(
                status="success",
                deployment_id="local",
                message="Local execution completed",
                metrics=summary
            )
        except Exception as e:
            return DeploymentResult(
                status="failed",
                deployment_id="local",
                message=f"Execution failed: {e}"
            )
    
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Local deployment completes immediately"""
        return DeploymentResult(
            status="success",
            deployment_id=deployment_id,
            message="Complete"
        )
    
    def collect_results(self, deployment_id: str) -> Dict:
        """Results already collected during execution"""
        return {}
    
    def cleanup(self, deployment_id: str) -> bool:
        """No cleanup needed for local"""
        return True
```

#### 4.1.3 Create DeploymentFactory

**File**: `src/madengine/deployment/factory.py`

```python
from typing import Dict, Type
from .base import BaseDeployment, DeploymentConfig


class DeploymentFactory:
    """Factory for creating deployment instances"""
    
    _deployments: Dict[str, Type[BaseDeployment]] = {}
    
    @classmethod
    def register(cls, deployment_type: str, deployment_class: Type[BaseDeployment]):
        """Register a deployment type"""
        cls._deployments[deployment_type] = deployment_class
    
    @classmethod
    def create(cls, config: DeploymentConfig) -> BaseDeployment:
        """Create deployment instance based on config"""
        deployment_class = cls._deployments.get(config.target)
        
        if not deployment_class:
            available = ", ".join(cls._deployments.keys())
            raise ValueError(
                f"Unknown deployment target: {config.target}. "
                f"Available: {available}"
            )
        
        return deployment_class(config)
    
    @classmethod
    def available_deployments(cls) -> list:
        """Get list of available deployment types"""
        return list(cls._deployments.keys())


def register_default_deployments():
    """Register default deployment types"""
    from .local import LocalDeployment
    DeploymentFactory.register("local", LocalDeployment)
    
    try:
        from .slurm import SlurmDeployment
        DeploymentFactory.register("slurm", SlurmDeployment)
    except ImportError:
        pass
    
    try:
        from .kubernetes import KubernetesDeployment
        DeploymentFactory.register("k8s", KubernetesDeployment)
        DeploymentFactory.register("kubernetes", KubernetesDeployment)
    except ImportError:
        pass


# Auto-register on import
register_default_deployments()
```

---

### 4.2 Phase 2: SLURM Deployment (Week 3-4)

#### 4.2.1 SLURM Template

**File**: `src/madengine/deployment/templates/slurm/job.sh.j2`

```bash
#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --output={{ output_dir }}/{{ job_name }}_%A_%a.out
#SBATCH --error={{ output_dir }}/{{ job_name }}_%A_%a.err
#SBATCH --partition={{ partition }}
#SBATCH --nodes={{ nnodes }}
#SBATCH --ntasks-per-node={{ nproc_per_node }}
#SBATCH --gres=gpu:{{ nproc_per_node }}
#SBATCH --time={{ time_limit }}
{% if array_tasks %}
#SBATCH --array={{ array_tasks }}
{% endif %}
{% if qos %}
#SBATCH --qos={{ qos }}
{% endif %}
{% if account %}
#SBATCH --account={{ account }}
{% endif %}

# Load modules
{% for module in modules %}
module load {{ module }}
{% endfor %}

# Set environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT={{ master_port }}
export WORLD_SIZE={{ world_size }}
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# GPU visibility
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Model-specific environment
{% for key, value in env_vars.items() %}
export {{ key }}="{{ value }}"
{% endfor %}

# Docker pull (if using Singularity/Apptainer)
{% if use_container %}
singularity pull {{ container_image }}
{% endif %}

# Execute model
cd {{ work_dir }}

{% if launcher == "torchrun" %}
torchrun \
    --nnodes={{ nnodes }} \
    --nproc_per_node={{ nproc_per_node }} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    {{ script_path }} {{ script_args }}
{% elif launcher == "deepspeed" %}
deepspeed \
    --num_nodes={{ nnodes }} \
    --num_gpus={{ nproc_per_node }} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    {{ script_path }} {{ script_args }}
{% elif launcher == "docker" %}
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --network=host \
    -v {{ work_dir }}:/workspace \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e WORLD_SIZE=$WORLD_SIZE \
    -e RANK=$RANK \
    {{ container_image }} \
    bash -c "cd /workspace && {{ run_command }}"
{% else %}
# Direct execution
{{ run_command }}
{% endif %}

# Collect results
echo "Job completed with exit code $?"
```

#### 4.2.2 SLURM Deployment Implementation

**File**: `src/madengine/deployment/slurm.py`

```python
import os
import subprocess
import time
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from .base import BaseDeployment, DeploymentConfig, DeploymentResult


class SlurmDeployment(BaseDeployment):
    """SLURM HPC deployment"""
    
    DEPLOYMENT_TYPE = "slurm"
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        
        # SLURM-specific config
        slurm_config = config.context.get("slurm", {})
        self.login_node = slurm_config.get("login_node")
        self.partition = config.partition or slurm_config.get("partition", "gpu")
        self.output_dir = slurm_config.get("output_dir", "./slurm_output")
        self.work_dir = slurm_config.get("work_dir", os.getcwd())
        
        # Setup Jinja2 for template rendering
        template_dir = Path(__file__).parent / "templates" / "slurm"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    def validate(self) -> bool:
        """Validate SLURM deployment requirements"""
        # Check if sbatch is available (or SSH to login node)
        if self.login_node:
            # SSH validation
            result = subprocess.run(
                ["ssh", self.login_node, "which", "sbatch"],
                capture_output=True
            )
            return result.returncode == 0
        else:
            # Local sbatch
            result = subprocess.run(["which", "sbatch"], capture_output=True)
            return result.returncode == 0
    
    def prepare(self) -> bool:
        """Prepare SLURM deployment (generate sbatch scripts)"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate sbatch script for each model
        for model_name, model_info in self.manifest["built_images"].items():
            job_script = self._generate_job_script(model_name, model_info)
            
            script_path = Path(self.output_dir) / f"{model_name}_job.sh"
            with open(script_path, "w") as f:
                f.write(job_script)
            
            # Make executable
            os.chmod(script_path, 0o755)
        
        return True
    
    def _generate_job_script(self, model_name: str, model_info: dict) -> str:
        """Generate sbatch script using Jinja2 template"""
        template = self.jinja_env.get_template("job.sh.j2")
        
        # Prepare template context
        execution = model_info.get("execution", {})
        
        context = {
            "job_name": model_name,
            "output_dir": self.output_dir,
            "partition": self.partition,
            "nnodes": execution.get("nnodes", self.config.nnodes),
            "nproc_per_node": execution.get("nproc_per_node", self.config.nproc_per_node),
            "time_limit": self._format_time(self.config.timeout),
            "master_port": execution.get("master_port", 29500),
            "world_size": execution.get("nnodes", 1) * execution.get("nproc_per_node", 1),
            "modules": self.config.context.get("slurm", {}).get("modules", []),
            "env_vars": self.config.context.get("env_vars", {}),
            "launcher": self.config.launcher,
            "container_image": model_info.get("registry_image"),
            "work_dir": self.work_dir,
            "run_command": self._get_run_command(model_info),
        }
        
        return template.render(**context)
    
    def _get_run_command(self, model_info: dict) -> str:
        """Get the run command from model info"""
        # Default: run.sh from model scripts
        return "./run.sh"
    
    def _format_time(self, seconds: int) -> str:
        """Format timeout in SLURM time format (HH:MM:SS)"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def deploy(self) -> DeploymentResult:
        """Submit SLURM jobs"""
        job_ids = []
        
        for model_name in self.manifest["built_images"].keys():
            script_path = Path(self.output_dir) / f"{model_name}_job.sh"
            
            # Submit job
            if self.login_node:
                cmd = ["ssh", self.login_node, "sbatch", str(script_path)]
            else:
                cmd = ["sbatch", str(script_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse job ID from output: "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
            else:
                return DeploymentResult(
                    status="failed",
                    deployment_id="",
                    message=f"Failed to submit {model_name}: {result.stderr}"
                )
        
        return DeploymentResult(
            status="success",
            deployment_id=",".join(job_ids),
            message=f"Submitted {len(job_ids)} SLURM jobs"
        )
    
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Monitor SLURM job status"""
        job_ids = deployment_id.split(",")
        
        # Check status using squeue
        if self.login_node:
            cmd = ["ssh", self.login_node, "squeue", "-j", deployment_id, "-h"]
        else:
            cmd = ["squeue", "-j", deployment_id, "-h"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if not result.stdout.strip():
            # Job completed or not found
            return DeploymentResult(
                status="success",
                deployment_id=deployment_id,
                message="Jobs completed"
            )
        else:
            # Jobs still running
            return DeploymentResult(
                status="pending",
                deployment_id=deployment_id,
                message=f"{len(job_ids)} jobs running"
            )
    
    def collect_results(self, deployment_id: str) -> Dict:
        """Collect results from SLURM output files"""
        results = {}
        
        for model_name in self.manifest["built_images"].keys():
            # Parse output files
            pattern = f"{self.output_dir}/{model_name}_job_*.out"
            output_files = glob.glob(pattern)
            
            for output_file in output_files:
                # Parse performance metrics from output
                # This depends on model output format
                pass
        
        return results
    
    def cleanup(self, deployment_id: str) -> bool:
        """Cleanup SLURM jobs if needed"""
        # Cancel any remaining jobs
        job_ids = deployment_id.split(",")
        
        if self.login_node:
            cmd = ["ssh", self.login_node, "scancel"] + job_ids
        else:
            cmd = ["scancel"] + job_ids
        
        subprocess.run(cmd, capture_output=True)
        return True
```

---

### 4.3 Phase 3: Kubernetes Deployment (Week 5-6)

#### 4.3.1 Kubernetes Templates

**File**: `src/madengine/deployment/templates/kubernetes/job.yaml.j2`

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ job_name }}
  namespace: {{ namespace }}
  labels:
    app: madengine
    model: {{ model_name }}
spec:
  completions: {{ nnodes }}
  parallelism: {{ nnodes }}
  template:
    metadata:
      labels:
        app: madengine
        model: {{ model_name }}
    spec:
      restartPolicy: OnFailure
      
      {% if node_selector %}
      nodeSelector:
        {% for key, value in node_selector.items() %}
        {{ key }}: {{ value }}
        {% endfor %}
      {% endif %}
      
      containers:
      - name: {{ model_name }}
        image: {{ container_image }}
        imagePullPolicy: Always
        
        command: ["/bin/bash", "-c"]
        args:
          - |
            # Set distributed environment
            export MASTER_ADDR={{ master_addr }}
            export MASTER_PORT={{ master_port }}
            export WORLD_SIZE={{ world_size }}
            export RANK=${JOB_COMPLETION_INDEX:-0}
            export LOCAL_RANK=0
            
            {% for key, value in env_vars.items() %}
            export {{ key }}="{{ value }}"
            {% endfor %}
            
            # Execute model
            cd /workspace
            {% if launcher == "torchrun" %}
            torchrun \
              --nnodes={{ nnodes }} \
              --nproc_per_node={{ nproc_per_node }} \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
              --node_rank=$RANK \
              {{ script_path }} {{ script_args }}
            {% else %}
            {{ run_command }}
            {% endif %}
        
        resources:
          requests:
            {% if gpu_vendor == "AMD" %}
            amd.com/gpu: {{ nproc_per_node }}
            {% elif gpu_vendor == "NVIDIA" %}
            nvidia.com/gpu: {{ nproc_per_node }}
            {% endif %}
            memory: {{ memory }}
            cpu: {{ cpu }}
          limits:
            {% if gpu_vendor == "AMD" %}
            amd.com/gpu: {{ nproc_per_node }}
            {% elif gpu_vendor == "NVIDIA" %}
            nvidia.com/gpu: {{ nproc_per_node }}
            {% endif %}
            memory: {{ memory_limit }}
            cpu: {{ cpu_limit }}
        
        volumeMounts:
        - name: workspace
          mountPath: /workspace
        {% for volume in volumes %}
        - name: {{ volume.name }}
          mountPath: {{ volume.mount_path }}
        {% endfor %}
      
      volumes:
      - name: workspace
        emptyDir: {}
      {% for volume in volumes %}
      - name: {{ volume.name }}
        {% if volume.type == "pvc" %}
        persistentVolumeClaim:
          claimName: {{ volume.claim_name }}
        {% elif volume.type == "configmap" %}
        configMap:
          name: {{ volume.config_name }}
        {% endif %}
      {% endfor %}
```

#### 4.3.2 Kubernetes Deployment Implementation

**File**: `src/madengine/deployment/kubernetes.py`

```python
import os
import yaml
import subprocess
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from .base import BaseDeployment, DeploymentConfig, DeploymentResult


class KubernetesDeployment(BaseDeployment):
    """Kubernetes deployment"""
    
    DEPLOYMENT_TYPE = "k8s"
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        
        # K8s-specific config
        k8s_config = config.context.get("k8s", {})
        self.namespace = config.namespace or k8s_config.get("namespace", "default")
        self.kubeconfig = k8s_config.get("kubeconfig")
        self.output_dir = k8s_config.get("output_dir", "./k8s_manifests")
        
        # Setup Jinja2 for template rendering
        template_dir = Path(__file__).parent / "templates" / "kubernetes"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    def validate(self) -> bool:
        """Validate Kubernetes deployment requirements"""
        # Check kubectl is available
        result = subprocess.run(["which", "kubectl"], capture_output=True)
        if result.returncode != 0:
            return False
        
        # Check cluster connectivity
        cmd = ["kubectl", "cluster-info"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    def prepare(self) -> bool:
        """Prepare Kubernetes deployment (generate manifests)"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate Job manifest for each model
        for model_name, model_info in self.manifest["built_images"].items():
            job_manifest = self._generate_job_manifest(model_name, model_info)
            
            manifest_path = Path(self.output_dir) / f"{model_name}_job.yaml"
            with open(manifest_path, "w") as f:
                f.write(job_manifest)
        
        return True
    
    def _generate_job_manifest(self, model_name: str, model_info: dict) -> str:
        """Generate Kubernetes Job manifest using Jinja2 template"""
        template = self.jinja_env.get_template("job.yaml.j2")
        
        # Prepare template context
        execution = model_info.get("execution", {})
        k8s_config = self.config.context.get("k8s", {})
        
        context = {
            "job_name": self._sanitize_name(model_name),
            "model_name": model_name,
            "namespace": self.namespace,
            "nnodes": execution.get("nnodes", self.config.nnodes),
            "nproc_per_node": execution.get("nproc_per_node", self.config.nproc_per_node),
            "container_image": model_info.get("registry_image"),
            "master_addr": "madengine-master",  # Service name
            "master_port": execution.get("master_port", 29500),
            "world_size": execution.get("nnodes", 1) * execution.get("nproc_per_node", 1),
            "launcher": self.config.launcher,
            "env_vars": self.config.context.get("env_vars", {}),
            "gpu_vendor": k8s_config.get("gpu_vendor", "AMD"),
            "memory": k8s_config.get("memory", "32Gi"),
            "memory_limit": k8s_config.get("memory_limit", "64Gi"),
            "cpu": k8s_config.get("cpu", "8"),
            "cpu_limit": k8s_config.get("cpu_limit", "16"),
            "node_selector": k8s_config.get("node_selector", {}),
            "volumes": k8s_config.get("volumes", []),
            "run_command": self._get_run_command(model_info),
        }
        
        return template.render(**context)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for Kubernetes (lowercase, no underscores)"""
        return name.lower().replace("_", "-").replace("/", "-")
    
    def _get_run_command(self, model_info: dict) -> str:
        """Get the run command from model info"""
        return "./run.sh"
    
    def deploy(self) -> DeploymentResult:
        """Deploy to Kubernetes cluster"""
        job_names = []
        
        for model_name in self.manifest["built_images"].keys():
            manifest_path = Path(self.output_dir) / f"{model_name}_job.yaml"
            
            # Apply manifest
            cmd = ["kubectl", "apply", "-f", str(manifest_path)]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            cmd.extend(["-n", self.namespace])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                job_names.append(self._sanitize_name(model_name))
            else:
                return DeploymentResult(
                    status="failed",
                    deployment_id="",
                    message=f"Failed to deploy {model_name}: {result.stderr}"
                )
        
        return DeploymentResult(
            status="success",
            deployment_id=",".join(job_names),
            message=f"Deployed {len(job_names)} Kubernetes jobs"
        )
    
    def monitor(self, deployment_id: str) -> DeploymentResult:
        """Monitor Kubernetes job status"""
        job_names = deployment_id.split(",")
        
        # Check job status
        cmd = ["kubectl", "get", "jobs"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        cmd.extend(["-n", self.namespace, "-o", "json"])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return DeploymentResult(
                status="failed",
                deployment_id=deployment_id,
                message="Failed to get job status"
            )
        
        jobs_data = yaml.safe_load(result.stdout)
        
        # Check if all jobs completed
        all_completed = True
        for job in jobs_data.get("items", []):
            if job["metadata"]["name"] in job_names:
                status = job.get("status", {})
                if not status.get("succeeded"):
                    all_completed = False
                    break
        
        if all_completed:
            return DeploymentResult(
                status="success",
                deployment_id=deployment_id,
                message="All jobs completed"
            )
        else:
            return DeploymentResult(
                status="pending",
                deployment_id=deployment_id,
                message="Jobs running"
            )
    
    def collect_results(self, deployment_id: str) -> Dict:
        """Collect results from Kubernetes pods"""
        results = {}
        job_names = deployment_id.split(",")
        
        for job_name in job_names:
            # Get logs from completed pods
            cmd = [
                "kubectl", "logs",
                f"job/{job_name}",
                "-n", self.namespace
            ]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse logs for metrics
                results[job_name] = {
                    "logs": result.stdout
                }
        
        return results
    
    def cleanup(self, deployment_id: str) -> bool:
        """Cleanup Kubernetes resources"""
        job_names = deployment_id.split(",")
        
        for job_name in job_names:
            cmd = [
                "kubectl", "delete", "job", job_name,
                "-n", self.namespace
            ]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            subprocess.run(cmd, capture_output=True)
        
        return True
```

---

### 4.4 Phase 4: CLI Integration (Week 7)

#### 4.4.1 Refactor mad_cli.py

```python
# mad_cli.py updates

@app.command(name="run")
def run_command(
    tags: List[str] = typer.Option([], "--tags", "-t"),
    manifest_file: str = typer.Option("", "--manifest-file", "-m"),
    timeout: int = typer.Option(3600, "--timeout"),
    additional_context: str = typer.Option("{}", "--additional-context", "-c"),
    additional_context_file: Optional[str] = typer.Option(None, "--additional-context-file", "-f"),
    live_output: bool = typer.Option(False, "--live-output", "-l"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Run models locally or deploy to infrastructure (SLURM/K8s).
    
    Deployment mode is determined by --additional-context:
    
    Local (default):
      madengine-cli run --tags dummy
    
    SLURM deployment:
      madengine-cli run --tags dummy --additional-context '{"deploy": "slurm", "slurm": {...}}'
    
    Kubernetes deployment:
      madengine-cli run --tags dummy --additional-context '{"deploy": "k8s", "k8s": {...}}'
    """
    setup_logging(verbose)
    
    # Parse additional context
    context = validate_additional_context(additional_context, additional_context_file)
    
    # Determine deployment mode
    deploy_target = context.get("deploy", "local")
    
    # Build manifest if needed
    if not manifest_file:
        if not tags:
            console.print("[red]Error:[/red] Either --tags or --manifest-file must be provided")
            raise typer.Exit(ExitCode.INVALID_ARGS)
        
        # Execute build phase first
        console.print("[bold blue]Building Docker images...[/bold blue]")
        manifest_file = _execute_build_phase(tags, context)
    
    # Create deployment configuration
    from madengine.deployment.base import DeploymentConfig
    from madengine.deployment.factory import DeploymentFactory
    
    config = DeploymentConfig(
        target=deploy_target,
        manifest_file=manifest_file,
        timeout=timeout,
        namespace=context.get("k8s", {}).get("namespace"),
        partition=context.get("slurm", {}).get("partition"),
        launcher=context.get("launcher", "python"),
        nnodes=context.get("nnodes", 1),
        nproc_per_node=context.get("nproc_per_node", 1),
        context=context
    )
    
    # Create and execute deployment
    try:
        deployment = DeploymentFactory.create(config)
        
        console.print(f"\n[bold blue]Deploying to {deploy_target}...[/bold blue]")
        result = deployment.execute()
        
        if result.status == "success":
            console.print(f"[green]✓[/green] Deployment successful: {result.message}")
            if result.metrics:
                _display_metrics(result.metrics)
        else:
            console.print(f"[red]✗[/red] Deployment failed: {result.message}")
            raise typer.Exit(ExitCode.RUN_FAILURE)
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)
```

---

### 4.5 Phase 5: Deprecation & Documentation (Week 8)

#### 4.5.1 Mark Old Runners as Deprecated

```python
# src/madengine/runners/__init__.py

import warnings

warnings.warn(
    "The madengine.runners module is deprecated and will be removed in v2.0. "
    "Please use the new deployment API: madengine.deployment",
    DeprecationWarning,
    stacklevel=2
)
```

#### 4.5.2 Update Documentation

Create `docs/DEPLOYMENT_GUIDE.md` with examples for all three modes.

---

## 5. MIGRATION STRATEGY

### 5.1 Backward Compatibility

**Legacy madengine (mad.py)**:
- ✅ No changes required
- ✅ Continue to use existing core components
- ✅ All existing tests pass
- ⚠️  Mark as deprecated in documentation
- 📅 Remove in v3.0 (12+ months)

**Existing madengine-cli users**:
- ✅ Local execution unchanged
- ✅ `build` command unchanged
- ⚠️  `runner` commands deprecated (print warning)
- ⚠️  `generate` commands deprecated (auto-generated now)
- 📋 Provide migration guide

### 5.2 Migration Path

**For SSH/Ansible users** → Use Local deployment + your own orchestration:
```bash
# Old way (deprecated)
madengine-cli runner ssh --inventory nodes.yml

# New way (v2.0+)
# 1. Build on central node
madengine-cli build --tags models --registry your-registry

# 2. Deploy to each node using your orchestration
ansible-playbook -i inventory.yml deploy_local.yml
  # Playbook runs: madengine-cli run --manifest-file build_manifest.json

# Or use SSH loop
for node in node1 node2 node3; do
  ssh $node "madengine-cli run --manifest-file build_manifest.json"
done
```

**For K8s users** → Use K8s deployment:
```bash
# Old way (complex setup)
madengine-cli generate k8s --manifest-file manifest.json
madengine-cli runner k8s --inventory k8s.yml

# New way (simple)
madengine-cli run --tags models \
  --additional-context '{"deploy": "k8s", "k8s": {"namespace": "prod"}}'
```

**For SLURM users** → Use SLURM deployment:
```bash
# Old way (manual sbatch)
madengine-cli generate slurm --manifest-file manifest.json
# Then manually submit sbatch scripts

# New way (automated)
madengine-cli run --tags models \
  --additional-context '{"deploy": "slurm", "slurm": {"partition": "gpu"}}'
```

---

## 6. TESTING STRATEGY

### 6.1 Unit Tests

```python
# tests/deployment/test_local.py
def test_local_deployment(mock_container_runner):
    config = DeploymentConfig(
        target="local",
        manifest_file="test_manifest.json"
    )
    deployment = LocalDeployment(config)
    
    result = deployment.execute()
    assert result.status == "success"

# tests/deployment/test_slurm.py
def test_slurm_job_generation(mock_manifest):
    config = DeploymentConfig(
        target="slurm",
        manifest_file="test_manifest.json",
        partition="gpu"
    )
    deployment = SlurmDeployment(config)
    deployment.prepare()
    
    # Check sbatch script generated
    assert os.path.exists("slurm_output/model_job.sh")

# tests/deployment/test_kubernetes.py
def test_k8s_manifest_generation(mock_manifest):
    config = DeploymentConfig(
        target="k8s",
        manifest_file="test_manifest.json",
        namespace="test"
    )
    deployment = KubernetesDeployment(config)
    deployment.prepare()
    
    # Check Job manifest generated
    assert os.path.exists("k8s_manifests/model_job.yaml")
```

### 6.2 Integration Tests

```python
# tests/integration/test_end_to_end.py
@pytest.mark.integration
def test_local_end_to_end():
    """Test full workflow: build + local run"""
    # Build phase
    result = subprocess.run([
        "madengine-cli", "build",
        "--tags", "dummy",
        "--registry", "localhost:5000"
    ])
    assert result.returncode == 0
    
    # Run phase (local)
    result = subprocess.run([
        "madengine-cli", "run",
        "--manifest-file", "build_manifest.json"
    ])
    assert result.returncode == 0

@pytest.mark.slurm
def test_slurm_deployment():
    """Test SLURM deployment (requires SLURM cluster)"""
    result = subprocess.run([
        "madengine-cli", "run",
        "--manifest-file", "build_manifest.json",
        "--additional-context", '{"deploy": "slurm"}'
    ])
    assert result.returncode == 0
```

---

## 7. TIMELINE & MILESTONES

### Week 1-2: Foundation
- [x] Design review (this document)
- [ ] Create deployment/ module structure
- [ ] Implement BaseDeployment abstract class
- [ ] Implement LocalDeployment (wrap existing)
- [ ] Create DeploymentFactory
- [ ] Unit tests for foundation

**Deliverable**: Local deployment working via new API

### Week 3-4: SLURM
- [ ] Design SLURM Jinja2 templates
- [ ] Implement SlurmDeployment class
- [ ] Test template generation
- [ ] Test job submission (mock + real)
- [ ] Documentation

**Deliverable**: SLURM deployment working end-to-end

### Week 5-6: Kubernetes
- [ ] Design K8s Jinja2 templates (Job, Deployment)
- [ ] Implement KubernetesDeployment class
- [ ] Test manifest generation
- [ ] Test kubectl deployment (mock + real)
- [ ] Documentation

**Deliverable**: K8s deployment working end-to-end

### Week 7: CLI Integration
- [ ] Refactor mad_cli.py run command
- [ ] Add deployment mode detection
- [ ] Update argument parsing
- [ ] Integration tests
- [ ] CLI documentation

**Deliverable**: Unified CLI with all three deployment modes

### Week 8: Polish & Documentation
- [ ] Mark old runners as deprecated
- [ ] Create migration guide
- [ ] Update README.md
- [ ] Create DEPLOYMENT_GUIDE.md
- [ ] Add examples for all deployment modes
- [ ] Final testing

**Deliverable**: Production-ready v2.0 release

---

## 8. SUCCESS CRITERIA

### Technical
- [ ] All existing tests pass (backward compatibility)
- [ ] New deployment tests pass (local, SLURM, K8s)
- [ ] Template generation works correctly
- [ ] Performance equivalent or better than v1.x

### Usability
- [ ] Simpler CLI (fewer commands)
- [ ] Clear deployment model (3 modes)
- [ ] Better error messages
- [ ] Comprehensive documentation

### Maintainability
- [ ] Reduced code complexity
- [ ] Better separation of concerns
- [ ] Easier to add new deployment targets
- [ ] Clear deprecation path

---

## 9. RISKS & MITIGATION

### Risk 1: Breaking Changes
**Mitigation**: Extensive testing, deprecation warnings, migration guide

### Risk 2: Template Complexity
**Mitigation**: Start with simple templates, iterate based on real usage

### Risk 3: Cluster Access for Testing
**Mitigation**: Mock-based unit tests + optional integration tests

### Risk 4: User Adoption
**Mitigation**: Clear documentation, migration examples, both APIs work during transition

---

## APPENDIX A: Example Usage

### A.1 Local Execution

```bash
# Simple local run (unchanged)
madengine-cli run --tags dummy

# With explicit context
madengine-cli run --tags dummy \
  --additional-context '{"deploy": "local"}'
```

### A.2 SLURM Deployment

```bash
# Basic SLURM deployment
madengine-cli run --tags bert_training \
  --additional-context '{
    "deploy": "slurm",
    "launcher": "torchrun",
    "nnodes": 4,
    "nproc_per_node": 8,
    "slurm": {
      "partition": "gpu",
      "time_limit": 7200,
      "modules": ["rocm/5.7.0", "python/3.10"]
    }
  }'

# With config file
madengine-cli run --tags bert_training \
  --additional-context-file slurm_config.json
```

### A.3 Kubernetes Deployment

```bash
# Basic K8s deployment
madengine-cli run --tags llama_inference \
  --additional-context '{
    "deploy": "k8s",
    "launcher": "python",
    "nnodes": 2,
    "nproc_per_node": 4,
    "k8s": {
      "namespace": "ml-workloads",
      "gpu_vendor": "AMD",
      "memory": "64Gi",
      "node_selector": {"gpu-type": "mi250x"}
    }
  }'
```

---

## APPENDIX B: Configuration Examples

### B.1 SLURM Configuration

```json
{
  "deploy": "slurm",
  "launcher": "torchrun",
  "nnodes": 4,
  "nproc_per_node": 8,
  "slurm": {
    "login_node": "hpc-login.example.com",
    "partition": "gpu",
    "qos": "high",
    "account": "ml-research",
    "time_limit": 14400,
    "modules": [
      "rocm/5.7.0",
      "python/3.10",
      "git/2.40"
    ],
    "output_dir": "./slurm_jobs",
    "work_dir": "/projects/ml/experiments"
  },
  "env_vars": {
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_HCA": "mlx5_0"
  }
}
```

### B.2 Kubernetes Configuration

```json
{
  "deploy": "k8s",
  "launcher": "torchrun",
  "nnodes": 2,
  "nproc_per_node": 4,
  "k8s": {
    "namespace": "ml-prod",
    "kubeconfig": "~/.kube/config",
    "gpu_vendor": "AMD",
    "memory": "64Gi",
    "memory_limit": "128Gi",
    "cpu": "16",
    "cpu_limit": "32",
    "node_selector": {
      "gpu-type": "mi250x",
      "zone": "us-west1-a"
    },
    "volumes": [
      {
        "name": "data",
        "type": "pvc",
        "claim_name": "ml-data",
        "mount_path": "/data"
      }
    ],
    "output_dir": "./k8s_manifests"
  },
  "env_vars": {
    "NCCL_DEBUG": "INFO"
  }
}
```

---

**Document Status**: Ready for Review  
**Next Steps**: Approve plan → Begin Phase 1 implementation



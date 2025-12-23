# Execution Layer

**Status**: Active  
**Purpose**: Local Docker execution primitives for building and running containers

---

## 🎯 Responsibility

This layer handles low-level Docker operations:
- **Building** Docker images from Dockerfiles
- **Running** Docker containers locally
- **Managing** Docker lifecycle (create, start, stop, cleanup)

Used by the orchestration layer to execute Docker operations.

---

## 📦 Components

### **`docker_builder.py`**

Builds Docker images for models.

**Key Features:**
- Multi-architecture builds (GPU-specific compilation)
- Build argument injection (ROCm/CUDA versions, architectures)
- Registry push support (DockerHub, local registries)
- Build manifest generation
- Credential management

**Usage:**
```python
from madengine.execution.docker_builder import DockerBuilder

builder = DockerBuilder(context, console)

# Build single model
result = builder.build_image(
    model_info={"name": "model1", "dockerfile": "docker/model1.Dockerfile"},
    dockerfile="docker/model1.Dockerfile",
    phase_suffix="gfx90a"
)

# Build all models
results = builder.build_all_models(
    models_list=[model1, model2, model3],
    target_archs=["gfx90a", "gfx942"]
)

# Export build manifest
builder.export_build_manifest(output_file="build_manifest.json")
```

### **`container_runner.py`**

Runs Docker containers locally for model execution.

**Key Features:**
- GPU passthrough (ROCm, CUDA)
- Volume mounting (data, scripts, results)
- Resource limits (GPU, CPU, memory)
- Timeout management
- Performance metrics collection
- Container cleanup

**Usage:**
```python
from madengine.execution.container_runner import ContainerRunner

runner = ContainerRunner(context, data, console)

# Run model in container
result = runner.run_container(
    model_info=model_dict,
    model_docker=docker_client,
    gpu_ids="0,1",
    timeout=3600
)

# Result includes status, metrics, logs
print(result["status"])  # "successful", "failed", "timeout"
print(result["duration"])
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│   Orchestration Layer               │
│   (build_orchestrator.py,           │
│    run_orchestrator.py)             │
└─────────────┬───────────────────────┘
              │ uses
    ┌─────────┴─────────┐
    │                   │
┌───▼──────────┐  ┌─────▼──────────┐
│ docker_builder│  │container_runner│  ← This Layer
│  (build)      │  │    (run)       │
└───┬──────────┘  └─────┬──────────┘
    │                   │
    └─────────┬─────────┘
              │ uses
    ┌─────────▼─────────┐
    │   Core Layer      │
    │   (docker.py,     │
    │    context.py)    │
    └───────────────────┘
```

---

## 🔄 Workflow

### **Build Phase**

1. `BuildOrchestrator` discovers models
2. `BuildOrchestrator` calls `DockerBuilder.build_all_models()`
3. `DockerBuilder` builds each model with target architectures
4. `DockerBuilder` generates `build_manifest.json`

### **Run Phase**

1. `RunOrchestrator` loads `build_manifest.json`
2. `RunOrchestrator` calls `ContainerRunner.run_container()`
3. `ContainerRunner` executes model in Docker container
4. `ContainerRunner` collects metrics and writes results
5. Performance data saved via `reporting/update_perf_csv.py`

---

## 🎯 Design Principles

1. **Single Responsibility**: Each component does ONE thing
   - `docker_builder.py` = Build images
   - `container_runner.py` = Run containers

2. **Separation from Logic**: This layer is **execution only**
   - ❌ No workflow decisions (that's orchestration)
   - ❌ No model discovery (that's utils)
   - ✅ Pure Docker operations

3. **Reusability**: Can be used by:
   - Modern `madengine` CLI (via orchestrators)
   - Future automation scripts

4. **Testability**: Mock Docker client for unit tests

---

## 🧪 Testing

```bash
# Test docker builder
pytest tests/test_docker_builder.py -v

# Test container runner
pytest tests/test_container_runner.py -v

# Test multi-GPU architecture support
pytest tests/test_multi_gpu_arch.py -v
```

---

## 📚 Related Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Orchestration** | `orchestration/` | High-level workflow coordination |
| **Deployment** | `deployment/` | Distributed execution (SLURM, K8s) |
| **Core** | `core/` | Docker client, Context, Console |
| **Utils** | `utils/` | GPU tools, validators |

---

## 🔍 Key Differences

**Execution vs Deployment:**

| Aspect | Execution Layer | Deployment Layer |
|--------|----------------|------------------|
| **Scope** | Local Docker | Distributed systems |
| **Examples** | Build image, run container | SLURM jobs, K8s pods |
| **Location** | `execution/` | `deployment/` |
| **Complexity** | Simple (direct Docker) | Complex (cluster orchestration) |

---

## ⚙️ Configuration

Both components use `Context` for configuration:

```python
# GPU vendor, architecture, ROCm version
context.get_gpu_vendor()  # "AMD" or "NVIDIA"
context.get_system_gpu_architecture()  # "gfx90a", "sm_80"

# Docker settings
context.ctx["docker_env_vars"]  # Environment variables
context.ctx["docker_build_arg"]  # Build arguments
context.ctx["docker_mounts"]  # Volume mounts
```

---

**Last Updated**: November 30, 2025  
**Maintainer**: madengine Team


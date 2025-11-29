# ⚠️ DEPRECATED - This folder is no longer used

**Status**: DEPRECATED (Phase 5 - November 29, 2025)  
**Replaced By**: `src/madengine/deployment/` architecture

---

## ⛔ DO NOT USE

This entire `runners/` directory has been replaced by the new `deployment/` architecture.

The old runner system included:
- `base.py` - Base runner classes
- `factory.py` - Runner factory
- `ssh_runner.py` - SSH-based execution
- `ansible_runner.py` - Ansible orchestration  
- `k8s_runner.py` - Kubernetes execution
- `slurm_runner.py` - SLURM execution
- `orchestrator_generation.py` - Config generators
- `template_generator.py` - Template engine

---

## ✅ New Architecture (Use Instead)

### For SLURM Deployment:
```bash
madengine-cli run --tags model \
  --additional-context '{
    "deploy": "slurm",
    "slurm": {"partition": "gpu", "nodes": 4, "gpus_per_node": 8}
  }'
```

**Implementation**: `src/madengine/deployment/slurm.py`
- Uses CLI commands (sbatch, squeue, scancel)
- Zero Python dependencies
- Jinja2 templates in `deployment/templates/slurm/`

### For Kubernetes Deployment:
```bash
madengine-cli run --tags model \
  --additional-context '{
    "deploy": "k8s",
    "k8s": {"namespace": "default", "gpu_resource_name": "amd.com/gpu"}
  }'
```

**Implementation**: `src/madengine/deployment/kubernetes.py`
- Uses Kubernetes Python library
- Type-safe Job creation
- AMD GPU Device Plugin integration

---

## 🗑️ Planned Removal

This folder will be **DELETED** in a future release after thorough testing of the new architecture.

**Do not add new code to this folder.**  
**Do not fix bugs in this folder.**  
**Migrate to the new `deployment/` architecture instead.**

---

## 📚 Migration Guide

| Old Command | New Command |
|-------------|-------------|
| `madengine-cli generate slurm` | **REMOVED** - automatic via `--additional-context` |
| `madengine-cli runner slurm` | `madengine-cli run --additional-context '{"deploy": "slurm"}'` |
| `madengine-cli generate k8s` | **REMOVED** - automatic via `--additional-context` |
| `madengine-cli runner k8s` | `madengine-cli run --additional-context '{"deploy": "k8s"}'` |

---

**See**: `REFACTOR_COMPLETE.md` for complete implementation details


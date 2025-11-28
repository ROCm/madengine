# MADEngine Refactoring - Quick Summary

> **TL;DR**: Simplify from 4 complex runners to 3 clear deployment modes, clarify terminology, keep what works.

---

## 🔑 Key Changes

### Before (Current v1.x - Complex)
```
❌ Confusing: "Runner" distributes madengine itself, not model workloads
❌ 4 Runner types: SSH, Ansible, K8s, SLURM
❌ Complex setup: Clone MAD → venv → install madengine on each node
❌ Separate commands: generate + runner
❌ Not how K8s/SLURM are actually used in practice
```

### After (New v2.0 - Simple)
```
✅ Clear: Infrastructure layer (where) vs Execution layer (how)
✅ 3 Deployment modes: Local, SLURM, K8s
✅ Simple: Docker image → Deploy directly
✅ Unified command: run with --additional-context
✅ Aligned with industry best practices
```

---

## 📊 Architecture Comparison

### Old Architecture (v1.x)
```
User → madengine-cli runner → Setup madengine on nodes → Run madengine → Pull image → Run model
      (Complex indirection)
```

### New Architecture (v2.0)
```
User → madengine-cli run → Deploy model container → Run model
      (Direct, simple)
```

---

## 🎯 Three Deployment Modes

### 1️⃣ Local (Keep existing - works great!)
```bash
madengine-cli run --tags dummy
```
**What happens**: Docker run on current node (unchanged)

### 2️⃣ SLURM (New - proper HPC workflow)
```bash
madengine-cli run --tags bert \
  --additional-context '{"deploy": "slurm", "slurm": {"partition": "gpu"}}'
```
**What happens**: 
1. Generate sbatch script from template
2. Submit to SLURM
3. SLURM allocates nodes
4. Each node runs model container directly

### 3️⃣ Kubernetes (New - proper cloud workflow)
```bash
madengine-cli run --tags llama \
  --additional-context '{"deploy": "k8s", "k8s": {"namespace": "prod"}}'
```
**What happens**:
1. Generate pod.yaml from template
2. kubectl apply
3. K8s schedules pods
4. Each pod runs model container directly

---

## 🏗️ Terminology Clarification

### Infrastructure Layer (madengine's job)
**Where the workload runs**:
- Local: Docker on current node
- SLURM: HPC cluster job scheduler
- Kubernetes: Container orchestration

### Execution Layer (model's job, inside container)
**How the model runs**:
- Single GPU: `python train.py`
- Multi GPU: `torchrun --nproc_per_node=8`
- Multi Node: `torchrun --nnodes=4 --nproc_per_node=8`
- DeepSpeed: `deepspeed --hostfile=...`

**madengine orchestrates infrastructure, models handle execution**

---

## 🔄 Migration Path

### SSH/Ansible Users → Use your own orchestration
```bash
# Old (deprecated)
madengine-cli runner ssh --inventory nodes.yml

# New (use your tools)
# 1. Build once
madengine-cli build --tags models --registry your-registry

# 2. Deploy with your orchestration (Ansible, SSH, etc.)
ansible-playbook deploy.yml
  # Playbook runs: madengine-cli run --manifest-file manifest.json
```

### K8s Users → Use K8s deployment
```bash
# Old (complex)
madengine-cli generate k8s ...
madengine-cli runner k8s ...

# New (simple)
madengine-cli run --tags models \
  --additional-context '{"deploy": "k8s"}'
```

### SLURM Users → Use SLURM deployment
```bash
# Old (manual)
madengine-cli generate slurm ...
# Then manually submit sbatch

# New (automated)
madengine-cli run --tags models \
  --additional-context '{"deploy": "slurm"}'
```

---

## ✅ What We Keep (Working Well)

| Component | Status | Action |
|-----------|--------|--------|
| Build Phase | ✅ Excellent | Keep as-is |
| Run Phase (local) | ✅ Excellent | Keep as-is |
| Model Discovery | ✅ Excellent | Keep as-is |
| Core (Context, Docker, Data) | ✅ Stable | Keep as-is |
| Legacy madengine (mad.py) | ⚠️ Deprecated | Keep for now, remove in v3.0 |

---

## 🗂️ New Directory Structure

```
src/madengine/
├── mad.py                      # Legacy CLI (keep, deprecate)
├── mad_cli.py                  # Modern CLI (refactor run command)
│
├── core/                       # ✅ Keep as-is
├── tools/                      # ✅ Keep existing + enhance
│
├── deployment/                 # 🆕 NEW
│   ├── base.py                # Abstract deployment class
│   ├── local.py               # Wraps existing ContainerRunner
│   ├── slurm.py               # SLURM deployment
│   ├── kubernetes.py          # K8s deployment
│   ├── factory.py             # DeploymentFactory
│   └── templates/             # Jinja2 templates
│       ├── slurm/
│       │   └── job.sh.j2
│       └── kubernetes/
│           └── job.yaml.j2
│
└── runners/                    # ⚠️ DEPRECATED (mark, remove later)
```

---

## 🚀 Implementation Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1: Foundation** | Week 1-2 | Deployment framework, LocalDeployment |
| **Phase 2: SLURM** | Week 3-4 | SLURM deployment working |
| **Phase 3: Kubernetes** | Week 5-6 | K8s deployment working |
| **Phase 4: CLI Integration** | Week 7 | Unified CLI |
| **Phase 5: Documentation** | Week 8 | Production ready |

**Total**: 8 weeks to production-ready v2.0

---

## 📋 Quick Reference: Command Changes

### Commands That Stay
```bash
✅ madengine-cli build          # Unchanged
✅ madengine-cli run            # Enhanced (auto-detects mode)
✅ madengine discover           # Unchanged (legacy)
```

### Commands That Change
```bash
❌ madengine-cli runner ssh     → ⚠️ Use your SSH/Ansible
❌ madengine-cli runner ansible → ⚠️ Use your SSH/Ansible
❌ madengine-cli runner k8s     → ✅ madengine-cli run --additional-context '{"deploy": "k8s"}'
❌ madengine-cli runner slurm   → ✅ madengine-cli run --additional-context '{"deploy": "slurm"}'

❌ madengine-cli generate k8s   → ✅ Auto-generated during run
❌ madengine-cli generate slurm → ✅ Auto-generated during run
```

---

## 🎓 Example: Full Workflow

### Local Development
```bash
# Build + Run in one command (unchanged)
madengine-cli run --tags dummy
```

### SLURM HPC Cluster
```bash
# 1. Build on login node or build node
madengine-cli build --tags bert_training \
  --registry your-registry \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'

# 2. Deploy to SLURM
madengine-cli run --manifest-file build_manifest.json \
  --additional-context '{
    "deploy": "slurm",
    "launcher": "torchrun",
    "nnodes": 4,
    "nproc_per_node": 8,
    "slurm": {
      "partition": "gpu",
      "modules": ["rocm/5.7.0"]
    }
  }'

# Result: Automatic sbatch generation + submission + monitoring
```

### Kubernetes Cloud
```bash
# 1. Build (anywhere with Docker)
madengine-cli build --tags llama_serving \
  --registry gcr.io/my-project

# 2. Deploy to K8s
madengine-cli run --manifest-file build_manifest.json \
  --additional-context '{
    "deploy": "k8s",
    "k8s": {
      "namespace": "ml-prod",
      "gpu_vendor": "AMD",
      "memory": "64Gi"
    }
  }'

# Result: Automatic pod.yaml generation + kubectl apply + monitoring
```

---

## ❓ FAQ

**Q: What about SSH/Ansible runners?**  
A: Removed. Use your own SSH/Ansible to orchestrate `madengine-cli run` on each node.

**Q: Will this break my existing workflows?**  
A: No. Legacy madengine and old commands will continue to work with deprecation warnings.

**Q: When will old runners be removed?**  
A: After v2.0 stable (6-12 months), giving time for migration.

**Q: Can I still use Primus/Megatron/etc?**  
A: Yes! These are execution frameworks (inside container). madengine handles infrastructure.

**Q: What about training vs inference?**  
A: Both supported. Configure via model's run.sh and --additional-context.

**Q: Does this work with vLLM/SGLang serving?**  
A: Yes! These are inference servers. Your model container runs them, madengine deploys.

---

## 🎯 Success Metrics

- ✅ Simpler: 3 modes instead of 4 runner types
- ✅ Clearer: Infrastructure vs Execution terminology
- ✅ Faster: Direct deployment, no setup overhead
- ✅ Better: Aligned with K8s/SLURM best practices
- ✅ Compatible: Zero breaking changes
- ✅ Maintainable: Less code, clearer structure

---

**Next Steps**: Review REFACTOR_PLAN.md for detailed implementation



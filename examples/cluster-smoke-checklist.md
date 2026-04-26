# Cluster Smoke Checklist (RDMA + GCM Phase 1)

This checklist validates:

- RDMA recommender on **SLURM + Kubernetes**
- GCM health checks + one-shot collector on **SLURM only**

If you prefer one-liners, use:

```bash
make -f examples/Makefile.smoke smoke-slurm MODEL_DIR=<path> MODEL_TAG=<tag>
make -f examples/Makefile.smoke smoke-k8s MODEL_DIR=<path> MODEL_TAG=<tag>
```

Or use the wrapper script:

```bash
examples/run-smoke.sh slurm MODEL_DIR=<path> MODEL_TAG=<tag>
examples/run-smoke.sh verify-slurm
examples/run-smoke.sh k8s MODEL_DIR=<path> MODEL_TAG=<tag>
examples/run-smoke.sh verify-k8s
```

## 0) Set shared variables

```bash
cd /home/ysha/amd/madengine
export MODEL_DIR="<path-to-your-model-dir>"
export MODEL_TAG="<your-model-tag>"
```

---

## 1) SLURM smoke (RDMA + GCM)

### 1.1 Build

```bash
madengine build \
  --tags "${MODEL_TAG}" \
  --additional-context-file "examples/slurm-configs/configs/smoke-rdma-gcm-slurm.json" \
  --manifest-output "build_manifest.slurm.smoke.json"
```

### 1.2 Run

```bash
madengine run \
  --manifest-file "build_manifest.slurm.smoke.json" \
  --timeout 3600
```

### 1.3 Verify artifacts

```bash
# GCM health summary
python3 - <<'PY'
import glob, json, os
matches = glob.glob("slurm_results/cluster_artifacts/**/gcm_health_summary.json", recursive=True)
print("gcm_health_summary files:", matches)
for p in matches[:2]:
    print(p, json.load(open(p)).get("status"))
PY

# GCM collector output
python3 - <<'PY'
import glob
matches = glob.glob("slurm_results/cluster_artifacts/**/gcm_collector_output.log", recursive=True)
print("gcm_collector_output files:", matches)
PY

# RDMA artifacts copied per node collection directory
python3 - <<'PY'
import glob, json
matches = glob.glob("slurm_results/**/rdma_recommendation.json", recursive=True)
print("rdma_recommendation files:", matches)
for p in matches[:3]:
    data = json.load(open(p))
    print(p, data.get("status"), sorted((data.get("recommended_env") or {}).keys())[:6])
PY
```

---

## 2) Kubernetes smoke (RDMA only)

### 2.1 Build

```bash
madengine build \
  --tags "${MODEL_TAG}" \
  --additional-context-file "examples/k8s-configs/configs/smoke-rdma-k8s.json" \
  --manifest-output "build_manifest.k8s.smoke.json"
```

### 2.2 Run

```bash
madengine run \
  --manifest-file "build_manifest.k8s.smoke.json" \
  --timeout 3600
```

### 2.3 Verify artifacts

```bash
python3 - <<'PY'
import glob, json
matches = glob.glob("k8s_results/**/rdma_recommendation.json", recursive=True)
print("rdma_recommendation files:", matches)
for p in matches[:3]:
    data = json.load(open(p))
    print(p, data.get("status"), sorted((data.get("recommended_env") or {}).keys())[:6])
PY
```

---

## 3) Optional strict-mode gate checks

### 3.1 SLURM GCM strict gate

Set in `examples/slurm-configs/configs/smoke-rdma-gcm-slurm.json`:

- `cluster.gcm.strict: true`

Then rerun section 1. If health checks fail, submission should fail early.

### 3.2 RDMA strict gate

Set in smoke config(s):

- `cluster.rdma.strict: true`

Then rerun section 1 or 2. Workload should fail when RDMA recommendation cannot be produced.

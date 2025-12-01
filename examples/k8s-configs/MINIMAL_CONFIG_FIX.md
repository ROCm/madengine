# Minimal Config Fix - Required Fields

## Issue

The initial `00-minimal.json` was missing required fields for build operations:

```bash
❌ Missing required fields: gpu_vendor, guest_os
💡 Both gpu_vendor and guest_os are required for build operations
```

## Root Cause

`madengine-cli build` requires `gpu_vendor` and `guest_os` to:
1. Select the correct base Docker image
2. Install GPU-specific packages (ROCm, CUDA)
3. Configure the build environment

These are **not optional** - they are required for any build operation.

## Fix Applied

### Before (Broken)
```json
{
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 1
  }
}
```

### After (Working) ✅
```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 1
  }
}
```

## Files Updated

1. **`00-minimal.json`** - Added `gpu_vendor` and `guest_os`
2. **`README.md`** - Marked `gpu_vendor` and `guest_os` as **Required**
3. **`INDEX.md`** - Updated minimal config example
4. **`EXAMPLES_SUMMARY.md`** - Updated description

## Validation

```bash
$ export MODEL_DIR=tests/fixtures/dummy
$ madengine-cli build --tags dummy \
    --additional-context-file examples/k8s-configs/00-minimal.json \
    --registry dockerhub

✅ Loaded additional context from file: examples/k8s-configs/00-minimal.json
✅ Context validated: AMD + UBUNTU
🔨 BUILD PHASE
```

## True "Minimal" Configuration

The actual minimal config for K8s deployment now includes **4 required fields**:

```json
{
  "gpu_vendor": "AMD",        // Required for build
  "guest_os": "UBUNTU",       // Required for build
  "deploy": "k8s",            // Required for K8s deployment
  "k8s": {
    "gpu_count": 1            // Required for GPU allocation
  }
}
```

All other fields use sensible defaults:
- `kubeconfig`: `~/.kube/config`
- `namespace`: `"default"`
- `memory`: `"128Gi"`
- `cpu`: `"32"`
- `image_pull_policy`: `"Always"`
- etc.

## For NVIDIA GPUs

If using NVIDIA instead of AMD:

```json
{
  "gpu_vendor": "NVIDIA",     // Changed from AMD
  "guest_os": "UBUNTU",
  "deploy": "k8s",
  "k8s": {
    "gpu_count": 1,
    "gpu_resource_name": "nvidia.com/gpu"  // NVIDIA resource name
  }
}
```

---

**Fixed**: December 1, 2025  
**Status**: Resolved ✅

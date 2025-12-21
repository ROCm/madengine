# Batch Build Guide

Complete guide to using batch manifests for selective model builds in CI/CD pipelines.

## Overview

Batch build mode enables selective builds with per-model configuration through a JSON manifest file. This is ideal for CI/CD pipelines where you need fine-grained control over which models to rebuild.

## Usage

```bash
madengine-cli build --batch-manifest examples/build-manifest/batch.json \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

## Manifest Format

### Basic Structure

```json
[
  {
    "model_name": "model1",
    "build_new": true,
    "registry": "docker.io/myorg",
    "registry_image": "myorg/model1"
  },
  {
    "model_name": "model2",
    "build_new": false
  }
]
```

### Field Reference

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Model tag to include in manifest |

#### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `build_new` | boolean | `false` | `true`: Build from source<br>`false`: Reference existing image |
| `registry` | string | - | Per-model Docker registry (overrides global `--registry`) |
| `registry_image` | string | - | Custom registry image name/namespace |

## Key Features

### Selective Building

- Models with `"build_new": true` are built from source
- Models with `"build_new": false` are referenced without building
- All models are included in the output `build_manifest.json`

### Per-Model Registry Override

Each model can specify its own registry:

```json
[
  {
    "model_name": "public_model",
    "build_new": true,
    "registry": "docker.io/myorg"
  },
  {
    "model_name": "private_model",
    "build_new": true,
    "registry": "gcr.io/myproject"
  }
]
```

### Mutual Exclusivity with --tags

Cannot use `--batch-manifest` and `--tags` together:

```bash
# ❌ Error
madengine-cli build --batch-manifest batch.json --tags model1

# ✅ Correct
madengine-cli build --batch-manifest batch.json
```

## Common Use Cases

### CI/CD Incremental Builds

Rebuild only changed models while referencing stable ones:

**Example:** [`examples/build-manifest/ci_incremental.json`](../examples/build-manifest/ci_incremental.json)

```json
[
  {"model_name": "changed_model", "build_new": true},
  {"model_name": "stable_model_1", "build_new": false},
  {"model_name": "stable_model_2", "build_new": false}
]
```

**Usage:**
```bash
madengine-cli build --batch-manifest examples/build-manifest/ci_incremental.json \
  --registry docker.io/myorg \
  --additional-context-file config.json
```

### Multi-Registry Deployment

Deploy models to different registries:

```json
[
  {
    "model_name": "public_model",
    "build_new": true,
    "registry": "docker.io/myorg"
  },
  {
    "model_name": "private_model",
    "build_new": true,
    "registry": "gcr.io/myproject"
  }
]
```

### Custom Image Names

Specify custom image names and tags:

```json
[
  {
    "model_name": "my_model",
    "build_new": true,
    "registry": "docker.io/myorg",
    "registry_image": "myorg/custom-name:v2.0"
  }
]
```

## Complete Workflow

### 1. Create Batch Manifest

```bash
cat > my_batch.json << 'EOF'
[
  {
    "model_name": "dummy",
    "build_new": true
  },
  {
    "model_name": "stable_model",
    "build_new": false,
    "registry": "docker.io/myorg",
    "registry_image": "myorg/stable:v1.0"
  }
]
EOF
```

### 2. Build with Batch Manifest

```bash
madengine-cli build --batch-manifest my_batch.json \
  --registry localhost:5000 \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU"
  }' \
  --verbose
```

### 3. Use Output Manifest

The command generates `build_manifest.json` containing:
- Built models with their new image names
- Referenced models with their existing image names
- Per-model registry configuration

Run the models:
```bash
madengine-cli run --manifest-file build_manifest.json
```

## Examples

See [`examples/build-manifest/`](../examples/build-manifest/) directory for:
- [`batch.json`](../examples/build-manifest/batch.json) - Basic example with all field types
- [`ci_incremental.json`](../examples/build-manifest/ci_incremental.json) - CI/CD incremental build pattern

## Command Reference

### Build Command

```bash
madengine-cli build [OPTIONS]
```

**Batch Build Options:**
- `--batch-manifest PATH` - Input batch manifest file (mutually exclusive with `--tags`)
- `--registry, -r URL` - Global Docker registry (can be overridden per model)
- `--additional-context, -c JSON` - Configuration as JSON string
- `--additional-context-file, -f PATH` - Configuration file
- `--manifest-output, -m PATH` - Output manifest file (default: `build_manifest.json`)
- `--verbose, -v` - Verbose logging

### Output

Creates `build_manifest.json` with:
```json
{
  "built_images": {
    "image_name": {
      "docker_image": "...",
      "registry": "...",
      ...
    }
  },
  "built_models": {...},
  "deployment_config": {...},
  "summary": {...}
}
```

## Best Practices

1. **Version Control**: Keep batch manifests in version control for reproducibility
2. **Start Simple**: Begin with basic manifests and add complexity as needed
3. **Test Locally**: Validate batch manifests locally before CI/CD deployment
4. **Consistent Naming**: Use descriptive model names and consistent registry paths
5. **Document Changes**: Add comments in commit messages explaining manifest changes

## See Also

- [Configuration Guide](configuration.md) - Additional context and build arguments
- [Usage Guide](usage.md) - General build and run workflows
- [Deployment Guide](deployment.md) - Kubernetes and SLURM deployment


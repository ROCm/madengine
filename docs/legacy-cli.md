# Legacy CLI Guide

> **⚠️ DEPRECATED**: The `madengine` CLI is the legacy v1.x interface. New projects should use `madengine-cli`.

This guide documents the legacy `madengine` CLI for backward compatibility. For new projects, see the [Usage Guide](usage.md) for `madengine-cli`.

## Overview

The legacy `madengine` CLI provides basic model execution and reporting capabilities without distributed deployment support.

```bash
madengine [COMMAND] [OPTIONS]
```

**Available Commands:**
- `run` - Run models locally
- `discover` - Discover models
- `report` - Generate performance reports
- `database` - Database operations

## Commands

### run - Execute Models

```bash
madengine run --tags model \
  --additional-context '{"guest_os": "UBUNTU"}' \
  --live-output
```

**Common Options:**
- `--tags` - Model tags to run
- `--timeout` - Execution timeout in seconds
- `--live-output` - Real-time output streaming
- `--additional-context` - Configuration JSON string
- `--additional-context-file` - Configuration file path
- `--keep-alive` - Keep containers alive after run
- `-o, --output` - Performance output file

### discover - Find Models

```bash
madengine discover --tags dummy
```

### report - Generate Reports

```bash
# Generate HTML report
madengine report to-html --csv-file-path perf.csv

# Send email report
madengine report to-email --csv-file-path perf.csv

# Update performance database
madengine report update-perf --perf-csv perf.csv
```

### database - Database Operations

```bash
# Create database table
madengine database create-table

# Update database table
madengine database update-table --csv-file-path perf.csv

# Upload to MongoDB
madengine database upload-mongodb --type perf --file-path perf.csv
```

## Configuration

The legacy CLI uses the same configuration format as `madengine-cli`:

```json
{
  "guest_os": "UBUNTU",
  "docker_env_vars": {
    "HSA_ENABLE_SDMA": "0"
  }
}
```

**Note:** The legacy CLI does not support:
- Kubernetes deployment
- SLURM deployment
- Distributed launchers
- Build-only operations
- Manifest-based execution

## Migration to madengine-cli

### Command Mapping

| Legacy (`madengine`) | Modern (`madengine-cli`) |
|---------------------|-------------------------|
| `madengine run --tags model` | `madengine-cli run --tags model` |
| `madengine discover --tags model` | `madengine-cli discover --tags model` |
| `madengine report to-html` | Use external tools or custom scripts |
| `madengine database create-table` | Use external tools or custom scripts |

### Migration Steps

1. **Update commands** from `madengine` to `madengine-cli`
2. **Add required context** - `madengine-cli` requires `gpu_vendor` and `guest_os` for local execution
3. **Update scripts** - Replace legacy commands with modern equivalents
4. **Test thoroughly** - Verify behavior matches expectations

### Example Migration

**Before (legacy):**
```bash
madengine run --tags pyt_huggingface_bert \
  --additional-context '{"guest_os": "UBUNTU"}' \
  --live-output
```

**After (modern):**
```bash
madengine-cli run --tags pyt_huggingface_bert \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}' \
  --live-output
```

## Differences from madengine-cli

| Feature | Legacy `madengine` | Modern `madengine-cli` |
|---------|-------------------|----------------------|
| **Local Execution** | ✅ Supported | ✅ Supported |
| **K8s Deployment** | ❌ Not supported | ✅ Supported |
| **SLURM Deployment** | ❌ Not supported | ✅ Supported |
| **Build Command** | ❌ Not available | ✅ Available |
| **Distributed Launchers** | ❌ Not supported | ✅ Supported |
| **Rich Output** | ❌ Basic output | ✅ Rich terminal UI |
| **Manifest Support** | ❌ Not available | ✅ Supported |
| **Report Generation** | ✅ Built-in | ⚠️ Use external tools |
| **Database Operations** | ✅ Built-in | ⚠️ Use external tools |

## When to Use Legacy CLI

The legacy CLI should only be used when:
- Maintaining existing scripts that haven't been migrated
- Using report generation features not yet available in `madengine-cli`
- Working with legacy database integration

**For all new projects, use `madengine-cli`.**

## Support Status

- **Legacy CLI (`madengine`)**: Maintenance mode, bug fixes only
- **Modern CLI (`madengine-cli`)**: Active development, new features

## Next Steps

- [Usage Guide](usage.md) - Learn `madengine-cli` commands
- [Configuration Guide](configuration.md) - Configure `madengine-cli`
- [Deployment Guide](deployment.md) - Deploy to clusters


# Profiling Guide

Complete guide to profiling model performance and analyzing library calls with madengine.

## Overview

madengine integrates multiple profiling and tracing tools to analyze GPU usage, library calls, and system performance. Tools are configured via `--additional-context` and applied in a stackable design pattern.

## Quick Start

### Basic GPU Profiling

```bash
madengine run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{"name": "rocprof"}]
  }'
```

**Output:** `rocprof_output/` directory with profiling results

### Using Configuration Files

For complex profiling setups, use configuration files:

**profiling-config.json:**
```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "tools": [
    {"name": "rocprof"}
  ]
}
```

```bash
madengine run --tags model --additional-context-file profiling-config.json
```

## Profiling Tools

### rocprof - GPU Profiling

Profile GPU kernels and HIP API calls:

```json
{
  "tools": [
    {"name": "rocprof"}
  ]
}
```

**Default Behavior:** HIP trace mode
**Output:** `rocprof_output/` directory

**Custom Configuration:**
```json
{
  "tools": [
    {
      "name": "rocprof",
      "cmd": "rocprof --timestamp on",
      "env_vars": {
        "NCCL_DEBUG": "INFO"
      }
    }
  ]
}
```

### rpd - ROCm Profiler Data

Collect comprehensive ROCm profiling data:

```json
{
  "tools": [
    {"name": "rpd"}
  ]
}
```

**Output:** ROCm profiler data files

### rocblas_trace - rocBLAS Library Tracing

Trace rocBLAS API calls and configurations:

```json
{
  "tools": [
    {"name": "rocblas_trace"}
  ]
}
```

**Output:** 
- Trace logs in execution output
- `library_trace.csv` with library call summary

**Use Case:** Analyze BLAS operations, identify optimization opportunities

### miopen_trace - MIOpen Library Tracing

Trace MIOpen API calls for deep learning operations:

```json
{
  "tools": [
    {"name": "miopen_trace"}
  ]
}
```

**Output:**
- Trace logs in execution output
- `library_trace.csv` with convolution, pooling, and other DNN operations

**Use Case:** Optimize deep learning layers, analyze convolution configurations

### tensile_trace - Tensile Library Tracing

Trace Tensile matrix operations:

```json
{
  "tools": [
    {"name": "tensile_trace"}
  ]
}
```

**Output:**
- Trace logs in execution output
- `library_trace.csv` with matrix operation details

**Use Case:** Analyze GEMM operations, optimize matrix multiplications

### rccl_trace - RCCL Communication Tracing

Trace RCCL collective communication operations:

```json
{
  "tools": [
    {"name": "rccl_trace"}
  ]
}
```

**Output:** Trace logs with communication patterns

**Use Case:** Debug multi-GPU communication, optimize distributed training

### gpu_info_power_profiler - Power Consumption

Profile real-time GPU power consumption:

```json
{
  "tools": [
    {"name": "gpu_info_power_profiler"}
  ]
}
```

**Output:** `gpu_info_power_profiler_output.csv`

**Configuration:**
```json
{
  "tools": [
    {
      "name": "gpu_info_power_profiler",
      "env_vars": {
        "DEVICE": "0",
        "SAMPLING_RATE": "0.1"
      }
    }
  ]
}
```

**Environment Variables:**
- `DEVICE` - GPU device(s): `"0"`, `"0,1,2"`, or `"all"` (default: `"0"`)
- `SAMPLING_RATE` - Sampling interval in seconds (default: `"0.1"`)
- `MODE` - Must be `"power"` for this tool
- `DUAL-GCD` - Enable dual-GCD mode: `"true"` or `"false"` (default: `"false"`)

**Supported Platforms:** ROCm and CUDA

### gpu_info_vram_profiler - VRAM Usage

Profile real-time GPU memory consumption:

```json
{
  "tools": [
    {"name": "gpu_info_vram_profiler"}
  ]
}
```

**Output:** `gpu_info_vram_profiler_output.csv`

**Configuration:**
```json
{
  "tools": [
    {
      "name": "gpu_info_vram_profiler",
      "env_vars": {
        "DEVICE": "all",
        "SAMPLING_RATE": "0.5",
        "MODE": "vram"
      }
    }
  ]
}
```

**Environment Variables:**
- `DEVICE` - GPU device(s): `"0"`, `"0,1,2"`, or `"all"`
- `SAMPLING_RATE` - Sampling interval in seconds
- `MODE` - Must be `"vram"` for this tool
- `DUAL-GCD` - Enable dual-GCD mode

**Supported Platforms:** ROCm and CUDA

## Stackable Design

Tools can be stacked to collect multiple types of profiling data simultaneously. Tools are applied in order, with the first tool being innermost:

```json
{
  "tools": [
    {"name": "rocprof"},
    {"name": "miopen_trace"},
    {"name": "rocblas_trace"}
  ]
}
```

**Execution Order:**
1. **Setup:** rocblas_trace → miopen_trace → rocprof
2. **Run:** Model execution
3. **Teardown:** rocprof → miopen_trace → rocblas_trace

**Example:**
```bash
madengine run --tags pyt_torchvision_alexnet \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [
      {"name": "rocprof"},
      {"name": "miopen_trace"}
    ]
  }'
```

## Competitive Library Performance Analysis

### Overview

Analyze and compare performance of different library configurations by:
1. Collecting library call traces
2. Measuring performance of different configurations
3. Comparing competitive implementations

### Step 1: Collect Library Traces

Collect library API call traces:

```bash
# Trace MIOpen calls
madengine run --tags pyt_torchvision_alexnet \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{"name": "miopen_trace"}]
  }'

# Trace rocBLAS calls
madengine run --tags pyt_torchvision_alexnet \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{"name": "rocblas_trace"}]
  }'
```

Or collect both in one run:

```bash
madengine run --tags pyt_torchvision_alexnet \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [
      {"name": "miopen_trace"},
      {"name": "rocblas_trace"}
    ]
  }'
```

**Output:** `library_trace.csv` containing library calls and configurations

### Step 2: Measure Library Configuration Performance

Use the collected traces to benchmark different library configurations:

```bash
madengine run --tags pyt_library_config_perf
```

**Prerequisites:**
- `library_trace.csv` must exist in the current directory
- Contains library call configurations from Step 1

**Output:** `library_perf.csv` with performance data for each configuration

**Platform Support:** Works on both AMD and NVIDIA GPUs

### Step 3: Analysis

Compare results from `library_perf.csv` to:
- Identify optimal library configurations
- Compare performance across different implementations
- Validate optimization opportunities

## Common Usage Patterns

### Full Performance Analysis

```bash
# Step 1: Collect comprehensive traces
madengine run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [
      {"name": "rocprof"},
      {"name": "gpu_info_power_profiler"},
      {"name": "gpu_info_vram_profiler"}
    ]
  }'

# Step 2: Analyze results
ls -lh rocprof_output/
cat gpu_info_power_profiler_output.csv
cat gpu_info_vram_profiler_output.csv
```

### Library Optimization Workflow

```bash
# 1. Profile current implementation
madengine run --tags model \
  --additional-context '{"tools": [{"name": "miopen_trace"}]}'

# 2. Test library configurations
madengine run --tags pyt_library_config_perf

# 3. Analyze and compare
python analyze_library_perf.py library_perf.csv
```

### Multi-GPU Profiling

```bash
madengine run --tags model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "docker_gpus": "0,1,2,3",
    "tools": [
      {
        "name": "gpu_info_power_profiler",
        "env_vars": {
          "DEVICE": "all",
          "SAMPLING_RATE": "0.1"
        }
      },
      {"name": "rccl_trace"}
    ]
  }'
```

## Output Files Reference

| Tool | Output File(s) | Content |
|------|---------------|---------|
| `rocprof` | `rocprof_output/*` | GPU kernel traces, HIP API calls |
| `rpd` | Various RPD files | ROCm profiler data |
| `rocblas_trace` | `library_trace.csv`, logs | rocBLAS API calls |
| `miopen_trace` | `library_trace.csv`, logs | MIOpen API calls |
| `tensile_trace` | `library_trace.csv`, logs | Tensile operations |
| `rccl_trace` | Execution logs | RCCL communication |
| `gpu_info_power_profiler` | `gpu_info_power_profiler_output.csv` | Power consumption over time |
| `gpu_info_vram_profiler` | `gpu_info_vram_profiler_output.csv` | VRAM usage over time |

## Tool Configuration Options

All tools support these configuration keys:

### cmd - Custom Command

Override the default profiling command:

```json
{
  "tools": [
    {
      "name": "rocprof",
      "cmd": "rocprof --timestamp on --hip-trace"
    }
  ]
}
```

**Note:** Tool binary name must be included in custom commands.

### env_vars - Environment Variables

Set tool-specific environment variables:

```json
{
  "tools": [
    {
      "name": "rocprof",
      "env_vars": {
        "NCCL_DEBUG": "INFO",
        "HSA_ENABLE_SDMA": "0"
      }
    }
  ]
}
```

## Best Practices

### 1. Profile Single Workloads

Profiling works best with single model tags:

```bash
# Good
madengine run --tags pyt_torchvision_alexnet \
  --additional-context '{"tools": [{"name": "rocprof"}]}'

# Avoid
madengine run --tags model1 model2 model3 \
  --additional-context '{"tools": [{"name": "rocprof"}]}'
```

### 2. Use Configuration Files

For complex profiling setups:

```json
{
  "gpu_vendor": "AMD",
  "guest_os": "UBUNTU",
  "tools": [
    {
      "name": "rocprof",
      "cmd": "rocprof --timestamp on"
    },
    {
      "name": "gpu_info_power_profiler",
      "env_vars": {
        "DEVICE": "all",
        "SAMPLING_RATE": "0.1"
      }
    }
  ]
}
```

### 3. Optimize Sampling Rates

Balance detail vs. overhead:

```json
{
  "tools": [
    {
      "name": "gpu_info_power_profiler",
      "env_vars": {
        "SAMPLING_RATE": "1.0"  // Less overhead, less detail
      }
    }
  ]
}
```

### 4. Stack Related Tools

Group related profiling tools:

```json
{
  "tools": [
    {"name": "miopen_trace"},
    {"name": "rocblas_trace"},
    {"name": "tensile_trace"}
  ]
}
```

### 5. Separate Profiling Runs

For performance-critical profiling:

```bash
# Baseline run (no profiling)
madengine run --tags model

# Profiling run
madengine run --tags model \
  --additional-context '{"tools": [{"name": "rocprof"}]}'
```

## Troubleshooting

### Profiling Tool Not Found

**Error:** Tool binary not available

**Solution:**
```bash
# Verify tool is installed
which rocprof
which rocblas-bench

# Check container has tools
docker run --rm rocm/pytorch:latest which rocprof
```

### Empty Output Files

**Error:** Profiling produces empty results

**Causes:**
- Model execution too fast
- Incorrect device selection
- Tool configuration error

**Solutions:**
- Increase workload size
- Verify GPU device IDs
- Check tool logs for errors

### High Profiling Overhead

**Error:** Profiling significantly slows execution

**Solutions:**
- Reduce sampling rate
- Use fewer stacked tools
- Profile subset of execution
- Use targeted profiling

### library_trace.csv Not Generated

**Error:** Library trace file missing

**Causes:**
- No library calls made
- Tool not properly initialized
- Output directory permission issues

**Solutions:**
- Verify model uses the library (e.g., uses convolutions for MIOpen)
- Check execution logs for errors
- Verify write permissions

## Developer Information

### Tool Implementation

Profiling functionality is implemented via pre/post scripts:

**Location:**
- Pre-scripts: `scripts/common/pre_scripts/`
- Post-scripts: `scripts/common/post_scripts/`

**Workflow:**
1. Pre-script: Tool setup and initialization
2. Model execution: Tool collects data
3. Post-script: Save results, cleanup

### Default Tool Configuration

Tool defaults are defined in `scripts/common/tools.json`:

```json
{
  "rocprof": {
    "cmd": "rocprof --hip-trace",
    "env_vars": {}
  },
  "gpu_info_power_profiler": {
    "env_vars": {
      "DEVICE": "0",
      "SAMPLING_RATE": "0.1",
      "MODE": "power",
      "DUAL-GCD": "false"
    }
  }
}
```

### Adding Custom Tools

To add new profiling tools:

1. Create pre-script: `scripts/common/pre_scripts/tool_name_pre.sh`
2. Create post-script: `scripts/common/post_scripts/tool_name_post.sh`
3. Add default config to `scripts/common/tools.json`
4. Test with madengine

## Next Steps

- [Configuration Guide](configuration.md) - Detailed profiling configuration
- [Usage Guide](usage.md) - Running models with profiling
- [Deployment Guide](deployment.md) - Profiling in distributed environments

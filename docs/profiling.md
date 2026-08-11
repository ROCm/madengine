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

#### ROCm Profiler Version Compatibility

madengine uses `rocprof_wrapper.sh` to automatically handle the transition between rocprof (legacy) and rocprofv3:

| ROCm Version | Profiler Used | Command Syntax |
|--------------|---------------|----------------|
| ROCm < 7.0   | rocprof (legacy) | `rocprof [options] <app>` |
| ROCm >= 7.0  | rocprofv3 (preferred) | `rocprofv3 [options] -- <app>` |

**Key Points:**

1. **Automatic Detection:** The wrapper detects which profiler is available and uses the appropriate syntax
2. **Separator Requirement:** When using custom commands with `rocprof_wrapper.sh`, always include the trailing `--`:
   ```json
   {
     "name": "rocprof",
     "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --sys-trace --"
   }
   ```
3. **Backward Compatibility:** The `--` works with both rocprof and rocprofv3, ensuring your configurations work across ROCm versions

**Example - Custom Command with Wrapper:**
```json
{
  "tools": [
    {
      "name": "rocprof",
      "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --hip-trace --sys-trace --",
      "env_vars": {
        "HSA_ENABLE_SDMA": "0"
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

**Output:** ROCm profiler data files (e.g. `rpd_output/trace.rpd`).

**Note:** The rpd pre-script installs build dependencies in the container (e.g. `nlohmann-json3-dev` on Ubuntu) so the rocmProfileData tracer can compile; the first run may take longer while packages are installed.

### rocm-trace-lite (RTL) — lightweight kernel dispatch trace

[rocm-trace-lite](https://sunway513.github.io/rocm-trace-lite/index.html) captures GPU kernel dispatch timestamps via HSA runtime interception and writes a **SQLite** `.db` file (RPD-compatible). It does **not** use rocprofiler-sdk or rocprofiler. Use it when you want a **low-overhead** dispatch timeline without installing the full rocprofv3 stack, or alongside workflows that already rely on RPD-style SQLite.

**Do not** wrap the same workload with both **rocprofv3** (or `rocprof` via `rocprof_wrapper.sh`) and **rocm_trace_lite** / **rocm_trace_lite_default** in one run: choose **one** primary GPU profiler.

```json
{
  "tools": [
    {"name": "rocm_trace_lite"}
  ]
}
```

Use **`rocm_trace_lite`** for RTL **`lite`** mode (lower overhead; skips some dispatches that already carry a completion signal) or **`rocm_trace_lite_default`** for RTL **`default`** mode (broader coverage; higher overhead). Both set `RTL_MODE` for `rtl_trace_wrapper.sh`, which passes `rtl trace --mode <mode> …` when supported by your installed rocm-trace-lite. See upstream [rocm-trace-lite](https://github.com/sunway513/rocm-trace-lite) (`--mode` / profiling modes). Example: `examples/profiling-configs/rocm_trace_lite_default.json`.

**How madengine runs it:** The tool prepends `bash ../scripts/common/tools/rtl_trace_wrapper.sh` around your model command. The wrapper runs `rtl trace` with `-o rocm_trace_lite_output/trace.db` and optional `--mode` from `RTL_MODE` (see the [RTL quick start](https://sunway513.github.io/rocm-trace-lite/quickstart.html)). If `rtl` is not on `PATH` but the Python package is installed, it falls back to `python3 -m rocm_trace_lite.cli`.

**Installing `rocm-trace-lite` in the container:** Upstream distributes **wheels on [GitHub Releases](https://github.com/sunway513/rocm-trace-lite/releases)**, not on PyPI. The trace **pre-script** (`scripts/common/pre_scripts/trace.sh` with args `rocm_trace_lite`) installs via `pip` from a **pinned** `linux_x86_64` wheel URL by default (reproducible; bump the pin in that script when you intentionally upgrade RTL). To follow upstream’s latest release instead, set **`ROCM_TRACE_LITE_FOLLOW_LATEST=1`** (uses the GitHub API; needs `curl`). For a specific wheel, set **`ROCM_TRACE_LITE_WHEEL_URL`** to the full URL of a `.whl` file (or bake the package into the image). You need **outbound HTTPS to `github.com`** for the default or latest path unless the wheel is already present. Published wheels target **linux x86_64**; other architectures require a compatible wheel and the env override.

**Output:** `rocm_trace_lite_output/trace.db` under the model workspace (and optionally `trace.json.gz`, `trace_summary.txt`, etc., depending on RTL version). The trace **post-script** copies `rocm_trace_lite_output/` to `/myworkspace/` like other profiling tools.

**RTL vs rocprofv3**

| Topic | rocprofv3 (this guide, presets `rocprofv3_*`) | rocm-trace-lite |
|-------|-----------------------------------------------|-----------------|
| Stack | rocprofiler-sdk, rich traces and counters | HSA interception, SQLite timeline |
| Multi-node (K8s/SLURM) | `rocprof` is upgraded to `rocprofv3` when available | Does not require `rocprofv3` on the submission host; other rocprof-family tools are omitted if `rocprofv3` is missing (see multi-node profiling behavior below) |
| When to prefer | Deep analysis, hardware counters, Perfetto from rocprofv3 | Minimal-deps dispatch trace, RPD-compatible `.db` |

**Multi-node profiling:** Multi-node runs that use **only** tools outside the rocprof/rocprofv3 family (such as `rocm_trace_lite` or `rocm_trace_lite_default`) keep profiling enabled even when `rocprofv3` is not installed on the machine submitting the job. If the tool list includes **rocprof** or any **`rocprofv3_*`** preset and `rocprofv3` is unavailable, those entries are dropped; if no tools remain, profiling is disabled and the usual rocprofiler-sdk installation guidance is logged.

### ROCprofv3 - Advanced GPU Profiling

ROCprofv3 is the next-generation profiler for ROCm 7.0+ with enhanced features and better performance. madengine provides pre-configured profiles for common bottleneck scenarios.

#### Available ROCprofv3 Profiles

**Compute-Bound Analysis:**
```json
{
  "tools": [
    {"name": "rocprofv3_compute"}
  ]
}
```
- **Use Case**: Models bottlenecked by ALU operations
- **Metrics**: Wave execution, VALU/SALU instructions, wait states
- **Output Format**: Perfetto trace with hardware counters

**Memory-Bound Analysis:**
```json
{
  "tools": [
    {"name": "rocprofv3_memory"}
  ]
}
```
- **Use Case**: Models bottlenecked by memory bandwidth
- **Metrics**: Cache hits/misses, memory transfers, LDS usage
- **Output Format**: Perfetto trace with memory counters

**Communication-Bound Analysis (Multi-GPU):**
```json
{
  "tools": [
    {"name": "rocprofv3_communication"}
  ]
}
```
- **Use Case**: Multi-GPU distributed training
- **Metrics**: RCCL traces, inter-GPU transfers, synchronization
- **Output Format**: Perfetto trace with RCCL data

**Comprehensive Profiling:**
```json
{
  "tools": [
    {"name": "rocprofv3_full"}
  ]
}
```
- **Use Case**: Complete analysis with all metrics (high overhead)
- **Metrics**: All traces + counters + stats
- **Output Format**: Perfetto trace with full instrumentation

**Lightweight Profiling:**
```json
{
  "tools": [
    {"name": "rocprofv3_lightweight"}
  ]
}
```
- **Use Case**: Production-like profiling with minimal overhead
- **Metrics**: HIP and kernel traces only
- **Output Format**: JSON (compact)

**Perfetto Visualization:**
```json
{
  "tools": [
    {"name": "rocprofv3_perfetto"}
  ]
}
```
- **Use Case**: Generate Perfetto-compatible traces
- **Metrics**: HIP, kernel, memory traces
- **Output Format**: Perfetto trace file (`.pftrace`)

**API Overhead Analysis:**
```json
{
  "tools": [
    {"name": "rocprofv3_api_overhead"}
  ]
}
```
- **Use Case**: Analyze HIP/HSA API call overhead
- **Metrics**: API call timing and statistics
- **Output Format**: JSON with stats

**PC Sampling (Hotspot Analysis):**
```json
{
  "tools": [
    {"name": "rocprofv3_pc_sampling"}
  ]
}
```
- **Use Case**: Identify kernel hotspots
- **Metrics**: Program counter sampling at 1000 Hz
- **Output Format**: Perfetto trace with PC samples

#### Using Pre-Configured Profiles

madengine provides ready-to-use configuration files in `examples/profiling-configs/`:

```bash
# Compute-bound profiling
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_compute_bound.json

# Memory-bound profiling
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_memory_bound.json

# Multi-GPU profiling
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_multi_gpu.json

# Comprehensive profiling
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_comprehensive.json

# rocm-trace-lite (RTL) — not a rocprofv3 preset; do not mix with rocprof on the same run
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocm_trace_lite.json
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocm_trace_lite_default.json
```

See `examples/profiling-configs/README.md` for complete documentation.

#### Custom ROCprofv3 Commands

For advanced users, customize rocprofv3 invocation:

```json
{
  "tools": [
    {
      "name": "rocprof",
      "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --hip-trace --kernel-trace --memory-copy-trace --rccl-trace --counter-collection -i custom_counters.txt --output-format pftrace --stats -d ./my_output --",
      "env_vars": {
        "RCCL_DEBUG": "TRACE",
        "HSA_ENABLE_SDMA": "0"
      }
    }
  ]
}
```

**Important:** The `--` separator at the end of the `cmd` string is **required** when using `rocprof_wrapper.sh`. This separator distinguishes between profiler options and the application command:

- **rocprofv3 (ROCm >= 7.0):** Requires `--` separator → `rocprofv3 [options] -- <app>`
- **rocprof (legacy):** Works with or without `--` → `rocprof [options] <app>`

The wrapper auto-detects which profiler is available and formats arguments correctly. Always include the trailing `--` in your custom commands to ensure compatibility with both versions.

#### Hardware Counter Collection

Custom counter files are in `scripts/common/tools/counters/`:
- `compute_bound.txt` - ALU and execution metrics
- `memory_bound.txt` - Cache and memory metrics
- `communication_bound.txt` - PCIe and synchronization metrics
- `full_profile.txt` - Comprehensive metrics

Create your own counter file:
```text
# my_counters.txt
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU
pmc: L2CacheHit
pmc: TCC_HIT_sum
```

Then use it:
```bash
madengine run --tags your_model \
  --additional-context '{
    "tools": [{
      "name": "rocprof",
      "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --counter-collection -i my_counters.txt --output-format pftrace -d ./output --"
    }]
  }'
```

### torch_profiler_dynolog - On-Demand PyTorch Traces

Capture `torch.profiler` (Kineto) traces from a running PyTorch workload without editing the model script. Instead of wrapping the command, this tool starts the [dynolog](https://github.com/facebookincubator/dynolog) daemon inside the container; PyTorch registers with it because the tool sets `KINETO_USE_DAEMON=1`, and a background script then requests a trace with `dyno gputrace`.

```json
{
  "tools": [
    {"name": "torch_profiler_dynolog"}
  ]
}
```

**Output:** `torch_profiler_output/libkineto_trace_<pid>.json` (one file per rank)

**Requirements:**

- The workload must be PyTorch >= 1.13. Nothing is captured from non-PyTorch models.
- Iteration-based capture counts `optimizer.step()` calls. Workloads without an optimizer (pure inference) should set `TORCH_PROFILE_ITERATIONS` to `0` to fall back to duration-based capture.
- The pre-script downloads the dynolog `.deb` from GitHub, so the container needs outbound HTTPS on the first run (x86_64 Debian/Ubuntu base image). Set `DYNOLOG_DEB_URL` to use a mirror, or bake `dynolog` and `dyno` into the image to skip the download entirely.

**Environment Variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `TORCH_PROFILE_WARMUP_S` | `60` | Delay before the first trace request, so the workload reaches steady state |
| `TORCH_PROFILE_ITERATIONS` | `5` | Iterations to capture. `0` switches to `TORCH_PROFILE_DURATION_MS` |
| `TORCH_PROFILE_DURATION_MS` | `500` | Capture window when iteration counting is not used |
| `TORCH_PROFILE_RETRY_INTERVAL_S` | `15` | Wait between attempts while no PyTorch process has registered |
| `TORCH_PROFILE_MAX_ATTEMPTS` | `40` | Attempts before giving up |
| `TORCH_PROFILE_PROCESS_LIMIT` | `64` | Ranks to trace. Upstream defaults to 3, which silently drops most ranks |
| `TORCH_PROFILE_RECORD_SHAPES` | `1` | Record input shapes (needed for TraceLens per-operator analysis) |
| `TORCH_PROFILE_WITH_STACKS` | `1` | Record CPU call stacks |
| `TORCH_PROFILE_WITH_MODULES` | `1` | Record the `nn.Module` hierarchy |
| `TORCH_PROFILE_WITH_FLOPS` | `0` | Estimate FLOPs per operator |
| `TORCH_PROFILE_PROFILE_MEMORY` | `0` | Record allocator events |

For a short-lived workload, shorten the warmup so the request lands while the model is still running:

```json
{
  "tools": [
    {
      "name": "torch_profiler_dynolog",
      "env_vars": {
        "TORCH_PROFILE_WARMUP_S": "20",
        "TORCH_PROFILE_MAX_ATTEMPTS": "10"
      }
    }
  ]
}
```

**No trace produced?** The teardown script reports how many traces it found. `no PyTorch process registered` means `dyno gputrace` never matched the workload: confirm the model is PyTorch, and raise `TORCH_PROFILE_WARMUP_S` and `TORCH_PROFILE_MAX_ATTEMPTS` for slow-starting jobs.

### tracelens - TraceLens Trace Analysis

[TraceLens](https://github.com/AMD-AGI/TraceLens) turns raw traces into operator, kernel, roofline, and collective reports. It analyzes traces the other profiling tools produce, so always pair it with a profiler; on its own it has nothing to read.

Each trace format is routed to the matching TraceLens report:

| Trace produced by | Format | TraceLens report |
|-------------------|--------|------------------|
| `torch_profiler_dynolog` | Kineto JSON | Per-operator, kernel summary, roofline, `nn.Module` breakdown |
| `rocprofv3_lightweight` | rocprofv3 JSON | Kernel summary and details |
| `rocprofv3_perfetto` | `.pftrace` | HIP activity, HIP API, and memory-copy reports |
| Multiple per-rank Kineto traces | Kineto JSON | Multi-rank collective report |

**Unreadable formats:** TraceLens cannot read rocprofv3's default SQLite (`*_results.db`) or RPD (`.rpd`) databases. Those are listed in the summary as `SKIPPED` with a pointer at a preset that works — use `rocprofv3_lightweight` for JSON or `rocprofv3_perfetto` for `.pftrace`. For `rpd`, point TraceLens at the `trace.json` its post-script writes alongside the database.

#### Analyzing on the Host (Recommended)

Running analysis on the host keeps TraceLens' pinned `protobuf` and `xprof` out of your workload image:

```bash
pip install 'madengine[tracelens]'

# Analyze every trace a run left behind
madengine report tracelens

# See what would be analyzed, without running TraceLens
madengine report tracelens --discover-only

# Enable roofline bound classification
madengine report tracelens --gpu-arch MI300X

# Analyze a distributed run's collected artifacts
madengine report tracelens --root slurm_results --mode collective --world-size 8
```

**Output:** `tracelens_output/` with per-trace reports plus `tracelens_summary.csv` and `tracelens_summary.json`

Quantify the effect of a change by diffing two runs' reports. The first report is the baseline; every metric gains `_diff` and `_pct` columns relative to it:

```bash
madengine report tracelens-compare baseline.xlsx candidate.xlsx \
  --names before --names after -o diff.xlsx
```

#### Analyzing in the Container

Stack the `tracelens` tool **after** a profiler to get reports as part of the run itself:

```json
{
  "tools": [
    {"name": "rocprofv3_lightweight"},
    {"name": "tracelens"}
  ]
}
```

**Output:** `tracelens_output/` copied to the working directory alongside the profiler's own output

The pre-script installs TraceLens into an isolated virtualenv at `/opt/madengine-tracelens-venv` (no system site-packages), so its dependency pins cannot disturb the workload's Python environment. This needs outbound HTTPS to `github.com` on the first run. Analysis failures are reported as warnings and never fail a passing model run.

Use a mode-specific variant to restrict analysis to one trace kind:

| Tool | Analyzes |
|------|----------|
| `tracelens` | Every supported trace found (default) |
| `tracelens_pytorch` | Kineto traces only |
| `tracelens_rocprof` | rocprofv3 JSON only |
| `tracelens_pftrace` | Perfetto traces only |
| `tracelens_collective` | Multi-rank collective report only |

**Full PyTorch pipeline** — capture and analyze in one run:

```bash
madengine run --tags your_model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [
      {"name": "torch_profiler_dynolog"},
      {"name": "tracelens"}
    ]
  }'
```

**Environment Variables:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRACELENS_VENV` | `/opt/madengine-tracelens-venv` | Virtualenv holding TraceLens |
| `TRACELENS_OUTPUT_DIR` | `tracelens_output` | Directory for generated reports |
| `TRACELENS_MODE` | `auto` | `auto`, `pytorch`, `rocprof`, `pftrace`, or `collective` |
| `TRACELENS_GPU_ARCH` | unset | GPU arch (e.g. `MI300X`) enabling roofline bound classification |
| `TRACELENS_WORLD_SIZE` | trace count | Rank count for the collective report |
| `TRACELENS_MAX_TRACES` | unset | Cap traces analyzed per kind |
| `TRACELENS_GIT_REF` | pinned commit | TraceLens revision to install |
| `TRACELENS_PIP_SPEC` | git URL at `TRACELENS_GIT_REF` | Full pip spec, for private mirrors |

#### Distributed Runs

SLURM and Kubernetes runs collect `torch_profiler_output/` and `tracelens_output/` alongside the other profiling directories, per node and per rank. Analyze the collected tree on the host by pointing `--root` at it:

```bash
madengine report tracelens --root slurm_results
madengine report tracelens --root k8s_results --mode collective --world-size 16
```

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
        "POWER_DEVICE": "0",
        "POWER_SAMPLING_RATE": "0.1"
      }
    }
  ]
}
```

**Environment Variables:**
- `POWER_DEVICE` - GPU device(s): `"0"`, `"0,1,2"`, or `"all"` (default: `"all"`)
- `POWER_SAMPLING_RATE` - Sampling interval in seconds (default: `"0.1"`)
- `POWER_MODE` - Must be `"power"` for this tool (default: `"power"`)
- `POWER_DUAL_GCD` - Enable dual-GCD mode: `"true"` or `"false"` (default: `"false"`)

**Note:** To customize, override in tools configuration:
```json
{
  "tools": [
    {
      "name": "gpu_info_power_profiler",
      "env_vars": {
        "POWER_DEVICE": "0,1",
        "POWER_SAMPLING_RATE": "0.2"
      }
    }
  ]
}
```

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
        "VRAM_DEVICE": "all",
        "VRAM_SAMPLING_RATE": "0.5"
      }
    }
  ]
}
```

**Environment Variables:**
- `VRAM_DEVICE` - GPU device(s): `"0"`, `"0,1,2"`, or `"all"` (default: `"all"`)
- `VRAM_SAMPLING_RATE` - Sampling interval in seconds (default: `"0.1"`)
- `VRAM_MODE` - Must be `"vram"` for this tool (default: `"vram"`)
- `VRAM_DUAL_GCD` - Enable dual-GCD mode: `"true"` or `"false"` (default: `"false"`)

**Using Both Profilers Together:**
```json
{
  "tools": [
    {"name": "gpu_info_power_profiler"},
    {"name": "gpu_info_vram_profiler"}
  ]
}
```
This will generate both `gpu_info_power_profiler_output.csv` and `gpu_info_vram_profiler_output.csv`.
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
| `torch_profiler_dynolog` | `torch_profiler_output/*.json` | Kineto traces, one per rank |
| `tracelens` | `tracelens_output/*` | TraceLens reports plus `tracelens_summary.csv` |
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
        "POWER_DEVICE": "all",
        "POWER_SAMPLING_RATE": "0.1"
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

### False Failure Detection with ROCProf

**Issue:** Model runs marked as FAILURE despite successful execution

**Symptoms:**
- Status shows FAILURE but performance metrics are reported
- Log contains ROCProf messages like `E20251230 ... Opened result file`
- Error pattern `Error:` detected in logs

**Root Cause:**
ROCProf uses glog-style logging where `E` prefix means "Error level log" (not an actual error). These informational messages were incorrectly triggering failure detection.

**Fixed in:** madengine v2.0+

For false failures **not** caused by ROCProf (for example workloads that print benign `RuntimeError:` text), see [Configuration — Run phase: log error pattern scan](configuration.md#run-phase-log-error-pattern-scan) (`log_error_pattern_scan`, `log_error_benign_patterns`).

**Verification:**
```bash
# Run with profiling - should show SUCCESS status
madengine run --tags pyt_huggingface_gpt2 \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{"name": "rocprof"}, {"name": "rpd"}]
  }'

# Check status in output
# ✅ Expected: Status = SUCCESS, Performance = ~38-40 samples/second
```

**Technical Details:**
- ROCProf log patterns now excluded from error detection
- Error patterns made more specific (e.g., `RuntimeError:` vs `Error:`)
- Performance extraction hardened against bash segfaults during profiling
- Tests: `pytest tests/unit/test_error_handling.py::TestErrorPatternMatching`

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

## Environment Validation Tools

### TheRock Detection

Validate [TheRock](https://github.com/ROCm/TheRock) ROCm installations before running models. TheRock is AMD's lightweight build system for HIP and ROCm, distributed via Python pip packages.

**Enable TheRock validation:**

```bash
madengine run --tags dummy_therock \
  --tools therock_check \
  --additional-context '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
```

**Standalone detection:**

```bash
# Shell script (quick check)
bash src/madengine/scripts/common/tools/detect_therock.sh

# Python script (detailed output)
python3 src/madengine/scripts/common/tools/therock_detector.py --verbose

# JSON output (for scripting)
python3 src/madengine/scripts/common/tools/therock_detector.py --json
```

**Detection methods:**
- Python pip installations (`~/.local/lib/python*/site-packages/rocm`)
- Virtual environments with rocm packages
- System packages (`/usr/lib/python*/site-packages/rocm`)
- Tarball installations
- Local build directories
- Environment variables (`ROCM_PATH`, `HIP_PATH`)

**Configuration in tools.json:**

```json
{
  "therock_check": {
    "pre_scripts": [
      {
        "path": "scripts/common/tools/detect_therock.sh"
      }
    ],
    "cmd": "",
    "env_vars": {},
    "post_scripts": []
  }
}
```

**Features:**
- Non-blocking validation (warnings only)
- Automatic integration in `dummy_therock` model
- Reports GPU targets and installation paths
- Exit code 0 = found, 1 = not found

**Resources:**
- [TheRock GitHub](https://github.com/ROCm/TheRock)
- [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)

## Next Steps

- [Configuration Guide](configuration.md) - Detailed profiling configuration
- [Usage Guide](usage.md) - Running models with profiling
- [Deployment Guide](deployment.md) - Profiling in distributed environments

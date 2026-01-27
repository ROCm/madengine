# ROCprofv3 Profiling Configurations

This directory contains pre-configured profiling setups for different AI model benchmarking scenarios using madengine and ROCprofv3.

## Available Profiles

### 1. Compute-Bound Profiling (`rocprofv3_compute_bound.json`)

**Use Case**: Models bottlenecked by ALU operations (e.g., large transformers with dense matrix operations)

**Collected Metrics**:
- Wave execution and cycles
- VALU (Vector ALU) instructions
- SALU (Scalar ALU) instructions
- Wait states
- GPU power consumption

**Usage**:
```bash
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_compute_bound.json
```

### 2. Memory-Bound Profiling (`rocprofv3_memory_bound.json`)

**Use Case**: Models bottlenecked by memory bandwidth (e.g., large batch sizes, high-resolution inputs)

**Collected Metrics**:
- L1/L2 cache hit rates
- Memory read/write requests
- Cache efficiency
- VRAM usage over time

**Usage**:
```bash
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_memory_bound.json
```

### 3. Multi-GPU Profiling (`rocprofv3_multi_gpu.json`)

**Use Case**: Multi-GPU training with data parallel or model parallel

**Collected Metrics**:
- RCCL communication traces
- Inter-GPU memory transfers
- Scratch memory allocation
- Per-GPU power and VRAM

**Usage**:
```bash
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_multi_gpu.json
```

### 4. Comprehensive Profiling (`rocprofv3_comprehensive.json`)

**Use Case**: Full analysis with all available metrics (high overhead!)

**Collected Metrics**:
- All kernel traces (HIP, HSA, kernel, memory)
- Hardware performance counters
- Library call traces (MIOpen, rocBLAS)
- Power and VRAM monitoring
- Statistical summaries

**Usage**:
```bash
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_comprehensive.json
```

**Warning**: This profile has significant overhead. Use for detailed analysis only.

### 5. Lightweight Profiling (`rocprofv3_lightweight.json`)

**Use Case**: Production-like workloads with minimal profiling overhead

**Collected Metrics**:
- Basic HIP and kernel traces
- JSON output format (compact)

**Usage**:
```bash
madengine run --tags your_model \
  --additional-context-file examples/profiling-configs/rocprofv3_lightweight.json
```

### 6. Multi-Node Distributed (`rocprofv3_multinode.json`)

**Use Case**: Large-scale distributed training on SLURM clusters

**Collected Metrics**:
- RCCL communication patterns
- Cross-node synchronization
- Per-node power monitoring

**Usage**:
```bash
# Build phase
madengine build --tags your_model --registry your-registry:5000

# Deploy to SLURM
madengine run --manifest-file build_manifest.json \
  --additional-context-file examples/profiling-configs/rocprofv3_multinode.json
```

## Direct Tool Usage (Without Config Files)

### Single GPU - Compute Analysis
```bash
madengine run --tags dummy_prof \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{"name": "rocprofv3_compute"}]
  }'
```

### Multi-GPU - Communication Analysis
```bash
madengine run --tags your_model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "docker_gpus": "all",
    "distributed": {
      "launcher": "torchrun",
      "nproc_per_node": 8
    },
    "tools": [{"name": "rocprofv3_communication"}]
  }'
```

### Custom ROCprofv3 Command
```bash
madengine run --tags your_model \
  --additional-context '{
    "gpu_vendor": "AMD",
    "guest_os": "UBUNTU",
    "tools": [{
      "name": "rocprof",
      "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --hip-trace --kernel-trace --memory-copy-trace --output-format pftrace -d ./my_traces --",
      "env_vars": {
        "RCCL_DEBUG": "TRACE",
        "HSA_ENABLE_SDMA": "0"
      }
    }]
  }'
```

## Best Practices for Custom Commands

### Always Include the `--` Separator

When using custom profiling commands with `rocprof_wrapper.sh`, **always include the trailing `--`**:

```json
{
  "name": "rocprof",
  "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --sys-trace --"
}
```

**Why?** The `--` separator is critical for rocprofv3 (ROCm >= 7.0):
- **rocprofv3** requires: `rocprofv3 [options] -- <application>`
- **rocprof (legacy)** accepts: `rocprof [options] <application>`

The wrapper script auto-detects which profiler is available and formats the command correctly. Without the `--`, rocprofv3 will fail to parse arguments when the application command is appended.

**❌ Wrong:**
```json
{"cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --sys-trace"}
```

**✅ Correct:**
```json
{"cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --sys-trace --"}
```

## Available ROCprofv3 Tools

| Tool Name | Description | Key Options | Overhead |
|-----------|-------------|-------------|----------|
| `rocprofv3_compute` | Compute-bound analysis | Counter collection, VALU/SALU metrics | Medium |
| `rocprofv3_memory` | Memory bandwidth analysis | Cache hits/misses, memory transfers | Medium |
| `rocprofv3_communication` | Multi-GPU communication | RCCL trace, scratch memory | Medium |
| `rocprofv3_full` | Comprehensive profiling | All traces + counters + stats | High |
| `rocprofv3_lightweight` | Minimal overhead | HIP + kernel trace only | Low |
| `rocprofv3_perfetto` | Perfetto visualization | Perfetto-compatible output | Medium |
| `rocprofv3_api_overhead` | API call analysis | HIP/HSA/marker traces with stats | Low |
| `rocprofv3_pc_sampling` | Kernel hotspot analysis | PC sampling at 1000 Hz | Medium |

## Counter Definition Files

Counter files are located at `src/madengine/scripts/common/tools/counters/`:

- **`compute_bound.txt`**: Wave execution, VALU/SALU instructions, wait states
- **`memory_bound.txt`**: Cache metrics, memory controller traffic, LDS usage
- **`communication_bound.txt`**: PCIe traffic, atomic operations, synchronization
- **`full_profile.txt`**: Comprehensive set of all important metrics

You can create custom counter files and reference them in your profiling commands.

## Output Files

After profiling, madengine writes outputs to the working directory:

```
rocprof_output/
├── <timestamp>/
│   ├── *_results.db          # ROCprofv3 database (SQLite)
│   ├── kernel_trace.csv      # Kernel execution traces
│   ├── hip_api_trace.csv     # HIP API calls
│   └── memory_copy_trace.csv # Memory transfers
├── model_trace.pftrace       # Perfetto format (if using rocprofv3_perfetto)
└── trace.json                # JSON format (if using rocprofv3_lightweight)

gpu_info_power_profiler_output.csv  # Power consumption over time
gpu_info_vram_profiler_output.csv   # VRAM usage over time
library_trace.csv                    # Library API calls (if library tracing enabled)
```

## Visualization

### Perfetto UI (Recommended)
```bash
# If using rocprofv3_perfetto or output-format pftrace
# Upload files to https://ui.perfetto.dev/
```

### Custom Analysis
```python
import sqlite3
import pandas as pd

# Parse ROCprofv3 database
conn = sqlite3.connect('rocprof_output/<timestamp>/*_results.db')
kernels = pd.read_sql_query("SELECT * FROM kernels", conn)
print(kernels.head())
```

## Best Practices

1. **Start lightweight**: Use `rocprofv3_lightweight` for initial profiling
2. **Target your bottleneck**: Use specific profiles (compute/memory/communication) based on initial findings
3. **Avoid full profiling in production**: `rocprofv3_full` adds 20-50% overhead
4. **Multi-GPU**: Always enable RCCL tracing for distributed workloads
5. **Sampling rates**: Reduce sampling rates for long-running jobs (e.g., 1.0 instead of 0.1)
6. **Counter multiplexing**: ROCprofv3 may need multiple runs if too many counters are requested

## Troubleshooting

### No output files generated
```bash
# Check if rocprofv3 is available
which rocprofv3
rocprofv3 --version

# Verify ROCm version (>= 7.0 recommended for rocprofv3)
rocm-smi --version
```

### "Counter not available" errors
Some counters may not be available on all GPU architectures. Check available counters:
```bash
rocprofv3-avail
```

### High overhead affecting results
Use `rocprofv3_lightweight` or reduce counter collection:
```bash
# Remove counter collection for minimal overhead
madengine run --tags your_model \
  --additional-context '{
    "tools": [{
      "name": "rocprof",
      "cmd": "bash ../scripts/common/tools/rocprof_wrapper.sh --hip-trace --kernel-trace --output-format json -d ./traces --"
    }]
  }'
```

## Additional Resources

- [ROCprofv3 Official Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [madengine Profiling Guide](../../docs/profiling.md)
- [ROCm Developer Hub](https://rocm.docs.amd.com/)
- [Perfetto Trace Viewer](https://ui.perfetto.dev/)

## Examples

### Example 1: Profile LLM Inference (Compute-Bound)
```bash
madengine run --tags pyt_vllm_llama2_7b \
  --additional-context-file examples/profiling-configs/rocprofv3_compute_bound.json
```

### Example 2: Profile Multi-GPU Training (Communication-Bound)
```bash
madengine run --tags pyt_torchtitan_llama3_8b \
  --additional-context-file examples/profiling-configs/rocprofv3_multi_gpu.json
```

### Example 3: Profile Image Model (Memory-Bound)
```bash
madengine run --tags pyt_torchvision_resnet50 \
  --additional-context-file examples/profiling-configs/rocprofv3_memory_bound.json
```

### Example 4: Quick Test with Dummy Model
```bash
madengine run --tags dummy_prof \
  --additional-context-file examples/profiling-configs/rocprofv3_lightweight.json
```

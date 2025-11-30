# GPU Tool Managers

This directory contains the GPU tool management architecture for madengine, providing version-aware tool selection and robust fallback mechanisms for AMD ROCm and NVIDIA CUDA environments.

## Overview

The tool manager architecture provides a clean abstraction layer for interacting with vendor-specific GPU management tools, with automatic version detection and intelligent fallback strategies.

## Architecture

```
BaseGPUToolManager (Abstract)
├── ROCmToolManager (AMD)
└── NvidiaToolManager (NVIDIA)

GPUToolFactory
└── get_gpu_tool_manager(vendor) → BaseGPUToolManager
```

## Key Features

### Version-Aware Tool Selection (AMD ROCm)

Based on [PR #54](https://github.com/ROCm/madengine/pull/54), ROCm tool selection follows these rules:

- **ROCm >= 6.4.1**: Prefer `amd-smi`, fallback to `rocm-smi` with warning
- **ROCm < 6.4.1**: Use `rocm-smi`
- **Unknown version**: Try `amd-smi` first (conservative choice)

### Robust Fallback Strategy

1. Try preferred tool based on version
2. Log WARNING if primary tool fails
3. Attempt fallback tool with alternative command syntax
4. Raise comprehensive error with troubleshooting suggestions if both fail

### Comprehensive Error Messages

When tools fail, errors include:
- What was attempted
- Why it failed
- Actionable suggestions for fixing the issue
- Links to ROCm best practices

## Files

### Core Architecture

- **`gpu_tool_manager.py`**: Base abstract class with common infrastructure
  - Tool availability checking
  - Command execution with timeout
  - Result caching (thread-safe)
  - Consistent logging

- **`gpu_tool_factory.py`**: Factory pattern for creating tool managers
  - Singleton management per vendor
  - Auto-detection support
  - Cache management

### Vendor Implementations

- **`rocm_tool_manager.py`**: AMD ROCm tool manager
  - ROCm version detection (multiple methods)
  - Version-aware amd-smi/rocm-smi selection
  - GPU count, product name, architecture queries
  - Fallback support for all operations

- **`nvidia_tool_manager.py`**: NVIDIA CUDA tool manager
  - Basic nvidia-smi and nvcc wrappers
  - CUDA/driver version detection
  - GPU queries
  - Placeholder for future version-aware logic

## Usage Examples

### Basic Usage

```python
from madengine.utils.gpu_tool_factory import get_gpu_tool_manager

# Auto-detect vendor and get appropriate manager
manager = get_gpu_tool_manager()

# Get GPU count
num_gpus = manager.get_gpu_count()

# Get GPU product name
product = manager.get_gpu_product_name(gpu_id=0)

# Get version
version = manager.get_version()
```

### Explicit Vendor Selection

```python
from madengine.utils.gpu_tool_factory import get_gpu_tool_manager
from madengine.utils.gpu_validator import GPUVendor

# AMD ROCm
amd_manager = get_gpu_tool_manager(GPUVendor.AMD)
rocm_version = amd_manager.get_rocm_version()  # Returns tuple: (6, 4, 1)
preferred_tool = amd_manager.get_preferred_smi_tool()  # "amd-smi" or "rocm-smi"

# NVIDIA CUDA
nvidia_manager = get_gpu_tool_manager(GPUVendor.NVIDIA)
cuda_version = nvidia_manager.get_cuda_version()  # Returns string: "12.0"
```

### Integration with Context

```python
from madengine.core.context import Context

context = Context()
# Tool manager is automatically created and cached
num_gpus = context.get_system_ngpus()  # Uses tool manager internally
product_name = context.get_system_gpu_product_name()  # With PR #54 fallback
```

## ROCm Version Detection

The ROCmToolManager tries multiple methods in order:

1. **hipconfig --version** (primary, most reliable)
2. **/opt/rocm/.info/version** file (fallback)
3. **rocminfo** parsing (last resort)

Results are cached for performance.

## ROCm Tool Selection Logic

```python
# Example: ROCm 6.4.1 system with amd-smi
manager = ROCmToolManager()
manager.get_preferred_smi_tool()  # Returns "amd-smi"

# If amd-smi fails, automatically tries rocm-smi
count = manager.get_gpu_count()  
# Logs: "WARNING: amd-smi failed, trying fallback rocm-smi"
```

## Error Handling Example

```python
try:
    manager = get_gpu_tool_manager(GPUVendor.AMD)
    product = manager.get_gpu_product_name(0)
except RuntimeError as e:
    # Error includes:
    # - What commands were tried
    # - Why they failed
    # - Suggestions for fixing
    # - Links to documentation
    print(e)
```

Example error output:
```
Unable to get GPU product name for GPU 0.

ROCm Version Detected: 6.4.1 (preferred tool: amd-smi)

Attempted:
1. amd-smi static -g 0 | grep MARKET_NAME:
   Error: /opt/rocm/bin/amd-smi not found
2. rocm-smi --showproductname (fallback)
   Error: Permission denied on /dev/kfd

Suggestions:
- Verify ROCm 6.4.1 installation includes amd-smi
- Check GPU device permissions: ls -la /dev/kfd /dev/dri
- Ensure user is in 'video' and 'render' groups
- See: https://github.com/ROCm/TheRock for ROCm best practices
```

## Testing

Run unit tests:
```bash
pytest tests/test_gpu_tool_managers.py -v
```

Key test scenarios:
- ROCm version detection (6.4.0, 6.4.1, 6.5.0)
- Tool selection based on version
- Fallback behavior when tools unavailable
- Error messages and suggestions

## ROCm Best Practices

This implementation follows best practices from:
- [ROCm/TheRock](https://github.com/ROCm/TheRock) - Build system and tool migration
- [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems) - System tools
- [PR #54](https://github.com/ROCm/madengine/pull/54) - Tool migration guide

### Key Recommendations

1. **Version Detection**: Always check ROCm version before selecting tools
2. **Fallback Support**: Provide rocm-smi fallback for amd-smi in ROCm >= 6.4.1
3. **Error Messages**: Include actionable troubleshooting steps
4. **Tool Paths**: Use standard ROCm paths (/opt/rocm/bin/)

## Backward Compatibility

- Legacy madengine (`mad.py`, `run_models.py`) continues to work unchanged
- Context methods maintain same signatures
- Shared code works for both legacy and new madengine-cli

## Future Enhancements

### NVIDIA Tool Manager
- Version-aware tool selection for different CUDA versions
- Fallback strategies for nvidia-smi variations
- Enhanced error handling similar to ROCm

### Additional Features
- Tool manager plugins for other GPU vendors (Intel, etc.)
- Performance profiling tool integration
- Remote GPU tool execution support


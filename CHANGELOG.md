# Changelog

All notable changes to madengine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **ROCprofv3 Argument Parsing**: Fixed rocprof_wrapper.sh argument parsing with custom commands
  - Test `test_can_change_default_behavior_of_profiling_tool_with_additionalContext` now includes required `--` separator
  - Without `--`, rocprofv3 would incorrectly parse application command as profiler boolean option
  - Error manifested as: `ValueError: invalid truth value bash (type=str)`
  - Fix ensures compatibility with both rocprof (legacy) and rocprofv3 (ROCm >= 7.0)
- **Error Pattern Detection**: Fixed false failure detection in HuggingFace GPT2/BERT models
  - ROCProf logging messages (E20251230/W20251230 prefixes) no longer trigger false failures
  - Added benign pattern list to exclude profiling tool output from error detection
  - Made error patterns more specific (e.g., `RuntimeError:` instead of `Error:`)
  - Improved performance metric extraction robustness to prevent bash segfaults during profiling
  - Tests: Added `TestErrorPatternMatching` class in `tests/unit/test_error_handling.py`
- Removed stale compiled Python file (`__init__.pyc`) from source tree
- Cleaned up unused `typing_extensions` import in `core/console.py`
- Improved type hint accuracy in `Console.sh()` method docstring

### Documentation
- **ROCprofv3 Usage Guide**: Enhanced documentation for custom profiling commands
  - Added section in `docs/profiling.md` explaining the `--` separator requirement
  - Added "Best Practices" section in `examples/profiling-configs/README.md`
  - Enhanced `rocprof_wrapper.sh` header comments with usage examples
  - Clarified that `--` must always be included when using custom rocprof commands
  - Documented auto-detection behavior between rocprof (legacy) and rocprofv3

### Breaking Changes
- **CLI Unification**: Simplified command-line interface
  - ✅ `madengine` is now the unified CLI command (previously `madengine-cli`)
  - ❌ Removed legacy `madengine` v1.x CLI (previously `mad.py`)
  - ❌ Removed `madengine-cli` alias (use `madengine` instead)
  - **Migration**: Simply replace `madengine-cli` with `madengine` in your scripts
  - All functionality remains identical, just cleaner command naming

### Removed
- **Legacy CLI Components**:
  - `src/madengine/mad.py` - Legacy CLI entry point (v1.x)
  - `src/madengine/tools/run_models.py` - Legacy model runner
  - `docs/legacy-cli.md` - Legacy CLI documentation
- Justification: Modern `madengine` CLI (formerly `madengine-cli`) provides all functionality plus K8s, SLURM, and distributed support

### Security
- **CRITICAL:** Fixed SQL injection vulnerability in legacy database module (`src/madengine/db/database_functions.py`)
  - Replaced string formatting with parameterized queries using SQLAlchemy `text()`
  - Prevents potential SQL injection attacks in `get_matching_db_entries()` function
- Fixed 4 instances of bare `except:` blocks that could mask critical exceptions
  - `kubernetes.py`: Replaced with specific exception types (`ConfigException`, `FileNotFoundError`, `ApiException`)
  - `console.py`: Replaced with specific exception types (`OSError`, `ValueError`) for resource cleanup

### Added
- **ROCprofv3 Profiling Suite** (ROCm 7.0+): 8 pre-configured profiling profiles for AI model benchmarking
  - `rocprofv3_compute` - Compute-bound analysis (VALU/SALU instructions, wave execution)
  - `rocprofv3_memory` - Memory-bound analysis (cache metrics, memory bandwidth)
  - `rocprofv3_communication` - Multi-GPU communication analysis (RCCL traces, inter-GPU transfers)
  - `rocprofv3_full` - Comprehensive profiling with all metrics (high overhead)
  - `rocprofv3_lightweight` - Minimal overhead profiling (production-friendly)
  - `rocprofv3_perfetto` - Perfetto UI compatible trace generation
  - `rocprofv3_api_overhead` - API call timing analysis (HIP/HSA/marker traces)
  - `rocprofv3_pc_sampling` - Kernel hotspot identification (PC sampling at 1000 Hz)
- **Hardware Counter Definitions**: 4 counter files for targeted profiling scenarios
  - `compute_bound.txt` - Wave execution, ALU instructions, wait states
  - `memory_bound.txt` - Cache hit rates, memory controller traffic, LDS usage
  - `communication_bound.txt` - PCIe traffic, atomic operations, synchronization
  - `full_profile.txt` - Comprehensive metrics for complete analysis
- **Profiling Configuration Examples**: 6 ready-to-use JSON configs in `examples/profiling-configs/`
  - Single-GPU profiles (compute, memory, lightweight)
  - Multi-GPU distributed training profile
  - Comprehensive full-stack profiling
  - Multi-node SLURM deployment config
- **Comprehensive Launcher Support**: Full K8s and SLURM support for 6 distributed frameworks
  - TorchTitan: LLM pre-training with FSDP2+TP+PP+CP parallelism
  - vLLM: High-throughput LLM inference with continuous batching
  - SGLang: Fast LLM inference with structured generation
  - DeepSpeed: ZeRO optimization training (K8s support added)
  - Megatron-LM: Large-scale transformer training (K8s + SLURM)
  - torchrun: Standard PyTorch DDP/FSDP
- **Centralized Launcher Documentation**: `docs/distributed-launchers.md` with comprehensive guide
- **Example Configurations**: 6 new minimal configs for distributed launchers (K8s)
- Comprehensive development tooling and configuration
- Pre-commit hooks for code quality
- Makefile for common development tasks
- Developer guide with coding standards
- Type checking with mypy
- Code formatting with black and isort
- Enhanced .gitignore for better file exclusions
- CI/CD configuration templates
- **Major Documentation Refactor**: Complete integration of distributed execution and CLI guides into README.md
- Professional open-source project structure with badges and table of contents
- Comprehensive MAD package integration documentation
- Enhanced model discovery and tag system documentation
- Modern deployment scenarios and configuration examples

### Changed
- **README.md**: Added launcher ecosystem highlights to v2.0 features
- **K8s README**: Updated with new launcher configs and comprehensive launcher section
- **Documentation Structure**: Consolidated all launcher docs into single comprehensive guide
- Improved package initialization and imports
- Replaced print statements with proper logging in main CLI
- Enhanced error handling and logging throughout codebase
- Cleaned up setup.py for better maintainability
- Updated development dependencies in pyproject.toml
- **Complete README.md overhaul**: Merged all documentation into a single, comprehensive source
- Restructured documentation to emphasize MAD package integration
- Enhanced CLI usage examples and distributed execution workflows
- Improved developer contribution guidelines and legacy compatibility notes

### Changed (Previous)
- Removed Python cache files from repository
- Fixed import organization and structure
- Improved docstring formatting and consistency
- Cleaned up documentation fragmentation

### Removed
- Unnecessary debug print statements
- Python cache files and build artifacts
- **Legacy documentation files**: `docs/distributed-execution-solution.md` and `docs/madengine-cli-guide.md`
- **Duplicate documentation**: `docs/TORCHTITAN_LAUNCHER.md` (consolidated into distributed-launchers.md)
- Redundant documentation scattered across multiple files

## [Previous Versions]

For changes in previous versions, please refer to the git history.

---

## Guidelines for Changelog Updates

### Categories
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

### Format
- Keep entries brief but descriptive
- Include ticket/issue numbers when applicable
- Group related changes together
- Use present tense ("Add feature" not "Added feature")
- Target audience: users and developers of the project

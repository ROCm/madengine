# Changelog

All notable changes to madengine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed
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

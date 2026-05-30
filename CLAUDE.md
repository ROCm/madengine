# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Optional: install Kubernetes support
pip install -e ".[all]"

# Setup pre-commit hooks
pre-commit install
```

## Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_error_handling.py -v

# Run specific test class or function
pytest tests/unit/test_error_handling.py::TestErrorPatternMatching -v

# Run tests with coverage
pytest --cov=src/madengine --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/madengine

# Run all pre-commit checks
pre-commit run --all-files
```

## Architecture

madengine is a CLI tool for running AI/ML models in local Docker, Kubernetes, and SLURM environments. The entry point is `madengine.cli.app:cli_main` (registered as the `madengine` console script).

### Layer Structure

**CLI Layer** (`src/madengine/cli/`)
- `app.py` — Typer app wiring, registers 5 commands: `discover`, `build`, `run`, `report`, `database`
- `commands/` — One file per command (build, run, discover, report, database)
- `constants.py` — `ExitCode` enum (`SUCCESS=0`, `FAILURE=1`, `BUILD_FAILURE=2`, `RUN_FAILURE=3`, `INVALID_ARGS=4`)

**Orchestration Layer** (`src/madengine/orchestration/`)
- `build_orchestrator.py` — `BuildOrchestrator`: discovers models, builds Docker images, writes `build_manifest.json`
- `run_orchestrator.py` — `RunOrchestrator`: reads or triggers builds, infers deployment target, delegates to local or distributed execution

**Core Layer** (`src/madengine/core/`)
- `context.py` — `Context` class: merges `additional_context` with system detection (GPU vendor, architecture, OS, ROCm path). Uses `ast.literal_eval()` to parse additional_context strings (not `json.loads` — pass Python dict repr, not JSON)
- `console.py` — `Console`: shell execution wrapper with live output support
- `docker.py` — Docker command wrapper

**Execution Layer** (`src/madengine/execution/`)
- `container_runner.py` — `ContainerRunner`: runs models from manifest via `docker run`, writes results to `perf.csv`
- `docker_builder.py` — `DockerBuilder`: builds images from Dockerfiles
- `container_runner_helpers.py` — Log error pattern scanning, timeout resolution

**Deployment Layer** (`src/madengine/deployment/`)
- `factory.py` — `DeploymentFactory`: Factory pattern, registers `SlurmDeployment` and `KubernetesDeployment`
- `base.py` — `BaseDeployment` abstract class, `DeploymentConfig` dataclass
- `kubernetes.py` / `slurm.py` — Concrete deployments; target is inferred by Convention over Configuration: presence of `"k8s"` or `"kubernetes"` key → K8s; `"slurm"` key → SLURM; neither → local
- `presets/` — JSON preset files for K8s/SLURM default configurations; auto-merged with minimal user configs
- `config_loader.py` — Loads and merges preset JSON with user-supplied config

**Utils** (`src/madengine/utils/`)
- `discover_models.py` — `DiscoverModels`: three discovery methods: root `models.json`, `scripts/{dir}/models.json`, or `scripts/{dir}/get_models_json.py` (dynamic)
- `gpu_tool_factory.py` / `gpu_tool_manager.py` — GPU vendor abstraction (AMD/NVIDIA)
- `gpu_validator.py` — ROCm installation detection, GPU vendor detection
- `config_parser.py` — `ConfigParser`: parses `--additional-context` and tools config

**Reporting** (`src/madengine/reporting/`)
- `update_perf_csv.py` — Writes/appends to `perf.csv` and `perf_entry.csv`
- `csv_to_html.py` / `csv_to_email.py` — Report generation

### Key Data Flows

1. **Build flow**: CLI → `BuildOrchestrator` → `DiscoverModels` (finds models by tags) → `DockerBuilder` (builds images) → writes `build_manifest.json`

2. **Run flow**: CLI → `RunOrchestrator` → loads/generates `build_manifest.json` → infers target → `ContainerRunner` (local) or `DeploymentFactory` (K8s/SLURM) → writes `perf.csv`

3. **`additional_context`**: User JSON/Python-dict string merged into `Context.ctx`. Context is parsed with `ast.literal_eval()`, so values can use Python dict syntax. Keys like `k8s`, `slurm`, `distributed`, `tools`, `pre_scripts`, `post_scripts` drive behavior.

4. **Model definition**: Models defined in `models.json` with fields: `name`, `tags`, `dockerfile`, `scripts`, `n_gpus`, `args`, `timeout`, `skip_gpu_arch`, etc.

5. **Script isolation**: During run, `scripts/common/` is populated from the madengine package (pre_scripts, post_scripts, tools) and cleaned up afterwards. The MAD project's own `scripts/` and `docker/` directories are preserved.

### Deployment Target Inference

No explicit `"deploy"` field is needed. Target is inferred from config structure:
- `"k8s"` or `"kubernetes"` key present → Kubernetes deployment
- `"slurm"` key present → SLURM deployment
- Neither → local Docker execution

### Test Structure

```
tests/
├── unit/         # Fast isolated tests with mocking
├── integration/  # End-to-end with real Docker/system calls
├── e2e/          # Full workflow tests
└── fixtures/     # Dummy models, scripts, and data for testing
```

Pytest config is in `pyproject.toml` under `[tool.pytest.ini_options]`. Test markers: `slow`, `integration`.

### Code Style

- Black formatting, 88-character line length
- isort with `profile = "black"`
- Google-style docstrings
- Type hints required for public functions
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `style:`, `perf:`, `chore:`

#!/usr/bin/env python3
"""
Docker Container Runner Module for madengine

This module handles the Docker container execution phase separately from building,
enabling distributed workflows where containers are run on remote nodes
using pre-built images.
"""

import json
import os
import re
import shlex
import subprocess
import time
import typing
import warnings
from contextlib import redirect_stderr, redirect_stdout

from rich.console import Console as RichConsole

from madengine.core.auth import login_to_registry
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.dataprovider import Data
from madengine.core.docker import Docker
from madengine.core.timeout import Timeout
from madengine.deployment.base import PERFORMANCE_LOG_PATTERN
from madengine.execution.container_runner_helpers import (
    log_text_has_error_pattern,
    make_run_log_file_path,
    resolve_log_error_scan_config,
    resolve_run_timeout,
)
from madengine.reporting.update_perf_csv import (
    PERF_CSV_HEADER,
    flatten_tags,
    update_perf_csv,
)
from madengine.reporting.update_perf_super import (
    update_perf_super_csv,
    update_perf_super_json,
)
from madengine.utils.config_parser import ConfigParser
from madengine.utils.gpu_config import resolve_runtime_gpus
from madengine.utils.ops import PythonicTee, file_print
from madengine.utils.path_utils import scripts_base_dir_from
from madengine.utils.run_details import get_build_number, get_pipeline
from madengine.utils.therock_markers import is_therock_tree


def _print_run_env_table(
    gpu_vendor: str,
    context,
    model_docker,
    rich_console,
) -> None:
    """Print a side-by-side environment table for host and container.

    Covers AMD (installation type, ROCm root, ROCm version) and NVIDIA
    (installation type, CUDA root, CUDA version) for both sides.
    Follows container_runner.py convention: rich_console for styled borders,
    plain print() for data rows (so they appear in the run log file).
    """
    from pathlib import Path

    COL_W = 36  # width of each data column

    def row(label: str, host_val: str, container_val: str) -> str:
        return f"  {label:<26}  {host_val:<{COL_W}}  {container_val:<{COL_W}}"

    def separator() -> str:
        return "  " + "-" * (26 + 2 + COL_W + 2 + COL_W)

    def _sh(cmd: str) -> str:
        """Run a command in the container and return stripped output."""
        try:
            return (model_docker.sh(cmd) or "").strip()
        except Exception:
            return "N/A"

    rich_console.print("\n[bold blue]🖥️   RUN PHASE ENVIRONMENT[/bold blue]")
    rich_console.print(f"[dim]{'=' * 80}[/dim]")
    print(row("", "HOST", "CONTAINER"))
    print(separator())

    if "AMD" in gpu_vendor:
        # ── Host side ──────────────────────────────────────────────
        host_rocm_root = getattr(context, "_rocm_path", None) or "unknown"
        _host_rocm_path = Path(host_rocm_root)
        host_install_type = (
            "therock"
            if _host_rocm_path.is_dir() and is_therock_tree(_host_rocm_path)
            else "apt install" if _host_rocm_path.is_dir() else "unknown"
        )
        try:
            host_rocm_ver = context._get_tool_manager().get_version() or "unknown"
        except Exception:
            host_rocm_ver = "unknown"

        # ── Container side ─────────────────────────────────────────
        # Installation type: if rocm-sdk resolves a root it is TheRock;
        # otherwise fall back to the traditional .info/version marker.
        # Avoids nested quoting issues by not embedding $(...) inside [ -f "..." ].
        ctr_install_type = _sh(
            "if command -v rocm-sdk >/dev/null 2>&1 "
            "&& rocm-sdk path --root >/dev/null 2>&1; "
            "then echo therock; "
            "elif [ -f /opt/rocm/.info/version ]; then echo 'apt install'; "
            "else echo unknown; fi"
        )

        # ROCm root: prefer rocm-sdk, then ROCM_PATH env, then /opt/rocm
        ctr_rocm_root = _sh(
            "rocm-sdk path --root 2>/dev/null " '|| echo "${ROCM_PATH:-/opt/rocm}"'
        )

        # ROCm version: prefer rocm-sdk, then .info/version, then rocminfo
        ctr_rocm_ver = _sh(
            "rocm-sdk version 2>/dev/null "
            '|| cat "${ROCM_PATH:-/opt/rocm}/.info/version" 2>/dev/null '
            "|| rocminfo 2>/dev/null | grep -i 'ROCm Version' | head -n1 | sed 's/.*[Vv]ersion:[[:space:]]*//;s/[[:space:]].*//;s/[^0-9.]//g' 2>/dev/null "
            "|| echo unknown"
        )

        print(row("GPU Vendor", "AMD", "AMD"))
        print(row("Installation Type", host_install_type, ctr_install_type))
        print(row("ROCm Root", host_rocm_root, ctr_rocm_root))
        print(row("ROCm Version", host_rocm_ver, ctr_rocm_ver))

    elif "NVIDIA" in gpu_vendor:
        # ── Host side ──────────────────────────────────────────────
        def _host_sh(cmd: str) -> str:
            try:
                return subprocess.check_output(
                    cmd, shell=True, stderr=subprocess.DEVNULL, text=True
                ).strip()
            except Exception:
                return "unknown"

        host_cuda_root = _host_sh(
            "nvcc --version 2>/dev/null | sed -n 's/.*release \\([0-9][0-9.]*\\).*/\\1/p' | head -1 | "
            "xargs -I{} dirname $(which nvcc 2>/dev/null) 2>/dev/null | xargs dirname 2>/dev/null "
            '|| echo "${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"'
        )
        host_cuda_ver = _host_sh(
            "nvcc --version 2>/dev/null | sed -n 's/.*release \\([0-9][0-9.]*\\).*/\\1/p' | head -1 "
            "|| nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \\([0-9][0-9.]*\\).*/\\1/p' | head -1 "
            "|| echo unknown"
        )

        # ── Container side ─────────────────────────────────────────
        ctr_cuda_root = _sh(
            "dirname $(which nvcc 2>/dev/null) 2>/dev/null | xargs dirname 2>/dev/null "
            '|| echo "${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"'
        )
        ctr_cuda_ver = _sh(
            "nvcc --version 2>/dev/null | sed -n 's/.*release \\([0-9][0-9.]*\\).*/\\1/p' | head -1 "
            "|| nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \\([0-9][0-9.]*\\).*/\\1/p' | head -1 "
            "|| echo unknown"
        )

        print(row("GPU Vendor", "NVIDIA", "NVIDIA"))
        print(row("Installation Type", "CUDA toolkit", "CUDA toolkit"))
        print(row("CUDA Root", host_cuda_root, ctr_cuda_root))
        print(row("CUDA Version", host_cuda_ver, ctr_cuda_ver))

    print(separator())
    rich_console.print(f"[dim]{'=' * 80}[/dim]\n")


def _resolve_multiple_results_path(
    multiple_results: str, model_dir: str
) -> typing.Optional[str]:
    """Resolve multiple_results CSV path: try cwd then model_dir. Return first that exists."""
    if not multiple_results:
        return None
    if os.path.isfile(multiple_results):
        return multiple_results
    path_in_model_dir = os.path.join(model_dir, multiple_results)
    if os.path.isfile(path_in_model_dir):
        return path_in_model_dir
    return None


def _docker_image_exists_locally(image: str) -> bool:
    """Return True if ``docker image inspect`` succeeds for *image* (argv list; no shell)."""
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def _bash_quote_path(path: str) -> str:
    """Shell-escape a path for ``bash -c`` in the container (POSIX)."""
    return shlex.quote(os.path.normpath((path or "").replace("\\", "/")))


def _cp_model_dir_file_to_cwd_cmd(model_dir: str, relative_path: str) -> str:
    """``cp --`` from ``model_dir/relative`` to ``.`` with quoted paths (no injection)."""
    rel = (relative_path or "").strip()
    src = os.path.normpath(os.path.join(model_dir, rel)).replace("\\", "/")
    return f"cp -- {_bash_quote_path(src)} {_bash_quote_path('.')} 2>/dev/null || true"


class ContainerRunner:
    """Class responsible for running Docker containers with models."""

    def __init__(
        self,
        context: Context = None,
        data: Data = None,
        console: Console = None,
        live_output: bool = False,
        additional_context: typing.Dict = None,
    ):
        """Initialize the Container Runner.

        Args:
            context: The madengine context
            data: The data provider instance
            console: Optional console instance
            live_output: Whether to show live output
            additional_context: Additional configuration context (for GPU resolution)
        """
        self.context = context
        self.data = data
        self.console = console or Console(live_output=live_output)
        self.live_output = live_output
        self.rich_console = RichConsole()
        self.credentials = None
        self.perf_csv_path = "perf.csv"  # Default output path
        self.additional_context = additional_context or {}

        # Ensure runtime context is initialized for container operations
        if self.context:
            self.context.ensure_runtime_context()

    def set_perf_csv_path(self, path: str):
        """Set the path for the performance CSV output file.

        Args:
            path: Path to the performance CSV file
        """
        self.perf_csv_path = path

    def ensure_perf_csv_exists(self):
        """Ensure the performance CSV file exists with proper headers."""
        if not os.path.exists(self.perf_csv_path):
            file_print(
                PERF_CSV_HEADER,
                filename=self.perf_csv_path,
                mode="w",
            )
            print(f"Created performance CSV file: {self.perf_csv_path}")

    def create_run_details_dict(
        self, model_info: typing.Dict, build_info: typing.Dict, run_results: typing.Dict
    ) -> typing.Dict:
        """Create a run details dictionary similar to RunDetails class in run_models.py.

        Args:
            model_info: Model information dictionary
            build_info: Build information from manifest
            run_results: Container execution results

        Returns:
            dict: Run details dictionary for CSV generation
        """
        import os

        # Resolve GPU count using hierarchical resolution
        resolved_gpu_count = resolve_runtime_gpus(model_info, self.additional_context)

        # Convert -1 (all GPUs) to actual system GPU count for accurate reporting
        if resolved_gpu_count == -1 and self.context:
            try:
                system_ngpus = int(
                    self.context.ctx["docker_env_vars"]["MAD_SYSTEM_NGPUS"]
                )
                resolved_gpu_count = system_ngpus
                print(
                    f"ℹ️  Converted n_gpus=-1 to actual system GPU count: {system_ngpus}"
                )
            except (KeyError, ValueError, TypeError):
                # If system GPU count not available, keep -1
                pass

        # Determine number of nodes and GPUs per node
        # Priority: 1. SLURM env vars, 2. additional_context, 3. model_info, 4. default (1)
        nnodes = "1"  # Default for local execution
        gpus_per_node = str(resolved_gpu_count)

        # Check for SLURM multi-node environment
        if os.environ.get("MAD_DEPLOYMENT_TYPE") == "slurm":
            # Get from SLURM environment variables (most accurate for SLURM jobs)
            slurm_nnodes = os.environ.get("NNODES") or os.environ.get("SLURM_NNODES")
            slurm_gpus_per_node = os.environ.get("GPUS_PER_NODE") or os.environ.get(
                "SLURM_GPUS_PER_NODE"
            )

            if slurm_nnodes:
                nnodes = str(slurm_nnodes)
                print(f"ℹ️  Detected SLURM multi-node: {nnodes} nodes")

            if slurm_gpus_per_node:
                gpus_per_node = str(slurm_gpus_per_node)
                print(f"ℹ️  GPUs per node: {gpus_per_node}")

        # Fallback to additional_context (for non-SLURM or if env vars not set)
        if nnodes == "1" and self.additional_context:
            slurm_config = self.additional_context.get("slurm", {})
            if slurm_config:
                ctx_nodes = slurm_config.get("nodes")
                ctx_gpus = slurm_config.get("gpus_per_node")
                if ctx_nodes:
                    nnodes = str(ctx_nodes)
                if ctx_gpus:
                    gpus_per_node = str(ctx_gpus)

        # Final fallback to model_info
        if nnodes == "1":
            nnodes = model_info.get("nnodes", "1")

        # Calculate total GPUs
        try:
            total_gpus = int(nnodes) * int(gpus_per_node)
        except (ValueError, TypeError):
            total_gpus = resolved_gpu_count

        # Extract launcher from multiple sources in priority order:
        # 1. additional_context (passed via --additional-context CLI arg)
        # 2. model_info distributed config (in models.json)
        # 3. MAD_LAUNCHER environment variable
        # 4. Default to 'docker' for local deployments
        launcher = ""

        # Check additional_context first (highest priority)
        if self.additional_context:
            distributed_config = self.additional_context.get("distributed", {})
            launcher = distributed_config.get("launcher", "")
            if launcher:
                print(f"🚀 Launcher from additional_context: {launcher}")

        # Check model_info distributed config
        if not launcher and model_info.get("distributed"):
            launcher = model_info["distributed"].get("launcher", "")
            if launcher:
                print(f"🚀 Launcher from model_info: {launcher}")

        # Fallback to environment variable
        if not launcher:
            launcher = os.environ.get("MAD_LAUNCHER", "")
            if launcher:
                print(f"🚀 Launcher from MAD_LAUNCHER env: {launcher}")

        # Apply deployment-specific defaults if no launcher specified
        deployment_type = os.environ.get("MAD_DEPLOYMENT_TYPE", "local")
        if not launcher:
            if deployment_type == "kubernetes":
                launcher = "native"
                print(f"🚀 Launcher defaulted to 'native' for kubernetes deployment")
            elif deployment_type == "slurm":
                # For SLURM, try to get launcher type from environment or default to torchrun
                # Note: "slurm" is the deployment type, not the launcher
                launcher = os.environ.get("MAD_LAUNCHER_TYPE", "torchrun")
                print(f"🚀 Launcher defaulted to '{launcher}' for slurm deployment")
            elif deployment_type == "local":
                launcher = "docker"
                print(f"🚀 Launcher defaulted to 'docker' for local deployment")

        # Print final launcher selection
        if launcher:
            print(
                f"✅ Final launcher selected: '{launcher}' (deployment_type: {deployment_type})"
            )
        else:
            print(f"⚠️  No launcher specified (deployment_type: {deployment_type})")

        # Create run details dict with all required fields
        run_details = {
            "model": model_info["name"],
            "n_gpus": str(total_gpus),  # Total GPUs across all nodes
            "nnodes": nnodes,
            "gpus_per_node": gpus_per_node,
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": run_results.get(
                "docker_image", build_info.get("docker_image", "")
            ),
            "git_commit": run_results.get("git_commit", ""),
            "machine_name": run_results.get("machine_name", ""),
            "deployment_type": os.environ.get(
                "MAD_DEPLOYMENT_TYPE", "local"
            ),  # local, slurm, etc.
            "launcher": launcher,  # Distributed launcher: torchrun, vllm, sglang, deepspeed, etc.
            "gpu_architecture": (
                (self.context.ctx.get("docker_env_vars") or {}).get(
                    "MAD_SYSTEM_GPU_ARCHITECTURE", ""
                )
                if self.context
                else ""
            ),
            "performance": run_results.get("performance", ""),
            "metric": run_results.get("metric", ""),
            "relative_change": "",
            "status": run_results.get("status", "FAILURE"),
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": run_results.get("test_duration", ""),
            "dataname": run_results.get("dataname", ""),
            "data_provider_type": run_results.get("data_provider_type", ""),
            "data_size": run_results.get("data_size", ""),
            "data_download_duration": run_results.get("data_download_duration", ""),
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get(
                "additional_docker_run_options", ""
            ),
        }

        # Flatten tags if they are in list format
        flatten_tags(run_details)

        # Parse and load config file if present in args for perf_entry_super.json
        try:
            scripts_path = model_info.get("scripts", "")
            scripts_base_dir = scripts_base_dir_from(scripts_path)
            config_parser = ConfigParser(scripts_base_dir=scripts_base_dir)
            run_details["configs"] = config_parser.parse_and_load(
                model_info.get("args", ""), scripts_path
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not parse config file: {e}")
            run_details["configs"] = None

        return run_details

    def _create_setup_failure_perf_entry(
        self,
        model_info: typing.Dict,
        build_info: typing.Dict,
        image_name: str,
        error_message: str,
    ) -> typing.Dict:
        """Build a minimal perf entry for failures that occur before run_container (e.g. pull failed).

        Used so that every failed model is recorded in the performance table with status FAILURE.
        """
        machine_name = ""
        if self.console:
            try:
                machine_name = self.console.sh("hostname")
            except Exception:
                pass

        tags = model_info.get("tags", "")
        if isinstance(tags, list):
            tags = ",".join(str(t) for t in tags)

        return {
            "model": model_info.get("name", image_name),
            "n_gpus": str(model_info.get("n_gpus", "1")),
            "nnodes": "1",
            "gpus_per_node": str(model_info.get("n_gpus", "1")),
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": get_pipeline(),
            "args": model_info.get("args", ""),
            "tags": tags,
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", image_name),
            "git_commit": "",
            "machine_name": machine_name,
            "deployment_type": os.environ.get("MAD_DEPLOYMENT_TYPE", "local"),
            "launcher": "",
            "gpu_architecture": (
                self.context.ctx.get("docker_env_vars", {}).get(
                    "MAD_SYSTEM_GPU_ARCHITECTURE", ""
                )
                if self.context
                else ""
            ),
            "performance": "",
            "metric": "",
            "relative_change": "",
            "status": "FAILURE",
            "build_duration": build_info.get("build_duration", ""),
            "test_duration": "",
            "dataname": "",
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": get_build_number(),
            "additional_docker_run_options": model_info.get(
                "additional_docker_run_options", ""
            ),
        }

    def load_build_manifest(
        self, manifest_file: str = "build_manifest.json"
    ) -> typing.Dict:
        """Load build manifest from file.

        Args:
            manifest_file: Path to build manifest file

        Returns:
            dict: Build manifest data
        """
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        print(f"Loaded build manifest from: {manifest_file}")
        return manifest

    def login_to_registry(self, registry: str, credentials: typing.Dict = None) -> None:
        """Login to a Docker registry for pulling images.

        Delegates to :func:`madengine.core.auth.login_to_registry`.
        Does not raise on failure so public images can still be pulled.
        """
        login_to_registry(
            registry,
            credentials,
            console=self.console,
            rich_console=self.rich_console,
            raise_on_failure=False,
        )

    def pull_image(
        self,
        registry_image: str,
        local_name: str = None,
        registry: str = None,
        credentials: typing.Dict = None,
    ) -> str:
        """Pull an image from registry.

        Args:
            registry_image: Full registry image name
            local_name: Optional local name to tag the image
            registry: Optional registry URL for authentication
            credentials: Optional credentials dictionary for authentication

        Returns:
            str: Local image name
        """
        # Login to registry if credentials are provided
        if registry and credentials:
            self.login_to_registry(registry, credentials)

        self.rich_console.print(
            f"\n[bold blue]📥 Starting docker pull from registry...[/bold blue]"
        )
        print(f"📍 Registry: {registry or 'Default'}")
        print(f"🏷️  Image: {registry_image}")

        # Force fresh pull on SLURM compute nodes to avoid corrupted cached layers
        # This prevents "permission denied" errors from corrupted image layers
        deployment_type = os.environ.get("MAD_DEPLOYMENT_TYPE", "local")
        in_slurm_job = os.environ.get("MAD_IN_SLURM_JOB", "0") == "1"

        if deployment_type == "slurm" and in_slurm_job:
            print(
                f"🔄 Using fresh pull policy for SLURM compute node (prevents cached layer corruption)"
            )
            # Remove any existing cached image to force fresh pull
            try:
                self.console.sh(f"docker rmi -f {registry_image} 2>/dev/null || true")
                print(f"✓ Removed cached image layers")
            except Exception:
                pass  # It's okay if image doesn't exist

        try:
            self.console.sh(f"docker pull {registry_image}")

            if local_name:
                self.console.sh(f"docker tag {registry_image} {local_name}")
                print(f"🏷️  Tagged as: {local_name}")
                self.rich_console.print(
                    f"[bold green]✅ Successfully pulled and tagged image[/bold green]"
                )
                self.rich_console.print(f"[dim]{'='*80}[/dim]")
                return local_name

            self.rich_console.print(
                f"[bold green]✅ Successfully pulled image:[/bold green] [cyan]{registry_image}[/cyan]"
            )
            self.rich_console.print(f"[dim]{'='*80}[/dim]")
            return registry_image

        except Exception as e:
            self.rich_console.print(
                f"[red]❌ Failed to pull image {registry_image}: {e}[/red]"
            )
            raise

    def get_gpu_arg(self, requested_gpus: str) -> str:
        """Get the GPU arguments for docker run.

        Args:
            requested_gpus: The requested GPUs.

        Returns:
            str: The GPU arguments.
        """
        gpu_arg = ""
        gpu_vendor = self.context.ctx["docker_env_vars"]["MAD_GPU_VENDOR"]
        n_system_gpus = self.context.ctx["docker_env_vars"]["MAD_SYSTEM_NGPUS"]
        gpu_strings = self.context.ctx["docker_gpus"].split(",")

        # Parse GPU string, example: '{0-4}' -> [0,1,2,3,4]
        docker_gpus = []
        for gpu_string in gpu_strings:
            if "-" in gpu_string:
                gpu_range = gpu_string.split("-")
                docker_gpus += [
                    item for item in range(int(gpu_range[0]), int(gpu_range[1]) + 1)
                ]
            else:
                docker_gpus.append(int(gpu_string))
        docker_gpus.sort()

        # Check GPU range is valid for system
        if requested_gpus == "-1":
            print("NGPUS requested is ALL (" + ",".join(map(str, docker_gpus)) + ").")
            requested_gpus = len(docker_gpus)

        print(
            "NGPUS requested is "
            + str(requested_gpus)
            + " out of "
            + str(n_system_gpus)
        )

        if int(requested_gpus) > int(n_system_gpus) or int(requested_gpus) > len(
            docker_gpus
        ):
            raise RuntimeError(
                f"Too many gpus requested({requested_gpus}). System has {n_system_gpus} gpus. Context has {len(docker_gpus)} gpus."
            )

        # Expose number of requested gpus
        self.context.ctx["docker_env_vars"]["MAD_RUNTIME_NGPUS"] = str(requested_gpus)

        # Create docker arg to assign requested GPUs
        if gpu_vendor.find("AMD") != -1:
            gpu_arg = "--device=/dev/kfd "
            gpu_renderDs = self.context.ctx["gpu_renderDs"]
            if gpu_renderDs is not None:
                for idx in range(0, int(requested_gpus)):
                    gpu_arg += (
                        f"--device=/dev/dri/renderD{gpu_renderDs[docker_gpus[idx]]} "
                    )

        elif gpu_vendor.find("NVIDIA") != -1:
            gpu_str = ""
            for idx in range(0, int(requested_gpus)):
                gpu_str += str(docker_gpus[idx]) + ","
            gpu_arg += f"--gpus '\"device={gpu_str}\"' "
        else:
            raise RuntimeError("Unable to determine gpu vendor.")

        print(f"GPU arguments: {gpu_arg}")
        return gpu_arg

    def get_cpu_arg(self) -> str:
        """Get the CPU arguments for docker run."""
        if "docker_cpus" not in self.context.ctx:
            return ""
        cpus = self.context.ctx["docker_cpus"].replace(" ", "")
        return f"--cpuset-cpus {cpus} "

    _ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def get_env_arg(self, run_env: typing.Dict) -> str:
        """Get the environment arguments for docker run."""
        env_args = ""

        # Add custom environment variables
        if run_env:
            for env_arg in run_env:
                if not self._ENV_KEY_RE.match(env_arg):
                    raise ValueError(f"Invalid environment variable name: {env_arg!r}")
                env_args += f"--env {env_arg}={shlex.quote(str(run_env[env_arg]))} "

        # Add context environment variables
        if "docker_env_vars" in self.context.ctx:
            for env_arg in self.context.ctx["docker_env_vars"].keys():
                if not self._ENV_KEY_RE.match(env_arg):
                    raise ValueError(f"Invalid environment variable name: {env_arg!r}")
                value = self.context.ctx["docker_env_vars"][env_arg]
                env_args += f"--env {env_arg}={shlex.quote(str(value))} "

        return env_args

    def get_mount_arg(self, mount_datapaths: typing.List) -> str:
        """Get the mount arguments for docker run."""
        mount_args = ""

        # Mount data paths
        if mount_datapaths:
            for mount_datapath in mount_datapaths:
                if mount_datapath:
                    mount_args += (
                        f"-v {mount_datapath['path']}:{mount_datapath['home']}"
                    )
                    if (
                        "readwrite" in mount_datapath
                        and mount_datapath["readwrite"] == "true"
                    ):
                        mount_args += " "
                    else:
                        mount_args += ":ro "

        # Mount context paths
        if "docker_mounts" in self.context.ctx:
            for mount_arg in self.context.ctx["docker_mounts"].keys():
                mount_args += (
                    f"-v {self.context.ctx['docker_mounts'][mount_arg]}:{mount_arg} "
                )

        return mount_args

    def apply_tools(
        self,
        pre_encapsulate_post_scripts: typing.Dict,
        run_env: typing.Dict,
        tools_json_file: str,
    ) -> None:
        """Apply tools configuration to the runtime environment."""
        if "tools" not in self.context.ctx:
            return

        # Read tool settings from tools.json
        with open(tools_json_file) as f:
            tool_file = json.load(f)

        # Track commands that have been added to avoid duplicates
        # Some tools (like trace tools) share the same wrapper script
        added_cmds = set()

        # Iterate over tools in context, apply tool settings
        for ctx_tool_config in self.context.ctx["tools"]:
            tool_name = ctx_tool_config["name"]
            tool_config = tool_file["tools"][tool_name]

            if "cmd" in ctx_tool_config:
                tool_config.update({"cmd": ctx_tool_config["cmd"]})

            if "env_vars" in ctx_tool_config:
                for env_var in ctx_tool_config["env_vars"]:
                    tool_config["env_vars"].update(
                        {env_var: ctx_tool_config["env_vars"][env_var]}
                    )

            print(f"Selected Tool, {tool_name}. Configuration : {str(tool_config)}.")

            # Setup tool before other existing scripts
            if "pre_scripts" in tool_config:
                pre_encapsulate_post_scripts["pre_scripts"] = (
                    tool_config["pre_scripts"]
                    + pre_encapsulate_post_scripts["pre_scripts"]
                )
            # Cleanup tool after other existing scripts
            if "post_scripts" in tool_config:
                pre_encapsulate_post_scripts["post_scripts"] += tool_config[
                    "post_scripts"
                ]
            # Update environment variables (always apply, even if cmd is duplicate)
            if "env_vars" in tool_config:
                run_env.update(tool_config["env_vars"])

            # Only add cmd if it hasn't been added yet
            # This prevents duplicate wrappers like get_library_trace.py
            if "cmd" in tool_config:
                cmd = tool_config["cmd"]
                if cmd not in added_cmds:
                    # Prepend encapsulate cmd
                    pre_encapsulate_post_scripts["encapsulate_script"] = (
                        cmd + " " + pre_encapsulate_post_scripts["encapsulate_script"]
                    )
                    added_cmds.add(cmd)
                else:
                    print(
                        f"  Note: Command '{cmd}' already added by another tool, skipping duplicate."
                    )

    def run_pre_post_script(
        self, model_docker: Docker, model_dir: str, pre_post: typing.List
    ) -> None:
        """Run pre/post scripts in the container."""
        for script in pre_post:
            script_path = script["path"].strip()
            model_docker.sh(
                f"cp -vLR --preserve=all {script_path} {model_dir}", timeout=600
            )
            script_name = os.path.basename(script_path)
            script_args = ""
            if "args" in script:
                script_args = script["args"].strip()
            model_docker.sh(
                f"cd {model_dir} && bash {script_name} {script_args}", timeout=600
            )

    def gather_system_env_details(
        self, pre_encapsulate_post_scripts: typing.Dict, model_name: str
    ) -> None:
        """Gather system environment details.

        Args:
            pre_encapsulate_post_scripts: The pre, encapsulate and post scripts.
            model_name: The model name.

        Returns:
            None

        Raises:
            Exception: An error occurred while gathering system environment details.

        Note:
            This function is used to gather system environment details.
        """
        # initialize pre_env_details
        pre_env_details = {}
        pre_env_details["path"] = "scripts/common/pre_scripts/run_rocenv_tool.sh"
        pre_env_details["args"] = model_name.replace("/", "_") + "_env"
        pre_encapsulate_post_scripts["pre_scripts"].append(pre_env_details)
        print(f"pre encap post scripts: {pre_encapsulate_post_scripts}")

    def _resolve_docker_image(self, docker_image: str, model_name: str) -> str:
        """Resolve Docker image: use requested image if present, else primus_pretrain fallback with clear error."""
        if _docker_image_exists_locally(docker_image):
            return docker_image
        if model_name.startswith("primus_pretrain/"):
            fallback = "ci-primus_pretrain_primus.ubuntu.amd"
            if _docker_image_exists_locally(fallback):
                print(
                    f"ℹ️  Using shared Primus image (one build for all primus_pretrain configs): {fallback}"
                )
                return fallback
            raise RuntimeError(
                f"Docker image '{docker_image}' not found and fallback '{fallback}' not found. "
                "Build the Primus image first: madengine build --tags primus_pretrain --additional-context-file <config>.json"
            ) from None
        raise RuntimeError(
            f"Docker image '{docker_image}' not found. "
            "Build it first: madengine build --tags <model_tag> --additional-context-file <config>.json"
        ) from None

    def run_container(
        self,
        model_info: typing.Dict,
        docker_image: str,
        build_info: typing.Dict = None,
        keep_alive: bool = False,
        keep_model_dir: bool = False,
        timeout: int = 7200,
        tools_json_file: str = "scripts/common/tools.json",
        phase_suffix: str = "",
        generate_sys_env_details: bool = True,
    ) -> typing.Dict:
        """Run a model in a Docker container.

        Args:
            model_info: Model information dictionary
            docker_image: Docker image name to run
            build_info: Optional build information from manifest
            keep_alive: Whether to keep container alive after execution
            keep_model_dir: Whether to keep model directory after execution
            timeout: Execution timeout in seconds
            tools_json_file: Path to tools configuration file
            phase_suffix: Suffix for log file name (e.g., ".run" or "")
            generate_sys_env_details: Whether to collect system environment details

        Returns:
            dict: Execution results including performance metrics
        """
        self.rich_console.print(
            f"[bold green]🏃 Running model:[/bold green] [bold cyan]{model_info['name']}[/bold cyan] [dim]in container[/dim] [yellow]{docker_image}[/yellow]"
        )

        # Resolve image: if model-specific image is missing, try shared primus_pretrain image (one build for all configs)
        docker_image = self._resolve_docker_image(docker_image, model_info["name"])

        timeout = resolve_run_timeout(model_info, timeout)
        log_file_path = make_run_log_file_path(model_info, docker_image, phase_suffix)
        print(f"Run log will be written to: {log_file_path}")

        # get machine name
        machine_name = self.console.sh("hostname")
        print(f"MACHINE NAME is {machine_name}")

        # Initialize results
        run_results = {
            "model": model_info["name"],
            "docker_image": docker_image,
            "status": "FAILURE",
            "performance": "",
            "metric": "",
            "test_duration": 0,
            "machine_name": machine_name,
            "log_file": log_file_path,
        }

        # If build info provided, merge it
        if build_info:
            run_results.update(build_info)
        # Preserve actual image used (resolved, possibly fallback) for perf reporting
        run_results["docker_image"] = docker_image

        # Prepare docker run options
        gpu_vendor = self.context.ctx["gpu_vendor"]
        docker_options = ""

        if gpu_vendor.find("AMD") != -1:
            docker_options = (
                "--network host -u root --group-add video "
                "--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse "
                "--security-opt seccomp=unconfined --security-opt apparmor=unconfined --ipc=host "
            )
        elif gpu_vendor.find("NVIDIA") != -1:
            docker_options = (
                "-u root --cap-add=SYS_PTRACE --cap-add SYS_ADMIN --cap-add SYS_NICE --device /dev/fuse "
                "--security-opt seccomp=unconfined --security-opt apparmor=unconfined "
                "--network host --ipc=host "
            )
        else:
            raise RuntimeError("Unable to determine gpu vendor.")

        # Initialize scripts
        pre_encapsulate_post_scripts = {
            "pre_scripts": [],
            "encapsulate_script": "",
            "post_scripts": [],
        }

        if "pre_scripts" in self.context.ctx:
            pre_encapsulate_post_scripts["pre_scripts"] = self.context.ctx[
                "pre_scripts"
            ]
        if "post_scripts" in self.context.ctx:
            pre_encapsulate_post_scripts["post_scripts"] = self.context.ctx[
                "post_scripts"
            ]
        if "encapsulate_script" in self.context.ctx:
            pre_encapsulate_post_scripts["encapsulate_script"] = self.context.ctx[
                "encapsulate_script"
            ]

        # Add environment variables
        docker_options += f"--env MAD_MODEL_NAME='{model_info['name']}' "
        if model_info.get("multiple_results"):
            docker_options += (
                f"--env MAD_OUTPUT_CSV='{model_info['multiple_results']}' "
            )
        docker_options += f"--env JENKINS_BUILD_NUMBER='{get_build_number()}' "

        # Gather data and environment
        run_env = {}
        mount_datapaths = None

        # Merge docker_env_vars from additional_context into context
        # Also check shell environment for SLURM-passed variables
        if "docker_env_vars" not in self.context.ctx:
            self.context.ctx["docker_env_vars"] = {}

        # For SLURM jobs, check shell environment and populate additional_context with GPU info
        # This ensures GPU resolution works correctly
        if os.environ.get("MAD_DEPLOYMENT_TYPE") == "slurm":
            if "NPROC_PER_NODE" in os.environ or "GPUS_PER_NODE" in os.environ:
                gpus_per_node_str = os.environ.get("NPROC_PER_NODE") or os.environ.get(
                    "GPUS_PER_NODE"
                )
                if gpus_per_node_str:
                    try:
                        gpus = int(gpus_per_node_str)
                        # Add gpus_per_node to additional_context for GPU resolution
                        # resolve_runtime_gpus looks for this field name
                        if not self.additional_context:
                            self.additional_context = {}
                        if "gpus_per_node" not in self.additional_context:
                            self.additional_context["gpus_per_node"] = gpus
                            print(
                                f"ℹ️  SLURM GPU override: {gpus} GPUs per node (from shell environment)"
                            )
                    except ValueError:
                        pass

        # List of environment variables to pass from shell to Docker (for SLURM jobs)
        slurm_env_vars = [
            "MASTER_ADDR",
            "MASTER_PORT",
            "WORLD_SIZE",
            "RANK",
            "NODE_RANK",
            "NNODES",
            "NPROC_PER_NODE",
            "MAD_MULTI_NODE_RUNNER",
            "MAD_COLLECT_METRICS",
            "NCCL_SOCKET_IFNAME",
            "GLOO_SOCKET_IFNAME",
            "NCCL_DEBUG",
            "NCCL_IB_DISABLE",
            "NCCL_NET_GDR_LEVEL",
            # Primus launcher (config path and optional CLI extra args)
            "PRIMUS_CONFIG_PATH",
            "PRIMUS_CLI_EXTRA",
            # Rendezvous timeout so all nodes can join after pull
            "TORCH_ELASTIC_RDZV_TIMEOUT",
            # GPU visibility variables for Ray-based launchers (vLLM, SGLang)
            # CRITICAL: These must be passed to Docker for proper GPU device mapping
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ]

        # Check shell environment and add to docker_env_vars
        merged_from_env = 0
        for var_name in slurm_env_vars:
            if var_name in os.environ:
                self.context.ctx["docker_env_vars"][var_name] = os.environ[var_name]
                merged_from_env += 1

        # CRITICAL FIX for rocm/vllm image: Override RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
        # The rocm/vllm Docker image has RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 baked in,
        # which tells Ray to IGNORE HIP_VISIBLE_DEVICES. We must explicitly override it.
        # This is only needed if HIP_VISIBLE_DEVICES is set (indicating AMD GPU usage with Ray)
        if "HIP_VISIBLE_DEVICES" in self.context.ctx["docker_env_vars"]:
            # Set to empty string to disable Ray's behavior of ignoring HIP_VISIBLE_DEVICES
            self.context.ctx["docker_env_vars"][
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"
            ] = ""
            print(
                "ℹ️  Overriding RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES to enable HIP_VISIBLE_DEVICES"
            )

        if merged_from_env > 0:
            print(
                f"ℹ️  Inherited {merged_from_env} environment variables from shell for Docker"
            )

        # Also merge from additional_context if present
        if self.additional_context and "docker_env_vars" in self.additional_context:
            merged_count = 0
            for key, value in self.additional_context["docker_env_vars"].items():
                self.context.ctx["docker_env_vars"][key] = value
                merged_count += 1
            if merged_count > 0:
                print(
                    f"ℹ️  Merged {merged_count} environment variables from additional_context"
                )

        if (
            self.context
            and str(self.context.ctx.get("gpu_vendor", "")).upper().find("AMD") != -1
        ):
            from madengine.utils.rocm_path_resolver import finalize_container_rocm_path

            # Determine whether the user explicitly supplied ROCM_PATH for the container.
            # If they did (via docker_env_vars.ROCM_PATH in additional_context), the
            # re-merge above already restored it — keep it so finalize uses it directly.
            # If they did not, clear any ROCM_PATH left from a previous model run so
            # finalize always re-resolves for the current docker_image (OCI config →
            # in-image probe → /opt/rocm default).
            user_supplied_rocm_path = str(
                (self.additional_context or {})
                .get("docker_env_vars", {})
                .get("ROCM_PATH", "")
            ).strip()
            if not user_supplied_rocm_path:
                self.context.ctx["docker_env_vars"].pop("ROCM_PATH", None)

            finalize_container_rocm_path(
                self.context.ctx["docker_env_vars"],
                docker_image,
            )

        if "data" in model_info and model_info["data"] != "" and self.data:
            mount_datapaths = self.data.get_mountpaths(model_info["data"])
            model_dataenv = self.data.get_env(model_info["data"])
            if model_dataenv is not None:
                run_env.update(model_dataenv)
            run_env["MAD_DATANAME"] = model_info["data"]

        # Add credentials to environment
        if "cred" in model_info and model_info["cred"] != "" and self.credentials:
            if model_info["cred"] not in self.credentials:
                raise RuntimeError(f"Credentials({model_info['cred']}) not found")
            for key_cred, value_cred in self.credentials[model_info["cred"]].items():
                run_env[model_info["cred"] + "_" + key_cred.upper()] = value_cred

        # Apply tools if configured
        if os.path.exists(tools_json_file):
            self.apply_tools(pre_encapsulate_post_scripts, run_env, tools_json_file)

        # Add system environment collection script to pre_scripts
        # Context can explicitly disable via gen_sys_env_details: false in additional_context
        ctx_sys_env = self.context.ctx.get("gen_sys_env_details")
        should_collect_sys_env = (
            ctx_sys_env if ctx_sys_env is not None else generate_sys_env_details
        )
        if should_collect_sys_env:
            self.gather_system_env_details(
                pre_encapsulate_post_scripts, model_info["name"]
            )

        # Build docker options
        # Use hierarchical GPU resolution: runtime > deployment > model > default
        resolved_gpu_count = resolve_runtime_gpus(model_info, self.additional_context)
        docker_options += self.get_gpu_arg(str(resolved_gpu_count))
        docker_options += self.get_cpu_arg()

        # Filter out MIOPEN_USER_DB_PATH from run_env if it exists
        # It should be passed via docker_env_vars in context instead
        if "MIOPEN_USER_DB_PATH" in run_env:
            del run_env["MIOPEN_USER_DB_PATH"]
            print(
                "ℹ️  Removed MIOPEN_USER_DB_PATH from run_env (will use context.docker_env_vars)"
            )

        # Add MIOPEN_USER_DB_PATH from shell environment to context.docker_env_vars
        # This is set by SLURM script with ${LOCAL_RANK} variable for per-process paths
        if (
            "MIOPEN_USER_DB_PATH" in os.environ
            and "MIOPEN_USER_DB_PATH" not in self.context.ctx["docker_env_vars"]
        ):
            self.context.ctx["docker_env_vars"]["MIOPEN_USER_DB_PATH"] = os.environ[
                "MIOPEN_USER_DB_PATH"
            ]
            print(
                f"ℹ️  Added MIOPEN_USER_DB_PATH to docker_env_vars: {os.environ['MIOPEN_USER_DB_PATH']}"
            )

        docker_options += self.get_env_arg(run_env)
        docker_options += self.get_mount_arg(mount_datapaths)
        docker_options += f" {model_info.get('additional_docker_run_options', '')}"

        # Generate container name
        base_container_name = "container_" + re.sub(
            ".*:", "", docker_image.replace("/", "_").replace(":", "_")
        )

        # For multi-node SLURM jobs, add node rank to avoid name conflicts
        node_rank = os.environ.get("SLURM_PROCID") or os.environ.get("RANK")
        if node_rank is not None:
            container_name = f"{base_container_name}_node{node_rank}"
        else:
            container_name = base_container_name

        print(f"Docker options: {docker_options}")

        self.rich_console.print(
            f"\n[bold blue]🏃 Starting Docker container execution...[/bold blue]"
        )
        print(f"🏷️  Image: {docker_image}")
        print(f"📦 Container: {container_name}")
        print(f"📝 Log file: {log_file_path}")
        print(f"🎮 GPU Vendor: {gpu_vendor}")
        self.rich_console.print(f"[dim]{'='*80}[/dim]")

        # Run the container with logging
        try:
            with open(log_file_path, mode="w", buffering=1) as outlog:
                with (
                    redirect_stdout(PythonicTee(outlog, self.live_output)),
                    redirect_stderr(PythonicTee(outlog, self.live_output)),
                ):
                    # set timeout (print inside log redirection so it appears in log file)
                    print(f"⏰ Setting timeout to {str(timeout)} seconds.")

                    with Timeout(timeout):
                        model_docker = Docker(
                            docker_image,
                            container_name,
                            docker_options,
                            keep_alive=keep_alive,
                            console=self.console,
                        )

                        # Check user
                        whoami = model_docker.sh("whoami")
                        print(f"👤 Running as user: {whoami}")

                        # Show GPU info — let the container resolve its own tool paths via
                        # PATH rather than using host-resolved paths, which break on
                        # TheRock images where amd-smi/rocm-smi live in a Python venv
                        # (e.g. /opt/python/bin/) rather than /opt/rocm/bin/.
                        if gpu_vendor.find("AMD") != -1:
                            print(f"🎮 Checking AMD GPU status...")
                            model_docker.sh(
                                "amd-smi 2>/dev/null || rocm-smi 2>/dev/null || "
                                "echo 'GPU SMI tool not available in container PATH'"
                            )
                        elif gpu_vendor.find("NVIDIA") != -1:
                            print(f"🎮 Checking NVIDIA GPU status...")
                            model_docker.sh("/usr/bin/nvidia-smi || true")

                        # Print host vs container environment summary table
                        _print_run_env_table(
                            gpu_vendor, self.context, model_docker, self.rich_console
                        )

                        # Prepare model directory
                        model_dir = "run_directory"
                        if "url" in model_info and model_info["url"] != "":
                            model_dir = model_info["url"].rstrip("/").split("/")[-1]

                            # Validate model_dir
                            special_char = r"[^a-zA-Z0-9\-\_]"
                            if re.search(special_char, model_dir) is not None:
                                warnings.warn(
                                    "Model url contains special character. Fix url."
                                )

                        model_docker.sh(f"rm -rf {model_dir}", timeout=240)
                        model_docker.sh(
                            "git config --global --add safe.directory /myworkspace"
                        )

                        # Clone model repo if needed
                        if "url" in model_info and model_info["url"] != "":
                            if (
                                "cred" in model_info
                                and model_info["cred"] != ""
                                and self.credentials
                            ):
                                print(f"Using credentials for {model_info['cred']}")

                                if model_info["url"].startswith("ssh://"):
                                    model_docker.sh(
                                        f"git -c core.sshCommand='ssh -l {self.credentials[model_info['cred']]['username']} "
                                        f"-i {self.credentials[model_info['cred']]['ssh_key_file']} -o IdentitiesOnly=yes "
                                        f"-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no' "
                                        f"clone {model_info['url']}",
                                        timeout=240,
                                    )
                                else:  # http or https
                                    model_docker.sh(
                                        f"git clone -c credential.helper='!f() {{ echo username={self.credentials[model_info['cred']]['username']}; "
                                        f"echo password={self.credentials[model_info['cred']]['password']}; }};f' "
                                        f"{model_info['url']}",
                                        timeout=240,
                                        secret=f"git clone {model_info['url']}",
                                    )
                            else:
                                model_docker.sh(
                                    f"git clone {model_info['url']}", timeout=240
                                )

                            model_docker.sh(
                                f"git config --global --add safe.directory /myworkspace/{model_dir}"
                            )
                            run_results["git_commit"] = model_docker.sh(
                                f"cd {model_dir} && git rev-parse HEAD"
                            )
                            print(f"MODEL GIT COMMIT is {run_results['git_commit']}")
                            model_docker.sh(
                                f"cd {model_dir}; git submodule update --init --recursive"
                            )
                        else:
                            model_docker.sh(f"mkdir -p {model_dir}")

                        # Run pre-scripts
                        if pre_encapsulate_post_scripts["pre_scripts"]:
                            self.run_pre_post_script(
                                model_docker,
                                model_dir,
                                pre_encapsulate_post_scripts["pre_scripts"],
                            )

                        # Prepare script execution
                        scripts_arg = model_info["scripts"]
                        if scripts_arg.endswith(".sh"):
                            # Shell script specified directly
                            dir_path = os.path.dirname(scripts_arg)
                            script_name = "bash " + os.path.basename(scripts_arg)
                        elif scripts_arg.endswith(".py"):
                            # Python script specified directly
                            dir_path = os.path.dirname(scripts_arg)
                            script_name = "python3 " + os.path.basename(scripts_arg)
                        else:
                            # Directory specified (legacy behavior)
                            dir_path = model_info["scripts"]
                            script_name = "bash run.sh"

                        # Add script prepend command
                        script_name = (
                            pre_encapsulate_post_scripts["encapsulate_script"]
                            + " "
                            + script_name
                        )

                        # print repo hash
                        commit = model_docker.sh(
                            f"cd {dir_path}; git rev-parse HEAD || true"
                        )
                        print("======================================================")
                        print("MODEL REPO COMMIT: ", commit)
                        print("======================================================")

                        # Copy scripts to model directory
                        model_docker.sh(
                            f"cp -vLR --preserve=all {dir_path}/. {model_dir}/"
                        )

                        # Prepare data if needed
                        if (
                            "data" in model_info
                            and model_info["data"] != ""
                            and self.data
                        ):
                            self.data.prepare_data(model_info["data"], model_docker)

                            # Capture data provider information from selected_data_provider
                            if (
                                hasattr(self.data, "selected_data_provider")
                                and self.data.selected_data_provider
                            ):
                                if "dataname" in self.data.selected_data_provider:
                                    run_results["dataname"] = (
                                        self.data.selected_data_provider["dataname"]
                                    )
                                if (
                                    "data_provider_type"
                                    in self.data.selected_data_provider
                                ):
                                    run_results["data_provider_type"] = (
                                        self.data.selected_data_provider[
                                            "data_provider_type"
                                        ]
                                    )
                                if "duration" in self.data.selected_data_provider:
                                    run_results["data_download_duration"] = (
                                        self.data.selected_data_provider["duration"]
                                    )
                                if "size" in self.data.selected_data_provider:
                                    run_results["data_size"] = (
                                        self.data.selected_data_provider["size"]
                                    )
                                print(
                                    f"Data Provider Details: {run_results.get('dataname', '')}, "
                                    f"{run_results.get('data_provider_type', '')}, "
                                    f"{run_results.get('data_size', '')}, "
                                    f"{run_results.get('data_download_duration', '')}s"
                                )

                        # Set permissions
                        model_docker.sh(f"chmod -R a+rw {model_dir}")

                        # Run the model
                        test_start_time = time.time()
                        self.rich_console.print(
                            "[bold blue]Running model...[/bold blue]"
                        )

                        model_args = self.context.ctx.get(
                            "model_args", model_info["args"]
                        )
                        # Use the container timeout (default 7200s) for script execution
                        # to prevent indefinite hangs
                        model_output = model_docker.sh(
                            f"cd {model_dir} && {script_name} {model_args}",
                            timeout=timeout,
                        )
                        # When live_output is True, Console.sh() already streamed the output; avoid duplicate print.
                        if not self.live_output:
                            print(model_output)

                        run_results["test_duration"] = time.time() - test_start_time
                        print(f"Test Duration: {run_results['test_duration']} seconds")
                        # Parser-friendly line for SLURM log collection (test_duration: Xs)
                        print(f"test_duration: {run_results['test_duration']:.2f}s")

                        # Run post-scripts
                        if pre_encapsulate_post_scripts["post_scripts"]:
                            self.run_pre_post_script(
                                model_docker,
                                model_dir,
                                pre_encapsulate_post_scripts["post_scripts"],
                            )

                        # When model writes performance to a file in run_directory, copy to cwd
                        # so the host can read it (e.g. bind-mounted workspace) before extraction.
                        multiple_results_file = (
                            model_info.get("multiple_results") or ""
                        ).strip()
                        if multiple_results_file:
                            try:
                                model_docker.sh(
                                    _cp_model_dir_file_to_cwd_cmd(
                                        model_dir, multiple_results_file
                                    )
                                )
                            except Exception:
                                pass

                        # Extract performance metrics from logs
                        # Look for performance data in the log output similar to original run_models.py
                        try:
                            # Check if multiple results file is specified in model_info
                            multiple_results = model_info.get("multiple_results", None)
                            if multiple_results:
                                multiple_results = multiple_results.strip()

                            if multiple_results:
                                resolved_path = _resolve_multiple_results_path(
                                    multiple_results, model_dir
                                )
                                if not resolved_path:
                                    self.rich_console.print(
                                        f"[yellow]Warning: Could not find multiple results file "
                                        f"(tried cwd and {model_dir}/): {multiple_results}[/yellow]"
                                    )
                                    run_results["performance"] = None
                                else:
                                    run_results["performance"] = resolved_path
                                    # Validate multiple results file format using proper CSV parsing
                                    try:
                                        import csv

                                        with open(resolved_path, "r") as f:
                                            csv_reader = csv.DictReader(f)

                                            # Strip whitespace from fieldnames to handle headers like "model, performance, metric"
                                            csv_reader.fieldnames = [
                                                f.strip() for f in csv_reader.fieldnames
                                            ]

                                            # Check if 'performance' column exists
                                            if (
                                                "performance"
                                                not in csv_reader.fieldnames
                                            ):
                                                print(
                                                    "Error: 'performance' column not found in multiple results file."
                                                )
                                                run_results["performance"] = None
                                            else:
                                                # Check if at least one row has a non-empty performance value
                                                has_valid_perf = False
                                                for row in csv_reader:
                                                    if row.get(
                                                        "performance", ""
                                                    ).strip():
                                                        has_valid_perf = True
                                                        break

                                                if not has_valid_perf:
                                                    run_results["performance"] = None
                                                    print(
                                                        "Error: Performance metric is empty in all rows of multiple results file."
                                                    )
                                    except Exception as e:
                                        self.rich_console.print(
                                            f"[yellow]Warning: Could not validate multiple results file: {e}[/yellow]"
                                        )
                                        run_results["performance"] = None
                            else:
                                # Match the actual output format: "performance: 14164 samples_per_second"
                                # Simple pattern to capture number and metric unit

                                # Extract from log file
                                try:
                                    # Note: re and os are already imported at module level (lines 10, 15)

                                    # Verify log file exists and is readable
                                    if not os.path.exists(log_file_path):
                                        print(
                                            f"Warning: Log file not found: {log_file_path}"
                                        )
                                        run_results["performance"] = None
                                        run_results["metric"] = None
                                    else:
                                        # Read the log file once (avoids rocprofv3 crash from shell pipelines)
                                        # This approach matches the Kubernetes implementation pattern
                                        with open(
                                            log_file_path,
                                            "r",
                                            encoding="utf-8",
                                            errors="ignore",
                                        ) as f:
                                            log_content = f.read()

                                        # Try multiple patterns to match different log formats

                                        # Pattern 1: "performance: <value>[<unit>][,] <metric>"
                                        # See PERFORMANCE_LOG_PATTERN in deployment.base for accepted formats.
                                        match = re.search(
                                            PERFORMANCE_LOG_PATTERN, log_content
                                        )

                                        if match:
                                            run_results["performance"] = match.group(
                                                1
                                            ).strip()
                                            run_results["metric"] = match.group(
                                                2
                                            ).strip()
                                            print(
                                                f"✓ Extracted performance: {run_results['performance']} {run_results['metric']}"
                                            )
                                        else:
                                            # Pattern 2: HuggingFace format - "'train_samples_per_second': 4.23" or "train_samples_per_second = 4.23"
                                            # This matches the actual output from HuggingFace Trainer
                                            hf_pattern = r'train_samples_per_second[\'"\s:=]+([0-9][0-9.eE+-]*)'
                                            hf_match = re.search(
                                                hf_pattern, log_content
                                            )

                                            if hf_match:
                                                run_results["performance"] = (
                                                    hf_match.group(1).strip()
                                                )
                                                run_results["metric"] = (
                                                    "samples_per_second"
                                                )
                                                print(
                                                    f"✓ Extracted performance (HuggingFace format): {run_results['performance']} {run_results['metric']}"
                                                )
                                            else:
                                                # No performance metrics found
                                                print(
                                                    "Warning: Performance metric not found in expected format 'performance: NUMBER METRIC' or 'train_samples_per_second'"
                                                )
                                                run_results["performance"] = None
                                                run_results["metric"] = None

                                except Exception as e:
                                    print(
                                        f"Warning: Error extracting performance metrics: {e}"
                                    )
                                    run_results["performance"] = None
                                    run_results["metric"] = None
                                    # Performance extraction is optional - don't fail the entire run
                        except Exception as e:
                            print(
                                f"Warning: Could not extract performance metrics: {e}"
                            )

                        # Set status based on performance and error patterns
                        # First check for obvious failure patterns in the logs
                        try:
                            scan_logs, error_patterns, extra_benign = (
                                resolve_log_error_scan_config(
                                    model_info, self.additional_context
                                )
                            )

                            has_errors = False
                            if (
                                scan_logs
                                and log_file_path
                                and os.path.exists(log_file_path)
                            ):
                                try:
                                    # Benign: literal substrings (incl. user extra_benign) vs regex (ROCProf lines).
                                    benign_substrings = [
                                        "Failed to establish connection to the metrics exporter agent",
                                        "RpcError: Running out of retries to initialize the metrics agent",
                                        "Metrics will not be exported",
                                        "FutureWarning",
                                        "Opened result file:",
                                        "SQLite3 generation ::",
                                        "rocpd_op:",
                                        "rpd_tracer:",
                                    ]
                                    benign_substrings.extend(extra_benign)
                                    benign_regexes = [
                                        # ROCProf/glog: E/W prefixes are log levels, not app errors
                                        r"^E[0-9]{8}.*generateRocpd\.cpp",
                                        r"^W[0-9]{8}.*simple_timer\.cpp",
                                        r"^W[0-9]{8}.*generateRocpd\.cpp",
                                        r"^E[0-9]{8}.*tool\.cpp",
                                        r"\[rocprofv3\]",
                                    ]

                                    # Scan in Python (no shell; literals vs regex benign rules are explicit).
                                    with open(
                                        log_file_path,
                                        "r",
                                        encoding="utf-8",
                                        errors="ignore",
                                    ) as _lf:
                                        log_scan_text = _lf.read()

                                    for pattern in error_patterns:
                                        if log_text_has_error_pattern(
                                            log_scan_text,
                                            pattern,
                                            benign_substrings,
                                            benign_regexes,
                                        ):
                                            has_errors = True
                                            print(
                                                f"Found error pattern '{pattern}' in logs"
                                            )
                                            break
                                except Exception:
                                    pass  # Error checking is optional
                            elif not scan_logs:
                                self.rich_console.print(
                                    "[dim]ℹ️  Log error pattern scan disabled "
                                    "(log_error_pattern_scan).[/dim]"
                                )

                            # Status logic: Must have performance AND no errors to be considered success
                            # Exception: Worker nodes in multi-node training (MAD_COLLECT_METRICS=false)
                            # are not expected to report global performance metrics
                            performance_value = run_results.get("performance")
                            has_performance = (
                                performance_value
                                and performance_value.strip()
                                and performance_value.strip() != "N/A"
                            )

                            # Check if this is a worker node (not collecting metrics)
                            is_worker_node = (
                                os.environ.get("MAD_COLLECT_METRICS", "true").lower()
                                == "false"
                            )

                            if has_errors:
                                run_results["status"] = "FAILURE"
                                self.rich_console.print(
                                    f"[red]Status: FAILURE (error patterns detected in logs)[/red]"
                                )
                            elif has_performance:
                                run_results["status"] = "SUCCESS"
                                self.rich_console.print(
                                    f"[green]Status: SUCCESS (performance metrics found, no errors)[/green]"
                                )
                            elif is_worker_node:
                                # Worker nodes don't report global performance metrics - this is expected
                                run_results["status"] = "SUCCESS"
                                self.rich_console.print(
                                    f"[green]Status: SUCCESS (worker node, no errors detected)[/green]"
                                )
                            else:
                                run_results["status"] = "FAILURE"
                                self.rich_console.print(
                                    f"[red]Status: FAILURE (no performance metrics)[/red]"
                                )

                        except Exception as e:
                            self.rich_console.print(
                                f"[yellow]Warning: Error in status determination: {e}[/yellow]"
                            )
                            # Fallback to simple performance check
                            # Worker nodes don't need performance metrics
                            is_worker_node = (
                                os.environ.get("MAD_COLLECT_METRICS", "true").lower()
                                == "false"
                            )
                            run_results["status"] = (
                                "SUCCESS"
                                if run_results.get("performance") or is_worker_node
                                else "FAILURE"
                            )

                        print(
                            f"{model_info['name']} performance is {run_results.get('performance', 'N/A')} {run_results.get('metric', '')}"
                        )

                        # =============================================================================
                        # Multi-Node Performance Collection (Master Node Only)
                        # =============================================================================
                        # For distributed training, only master node should collect metrics
                        # Check skip_perf_collection flag from additional_context
                        skip_perf = self.additional_context.get(
                            "skip_perf_collection", False
                        )

                        if skip_perf:
                            self.rich_console.print(
                                "[cyan]ℹ️  Worker node: Skipping performance metric collection "
                                "(master node will collect results)[/cyan]"
                            )
                        else:
                            # Generate performance results and update perf.csv
                            self.ensure_perf_csv_exists()
                            try:
                                # Create run details dictionary for CSV generation
                                run_details_dict = self.create_run_details_dict(
                                    model_info, build_info, run_results
                                )

                                # Handle multiple results if specified
                                multiple_results = model_info.get(
                                    "multiple_results", None
                                )
                                resolved_multiple_results = (
                                    _resolve_multiple_results_path(
                                        multiple_results, model_dir
                                    )
                                    if multiple_results
                                    else None
                                )
                                if (
                                    resolved_multiple_results
                                    and run_results.get("status") == "SUCCESS"
                                ):
                                    # Generate common info JSON for multiple results
                                    common_info = run_details_dict.copy()
                                    # Remove model-specific fields for common info
                                    for key in [
                                        "model",
                                        "performance",
                                        "metric",
                                        "status",
                                    ]:
                                        common_info.pop(key, None)

                                    with open("common_info.json", "w") as f:
                                        json.dump(common_info, f)

                                    # Update perf.csv with multiple results
                                    update_perf_csv(
                                        multiple_results=resolved_multiple_results,
                                        perf_csv=self.perf_csv_path,
                                        model_name=run_details_dict["model"],
                                        common_info="common_info.json",
                                    )
                                    print(
                                        f"Updated perf.csv with multiple results for {model_info['name']}"
                                    )

                                    # Update perf_super.json with multiple results
                                    try:
                                        scripts_path = model_info.get("scripts", "")
                                        scripts_base_dir = scripts_base_dir_from(
                                            scripts_path
                                        )

                                        # Reuse common_info.json for super files (no need for duplicate)
                                        num_entries = update_perf_super_json(
                                            multiple_results=resolved_multiple_results,
                                            perf_super_json="perf_super.json",
                                            model_name=run_details_dict["model"],
                                            common_info="common_info.json",
                                            scripts_base_dir=scripts_base_dir,
                                        )

                                        # Generate CSV and JSON files from perf_super.json
                                        update_perf_super_csv(
                                            perf_super_json="perf_super.json",
                                            perf_super_csv="perf_super.csv",
                                            num_entries=num_entries,
                                        )
                                    except Exception as e:
                                        print(
                                            f"⚠️  Warning: Could not update perf_super files: {e}"
                                        )
                                else:
                                    # Generate single result JSON
                                    with open("perf_entry.json", "w") as f:
                                        json.dump(run_details_dict, f)

                                    # Update perf.csv with single result
                                    if run_results.get("status") == "SUCCESS":
                                        update_perf_csv(
                                            single_result="perf_entry.json",
                                            perf_csv=self.perf_csv_path,
                                        )
                                    else:
                                        update_perf_csv(
                                            exception_result="perf_entry.json",
                                            perf_csv=self.perf_csv_path,
                                        )
                                    print(
                                        f"Updated perf.csv with result for {model_info['name']}"
                                    )

                                    # Update perf_super.json with single result
                                    try:
                                        scripts_path = model_info.get("scripts", "")
                                        scripts_base_dir = scripts_base_dir_from(
                                            scripts_path
                                        )

                                        # Use perf_entry.json as input (already created above)
                                        if run_results.get("status") == "SUCCESS":
                                            num_entries = update_perf_super_json(
                                                single_result="perf_entry.json",
                                                perf_super_json="perf_super.json",
                                                scripts_base_dir=scripts_base_dir,
                                            )
                                        else:
                                            num_entries = update_perf_super_json(
                                                exception_result="perf_entry.json",
                                                perf_super_json="perf_super.json",
                                                scripts_base_dir=scripts_base_dir,
                                            )

                                        # Generate CSV and JSON files from perf_super.json
                                        update_perf_super_csv(
                                            perf_super_json="perf_super.json",
                                            perf_super_csv="perf_super.csv",
                                            num_entries=num_entries,
                                        )
                                    except Exception as e:
                                        print(
                                            f"⚠️  Warning: Could not update perf_super files: {e}"
                                        )

                            except Exception as e:
                                self.rich_console.print(
                                    f"[yellow]Warning: Could not update perf.csv: {e}[/yellow]"
                                )

                        # Copy profiler/trace output files from run_directory to base directory before cleanup
                        # This ensures test files like gpu_info_power_profiler_output.csv and library_trace.csv are accessible
                        try:
                            _md = model_dir.replace("\\", "/")
                            model_docker.sh(
                                f"cp -- {_bash_quote_path(_md)}/*_profiler_output.csv "
                                f"{_bash_quote_path('.')} 2>/dev/null || true"
                            )
                            model_docker.sh(
                                f"cp -- {_bash_quote_path(_md)}/*_output.csv "
                                f"{_bash_quote_path('.')} 2>/dev/null || true"
                            )
                            model_docker.sh(
                                f"cp -- {_bash_quote_path(_md)}/*_trace.csv "
                                f"{_bash_quote_path('.')} 2>/dev/null || true"
                            )
                            model_docker.sh(
                                _cp_model_dir_file_to_cwd_cmd(
                                    model_dir, "library_trace.csv"
                                )
                            )
                        except Exception:
                            # Ignore errors if no profiler/trace output files exist
                            pass

                        # Copy multiple_results CSV to workspace root before run_directory is removed
                        # so SLURM single-node copy can find it at $WORKSPACE/{{ multiple_results }}
                        mult_res = (model_info.get("multiple_results") or "").strip()
                        if mult_res:
                            try:
                                model_docker.sh(
                                    _cp_model_dir_file_to_cwd_cmd(model_dir, mult_res)
                                )
                            except Exception:
                                pass

                        # Cleanup if not keeping alive and not keeping model directory
                        if not keep_alive and not keep_model_dir:
                            model_docker.sh(f"rm -rf {model_dir}", timeout=240)
                        else:
                            model_docker.sh(f"chmod -R a+rw {model_dir}")
                            reason = "keep_alive" if keep_alive else "keep_model_dir"
                            print(
                                f"{reason} specified; model_dir({model_dir}) is not removed"
                            )

                        # Explicitly delete model docker to stop the container
                        del model_docker

        except Exception as e:
            self.rich_console.print("[bold red]===== EXCEPTION =====[/bold red]")
            self.rich_console.print(f"[red]Exception: {e}[/red]")
            import traceback

            traceback.print_exc()
            self.rich_console.print("[bold red]=============== =====[/bold red]")
            run_results["status"] = "FAILURE"

            # Also update perf.csv for failures
            self.ensure_perf_csv_exists()
            try:
                # Create run details dictionary for failed runs
                run_details_dict = self.create_run_details_dict(
                    model_info, build_info, run_results
                )

                # Generate exception result JSON
                with open("perf_entry.json", "w") as f:
                    json.dump(run_details_dict, f)

                # Update perf.csv with exception result
                update_perf_csv(
                    exception_result="perf_entry.json",
                    perf_csv=self.perf_csv_path,
                )
                print(
                    f"Updated perf.csv with exception result for {model_info['name']}"
                )

                # Update perf_super.json with exception result
                try:
                    scripts_path = model_info.get("scripts", "")
                    scripts_base_dir = scripts_base_dir_from(scripts_path)

                    # Use perf_entry.json as input (already created above)
                    num_entries = update_perf_super_json(
                        exception_result="perf_entry.json",
                        perf_super_json="perf_super.json",
                        scripts_base_dir=scripts_base_dir,
                    )

                    # Generate CSV and JSON files from perf_super.json
                    update_perf_super_csv(
                        perf_super_json="perf_super.json",
                        perf_super_csv="perf_super.csv",
                        num_entries=num_entries,
                    )
                except Exception as e:
                    print(f"⚠️  Warning: Could not update perf_super files: {e}")

            except Exception as csv_e:
                self.rich_console.print(
                    f"[yellow]Warning: Could not update perf.csv with exception: {csv_e}[/yellow]"
                )

        return run_results

    def set_credentials(self, credentials: typing.Dict) -> None:
        """Set credentials for model execution.

        Args:
            credentials: Credentials dictionary
        """
        self.credentials = credentials

    def run_models_from_manifest(
        self,
        manifest_file: str,
        registry: str = None,
        timeout: int = 7200,
        keep_alive: bool = False,
        keep_model_dir: bool = False,
        phase_suffix: str = "",
    ) -> typing.Dict:
        """Run all models from a build manifest file.

        This is the main entry point for running pre-built containers from a manifest.

        Args:
            manifest_file: Path to build_manifest.json
            registry: Optional registry override
            timeout: Execution timeout per model in seconds
            keep_alive: Whether to keep containers alive after execution
            keep_model_dir: Whether to keep model directory after execution
            phase_suffix: Suffix for log files (e.g., ".run")

        Returns:
            dict: Execution summary with successful and failed runs
        """
        self.rich_console.print(
            f"[bold blue]📦 Loading manifest:[/bold blue] {manifest_file}"
        )

        # Load manifest
        manifest = self.load_build_manifest(manifest_file)
        built_images = manifest.get("built_images", {})
        built_models = manifest.get("built_models", {})

        # Load deployment_config from manifest for GPU resolution
        if "deployment_config" in manifest and not self.additional_context:
            self.additional_context = {
                "deployment_config": manifest["deployment_config"]
            }
        # Merge manifest context (e.g. skip_perf_collection for multi-node SLURM aggregation)
        if "context" in manifest and isinstance(manifest["context"], dict):
            self.additional_context = {
                **(self.additional_context or {}),
                **manifest["context"],
            }

        if not built_images:
            self.rich_console.print("[yellow]⚠️  No images found in manifest[/yellow]")
            return {"successful_runs": [], "failed_runs": []}

        self.rich_console.print(
            f"[green]Found {len(built_images)} image(s) to run[/green]\n"
        )

        # Login to registry if needed
        if registry or any(img.get("registry") for img in built_images.values()):
            effective_registry = registry or next(
                (
                    img.get("registry")
                    for img in built_images.values()
                    if img.get("registry")
                ),
                None,
            )
            if effective_registry:
                try:
                    self.login_to_registry(effective_registry, self.credentials)
                except Exception as e:
                    self.rich_console.print(
                        f"[yellow]Warning: Registry login failed: {e}[/yellow]"
                    )
                    self.rich_console.print(
                        "[yellow]Proceeding with local images only[/yellow]\n"
                    )

        # Track results
        successful_runs = []
        failed_runs = []

        # Run each model
        for image_name, build_info in built_images.items():
            model_info = built_models.get(image_name, {})
            if not model_info:
                self.rich_console.print(
                    f"[yellow]⚠️  No model info for {image_name}, skipping[/yellow]"
                )
                continue

            try:
                # Handle different image sources
                if build_info.get("local_image"):
                    # Local image mode (MAD_CONTAINER_IMAGE): Use the provided image directly
                    run_image = build_info.get("docker_image")
                    self.rich_console.print(
                        f"[yellow]🏠 Using local image: {run_image}[/yellow]"
                    )

                    # Verify image exists
                    try:
                        self.console.sh(
                            f"docker image inspect {run_image} > /dev/null 2>&1"
                        )
                    except (subprocess.CalledProcessError, RuntimeError):
                        self.rich_console.print(
                            f"[yellow]⚠️  Image {run_image} not found, attempting to pull...[/yellow]"
                        )
                        try:
                            self.pull_image(run_image)
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to find or pull local image {run_image}: {e}"
                            )

                elif build_info.get("registry_image"):
                    # Registry image: Pull from registry
                    try:
                        self.pull_image(build_info["registry_image"])
                        # Update docker_image to use registry image
                        run_image = build_info["registry_image"]
                    except Exception:
                        self.rich_console.print(
                            f"[yellow]Warning: Could not pull from registry, using local image[/yellow]"
                        )
                        run_image = image_name
                else:
                    # Normal built image: Use the image name directly
                    run_image = image_name

                # Run the container
                run_results = self.run_container(
                    model_info=model_info,
                    docker_image=run_image,
                    build_info=build_info,
                    keep_alive=keep_alive,
                    keep_model_dir=keep_model_dir,
                    timeout=timeout,
                    phase_suffix=phase_suffix,
                )

                # Check actual status and track accordingly
                status = run_results.get("status", "SUCCESS")
                if status == "SUCCESS":
                    successful_runs.append(
                        {
                            "model": model_info["name"],
                            "image": run_image,
                            "status": status,
                            "performance": run_results.get("performance"),
                            "duration": run_results.get("test_duration"),
                        }
                    )
                else:
                    # Status is FAILURE - track as failed
                    failed_runs.append(
                        {
                            "model": model_info["name"],
                            "image": run_image,
                            "status": status,
                            "error": "Container execution failed - check logs for details",
                        }
                    )
                    self.rich_console.print(
                        f"[red]❌ Run failed for {model_info['name']}: Status={status}[/red]"
                    )

            except Exception as e:
                self.rich_console.print(
                    f"[red]❌ Failed to run {model_info['name']}: {e}[/red]"
                )
                error_msg = str(e)
                failed_runs.append(
                    {
                        "model": model_info.get("name", image_name),
                        "image": image_name,
                        "error": error_msg,
                    }
                )
                # Record failure in performance table so status is consistent and table is complete
                try:
                    import tempfile

                    self.ensure_perf_csv_exists()
                    perf_entry = self._create_setup_failure_perf_entry(
                        model_info=model_info,
                        build_info=build_info,
                        image_name=image_name,
                        error_message=error_msg,
                    )
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False
                    ) as f:
                        json.dump(perf_entry, f)
                        temp_path = f.name
                    try:
                        update_perf_csv(
                            exception_result=temp_path,
                            perf_csv=self.perf_csv_path,
                        )
                        print(
                            f"Updated perf.csv with setup failure for {model_info.get('name', image_name)}"
                        )
                    finally:
                        os.unlink(temp_path)
                except Exception as csv_e:
                    self.rich_console.print(
                        f"[yellow]Warning: Could not record setup failure to perf CSV: {csv_e}[/yellow]"
                    )

        # Summary
        self.rich_console.print(f"\n[bold]📊 Execution Summary:[/bold]")
        self.rich_console.print(
            f"  [green]✓ Successful:[/green] {len(successful_runs)}"
        )
        self.rich_console.print(f"  [red]✗ Failed:[/red] {len(failed_runs)}")

        return {
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "total_runs": len(successful_runs) + len(failed_runs),
        }

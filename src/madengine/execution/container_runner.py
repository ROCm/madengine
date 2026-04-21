#!/usr/bin/env python3
"""
Docker Container Runner Module for madengine

This module handles the Docker container execution phase separately from building,
enabling distributed workflows where containers are run on remote nodes
using pre-built images.
"""

import os
import re
import socket
import subprocess
import time
import json
import glob
import shlex
import typing
import warnings
from rich.console import Console as RichConsole
from contextlib import redirect_stdout, redirect_stderr
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.docker import Docker
from madengine.core.constants import get_rocm_path
from madengine.core.timeout import Timeout
from madengine.core.dataprovider import Data
from madengine.utils.ops import PythonicTee, file_print
from madengine.reporting.update_perf_csv import update_perf_csv, flatten_tags
from madengine.reporting.update_perf_super import update_perf_super_json, update_perf_super_csv
from madengine.utils.gpu_config import resolve_runtime_gpus
from madengine.utils.config_parser import ConfigParser


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
                "model,n_gpus,nnodes,gpus_per_node,training_precision,pipeline,args,tags,docker_file,base_docker,docker_sha,docker_image,git_commit,machine_name,deployment_type,launcher,gpu_architecture,performance,metric,relative_change,status,build_duration,test_duration,dataname,data_provider_type,data_size,data_download_duration,build_number,additional_docker_run_options",
                filename=self.perf_csv_path,
                mode="w",
            )
            print(f"Created performance CSV file: {self.perf_csv_path}")

    def _get_build_args(self) -> str:
        """Build docker --build-arg string from runtime context."""
        docker_build_arg = self.context.ctx.get("docker_build_arg", {}) if self.context else {}
        if not docker_build_arg:
            return ""

        build_args = ""
        for key, value in docker_build_arg.items():
            build_args += f"--build-arg {key}='{value}' "
        return build_args

    def _get_node_rank(self) -> int:
        """Return the current node rank for distributed runs."""
        node_rank_raw = os.environ.get("NODE_RANK") or os.environ.get("RANK") or "0"
        try:
            return int(node_rank_raw)
        except Exception:
            return 0

    def _local_image_exists(self, run_image: str) -> bool:
        """Check whether a Docker image already exists locally."""
        try:
            self.console.sh(
                f"docker image inspect {shlex.quote(run_image)} > /dev/null 2>&1"
            )
            return True
        except (subprocess.CalledProcessError, RuntimeError):
            return False

    def _get_local_image_tar_path(self, run_image: str) -> typing.Optional[str]:
        """Resolve the shared tar path for a local image, if configured."""
        builds_dir = (os.environ.get("MAD_DOCKER_BUILDS") or "").strip()
        if not builds_dir:
            return None

        safe_image_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_image).strip("._")
        if not safe_image_name:
            safe_image_name = "docker_image"
        return os.path.join(builds_dir, f"{safe_image_name}.tar")

    def _load_local_image_from_tar(self, run_image: str, tar_path: str) -> None:
        """Load a Docker image from a previously saved tar archive."""
        if not os.path.exists(tar_path):
            raise RuntimeError(f"Image tar not found for {run_image}: {tar_path}")

        self.rich_console.print(
            f"[yellow]📦 Loading local image tar:[/yellow] {tar_path}"
        )
        self.console.sh(f"docker load -i {shlex.quote(tar_path)}", timeout=None)
        self.console.sh(
            f"docker image inspect {shlex.quote(run_image)} > /dev/null 2>&1"
        )
        self.rich_console.print(
            f"[green]✅ Loaded local image from tar:[/green] {run_image}"
        )

    def _save_local_image_to_tar(self, run_image: str, tar_path: str) -> None:
        """Persist a local Docker image into the shared tar cache."""
        tar_dir = os.path.dirname(tar_path)
        if tar_dir:
            os.makedirs(tar_dir, exist_ok=True)

        self.rich_console.print(
            f"[yellow]💾 Saving local image tar:[/yellow] {tar_path}"
        )
        self.console.sh(
            f"docker save -o {shlex.quote(tar_path)} {shlex.quote(run_image)}",
            timeout=None,
        )
        self.rich_console.print(
            f"[green]✅ Saved local image tar:[/green] {tar_path}"
        )

    def _build_or_pull_local_image(
        self, run_image: str, build_info: typing.Dict, model_info: typing.Dict
    ) -> None:
        """Ensure the local image exists by building it first and pulling as fallback."""
        self.rich_console.print(
            f"[yellow]⚠️  Image {run_image} not found on this node.[/yellow]"
        )
        try:
            self._build_local_image_from_manifest(
                run_image=run_image,
                build_info=build_info,
                model_info=model_info,
            )
        except Exception as build_error:
            self.rich_console.print(
                "[yellow]⚠️  Local build failed, attempting pull as fallback...[/yellow]"
            )
            try:
                self.pull_image(run_image)
            except Exception as pull_error:
                raise RuntimeError(
                    f"Failed to build or pull local image {run_image}: "
                    f"build_error={build_error}; pull_error={pull_error}"
                )

    def _ensure_local_image_available(
        self, run_image: str, build_info: typing.Dict, model_info: typing.Dict
    ) -> None:
        """Prepare a local image with optional shared tar cache support."""
        tar_path = self._get_local_image_tar_path(run_image)
        node_rank = self._get_node_rank()
        is_primary_node = node_rank == 0
        image_exists = self._local_image_exists(run_image)
        tar_exists = bool(tar_path) and os.path.exists(tar_path)
        tar_missing_at_start = bool(tar_path) and not tar_exists

        # When shared cache is configured and no tar exists yet, only node 0
        # may produce the tar artifact. Other nodes wait and then load it.
        if tar_missing_at_start:
            if is_primary_node:
                if not image_exists:
                    self._build_or_pull_local_image(
                        run_image=run_image,
                        build_info=build_info,
                        model_info=model_info,
                    )
                    image_exists = True
                if not tar_exists:
                    self._save_local_image_to_tar(run_image, tar_path)
                    tar_exists = True

            self._sync_after_local_image_ready(run_image=run_image)

            if not image_exists:
                if not tar_exists and not os.path.exists(tar_path):
                    raise RuntimeError(
                        f"Node 0 did not produce image tar for {run_image}: {tar_path}"
                    )
                self._load_local_image_from_tar(run_image, tar_path)
                image_exists = True

        elif not image_exists:
            if tar_exists:
                self._load_local_image_from_tar(run_image, tar_path)
                image_exists = True
            else:
                self._build_or_pull_local_image(
                    run_image=run_image,
                    build_info=build_info,
                    model_info=model_info,
                )
                image_exists = True

        if tar_path and image_exists and is_primary_node and not tar_exists:
            self._save_local_image_to_tar(run_image, tar_path)

    def _build_local_image_from_manifest(
        self, run_image: str, build_info: typing.Dict, model_info: typing.Dict
    ) -> None:
        """Build image on current node using dockerfile from manifest.

        This is used by run --manifest-file in distributed mode when the image is
        not present on a compute node and pulling is not desired/possible.
        """
        dockerfile = build_info.get("dockerfile", "")
        if not dockerfile or dockerfile == "N/A (local image mode)":
            raise RuntimeError(
                f"Cannot build image {run_image}: dockerfile is missing in manifest"
            )

        if not os.path.exists(dockerfile):
            raise RuntimeError(
                f"Cannot build image {run_image}: dockerfile not found at '{dockerfile}'"
            )

        docker_context = model_info.get("dockercontext", "") or "./docker"
        if not os.path.exists(docker_context):
            # Fallback to dockerfile directory if provided context is unavailable.
            docker_context = os.path.dirname(dockerfile) or "."

        build_args = self._get_build_args()
        build_command = (
            f"docker build --network=host -t {run_image} --pull -f {dockerfile} "
            f"{build_args}{docker_context}"
        )

        self.rich_console.print(
            f"[yellow]🔨 Building missing local image on this node:[/yellow] {run_image}"
        )
        self.rich_console.print(f"[dim]  Dockerfile: {dockerfile}[/dim]")
        self.rich_console.print(f"[dim]  Context: {docker_context}[/dim]")
        self.console.sh(build_command, timeout=None)
        self.console.sh(f"docker image inspect {run_image} > /dev/null 2>&1")
        self.rich_console.print(
            f"[green]✅ Built local image on this node:[/green] {run_image}"
        )

    def _sync_after_local_image_ready(self, run_image: str, timeout_s: int = 1800) -> None:
        """Barrier for multi-node local-image runs so all nodes continue together."""
        nnodes_raw = os.environ.get("NNODES") or os.environ.get("WORLD_SIZE") or "1"
        node_rank = os.environ.get("NODE_RANK") or os.environ.get("RANK") or "0"
        try:
            nnodes = int(nnodes_raw)
        except Exception:
            nnodes = 1
        if nnodes <= 1:
            return

        self._tcp_image_ready_barrier(
            nnodes=nnodes,
            node_rank=node_rank,
            timeout_s=timeout_s,
        )
        return

    def _tcp_image_ready_barrier(self, nnodes: int, node_rank: str, timeout_s: int) -> None:
        """Fallback barrier that does not require shared filesystem visibility."""
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        job_id_raw = os.environ.get("SLURM_JOB_ID", "0")
        try:
            job_id = int(job_id_raw)
        except Exception:
            job_id = 0
        token = f"JOB{job_id}"
        master_port_raw = os.environ.get("MASTER_PORT", "29500")
        try:
            master_port = int(master_port_raw)
        except Exception:
            master_port = 29500
        base_port = 43000 + ((master_port + job_id) % 1000)
        candidate_ports = [base_port + i for i in range(0, 16)]
        deadline = time.time() + timeout_s
        rank_int = int(node_rank)

        if rank_int == 0:
            accepted = 0
            peers = []
            waiting: typing.Dict[int, socket.socket] = {}
            server = None
            port = None
            try:
                bind_errors = []
                for candidate in candidate_ports:
                    trial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        trial.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        trial.bind(("0.0.0.0", candidate))
                        server = trial
                        port = candidate
                        break
                    except Exception as e:
                        bind_errors.append({"port": candidate, "error": str(e)})
                        try:
                            trial.close()
                        except Exception:
                            pass
                if server is None or port is None:
                    raise RuntimeError(f"TCP barrier bind failed on all candidate ports: {bind_errors}")
                server.listen(max(1, nnodes - 1))
                server.settimeout(2.0)
                while accepted < max(0, nnodes - 1) and time.time() < deadline:
                    try:
                        conn, addr = server.accept()
                        conn.settimeout(2.0)
                        payload = conn.recv(128).decode("utf-8", errors="ignore").strip()
                        parts = payload.split()
                        if len(parts) != 3 or parts[0] != "READY" or parts[1] != token:
                            conn.close()
                            continue
                        try:
                            worker_rank = int(parts[2])
                        except Exception:
                            conn.close()
                            continue
                        if worker_rank <= 0 or worker_rank >= nnodes:
                            conn.close()
                            continue
                        if worker_rank in waiting:
                            try:
                                waiting[worker_rank].close()
                            except Exception:
                                pass
                        waiting[worker_rank] = conn
                        peers.append(f"{addr[0]}:r{worker_rank}")
                        accepted = len(waiting)
                    except socket.timeout:
                        continue
                if accepted < max(0, nnodes - 1):
                    raise RuntimeError(
                        f"TCP barrier timeout on master: accepted={accepted}/{max(0, nnodes - 1)} port={port}"
                    )
                for worker_rank, conn in waiting.items():
                    try:
                        conn.sendall(f"GO {token} {worker_rank}\n".encode("utf-8"))
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
                return
            finally:
                try:
                    if server is not None:
                        server.close()
                except Exception:
                    pass

        last_error = ""
        connect_attempts = 0
        while time.time() < deadline:
            for candidate in candidate_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                connect_attempts += 1
                try:
                    sock.settimeout(1.5)
                    sock.connect((master_addr, candidate))
                    sock.sendall(f"READY {token} {node_rank}\n".encode("utf-8"))
                    remaining_s = max(1.0, deadline - time.time())
                    sock.settimeout(remaining_s)
                    ack = sock.recv(128).decode("utf-8", errors="ignore").strip()
                    if ack == f"GO {token} {node_rank}":
                        return
                    last_error = f"unexpected_ack={ack!r} port={candidate}"
                except Exception as e:
                    last_error = f"{e} port={candidate}"
                finally:
                    try:
                        sock.close()
                    except Exception:
                        pass
            time.sleep(1)

        raise RuntimeError(
            f"TCP barrier timeout on worker rank={node_rank} master={master_addr} "
            f"ports={candidate_ports} attempts={connect_attempts} last_error={last_error}"
        )

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
                system_ngpus = int(self.context.ctx["docker_env_vars"]["MAD_SYSTEM_NGPUS"])
                resolved_gpu_count = system_ngpus
                print(f"ℹ️  Converted n_gpus=-1 to actual system GPU count: {system_ngpus}")
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
            slurm_gpus_per_node = os.environ.get("GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_PER_NODE")
            
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
            print(f"✅ Final launcher selected: '{launcher}' (deployment_type: {deployment_type})")
        else:
            print(f"⚠️  No launcher specified (deployment_type: {deployment_type})")
        
        # Create run details dict with all required fields
        run_details = {
            "model": model_info["name"],
            "n_gpus": str(total_gpus),  # Total GPUs across all nodes
            "nnodes": nnodes,
            "gpus_per_node": gpus_per_node,
            "training_precision": model_info.get("training_precision", ""),
            "pipeline": os.environ.get("pipeline", ""),
            "args": model_info.get("args", ""),
            "tags": model_info.get("tags", ""),
            "docker_file": build_info.get("dockerfile", ""),
            "base_docker": build_info.get("base_docker", ""),
            "docker_sha": build_info.get("docker_sha", ""),
            "docker_image": build_info.get("docker_image", ""),
            "git_commit": run_results.get("git_commit", ""),
            "machine_name": run_results.get("machine_name", ""),
            "deployment_type": os.environ.get("MAD_DEPLOYMENT_TYPE", "local"),  # local, slurm, etc.
            "launcher": launcher,  # Distributed launcher: torchrun, vllm, sglang, deepspeed, etc.
            "gpu_architecture": (
                self.context.ctx["docker_env_vars"]["MAD_SYSTEM_GPU_ARCHITECTURE"]
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
            "build_number": os.environ.get("BUILD_NUMBER", "0"),
            "additional_docker_run_options": model_info.get(
                "additional_docker_run_options", ""
            ),
        }

        # Flatten tags if they are in list format
        flatten_tags(run_details)

        # Parse and load config file if present in args for perf_entry_super.json
        try:
            scripts_path = model_info.get("scripts", "")
            scripts_base_dir = os.path.dirname(scripts_path) if scripts_path else None
            config_parser = ConfigParser(scripts_base_dir=scripts_base_dir)
            run_details["configs"] = config_parser.parse_and_load(
                model_info.get("args", ""),
                scripts_path
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not parse config file: {e}")
            run_details["configs"] = None

        return run_details

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

        Args:
            registry: Registry URL (e.g., "localhost:5000", "docker.io")
            credentials: Optional credentials dictionary containing username/password
        """
        if not credentials:
            self.rich_console.print("[yellow]No credentials provided for registry login[/yellow]")
            return

        # Check if registry credentials are available
        registry_key = registry if registry else "dockerhub"

        # Handle docker.io as dockerhub
        if registry and registry.lower() == "docker.io":
            registry_key = "dockerhub"

        if registry_key not in credentials:
            error_msg = f"No credentials found for registry: {registry_key}"
            if registry_key == "dockerhub":
                error_msg += f"\nPlease add dockerhub credentials to credential.json:\n"
                error_msg += "{\n"
                error_msg += '  "dockerhub": {\n'
                error_msg += '    "repository": "your-repository",\n'
                error_msg += '    "username": "your-dockerhub-username",\n'
                error_msg += '    "password": "your-dockerhub-password-or-token"\n'
                error_msg += "  }\n"
                error_msg += "}"
            else:
                error_msg += (
                    f"\nPlease add {registry_key} credentials to credential.json:\n"
                )
                error_msg += "{\n"
                error_msg += f'  "{registry_key}": {{\n'
                error_msg += f'    "repository": "your-repository",\n'
                error_msg += f'    "username": "your-{registry_key}-username",\n'
                error_msg += f'    "password": "your-{registry_key}-password"\n'
                error_msg += "  }\n"
                error_msg += "}"
            print(error_msg)
            raise RuntimeError(error_msg)

        creds = credentials[registry_key]

        if "username" not in creds or "password" not in creds:
            error_msg = f"Invalid credentials format for registry: {registry_key}"
            error_msg += f"\nCredentials must contain 'username' and 'password' fields"
            print(error_msg)
            raise RuntimeError(error_msg)

        # Ensure credential values are strings
        username = str(creds["username"])
        password = str(creds["password"])

        # Perform docker login
        login_command = f"echo '{password}' | docker login"

        if registry and registry.lower() not in ["docker.io", "dockerhub"]:
            login_command += f" {registry}"

        login_command += f" --username {username} --password-stdin"

        try:
            self.console.sh(login_command, secret=True)
            self.rich_console.print(f"[green]✅ Successfully logged in to registry: {registry or 'DockerHub'}[/green]")
        except Exception as e:
            self.rich_console.print(f"[red]❌ Failed to login to registry {registry}: {e}[/red]")
            # Don't raise exception here, as public images might still be pullable

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

        self.rich_console.print(f"\n[bold blue]📥 Starting docker pull from registry...[/bold blue]")
        print(f"📍 Registry: {registry or 'Default'}")
        print(f"🏷️  Image: {registry_image}")
        
        # Force fresh pull on SLURM compute nodes to avoid corrupted cached layers
        # This prevents "permission denied" errors from corrupted image layers
        deployment_type = os.environ.get("MAD_DEPLOYMENT_TYPE", "local")
        in_slurm_job = os.environ.get("MAD_IN_SLURM_JOB", "0") == "1"
        
        if deployment_type == "slurm" and in_slurm_job:
            print(f"🔄 Using fresh pull policy for SLURM compute node (prevents cached layer corruption)")
            # Remove any existing cached image to force fresh pull
            try:
                self.console.sh(f"docker rmi -f {registry_image} 2>/dev/null || true")
                print(f"✓ Removed cached image layers")
            except:
                pass  # It's okay if image doesn't exist
        
        try:
            self.console.sh(f"docker pull {registry_image}")

            if local_name:
                self.console.sh(f"docker tag {registry_image} {local_name}")
                print(f"🏷️  Tagged as: {local_name}")
                self.rich_console.print(f"[bold green]✅ Successfully pulled and tagged image[/bold green]")
                self.rich_console.print(f"[dim]{'='*80}[/dim]")
                return local_name

            self.rich_console.print(f"[bold green]✅ Successfully pulled image:[/bold green] [cyan]{registry_image}[/cyan]")
            self.rich_console.print(f"[dim]{'='*80}[/dim]")
            return registry_image

        except Exception as e:
            self.rich_console.print(f"[red]❌ Failed to pull image {registry_image}: {e}[/red]")
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

    def get_env_arg(self, run_env: typing.Dict) -> str:
        """Get the environment arguments for docker run."""
        env_args = ""

        # Add custom environment variables
        if run_env:
            for env_arg in run_env:
                env_args += f"--env {env_arg}='{str(run_env[env_arg])}' "

        # Add context environment variables
        if "docker_env_vars" in self.context.ctx:
            for env_arg in self.context.ctx["docker_env_vars"].keys():
                env_args += f"--env {env_arg}='{str(self.context.ctx['docker_env_vars'][env_arg])}' "

        print(f"Env arguments: {env_args}")
        return env_args

    def _extract_additional_mount_targets(self, additional_opts: str) -> typing.Set[str]:
        """Extract container mount targets from free-form docker options."""
        targets: typing.Set[str] = set()
        if not additional_opts:
            return targets
        try:
            tokens = shlex.split(additional_opts)
        except Exception:
            return targets

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ("-v", "--volume") and i + 1 < len(tokens):
                spec = tokens[i + 1]
                i += 2
            elif token.startswith("-v") and len(token) > 2:
                spec = token[2:]
                i += 1
            elif token.startswith("--volume="):
                spec = token.split("=", 1)[1]
                i += 1
            else:
                i += 1
                continue

            # Parse spec: /host:/container[:mode]
            parts = spec.split(":")
            if len(parts) >= 2:
                targets.add(parts[1])
        return targets

    def get_mount_arg(
        self,
        mount_datapaths: typing.List,
        excluded_container_targets: typing.Optional[typing.Set[str]] = None,
    ) -> str:
        """Get the mount arguments for docker run."""
        mount_args = ""
        excluded_container_targets = excluded_container_targets or set()

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
                # Avoid duplicate mount points when additional_docker_run_options
                # already mounts the same container target.
                if mount_arg in excluded_container_targets:
                    continue
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
                        cmd
                        + " "
                        + pre_encapsulate_post_scripts["encapsulate_script"]
                    )
                    added_cmds.add(cmd)
                else:
                    print(f"  Note: Command '{cmd}' already added by another tool, skipping duplicate.")

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
        self.rich_console.print(f"[bold green]🏃 Running model:[/bold green] [bold cyan]{model_info['name']}[/bold cyan] [dim]in container[/dim] [yellow]{docker_image}[/yellow]")

        # Apply timeout logic: model timeout can override default timeout
        # If model has a timeout in models.json and CLI timeout is default (7200), use model's timeout
        # If CLI timeout is explicitly set (not default), it overrides model timeout
        if "timeout" in model_info and model_info["timeout"] is not None and model_info["timeout"] > 0 and timeout == 7200:
            # Model has a timeout and CLI is using default, so use model's timeout
            timeout = model_info["timeout"]

        # Create log file for this run
        # Extract dockerfile part from docker image name (remove "ci-" prefix and model name prefix)
        image_name_without_ci = docker_image.replace("ci-", "")
        model_name_clean = model_info["name"].replace("/", "_").lower()

        # Remove model name from the beginning to get the dockerfile part
        if image_name_without_ci.startswith(model_name_clean + "_"):
            dockerfile_part = image_name_without_ci[len(model_name_clean + "_") :]
        else:
            dockerfile_part = image_name_without_ci

        log_file_path = (
            model_info["name"].replace("/", "_")
            + "_"
            + dockerfile_part
            + phase_suffix
            + ".live.log"
        )
        # Replace / with _ in log file path (already done above, but keeping for safety)
        log_file_path = log_file_path.replace("/", "_")

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
        docker_options += (
            f"--env JENKINS_BUILD_NUMBER='{os.environ.get('BUILD_NUMBER','0')}' "
        )

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
                gpus_per_node_str = os.environ.get("NPROC_PER_NODE") or os.environ.get("GPUS_PER_NODE")
                if gpus_per_node_str:
                    try:
                        gpus = int(gpus_per_node_str)
                        # Add gpus_per_node to additional_context for GPU resolution
                        # resolve_runtime_gpus looks for this field name
                        if not self.additional_context:
                            self.additional_context = {}
                        if "gpus_per_node" not in self.additional_context:
                            self.additional_context["gpus_per_node"] = gpus
                            print(f"ℹ️  SLURM GPU override: {gpus} GPUs per node (from shell environment)")
                    except ValueError:
                        pass
        
        # List of environment variables to pass from shell to Docker (for SLURM jobs)
        slurm_env_vars = [
            'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'NODE_RANK',
            'NNODES', 'NPROC_PER_NODE', 'MAD_MULTI_NODE_RUNNER',
            'MAD_COLLECT_METRICS', 'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME',
            'NCCL_DEBUG', 'NCCL_IB_DISABLE', 'NCCL_NET_GDR_LEVEL',
            # Workload-level settings commonly provided via deployment_config.env_vars
            # (required for disaggregated launchers like vLLM/sglang disagg)
            'MODEL_NAME', 'MODEL_DIR', 'xP', 'yD', 'PD_SYNC_ROOT', 'PD_RUN_ID',
            'PROXY_TYPE', 'ROUTER_PORT', 'BENCHMARK_PORT', 'SLURM_JOB_ID',
            'OUTPUT_DIR', 'BARRIER_TIMEOUT_S', 'PROXY_CLOSE_TIMEOUT_S',
            'REQUIRE_RDMA', 'KV_UCX_TLS', 'KV_UCX_SOCKADDR_TLS_PRIORITY',
            # GPU visibility variables for Ray-based launchers (vLLM, SGLang)
            # CRITICAL: These must be passed to Docker for proper GPU device mapping
            'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'
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
        if 'HIP_VISIBLE_DEVICES' in self.context.ctx["docker_env_vars"]:
            # Set to empty string to disable Ray's behavior of ignoring HIP_VISIBLE_DEVICES
            self.context.ctx["docker_env_vars"]['RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES'] = ''
            print("ℹ️  Overriding RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES to enable HIP_VISIBLE_DEVICES")
        
        if merged_from_env > 0:
            print(f"ℹ️  Inherited {merged_from_env} environment variables from shell for Docker")
        
        # Also merge from additional_context if present
        if self.additional_context and "docker_env_vars" in self.additional_context:
            merged_count = 0
            for key, value in self.additional_context["docker_env_vars"].items():
                self.context.ctx["docker_env_vars"][key] = value
                merged_count += 1
            if merged_count > 0:
                print(f"ℹ️  Merged {merged_count} environment variables from additional_context")

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

        # Add system environment collection script to pre_scripts (equivalent to generate_sys_env_details)
        # This ensures distributed runs have the same system environment logging as standard runs
        if generate_sys_env_details or self.context.ctx.get("gen_sys_env_details"):
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
            print("ℹ️  Removed MIOPEN_USER_DB_PATH from run_env (will use context.docker_env_vars)")
        
        # Add MIOPEN_USER_DB_PATH from shell environment to context.docker_env_vars
        # This is set by SLURM script with ${LOCAL_RANK} variable for per-process paths
        if "MIOPEN_USER_DB_PATH" in os.environ and "MIOPEN_USER_DB_PATH" not in self.context.ctx["docker_env_vars"]:
            self.context.ctx["docker_env_vars"]["MIOPEN_USER_DB_PATH"] = os.environ["MIOPEN_USER_DB_PATH"]
            print(f"ℹ️  Added MIOPEN_USER_DB_PATH to docker_env_vars: {os.environ['MIOPEN_USER_DB_PATH']}")
        
        additional_docker_run_options = model_info.get("additional_docker_run_options", "")
        additional_mount_targets = self._extract_additional_mount_targets(additional_docker_run_options)
        docker_options += self.get_env_arg(run_env)
        docker_options += self.get_mount_arg(
            mount_datapaths,
            excluded_container_targets=additional_mount_targets,
        )
        docker_options += f" {additional_docker_run_options}"

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

        self.rich_console.print(f"\n[bold blue]🏃 Starting Docker container execution...[/bold blue]")
        print(f"🏷️  Image: {docker_image}")
        print(f"📦 Container: {container_name}")
        print(f"📝 Log file: {log_file_path}")
        print(f"🎮 GPU Vendor: {gpu_vendor}")
        self.rich_console.print(f"[dim]{'='*80}[/dim]")

        # Run the container with logging
        try:
            with open(log_file_path, mode="w", buffering=1) as outlog:
                with redirect_stdout(
                    PythonicTee(outlog, self.live_output)
                ), redirect_stderr(PythonicTee(outlog, self.live_output)):
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

                        # Show GPU info with version-aware tool selection (PR #54)
                        if gpu_vendor.find("AMD") != -1:
                            print(f"🎮 Checking AMD GPU status...")
                            rocm_path = self.context.ctx.get("rocm_path") or get_rocm_path()
                            amd_smi_path = os.path.join(rocm_path, "bin", "amd-smi")
                            rocm_smi_path = os.path.join(rocm_path, "bin", "rocm-smi")
                            try:
                                tool_manager = self.context._get_tool_manager()
                                preferred_tool = tool_manager.get_preferred_smi_tool()
                                if preferred_tool == "amd-smi":
                                    model_docker.sh(f"{amd_smi_path} || {rocm_smi_path} || true")
                                else:
                                    model_docker.sh(f"{rocm_smi_path} || {amd_smi_path} || true")
                            except Exception:
                                model_docker.sh(f"{amd_smi_path} || {rocm_smi_path} || true")
                        elif gpu_vendor.find("NVIDIA") != -1:
                            print(f"🎮 Checking NVIDIA GPU status...")
                            model_docker.sh("/usr/bin/nvidia-smi || true")

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
                                    run_results["dataname"] = self.data.selected_data_provider["dataname"]
                                if "data_provider_type" in self.data.selected_data_provider:
                                    run_results["data_provider_type"] = self.data.selected_data_provider["data_provider_type"]
                                if "duration" in self.data.selected_data_provider:
                                    run_results["data_download_duration"] = self.data.selected_data_provider["duration"]
                                if "size" in self.data.selected_data_provider:
                                    run_results["data_size"] = self.data.selected_data_provider["size"]
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
                        self.rich_console.print("[bold blue]Running model...[/bold blue]")

                        model_args = self.context.ctx.get(
                            "model_args", model_info["args"]
                        )
                        # Use the container timeout (default 7200s) for script execution
                        # to prevent indefinite hangs
                        try:
                            model_output = model_docker.sh(
                                f"cd {model_dir} && {script_name} {model_args}",
                                timeout=timeout,
                            )
                        except RuntimeError as run_err:
                            run_err_str = str(run_err)
                            container_id_match = re.search(
                                r"docker exec\s+([a-f0-9]+)\s+bash",
                                run_err_str,
                            )
                            failed_container_id = (
                                container_id_match.group(1)
                                if container_id_match
                                else None
                            )
                            if failed_container_id:
                                try:
                                    process_snapshot = self.console.sh(
                                        f"docker exec {failed_container_id} bash -lc \"ps -eo pid,ppid,stat,etime,cmd | sed -n '1,160p'\"",
                                        timeout=20,
                                    )
                                except Exception as diag_err:
                                    pass
                                try:
                                    net_snapshot = self.console.sh(
                                        f"docker exec {failed_container_id} bash -lc \"(ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null || lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null || true) | sed -n '1,200p'\"",
                                        timeout=20,
                                    )
                                except Exception as diag_err:
                                    pass
                                try:
                                    container_logs = self.console.sh(
                                        f"docker exec {failed_container_id} bash -lc \"for d in /run_logs /run_logs/${{SLURM_JOB_ID:-}} /myworkspace/{model_dir}; do if [ -d \\\"$d\\\" ]; then echo ===DIR:$d===; ls -lah \\\"$d\\\" | sed -n '1,80p'; fi; done; for f in /run_logs/*.log /run_logs/${{SLURM_JOB_ID:-}}/*.log /myworkspace/{model_dir}/*.log; do if [ -f \\\"$f\\\" ]; then echo ===$f===; tail -n 80 \\\"$f\\\"; fi; done\"",
                                        timeout=30,
                                    )
                                except Exception as diag_err:
                                    pass
                            if os.path.exists(log_file_path):
                                try:
                                    with open(log_file_path, "r", encoding="utf-8", errors="replace") as lf:
                                        log_lines = lf.readlines()
                                except Exception as log_tail_err:
                                    pass
                            raise
                        # Avoid duplicating full script output in live mode:
                        # Console.sh already streamed it line-by-line.
                        if not self.live_output:
                            print(model_output)

                        ts_after_model = int(time.time())
                        print(
                            f"[TS] stage=model_script_return epoch={ts_after_model} "
                            f"utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts_after_model))}"
                        )
                        run_results["test_duration"] = time.time() - test_start_time
                        print(f"Test Duration: {run_results['test_duration']} seconds")

                        # Run post-scripts
                        if pre_encapsulate_post_scripts["post_scripts"]:
                            self.run_pre_post_script(
                                model_docker,
                                model_dir,
                                pre_encapsulate_post_scripts["post_scripts"],
                            )

                        # Extract performance metrics from logs
                        # Look for performance data in the log output similar to original run_models.py
                        try:
                            # Check if multiple results file is specified in model_info
                            multiple_results = model_info.get("multiple_results", None)

                            if multiple_results:
                                multiple_results_name = os.path.basename(multiple_results)
                                multiple_results_resolved = multiple_results
                                if not os.path.exists(multiple_results_resolved):
                                    candidate_paths = [
                                        os.path.join(model_dir, multiple_results),
                                        os.path.join(model_dir, multiple_results_name),
                                        os.path.join(model_dir, "workdir", multiple_results),
                                        os.path.join(model_dir, "workdir", multiple_results_name),
                                    ]
                                    for candidate_path in candidate_paths:
                                        if os.path.exists(candidate_path):
                                            multiple_results_resolved = candidate_path
                                            break

                                run_results["performance"] = multiple_results_resolved
                                host_multiple_results_path = os.path.abspath(multiple_results_resolved)
                                host_run_dir_multiple = os.path.abspath(os.path.join(model_dir, multiple_results_name))
                                host_run_dir_workdir_multiple = os.path.abspath(
                                    os.path.join(model_dir, "workdir", multiple_results_name)
                                )
                                perf_candidates = {
                                    "host_multiple_results_path": host_multiple_results_path,
                                    "host_run_dir_multiple": host_run_dir_multiple,
                                    "host_run_dir_workdir_multiple": host_run_dir_workdir_multiple,
                                }

                                try:
                                    container_checks = {}
                                    for candidate in [
                                        f"{model_dir}/{multiple_results_name}",
                                        f"{model_dir}/workdir/{multiple_results_name}",
                                    ]:
                                        probe_cmd = f"if [ -f {candidate} ]; then echo EXISTS; else echo MISSING; fi"
                                        container_checks[candidate] = (model_docker.sh(probe_cmd) or "").strip()
                                    csv_inventory = (
                                        model_docker.sh(
                                            f"sh -c 'ls -lah {model_dir}/*.csv 2>/dev/null; "
                                            f"ls -lah {model_dir}/workdir/*.csv 2>/dev/null; "
                                            f"ls -lah {model_dir}/benchmark_*_CONCURRENCY.log 2>/dev/null'"
                                        )
                                        or ""
                                    )
                                except Exception as probe_err:
                                    pass

                                # Validate multiple results file format using proper CSV parsing
                                try:
                                    import csv
                                    with open(multiple_results_resolved, "r") as f:
                                        csv_reader = csv.DictReader(f)
                                        total_rows = 0
                                        perf_non_empty_rows = 0
                                        sample_rows = []
                                        
                                        # Check if 'performance' column exists
                                        if 'performance' not in csv_reader.fieldnames:
                                            print("Error: 'performance' column not found in multiple results file.")
                                            run_results["performance"] = None
                                        else:
                                            # Check if at least one row has a non-empty performance value
                                            has_valid_perf = False
                                            for row in csv_reader:
                                                total_rows += 1
                                                perf_value = (row.get('performance', '') or '').strip()
                                                if len(sample_rows) < 3:
                                                    sample_rows.append(
                                                        {
                                                            "performance": perf_value,
                                                            "metric": (row.get("metric", "") or "").strip(),
                                                            "unit": (row.get("unit", "") or "").strip(),
                                                        }
                                                    )
                                                if perf_value:
                                                    perf_non_empty_rows += 1
                                                    has_valid_perf = True
                                                    break

                                            if total_rows == 0:
                                                try:
                                                    bench_log_dump = model_docker.sh(
                                                        f"for f in {model_dir}/benchmark_*_CONCURRENCY.log; do "
                                                        f"if [ -f \"$f\" ]; then echo ===$f===; tail -n 80 \"$f\"; fi; done"
                                                    )
                                                except Exception as bench_log_err:
                                                    pass
                                            
                                            if not has_valid_perf:
                                                nnodes_env = os.environ.get("NNODES", "1")
                                                try:
                                                    nnodes = int(nnodes_env)
                                                except (TypeError, ValueError):
                                                    nnodes = 1

                                                if nnodes > 1:
                                                    # In multi-node runs perf CSV may be populated by another node
                                                    # moments later (shared workspace race). Keep the path so
                                                    # downstream aggregation can consume finalized file content.
                                                    print(
                                                        "Warning: Performance metric is currently empty in "
                                                        "multiple results file during multi-node run; "
                                                        "deferring final decision to aggregation step."
                                                    )
                                                else:
                                                    run_results["performance"] = None
                                                    print("Error: Performance metric is empty in all rows of multiple results file.")
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
                                        print(f"Warning: Log file not found: {log_file_path}")
                                        run_results["performance"] = None
                                        run_results["metric"] = None
                                    else:
                                        # Read the log file once (avoids rocprofv3 crash from shell pipelines)
                                        # This approach matches the Kubernetes implementation pattern
                                        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                            log_content = f.read()
                                        
                                        # Try multiple patterns to match different log formats
                                        
                                        # Pattern 1: "performance: 12345 metric_name" (original expected format)
                                        perf_pattern = r'performance:\s+([0-9][0-9.eE+-]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
                                        match = re.search(perf_pattern, log_content)
                                        
                                        if match:
                                            run_results["performance"] = match.group(1).strip()
                                            run_results["metric"] = match.group(2).strip()
                                            print(f"✓ Extracted performance: {run_results['performance']} {run_results['metric']}")
                                        else:
                                            # Pattern 2: HuggingFace format - "'train_samples_per_second': 4.23" or "train_samples_per_second = 4.23"
                                            # This matches the actual output from HuggingFace Trainer
                                            hf_pattern = r'train_samples_per_second[\'"\s:=]+([0-9][0-9.eE+-]*)'
                                            hf_match = re.search(hf_pattern, log_content)
                                            
                                            if hf_match:
                                                run_results["performance"] = hf_match.group(1).strip()
                                                run_results["metric"] = "samples_per_second"
                                                print(f"✓ Extracted performance (HuggingFace format): {run_results['performance']} {run_results['metric']}")
                                            else:
                                                # No performance metrics found
                                                print("Warning: Performance metric not found in expected format 'performance: NUMBER METRIC' or 'train_samples_per_second'")
                                                run_results["performance"] = None
                                                run_results["metric"] = None
                                            
                                except Exception as e:
                                    print(f"Warning: Error extracting performance metrics: {e}")
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
                            # Check for common failure patterns in the log file
                            # Note: Patterns should be specific enough to avoid false positives
                            # from profiling tools (rocprof, etc.) that use "Error:" as log level
                            error_patterns = [
                                "OutOfMemoryError",
                                "HIP out of memory",
                                "CUDA out of memory",
                                "RuntimeError:",  # More specific with colon
                                "AssertionError:",
                                "ValueError:",
                                "KeyError:",
                                "SystemExit",
                                "failed (exitcode:",  # Literal text in logs
                                "FAILED",
                                "Exception:",
                                "ImportError:",
                                "ModuleNotFoundError:",
                            ]

                            has_errors = False
                            if log_file_path and os.path.exists(log_file_path):
                                try:
                                    # Define benign patterns to exclude from error detection
                                    # These are known warnings/info messages that should not trigger failures
                                    benign_patterns = [
                                        "Failed to establish connection to the metrics exporter agent",
                                        "RpcError: Running out of retries to initialize the metrics agent",
                                        "Metrics will not be exported",
                                        "FutureWarning",
                                        # rocEnvTool pre-script can timeout rocm-smi without affecting run correctness.
                                        "RuntimeError: Console script timeout",
                                        "rocEnvTool/console.py",
                                        "rocEnvTool/rocenv_tool.py",
                                        # ROCProf/glog logging patterns (E/W/I prefixes are log levels, not errors)
                                        r"^E[0-9]{8}.*generateRocpd\.cpp",  # ROCProf error-level logs
                                        r"^W[0-9]{8}.*simple_timer\.cpp",    # ROCProf warning-level logs
                                        r"^W[0-9]{8}.*generateRocpd\.cpp",   # ROCProf warning-level logs
                                        r"^E[0-9]{8}.*tool\.cpp",            # ROCProf tool logs
                                        "Opened result file:",                # ROCProf result file messages
                                        "SQLite3 generation ::",              # ROCProf SQLite messages
                                        r"\[rocprofv3\]",                     # ROCProf v3 messages
                                        "rocpd_op:",                          # ROCProf operation logs
                                        "rpd_tracer:",                        # ROCProf tracer logs
                                    ]
                                    
                                    # Check for error patterns in the log (exclude our own grep commands, output messages, and benign patterns).
                                    # Use subprocess (not console.sh) so the check runs silently and does not clutter console output.
                                    for pattern in error_patterns:
                                        # Build exclusion regex: our own commands, output messages, and benign patterns.
                                        # Use re.escape(pattern) so parentheses and other special chars are safe in grep -E.
                                        pattern_escaped = re.escape(pattern)
                                        exclusions = f"(grep -q.*{pattern_escaped}|Found error pattern.*{pattern_escaped}"
                                        for benign in benign_patterns:
                                            # Escape special regex characters in benign patterns
                                            escaped_benign = benign.replace(".", r"\.").replace("(", r"\(").replace(")", r"\)")
                                            exclusions += f"|{escaped_benign}"
                                        exclusions += ")"
                                        # Match pattern literally in the filtered log (grep -F avoids regex issues)
                                        error_check_cmd = [
                                            "sh",
                                            "-c",
                                            f"grep -v -E '{exclusions}' {log_file_path} | grep -F -q -- '{pattern}' && echo 'FOUND' || echo 'NOT_FOUND'",
                                        ]
                                        try:
                                            proc = subprocess.run(
                                                error_check_cmd,
                                                capture_output=True,
                                                text=True,
                                                timeout=60,
                                            )
                                            result = (proc.stdout or "").strip()
                                            if result == "FOUND":
                                                has_errors = True
                                                print(
                                                    f"Found error pattern '{pattern}' in logs"
                                                )
                                                break
                                        except (subprocess.TimeoutExpired, OSError):
                                            pass  # Error checking is optional; treat as no match
                                except Exception:
                                    pass  # Error checking is optional

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
                            is_worker_node = os.environ.get("MAD_COLLECT_METRICS", "true").lower() == "false"

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
                                self.rich_console.print(f"[red]Status: FAILURE (no performance metrics)[/red]")

                        except Exception as e:
                            self.rich_console.print(f"[yellow]Warning: Error in status determination: {e}[/yellow]")
                            # Fallback to simple performance check
                            # Worker nodes don't need performance metrics
                            is_worker_node = os.environ.get("MAD_COLLECT_METRICS", "true").lower() == "false"
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
                        skip_perf = self.additional_context.get("skip_perf_collection", False)
                        
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
                                # Prefer resolved runtime path when available (e.g. run_directory/workdir/*)
                                # because model_info may only contain a basename that is not host-resolvable.
                                multiple_results = run_results.get(
                                    "performance", model_info.get("multiple_results", None)
                                )
                                if (
                                    multiple_results
                                    and run_results.get("status") == "SUCCESS"
                                ):
                                    # Generate common info JSON for multiple results
                                    common_info = run_details_dict.copy()
                                    # Remove model-specific fields for common info
                                    for key in ["model", "performance", "metric", "status"]:
                                        common_info.pop(key, None)

                                    with open("common_info.json", "w") as f:
                                        json.dump(common_info, f)

                                    # Update perf.csv with multiple results
                                    update_perf_csv(
                                        multiple_results=multiple_results,
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
                                        scripts_base_dir = os.path.dirname(scripts_path) if scripts_path else None
                                        
                                        # Reuse common_info.json for super files (no need for duplicate)
                                        num_entries = update_perf_super_json(
                                            multiple_results=multiple_results,
                                            perf_super_json="perf_super.json",
                                            model_name=run_details_dict["model"],
                                            common_info="common_info.json",
                                            scripts_base_dir=scripts_base_dir,
                                        )
                                        
                                        # Generate CSV and JSON files from perf_super.json
                                        update_perf_super_csv(
                                            perf_super_json="perf_super.json",
                                            perf_super_csv="perf_super.csv",
                                            num_entries=num_entries
                                        )
                                    except Exception as e:
                                        print(f"⚠️  Warning: Could not update perf_super files: {e}")
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
                                        scripts_base_dir = os.path.dirname(scripts_path) if scripts_path else None
                                        
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
                                            num_entries=num_entries
                                        )
                                    except Exception as e:
                                        print(f"⚠️  Warning: Could not update perf_super files: {e}")

                            except Exception as e:
                                self.rich_console.print(f"[yellow]Warning: Could not update perf.csv: {e}[/yellow]")

                        # Copy profiler/trace output files from run_directory to base directory before cleanup
                        # This ensures test files like gpu_info_power_profiler_output.csv and library_trace.csv are accessible
                        try:
                            model_docker.sh(f"cp {model_dir}/*_profiler_output.csv . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/*_output.csv . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/*_trace.csv . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/library_trace.csv . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/perf_*.csv . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/perf-*.csv . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/benchmark_*_CONCURRENCY.log . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/workdir/perf_*.csv . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/workdir/perf-*.csv . 2>/dev/null || true")
                            model_docker.sh(f"cp {model_dir}/workdir/benchmark_*_CONCURRENCY.log . 2>/dev/null || true")
                            model_docker.sh(f"cp /run_logs/{os.environ.get('SLURM_JOB_ID', '*')}/benchmark_*_CONCURRENCY.log . 2>/dev/null || true")
                        except Exception as e:
                            # Ignore errors if no profiler/trace output files exist
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
                    scripts_base_dir = os.path.dirname(scripts_path) if scripts_path else None
                    
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
                        num_entries=num_entries
                    )
                except Exception as e:
                    print(f"⚠️  Warning: Could not update perf_super files: {e}")

            except Exception as csv_e:
                self.rich_console.print(f"[yellow]Warning: Could not update perf.csv with exception: {csv_e}[/yellow]")

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
        self.rich_console.print(f"[bold blue]📦 Loading manifest:[/bold blue] {manifest_file}")
        
        # Load manifest
        manifest = self.load_build_manifest(manifest_file)
        built_images = manifest.get("built_images", {})
        built_models = manifest.get("built_models", {})
        
        # Load deployment_config from manifest for GPU resolution
        if "deployment_config" in manifest and not self.additional_context:
            self.additional_context = {"deployment_config": manifest["deployment_config"]}
        
        if not built_images:
            self.rich_console.print("[yellow]⚠️  No images found in manifest[/yellow]")
            return {"successful_runs": [], "failed_runs": []}
        
        self.rich_console.print(f"[green]Found {len(built_images)} image(s) to run[/green]\n")
        
        # Login to registry if needed
        if registry or any(img.get("registry") for img in built_images.values()):
            effective_registry = registry or next(
                (img.get("registry") for img in built_images.values() if img.get("registry")), 
                None
            )
            if effective_registry:
                try:
                    self.login_to_registry(effective_registry, self.credentials)
                except Exception as e:
                    self.rich_console.print(f"[yellow]Warning: Registry login failed: {e}[/yellow]")
                    self.rich_console.print("[yellow]Proceeding with local images only[/yellow]\n")
        
        # Track results
        successful_runs = []
        failed_runs = []
        
        # Run each model
        for image_name, build_info in built_images.items():
            model_info = built_models.get(image_name, {})
            if not model_info:
                self.rich_console.print(f"[yellow]⚠️  No model info for {image_name}, skipping[/yellow]")
                continue
            
            try:
                # Handle different image sources
                if build_info.get("local_image"):
                    # Local image mode (MAD_CONTAINER_IMAGE): Use the provided image directly
                    run_image = build_info.get("docker_image")
                    self.rich_console.print(f"[yellow]🏠 Using local image: {run_image}[/yellow]")

                    self._ensure_local_image_available(
                        run_image=run_image,
                        build_info=build_info,
                        model_info=model_info,
                    )
                    # Ensure all nodes reach this point before entering container run.
                    self._sync_after_local_image_ready(run_image=run_image)
                
                elif build_info.get("registry_image"):
                    # Registry image: Pull from registry
                    try:
                        self.pull_image(build_info["registry_image"])
                        # Update docker_image to use registry image
                        run_image = build_info["registry_image"]
                    except Exception as pull_error:
                        self.rich_console.print(f"[yellow]Warning: Could not pull from registry, using local image[/yellow]")
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
                    successful_runs.append({
                        "model": model_info["name"],
                        "image": run_image,
                        "status": status,
                        "performance": run_results.get("performance"),
                        "duration": run_results.get("test_duration"),
                    })
                else:
                    # Status is FAILURE - track as failed
                    failed_runs.append({
                        "model": model_info["name"],
                        "image": run_image,
                        "status": status,
                        "error": "Container execution failed - check logs for details",
                    })
                    self.rich_console.print(f"[red]❌ Run failed for {model_info['name']}: Status={status}[/red]")
                
            except Exception as e:
                self.rich_console.print(f"[red]❌ Failed to run {model_info['name']}: {e}[/red]")
                failed_runs.append({
                    "model": model_info.get("name", image_name),
                    "image": image_name,
                    "error": str(e),
                })
        
        # Summary
        self.rich_console.print(f"\n[bold]📊 Execution Summary:[/bold]")
        self.rich_console.print(f"  [green]✓ Successful:[/green] {len(successful_runs)}")
        self.rich_console.print(f"  [red]✗ Failed:[/red] {len(failed_runs)}")
        
        return {
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "total_runs": len(successful_runs) + len(failed_runs),
        }

#!/usr/bin/env python3
"""
Docker Container Runner Module for MADEngine

This module handles the Docker container execution phase separately from building,
enabling distributed workflows where containers are run on remote nodes
using pre-built images.
"""

import os
import time
import json
import typing
import warnings
import re
from rich.console import Console as RichConsole
from contextlib import redirect_stdout, redirect_stderr
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.docker import Docker
from madengine.core.timeout import Timeout
from madengine.core.dataprovider import Data
from madengine.utils.ops import PythonicTee, file_print
from madengine.reporting.update_perf_csv import update_perf_csv, flatten_tags


class ContainerRunner:
    """Class responsible for running Docker containers with models."""

    def __init__(
        self,
        context: Context = None,
        data: Data = None,
        console: Console = None,
        live_output: bool = False,
    ):
        """Initialize the Container Runner.

        Args:
            context: The MADEngine context
            data: The data provider instance
            console: Optional console instance
            live_output: Whether to show live output
        """
        self.context = context
        self.data = data
        self.console = console or Console(live_output=live_output)
        self.live_output = live_output
        self.rich_console = RichConsole()
        self.credentials = None
        self.perf_csv_path = "perf.csv"  # Default output path

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
                "model,n_gpus,training_precision,pipeline,args,tags,docker_file,base_docker,docker_sha,docker_image,git_commit,machine_name,deployment_type,gpu_architecture,performance,metric,relative_change,status,build_duration,test_duration,dataname,data_provider_type,data_size,data_download_duration,build_number,additional_docker_run_options",
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

        # Create run details dict with all required fields
        run_details = {
            "model": model_info["name"],
            "n_gpus": model_info.get("n_gpus", ""),
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
                # Skip individual MAD_MULTI_NODE_* env vars (except MAD_MULTI_NODE_RUNNER)
                # These are redundant since MAD_MULTI_NODE_RUNNER contains all necessary information
                if (
                    env_arg.startswith("MAD_MULTI_NODE_")
                    and env_arg != "MAD_MULTI_NODE_RUNNER"
                ):
                    continue
                env_args += f"--env {env_arg}='{str(self.context.ctx['docker_env_vars'][env_arg])}' "

        print(f"Env arguments: {env_args}")
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
            # Update environment variables
            if "env_vars" in tool_config:
                run_env.update(tool_config["env_vars"])
            if "cmd" in tool_config:
                # Prepend encapsulate cmd
                pre_encapsulate_post_scripts["encapsulate_script"] = (
                    tool_config["cmd"]
                    + " "
                    + pre_encapsulate_post_scripts["encapsulate_script"]
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

    def run_container(
        self,
        model_info: typing.Dict,
        docker_image: str,
        build_info: typing.Dict = None,
        keep_alive: bool = False,
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
            timeout: Execution timeout in seconds
            tools_json_file: Path to tools configuration file
            phase_suffix: Suffix for log file name (e.g., ".run" or "")
            generate_sys_env_details: Whether to collect system environment details

        Returns:
            dict: Execution results including performance metrics
        """
        self.rich_console.print(f"[bold green]🏃 Running model:[/bold green] [bold cyan]{model_info['name']}[/bold cyan] [dim]in container[/dim] [yellow]{docker_image}[/yellow]")

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
                "--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --cap-add SYS_NICE --device /dev/fuse "
                "--security-opt seccomp=unconfined --security-opt apparmor=unconfined "
                "--network host -u root --ipc=host "
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
        docker_options += self.get_gpu_arg(model_info["n_gpus"])
        docker_options += self.get_cpu_arg()
        docker_options += self.get_env_arg(run_env)
        docker_options += self.get_mount_arg(mount_datapaths)
        docker_options += f" {model_info.get('additional_docker_run_options', '')}"

        # Generate container name
        container_name = "container_" + re.sub(
            ".*:", "", docker_image.replace("/", "_").replace(":", "_")
        )

        print(f"Docker options: {docker_options}")

        # set timeout
        print(f"⏰ Setting timeout to {str(timeout)} seconds.")

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
                            # Use version-aware SMI tool selection
                            # Note: Use amd-smi without arguments to show full status table (same as legacy madengine)
                            try:
                                tool_manager = self.context._get_tool_manager()
                                preferred_tool = tool_manager.get_preferred_smi_tool()
                                if preferred_tool == "amd-smi":
                                    model_docker.sh("/opt/rocm/bin/amd-smi || /opt/rocm/bin/rocm-smi || true")
                                else:
                                    model_docker.sh("/opt/rocm/bin/rocm-smi || /opt/rocm/bin/amd-smi || true")
                            except Exception:
                                # Fallback: try both tools
                                model_docker.sh("/opt/rocm/bin/amd-smi || /opt/rocm/bin/rocm-smi || true")
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
                            dir_path = os.path.dirname(scripts_arg)
                            script_name = "bash " + os.path.basename(scripts_arg)
                        else:
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
                        model_docker.sh(
                            f"cd {model_dir} && {script_name} {model_args}",
                            timeout=None,
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
                                run_results["performance"] = multiple_results
                                # Validate multiple results file format
                                try:
                                    with open(multiple_results, "r") as f:
                                        header = f.readline().strip().split(",")
                                        for line in f:
                                            row = line.strip().split(",")
                                            for col in row:
                                                if col == "":
                                                    run_results["performance"] = None
                                                    print(
                                                        "Error: Performance metric is empty in multiple results file."
                                                    )
                                                    break
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
                                    # Extract performance number: capture digits (with optional decimal/scientific notation)
                                    perf_cmd = (
                                        "cat "
                                        + log_file_path
                                        + " | grep 'performance:' | sed -n 's/.*performance:[[:space:]]*\\([0-9][0-9.eE+-]*\\)[[:space:]].*/\\1/p'"
                                    )
                                    run_results["performance"] = self.console.sh(
                                        perf_cmd
                                    )

                                    # Extract metric unit: capture the word after the number
                                    metric_cmd = (
                                        "cat "
                                        + log_file_path
                                        + " | grep 'performance:' | sed -n 's/.*performance:[[:space:]]*[0-9][0-9.eE+-]*[[:space:]]*\\([a-zA-Z_][a-zA-Z0-9_]*\\).*/\\1/p'"
                                    )
                                    run_results["metric"] = self.console.sh(metric_cmd)
                                except Exception:
                                    pass  # Performance extraction is optional
                        except Exception as e:
                            print(
                                f"Warning: Could not extract performance metrics: {e}"
                            )

                        # Set status based on performance and error patterns
                        # First check for obvious failure patterns in the logs
                        try:
                            # Check for common failure patterns in the log file
                            error_patterns = [
                                "OutOfMemoryError",
                                "HIP out of memory",
                                "CUDA out of memory",
                                "RuntimeError",
                                "AssertionError",
                                "ValueError",
                                "SystemExit",
                                "failed (exitcode:",
                                "Error:",
                                "FAILED",
                                "Exception:",
                            ]

                            has_errors = False
                            if log_file_path and os.path.exists(log_file_path):
                                try:
                                    # Check for error patterns in the log (exclude our own grep commands and output messages)
                                    for pattern in error_patterns:
                                        # Use grep with -v to exclude our own commands and output to avoid false positives
                                        error_check_cmd = f"grep -v -E '(grep -q.*{pattern}|Found error pattern.*{pattern})' {log_file_path} | grep -q '{pattern}' && echo 'FOUND' || echo 'NOT_FOUND'"
                                        result = self.console.sh(
                                            error_check_cmd, canFail=True
                                        )
                                        if result.strip() == "FOUND":
                                            has_errors = True
                                            print(
                                                f"Found error pattern '{pattern}' in logs"
                                            )
                                            break
                                except Exception:
                                    pass  # Error checking is optional

                            # Status logic: Must have performance AND no errors to be considered success
                            performance_value = run_results.get("performance")
                            has_performance = (
                                performance_value
                                and performance_value.strip()
                                and performance_value.strip() != "N/A"
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
                            else:
                                run_results["status"] = "FAILURE"
                                self.rich_console.print(f"[red]Status: FAILURE (no performance metrics)[/red]")

                        except Exception as e:
                            self.rich_console.print(f"[yellow]Warning: Error in status determination: {e}[/yellow]")
                            # Fallback to simple performance check
                            run_results["status"] = (
                                "SUCCESS"
                                if run_results.get("performance")
                                else "FAILURE"
                            )

                        print(
                            f"{model_info['name']} performance is {run_results.get('performance', 'N/A')} {run_results.get('metric', '')}"
                        )

                        # Generate performance results and update perf.csv
                        self.ensure_perf_csv_exists()
                        try:
                            # Create run details dictionary for CSV generation
                            run_details_dict = self.create_run_details_dict(
                                model_info, build_info, run_results
                            )

                            # Handle multiple results if specified
                            multiple_results = model_info.get("multiple_results", None)
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

                        except Exception as e:
                            self.rich_console.print(f"[yellow]Warning: Could not update perf.csv: {e}[/yellow]")

                        # Cleanup if not keeping alive
                        if not keep_alive:
                            model_docker.sh(f"rm -rf {model_dir}", timeout=240)
                        else:
                            model_docker.sh(f"chmod -R a+rw {model_dir}")
                            print(
                                f"keep_alive specified; model_dir({model_dir}) is not removed"
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
        phase_suffix: str = "",
    ) -> typing.Dict:
        """Run all models from a build manifest file.

        This is the main entry point for running pre-built containers from a manifest.

        Args:
            manifest_file: Path to build_manifest.json
            registry: Optional registry override
            timeout: Execution timeout per model in seconds
            keep_alive: Whether to keep containers alive after execution
            phase_suffix: Suffix for log files (e.g., ".run")

        Returns:
            dict: Execution summary with successful and failed runs
        """
        self.rich_console.print(f"[bold blue]📦 Loading manifest:[/bold blue] {manifest_file}")
        
        # Load manifest
        manifest = self.load_build_manifest(manifest_file)
        built_images = manifest.get("built_images", {})
        built_models = manifest.get("built_models", {})
        
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
                # Pull image if registry is specified
                if build_info.get("registry_image"):
                    try:
                        self.pull_image(build_info["registry_image"])
                        # Update docker_image to use registry image
                        run_image = build_info["registry_image"]
                    except Exception as pull_error:
                        self.rich_console.print(f"[yellow]Warning: Could not pull from registry, using local image[/yellow]")
                        run_image = image_name
                else:
                    run_image = image_name
                
                # Run the container
                run_results = self.run_container(
                    model_info=model_info,
                    docker_image=run_image,
                    build_info=build_info,
                    keep_alive=keep_alive,
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

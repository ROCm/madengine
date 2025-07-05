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
from contextlib import redirect_stdout, redirect_stderr
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.docker import Docker
from madengine.core.timeout import Timeout
from madengine.core.dataprovider import Data
from madengine.utils.ops import PythonicTee, file_print
from madengine.tools.update_perf_csv import update_perf_csv, flatten_tags


class ContainerRunner:
    """Class responsible for running Docker containers with models."""
    
    def __init__(self, context: Context = None, data: Data = None, console: Console = None, live_output: bool = False):
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
        self.credentials = None
        self.perf_csv_path = "perf.csv"  # Default output path
        
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
                "model,n_gpus,training_precision,pipeline,args,tags,docker_file,base_docker,docker_sha,docker_image,git_commit,machine_name,gpu_architecture,performance,metric,relative_change,status,build_duration,test_duration,dataname,data_provider_type,data_size,data_download_duration,build_number,additional_docker_run_options",
                filename=self.perf_csv_path,
                mode="w",
            )
            print(f"Created performance CSV file: {self.perf_csv_path}")
    
    def create_run_details_dict(self, model_info: typing.Dict, build_info: typing.Dict, run_results: typing.Dict) -> typing.Dict:
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
            "docker_image": build_info.get("image_name", ""),
            "git_commit": run_results.get("git_commit", ""),
            "machine_name": run_results.get("machine_name", ""),
            "gpu_architecture": self.context.ctx["docker_env_vars"]["MAD_SYSTEM_GPU_ARCHITECTURE"] if self.context else "",
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
            "build_number": os.environ.get('BUILD_NUMBER', '0'),
            "additional_docker_run_options": model_info.get("additional_docker_run_options", "")
        }
        
        # Flatten tags if they are in list format
        flatten_tags(run_details)
        
        return run_details
    
    def load_build_manifest(self, manifest_file: str = "build_manifest.json") -> typing.Dict:
        """Load build manifest from file.
        
        Args:
            manifest_file: Path to build manifest file
            
        Returns:
            dict: Build manifest data
        """
        with open(manifest_file, 'r') as f:
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
            print("No credentials provided for registry login")
            return
            
        # Check if registry credentials are available
        registry_key = registry if registry else "dockerhub"
        
        if registry_key not in credentials:
            print(f"No credentials found for registry: {registry_key}")
            return
            
        creds = credentials[registry_key]
        
        if "username" not in creds or "password" not in creds:
            print(f"Invalid credentials format for registry: {registry_key}")
            return
            
        # Perform docker login
        login_command = f"echo '{creds['password']}' | docker login"
        
        if registry and registry != "docker.io":
            login_command += f" {registry}"
            
        login_command += f" --username {creds['username']} --password-stdin"
        
        try:
            self.console.sh(login_command, secret=True)
            print(f"Successfully logged in to registry: {registry or 'DockerHub'}")
        except Exception as e:
            print(f"Failed to login to registry {registry}: {e}")
            # Don't raise exception here, as public images might still be pullable

    def pull_image(self, registry_image: str, local_name: str = None, 
                   registry: str = None, credentials: typing.Dict = None) -> str:
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
        
        print(f"Pulling image: {registry_image}")
        try:
            self.console.sh(f"docker pull {registry_image}")
            
            if local_name:
                self.console.sh(f"docker tag {registry_image} {local_name}")
                print(f"Tagged as: {local_name}")
                return local_name
            
            return registry_image
            
        except Exception as e:
            print(f"Failed to pull image {registry_image}: {e}")
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
        n_system_gpus = self.context.ctx['docker_env_vars']['MAD_SYSTEM_NGPUS']
        gpu_strings = self.context.ctx["docker_gpus"].split(",")

        # Parse GPU string, example: '{0-4}' -> [0,1,2,3,4]
        docker_gpus = []
        for gpu_string in gpu_strings:
            if '-' in gpu_string:
                gpu_range = gpu_string.split('-')
                docker_gpus += [item for item in range(int(gpu_range[0]), int(gpu_range[1])+1)]
            else:
                docker_gpus.append(int(gpu_string))
        docker_gpus.sort()

        # Check GPU range is valid for system
        if requested_gpus == "-1":
            print("NGPUS requested is ALL (" + ','.join(map(str, docker_gpus)) + ").")
            requested_gpus = len(docker_gpus)

        print("NGPUS requested is " + str(requested_gpus) + " out of " + str(n_system_gpus))

        if int(requested_gpus) > int(n_system_gpus) or int(requested_gpus) > len(docker_gpus):
            raise RuntimeError(f"Too many gpus requested({requested_gpus}). System has {n_system_gpus} gpus. Context has {len(docker_gpus)} gpus.")

        # Expose number of requested gpus
        self.context.ctx['docker_env_vars']['MAD_RUNTIME_NGPUS'] = str(requested_gpus)

        # Create docker arg to assign requested GPUs
        if gpu_vendor.find("AMD") != -1:
            gpu_arg = '--device=/dev/kfd '
            gpu_renderDs = self.context.ctx['gpu_renderDs']
            if gpu_renderDs is not None:
                for idx in range(0, int(requested_gpus)):
                    gpu_arg += f"--device=/dev/dri/renderD{gpu_renderDs[docker_gpus[idx]]} "

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

    def get_mount_arg(self, mount_datapaths: typing.List) -> str:
        """Get the mount arguments for docker run."""
        mount_args = ""
        
        # Mount data paths
        if mount_datapaths:
            for mount_datapath in mount_datapaths:
                if mount_datapath:
                    mount_args += f"-v {mount_datapath['path']}:{mount_datapath['home']}"
                    if "readwrite" in mount_datapath and mount_datapath["readwrite"] == 'true':
                        mount_args += " "
                    else:
                        mount_args += ":ro "

        # Mount context paths
        if "docker_mounts" in self.context.ctx:
            for mount_arg in self.context.ctx["docker_mounts"].keys():
                mount_args += f"-v {self.context.ctx['docker_mounts'][mount_arg]}:{mount_arg} "

        return mount_args
    
    def apply_tools(self, pre_encapsulate_post_scripts: typing.Dict, run_env: typing.Dict, tools_json_file: str) -> None:
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
                    tool_config["env_vars"].update({env_var: ctx_tool_config["env_vars"][env_var]})

            print(f"Selected Tool, {tool_name}. Configuration : {str(tool_config)}.")

            # Setup tool before other existing scripts
            if "pre_scripts" in tool_config:
                pre_encapsulate_post_scripts["pre_scripts"] = (
                    tool_config["pre_scripts"] + pre_encapsulate_post_scripts["pre_scripts"]
                )
            # Cleanup tool after other existing scripts
            if "post_scripts" in tool_config:
                pre_encapsulate_post_scripts["post_scripts"] += tool_config["post_scripts"]
            # Update environment variables
            if "env_vars" in tool_config:
                run_env.update(tool_config["env_vars"])
            if "cmd" in tool_config:
                # Prepend encapsulate cmd
                pre_encapsulate_post_scripts["encapsulate_script"] = (
                    tool_config["cmd"] + " " + pre_encapsulate_post_scripts["encapsulate_script"]
                )
    
    def run_pre_post_script(self, model_docker: Docker, model_dir: str, pre_post: typing.List) -> None:
        """Run pre/post scripts in the container."""
        for script in pre_post:
            script_path = script["path"].strip()
            model_docker.sh(f"cp -vLR --preserve=all {script_path} {model_dir}", timeout=600)
            script_name = os.path.basename(script_path)
            script_args = ""
            if "args" in script:
                script_args = script["args"].strip()
            model_docker.sh(f"cd {model_dir} && bash {script_name} {script_args}", timeout=600)
    
    def run_container(self, model_info: typing.Dict, docker_image: str, 
                     build_info: typing.Dict = None, keep_alive: bool = False,
                     timeout: int = 7200, tools_json_file: str = "scripts/common/tools.json",
                     phase_suffix: str = "") -> typing.Dict:
        """Run a model in a Docker container.
        
        Args:
            model_info: Model information dictionary
            docker_image: Docker image name to run
            build_info: Optional build information from manifest
            keep_alive: Whether to keep container alive after execution
            timeout: Execution timeout in seconds
            tools_json_file: Path to tools configuration file
            phase_suffix: Suffix for log file name (e.g., ".run" or "")
            
        Returns:
            dict: Execution results including performance metrics
        """
        print(f"Running model {model_info['name']} in container {docker_image}")
        
        # Create log file for this run
        # Extract dockerfile part from docker image name (remove "ci-" prefix and model name prefix)
        image_name_without_ci = docker_image.replace("ci-", "")
        model_name_clean = model_info["name"].replace("/", "_").lower()
        
        # Remove model name from the beginning to get the dockerfile part
        if image_name_without_ci.startswith(model_name_clean + "_"):
            dockerfile_part = image_name_without_ci[len(model_name_clean + "_"):]
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
            "log_file": log_file_path
        }
        
        # If build info provided, merge it
        if build_info:
            run_results.update(build_info)
        
        # Prepare docker run options
        gpu_vendor = self.context.ctx["gpu_vendor"]
        docker_options = ""

        if gpu_vendor.find("AMD") != -1:
            docker_options = ("--network host -u root --group-add video "
                            "--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse "
                            "--security-opt seccomp=unconfined --security-opt apparmor=unconfined --ipc=host ")
        elif gpu_vendor.find("NVIDIA") != -1:
            docker_options = ("--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --cap-add SYS_NICE --device /dev/fuse "
                            "--security-opt seccomp=unconfined --security-opt apparmor=unconfined "
                            "--network host -u root --ipc=host ")
        else:
            raise RuntimeError("Unable to determine gpu vendor.")

        # Initialize scripts
        pre_encapsulate_post_scripts = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}

        if "pre_scripts" in self.context.ctx:
            pre_encapsulate_post_scripts["pre_scripts"] = self.context.ctx["pre_scripts"]
        if "post_scripts" in self.context.ctx:
            pre_encapsulate_post_scripts["post_scripts"] = self.context.ctx["post_scripts"]
        if "encapsulate_script" in self.context.ctx:
            pre_encapsulate_post_scripts["encapsulate_script"] = self.context.ctx["encapsulate_script"]

        # Add environment variables
        docker_options += f"--env MAD_MODEL_NAME='{model_info['name']}' "
        docker_options += f"--env JENKINS_BUILD_NUMBER='{os.environ.get('BUILD_NUMBER','0')}' "

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

        # Build docker options
        docker_options += self.get_gpu_arg(model_info["n_gpus"])
        docker_options += self.get_cpu_arg()
        docker_options += self.get_env_arg(run_env)
        docker_options += self.get_mount_arg(mount_datapaths)
        docker_options += f" {model_info.get('additional_docker_run_options', '')}"

        # Generate container name
        container_name = "container_" + re.sub('.*:', '', docker_image.replace("/", "_").replace(":", "_"))

        print(f"Docker options: {docker_options}")
        
        # set timeout
        print(f"Setting timeout to {str(timeout)} seconds.")

        # Run the container with logging
        try:
            with open(log_file_path, mode="w", buffering=1) as outlog:
                with redirect_stdout(PythonicTee(outlog, self.live_output)), redirect_stderr(PythonicTee(outlog, self.live_output)):
                    with Timeout(timeout):
                        model_docker = Docker(docker_image, container_name, docker_options, 
                                            keep_alive=keep_alive, console=self.console)
                        
                        # Check user
                        whoami = model_docker.sh("whoami")
                        print(f"USER is {whoami}")

                        # Show GPU info
                        if gpu_vendor.find("AMD") != -1:
                            smi = model_docker.sh("/opt/rocm/bin/rocm-smi || true")
                            print(smi)
                        elif gpu_vendor.find("NVIDIA") != -1:
                            smi = model_docker.sh("/usr/bin/nvidia-smi || true")
                            print(smi)

                        # Prepare model directory
                        model_dir = "run_directory"
                        if "url" in model_info and model_info["url"] != "":
                            model_dir = model_info['url'].rstrip('/').split('/')[-1]
                            
                            # Validate model_dir
                            special_char = r'[^a-zA-Z0-9\-\_]'
                            if re.search(special_char, model_dir) is not None:
                                warnings.warn("Model url contains special character. Fix url.")

                        model_docker.sh(f"rm -rf {model_dir}", timeout=240)
                        model_docker.sh("git config --global --add safe.directory /myworkspace")

                        # Clone model repo if needed
                        if "url" in model_info and model_info["url"] != "":
                            if "cred" in model_info and model_info["cred"] != "" and self.credentials:
                                print(f"Using credentials for {model_info['cred']}")
                                
                                if model_info['url'].startswith('ssh://'):
                                    model_docker.sh(
                                        f"git -c core.sshCommand='ssh -l {self.credentials[model_info['cred']]['username']} "
                                        f"-i {self.credentials[model_info['cred']]['ssh_key_file']} -o IdentitiesOnly=yes "
                                        f"-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no' "
                                        f"clone {model_info['url']}", timeout=240
                                    )
                                else:  # http or https
                                    model_docker.sh(
                                        f"git clone -c credential.helper='!f() {{ echo username={self.credentials[model_info['cred']]['username']}; "
                                        f"echo password={self.credentials[model_info['cred']]['password']}; }};f' "
                                        f"{model_info['url']}", timeout=240, secret=f"git clone {model_info['url']}"
                                    )
                            else:
                                model_docker.sh(f"git clone {model_info['url']}", timeout=240)

                            model_docker.sh(f"git config --global --add safe.directory /myworkspace/{model_dir}")
                            run_results["git_commit"] = model_docker.sh(f"cd {model_dir} && git rev-parse HEAD")
                            print(f"MODEL GIT COMMIT is {run_results['git_commit']}")
                            model_docker.sh(f"cd {model_dir}; git submodule update --init --recursive")
                        else:
                            model_docker.sh(f"mkdir -p {model_dir}")

                        # Run pre-scripts
                        if pre_encapsulate_post_scripts["pre_scripts"]:
                            self.run_pre_post_script(model_docker, model_dir, pre_encapsulate_post_scripts["pre_scripts"])

                        # Prepare script execution
                        scripts_arg = model_info['scripts']
                        if scripts_arg.endswith(".sh"):
                            dir_path = os.path.dirname(scripts_arg)
                            script_name = "bash " + os.path.basename(scripts_arg)
                        else:
                            dir_path = model_info['scripts']
                            script_name = "bash run.sh"

                        # Add script prepend command
                        script_name = pre_encapsulate_post_scripts["encapsulate_script"] + " " + script_name

                        # print repo hash
                        commit = model_docker.sh(f"cd {dir_path}; git rev-parse HEAD || true")
                        print("======================================================")
                        print("MODEL REPO COMMIT: ", commit)
                        print("======================================================")

                        # Copy scripts to model directory
                        model_docker.sh(f"cp -vLR --preserve=all {dir_path}/. {model_dir}/")

                        # Prepare data if needed
                        if 'data' in model_info and model_info['data'] != "" and self.data:
                            self.data.prepare_data(model_info['data'], model_docker)

                        # Set permissions
                        model_docker.sh(f"chmod -R a+rw {model_dir}")

                        # Run the model
                        test_start_time = time.time()
                        print("Running model...")
                        
                        model_args = self.context.ctx.get("model_args", model_info["args"])
                        model_docker.sh(f"cd {model_dir} && {script_name} {model_args}", timeout=None)
                        
                        run_results["test_duration"] = time.time() - test_start_time
                        print(f"Test Duration: {run_results['test_duration']} seconds")

                        # Run post-scripts
                        if pre_encapsulate_post_scripts["post_scripts"]:
                            self.run_pre_post_script(model_docker, model_dir, pre_encapsulate_post_scripts["post_scripts"])

                        # Extract performance metrics from logs
                        # Look for performance data in the log output similar to original run_models.py
                        try:
                            # Check if multiple results file is specified in model_info
                            multiple_results = model_info.get("multiple_results", None)
                            
                            if multiple_results:
                                run_results["performance"] = multiple_results
                                # Validate multiple results file format
                                try:
                                    with open(multiple_results, 'r') as f:
                                        header = f.readline().strip().split(',')
                                        for line in f:
                                            row = line.strip().split(',')
                                            for col in row:
                                                if col == '':
                                                    run_results["performance"] = None
                                                    print("Error: Performance metric is empty in multiple results file.")
                                                    break
                                except Exception as e:
                                    print(f"Warning: Could not validate multiple results file: {e}")
                                    run_results["performance"] = None
                            else:
                                # Check if this follows the same pattern as original run_models
                                perf_regex = r".*performance:\s*\([+|-]?[0-9]*[.]?[0-9]*\(e[+|-]?[0-9]\+\)?\)\s*.*\s*"
                                metric_regex = r".*performance:\s*[+|-]?[0-9]*[.]?[0-9]*\(e[+|-]?[0-9]\+\)?\s*\(\w*\)\s*"
                                
                                # Extract from log file
                                try:
                                    run_results["performance"] = self.console.sh("cat " + log_file_path +
                                                                " | sed -n 's/" + perf_regex + "/\\1/p'")
                                    run_results["metric"] = self.console.sh("cat " + log_file_path +
                                                            " | sed -n 's/" + metric_regex + "/\\2/p'")
                                except Exception:
                                    pass  # Performance extraction is optional
                        except Exception as e:
                            print(f"Warning: Could not extract performance metrics: {e}")
                        
                        # Set status based on performance
                        run_results["status"] = 'SUCCESS' if run_results.get("performance") else 'FAILURE'
                        print(f"{model_info['name']} performance is {run_results.get('performance', 'N/A')} {run_results.get('metric', '')}")

                        # Generate performance results and update perf.csv
                        self.ensure_perf_csv_exists()
                        try:
                            # Create run details dictionary for CSV generation
                            run_details_dict = self.create_run_details_dict(model_info, build_info, run_results)
                            
                            # Handle multiple results if specified
                            multiple_results = model_info.get("multiple_results", None)
                            if multiple_results and run_results.get("status") == "SUCCESS":
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
                                print(f"Updated perf.csv with multiple results for {model_info['name']}")
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
                                print(f"Updated perf.csv with result for {model_info['name']}")
                                
                        except Exception as e:
                            print(f"Warning: Could not update perf.csv: {e}")

                        # Cleanup if not keeping alive
                        if not keep_alive:
                            model_docker.sh(f"rm -rf {model_dir}", timeout=240)
                        else:
                            model_docker.sh(f"chmod -R a+rw {model_dir}")
                            print(f"keep_alive specified; model_dir({model_dir}) is not removed")

                        # Explicitly delete model docker to stop the container
                        del model_docker
            
        except Exception as e:
            print("===== EXCEPTION =====")
            print("Exception: ", e)
            import traceback
            traceback.print_exc()
            print("=============== =====")
            run_results["status"] = "FAILURE"
            
            # Also update perf.csv for failures
            self.ensure_perf_csv_exists()
            try:
                # Create run details dictionary for failed runs
                run_details_dict = self.create_run_details_dict(model_info, build_info, run_results)
                
                # Generate exception result JSON
                with open("perf_entry.json", "w") as f:
                    json.dump(run_details_dict, f)
                
                # Update perf.csv with exception result
                update_perf_csv(
                    exception_result="perf_entry.json",
                    perf_csv=self.perf_csv_path,
                )
                print(f"Updated perf.csv with exception result for {model_info['name']}")
                
            except Exception as csv_e:
                print(f"Warning: Could not update perf.csv with exception: {csv_e}")
        
        
        return run_results
    
    def set_credentials(self, credentials: typing.Dict) -> None:
        """Set credentials for model execution.
        
        Args:
            credentials: Credentials dictionary
        """
        self.credentials = credentials

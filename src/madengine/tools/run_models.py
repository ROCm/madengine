# lint as: python3
###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
"""Module of running models on container.

This module contains the RunModels class, which is responsible for running models on container.
It also contains the RunDetails class, which is responsible for storing the performance results of a model.
"""
# built-in modules
import sys
import os
import json
import time
import re
import traceback
from contextlib import redirect_stdout, redirect_stderr
import warnings
import typing

# MADEngine modules
from madengine.core.console import Console
from madengine.core.context import Context
from madengine.core.dataprovider import Data
from madengine.core.docker import Docker
from madengine.utils.ops import PythonicTee, file_print, substring_found, find_and_replace_pattern
from madengine.core.constants import MAD_MINIO, MAD_AWS_S3
from madengine.core.constants import MODEL_DIR, PUBLIC_GITHUB_ROCM_KEY
from madengine.core.timeout import Timeout
from madengine.tools.update_perf_csv import update_perf_csv
from madengine.tools.csv_to_html import convert_csv_to_html
from madengine.tools.discover_models import DiscoverModels


class RunDetails:
    """Class to store the performance results of a model.

    Attributes:
        model (str): The model name.
        pipeline (str): The pipeline used.
        n_gpus (str): The number of GPUs used.
        training_precision (str): The training precision used.
        args (str): The arguments used.
        tags (str): The tags used.
        docker_file (str): The docker file used.
        base_docker (str): The base docker used.
        docker_sha (str): The docker SHA used.
        docker_image (str): The docker image used.
        git_commit (str): The git commit used.
        machine_name (str): The machine name used.
        gpu_architecture (str): The GPU architecture used.
        performance (str): The performance of the model.
        metric (str): The metric used.
        relative_change (str): The relative change in performance.
        status (str): The status of the model.
        build_duration (str): The build duration.
        test_duration (str): The test duration.
        dataname (str): The data name used.
        data_provider_type (str): The data provider type used.
        data_size (str): The size of the data.
        data_download_duration (str): The duration of data download.
        build_number (str): The CI build number.
        additional_docker_run_options (str): The additional options used for docker run.
    """

    # Avoiding @property for ease of code, add if needed.
    def __init__(self):
        self.model = ""
        self.pipeline = ""
        self.n_gpus = ""
        self.training_precision = ""
        self.args = ""
        self.tags = ""
        self.docker_file = ""
        self.base_docker = ""
        self.docker_sha = ""
        self.docker_image = ""
        self.git_commit = ""
        self.machine_name = ""
        self.gpu_architecture = ""
        self.performance = ""
        self.metric = ""
        self.relative_change = ""
        self.status = "FAILURE"
        self.build_duration = ""
        self.test_duration = ""
        self.dataname = ""
        self.data_provider_type = ""
        self.data_size = ""
        self.data_download_duration = ""
        self.build_number = ""
        self.additional_docker_run_options = ""

    def print_perf(self):
        """Print the performance results of a model.

        Method to print stage perf results of a model.
        """
        print(f"{self.model} performance is {self.performance} {self.metric}")

    # Exports all info in json format to json_name
    # multiple_results excludes the "model,performance,metric,status" keys
    # to handle results more generically regardless of the multiple_results csv being passed in
    def generate_json(self, json_name: str, multiple_results: bool = False) -> None:
        """Generate JSON file for performance results of a model.

        Args:
            json_name (str): The name of the JSON file.
            multiple_results (bool): The status of multiple results. Default is False.

        Raises:
            Exception: An error occurred while generating JSON file for performance results of a model.
        """
        keys_to_exclude = (
            {"model", "performance", "metric", "status"} if multiple_results else {}
        )
        attributes = vars(self)
        output_dict = {x: attributes[x] for x in attributes if x not in keys_to_exclude}
        with open(json_name, "w") as outfile:
            json.dump(output_dict, outfile)


class RunModels:
    """Class to run models on container."""

    def __init__(self, args):
        """Constructor of the RunModels class.

        Args:
            args: The command-line arguments.
        """
        self.return_status = True
        self.args = args
        self.console = Console(live_output=True)
        self.context = Context(
            additional_context=args.additional_context,
            additional_context_file=args.additional_context_file,
        )
        # check the data.json file exists
        data_json_file = args.data_config_file_name

        if not os.path.exists(data_json_file):
            self.data = None
        else:
            self.data = Data(
                self.context,
                filename=args.data_config_file_name,
                force_mirrorlocal=args.force_mirror_local,
            )
        self.creds = None
        print(f"Context is {self.context.ctx}")

    def get_base_prefix_compat(self):
        """Get base/real prefix, or sys.prefix if there is none.

        Returns:
            str: The base/real prefix or sys.prefix if there is none.
        """
        return (
            getattr(sys, "base_prefix", None)
            or getattr(sys, "real_prefix", None)
            or sys.prefix
        )

    def in_virtualenv(self) -> bool:
        """Check if the current environment is a virtual environment.

        Returns:
            bool: The status of the current environment.
        """
        return self.get_base_prefix_compat() != sys.prefix

    def clean_up_docker_container(self, is_cleaned: bool = False) -> None:
        """Clean up docker container."""
        if is_cleaned:
            self.console.sh("docker ps -a || true")
            self.console.sh("docker kill $(docker ps -q) || true")

        # get gpu vendor
        gpu_vendor = self.context.ctx["docker_env_vars"]["MAD_GPU_VENDOR"]
        # show gpu info
        if gpu_vendor.find("AMD") != -1:
            self.console.sh("/opt/rocm/bin/amd-smi || true")
        elif gpu_vendor.find("NVIDIA") != -1:
            self.console.sh("nvidia-smi -L || true")

    # Either return the dockercontext path from the model info
    # or use the default of the ./docker directory if it doesn't exist
    def get_context_path(self, info: typing.Dict) -> str:
        """Get the context path.

        Args:
            info: The model info dict.

        Returns:
            str: The context path.

        Raises:
            Exception: An error occurred while getting the context path.
        """
        if "dockercontext" in info and info["dockercontext"] != "":
            return info["dockercontext"]
        else:
            return "./docker"

    def get_build_arg(self, run_build_arg: typing.Dict = {}) -> str:
        """Get the build arguments.

        Args:
            run_build_arg: The run build arguments.

        Returns:
            str: The build arguments.

        Raises:
            RuntimeError: An error occurred while getting the build arguments.
        """
        # check if docker_build_arg is provided in context, if not return empty string.
        if not run_build_arg and "docker_build_arg" not in self.context.ctx:
            return ""

        build_args = ""
        for build_arg in self.context.ctx["docker_build_arg"].keys():
            build_args += (
                "--build-arg "
                + build_arg
                + "='"
                + self.context.ctx["docker_build_arg"][build_arg]
                + "' "
            )

        # add model cred
        if run_build_arg:
            for key, value in run_build_arg.items():
                build_args += "--build-arg " + key + "='" + value + "' "

        return build_args

    def apply_tools(
            self,
            pre_encapsulate_post_scripts: typing.Dict,
            run_env: typing.Dict
        ) -> None:
        """Apply tools to the model.

        Args:
            pre_encapsulate_post_scripts: The pre, encapsulate and post scripts.
            run_env: The run environment.

        Raises:
            Exception: An error occurred while applying tools to the model.
        """
        if "tools" not in self.context.ctx:
            return

        # read tool setting from tools.json
        tool_file = None
        with open(self.args.tools_json_file_name) as f:
            tool_file = json.load(f)

        # iterate over tools in context, apply tool settings.
        for ctx_tool_config in self.context.ctx["tools"]:
            tool_name = ctx_tool_config["name"]
            tool_config = tool_file["tools"][tool_name]

            if "cmd" in ctx_tool_config:
                tool_config.update({"cmd": ctx_tool_config["cmd"]})

            if "env_vars" in ctx_tool_config:
                for env_var in ctx_tool_config["env_vars"]:
                    tool_config["env_vars"].update({env_var: ctx_tool_config["env_vars"][env_var]})

            print(f"Selected Tool, {tool_name}. Configuration : {str(tool_config)}.")

            # setup tool before other existing scripts
            if "pre_scripts" in tool_config:
                pre_encapsulate_post_scripts["pre_scripts"] = (
                    tool_config["pre_scripts"] + pre_encapsulate_post_scripts["pre_scripts"]
                )
            # cleanup tool after other existing scripts
            if "post_scripts" in tool_config:
                pre_encapsulate_post_scripts["post_scripts"] += tool_config["post_scripts"]
            # warning: this will update existing keys from env or other tools
            if "env_vars" in tool_config:
                run_env.update(tool_config["env_vars"])
            if "cmd" in tool_config:
                # prepend encapsulate cmd
                pre_encapsulate_post_scripts["encapsulate_script"] = (
                    tool_config["cmd"] + " " + pre_encapsulate_post_scripts["encapsulate_script"]
                )

    def gather_system_env_details(
            self,
            pre_encapsulate_post_scripts: typing.Dict,
            container_identifier: str
        ) -> None:
        """Gather system environment details.

        Args:
            pre_encapsulate_post_scripts: The pre, encapsulate and post scripts.
            container_identifier: The container name or image identifier (replaces model_name for container-specific env files).

        Returns:
            None

        Raises:
            Exception: An error occurred while gathering system environment details.

        Note:
            This function is used to gather system environment details.
            Uses container_identifier instead of model_name to ensure multiple models
            running on the same container share the same environment file.
        """
        # initialize pre_env_details
        pre_env_details = {}
        pre_env_details["path"] = "scripts/common/pre_scripts/run_rocenv_tool.sh"
        pre_env_details["args"] = container_identifier + "_env"
        pre_encapsulate_post_scripts["pre_scripts"].append(pre_env_details)
        print(f"pre encap post scripts: {pre_encapsulate_post_scripts}")

    def copy_scripts(self) -> None:
        """Copy scripts to the model directory."""
        scripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
        print(f"Package path: {scripts_path}")
        # copy the scripts to the model directory
        self.console.sh(f"cp -vLR --preserve=all {scripts_path} .")
        print(f"Scripts copied to {os.getcwd()}/scripts")

    def cleanup(self) -> None:
        """Cleanup the scripts/common directory."""
        # check the directory exists
        if os.path.exists("scripts/common"):
            # check tools.json exists in scripts/common directory
            if os.path.exists("scripts/common/tools.json"):
                # remove the scripts/common/tools.json file
                self.console.sh("rm -rf scripts/common/tools.json")
            # check test_echo.sh exists in scripts/common directory
            if os.path.exists("scripts/common/test_echo.sh"):
                # remove the scripts/common/test_echo.sh file
                self.console.sh("rm -rf scripts/common/test_echo.sh")
            # check folder pre_scripts exists in scripts/common directory
            if os.path.exists("scripts/common/pre_scripts"):
                # remove the scripts/common/pre_scripts directory
                self.console.sh("chmod -R +w scripts/common/pre_scripts 2>/dev/null || true")
                self.console.sh("rm -rf scripts/common/pre_scripts")
            # check folder post_scripts exists in scripts/common directory
            if os.path.exists("scripts/common/post_scripts"):
                # remove the scripts/common/post_scripts directory
                self.console.sh("chmod -R +w scripts/common/post_scripts 2>/dev/null || true")
                self.console.sh("rm -rf scripts/common/post_scripts")
            if os.path.exists("scripts/common/tools"):
                # remove the scripts/common/tools directory with robust permission fixes
                self.console.sh("find scripts/common/tools -type f -exec chmod +w {} \\; 2>/dev/null || true")
                self.console.sh("find scripts/common/tools -type d -exec chmod +wx {} \\; 2>/dev/null || true")
                self.console.sh("rm -rf scripts/common/tools 2>/dev/null || sudo rm -rf scripts/common/tools", canFail=True)
            print(f"scripts/common directory has been cleaned up.")

    def get_gpu_arg(self, requested_gpus: str) -> str:
        """Get the GPU arguments.

        Args:
            requested_gpus: The requested GPUs.

        Returns:
            str: The GPU arguments.

        Raises:
            RuntimeError: An error occurred while getting the GPU arguments.
        """
        # initialize gpu arg to empty string.
        gpu_arg = ""
        # get gpu vendor from context, if not raise exception.
        gpu_vendor = self.context.ctx["docker_env_vars"]["MAD_GPU_VENDOR"]
        n_system_gpus = self.context.ctx['docker_env_vars']['MAD_SYSTEM_NGPUS']
        gpu_strings = self.context.ctx["docker_gpus"].split(",")

        # parsing gpu string, example: '{0-4}' -> [0,1,2,3,4]
        docker_gpus = []
        # iterate over the list of gpu strings, split range and append to docker_gpus.
        for gpu_string in gpu_strings:
            # check if gpu string has range, if so split and append to docker_gpus.
            if '-' in gpu_string:
                gpu_range = gpu_string.split('-')
                docker_gpus += [item for item in range(int(gpu_range[0]),int(gpu_range[1])+1)]
            else:
                docker_gpus.append(int(gpu_string))
        # sort docker_gpus
        docker_gpus.sort()

        # Check GPU range is valid for system
        if requested_gpus == "-1":
            print("NGPUS requested is ALL (" + ','.join(map(str, docker_gpus)) + ")." )
            requested_gpus = len(docker_gpus)

        print("NGPUS requested is " + str(requested_gpus) + " out of " + str(n_system_gpus) )

        if int(requested_gpus) > int(n_system_gpus) or int(requested_gpus) > len(docker_gpus):
            raise RuntimeError("Too many gpus requested(" + str(requested_gpus) + "). System has " + str(n_system_gpus) + " gpus. Context has " + str(len(docker_gpus)) + " gpus." )

        # Exposing number of requested gpus
        self.context.ctx['docker_env_vars']['MAD_RUNTIME_NGPUS'] = str(requested_gpus)

        # Create docker arg to assign requested GPUs
        if gpu_vendor.find("AMD") != -1:
            gpu_arg = '--device=/dev/kfd '

            gpu_renderDs = self.context.ctx['gpu_renderDs']
            if gpu_renderDs is not None:
                for idx in range(0, int(requested_gpus)):
                    gpu_arg += "--device=/dev/dri/renderD" + str(gpu_renderDs[docker_gpus[idx]]) + " "

        elif gpu_vendor.find("NVIDIA") != -1:
            gpu_str = ""
            for idx in range(0, int(requested_gpus)):
                gpu_str += str( docker_gpus[idx] ) + ","
            gpu_arg += "--gpus '\"device=" + gpu_str + "\"' "
        else:
            raise RuntimeError("Unable to determine gpu vendor.")

        print(f"GPU arguments: {gpu_arg}")

        return gpu_arg

    def get_cpu_arg(self) -> str:
        """Get the CPU arguments.

        Returns:
            str: The CPU arguments.

        Raises:
            RuntimeError: An error occurred while getting the CPU arguments.
        """
        # get docker_cpus from context, if not return empty string.
        if "docker_cpus" not in self.context.ctx:
            return ""
        # get docker_cpus from context, remove spaces and return cpu arguments.
        cpus = self.context.ctx["docker_cpus"]
        cpus = cpus.replace(" ","")
        return "--cpuset-cpus " + cpus + " "

    def get_env_arg(self, run_env: typing.Dict) -> str:
        """Get the environment arguments.

        Args:
            run_env: The run environment.

        Returns:
            str: The environment arguments.

        Raises:
            RuntimeError: An error occurred while getting the environment arguments.
        """
        # initialize env_args to empty string.
        env_args = ""

        # aggregate environment variables
        if run_env:
            for env_arg in run_env:
                env_args += "--env " + env_arg + "='" + str(run_env[env_arg]) + "' "

        # get docker_env_vars from context, if not return env_args.
        if "docker_env_vars" in self.context.ctx:
            for env_arg in self.context.ctx["docker_env_vars"].keys():
                env_args += "--env " + env_arg + "='" + str(self.context.ctx["docker_env_vars"][env_arg]) + "' "

        print(f"Env arguments: {env_args}")
        return env_args

    def get_mount_arg(self, mount_datapaths: typing.List) -> str:
        """Get the mount arguments.

        Args:
            mount_datapaths: The mount data paths.

        Returns:
            str: The mount arguments.

        Raises:
            RuntimeError: An error occurred while getting the mount arguments.
        """
        # initialize mount_args to empty string.
        mount_args = ""
        # get mount_datapaths from context, if not return mount_args.
        if mount_datapaths:
            # iterate over mount_datapaths, if mount_datapath is not empty, mount data.
            for mount_datapath in mount_datapaths:
                if mount_datapath:
                    # uses --mount to enforce existence of parent directory; data is mounted readonly by default
                    mount_args += "-v " + mount_datapath["path"] + ":" +  mount_datapath["home"]
                    if "readwrite" in mount_datapath and mount_datapath["readwrite"] == 'true':
                        mount_args += " "
                    else:
                        mount_args += ":ro "

        if "docker_mounts" not in self.context.ctx:
            return mount_args

        # get docker_mounts from context, if not return mount_args.
        for mount_arg in self.context.ctx["docker_mounts"].keys():
            mount_args += "-v " + self.context.ctx["docker_mounts"][mount_arg] + ":" + mount_arg + " "

        return mount_args

    def run_pre_post_script(self, model_docker, model_dir, pre_post):
        for script in pre_post:
            script_path = script["path"].strip()
            model_docker.sh("cp -vLR --preserve=all " + script_path + " " + model_dir, timeout=600)
            script_name = os.path.basename(script_path)
            script_args = ""
            if "args" in script:
                script_args = script["args"]
                script_args.strip()
            model_docker.sh("cd " + model_dir + " && bash " + script_name + " " + script_args , timeout=600)

    def _cleanup_model_directory(self, model_docker, model_dir: str, max_retries: int = 3, retry_delay: float = 2.0) -> None:
        """Robustly cleanup model directory with retry logic.
        
        This method handles cleanup failures that can occur due to:
        - File system sync delays
        - Background processes holding file handles
        - Permission issues
        - Race conditions between test completion and cleanup
        
        Args:
            model_docker: The Docker instance for executing commands
            model_dir: The directory path to remove
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay in seconds between retries (default: 2.0)
        
        Raises:
            RuntimeError: If cleanup fails after all retry attempts (logged as warning, not raised)
        """
        import time
        
        print(f"Cleaning up model directory: {model_dir}")
        
        for attempt in range(max_retries):
            try:
                # First attempt: Try killing any processes that might be holding files
                if attempt > 0:
                    print(f"Cleanup attempt {attempt + 1}/{max_retries}...")
                    # Kill any processes using files in the directory (ignore errors)
                    model_docker.sh(f"fuser -k {model_dir} 2>/dev/null || true", timeout=30)
                    time.sleep(retry_delay)
                    
                    # Try to fix permissions before removal
                    try:
                        model_docker.sh(f"chmod -R 777 {model_dir} 2>/dev/null || true", timeout=60)
                    except RuntimeError:
                        # Permission change failed, but continue anyway
                        pass
                
                # Attempt to remove the directory
                model_docker.sh(f"rm -rf {model_dir}", timeout=240)
                print(f"Successfully cleaned up {model_dir}")
                return
                
            except RuntimeError as e:
                error_msg = str(e)
                print(f"Cleanup attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < max_retries - 1:
                    # Not the last attempt, wait and retry
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    # Last attempt failed, log warning but don't fail the build
                    # The test itself succeeded, cleanup failure is not critical
                    print("=" * 60)
                    print("WARNING: Failed to cleanup model directory after all retries")
                    print(f"Directory: {model_dir}")
                    print(f"Error: {error_msg}")
                    print("This is not a critical failure - the test completed successfully")
                    print("The directory will be cleaned up when the container is removed")
                    print("=" * 60)
                    # Don't raise the exception - cleanup failure shouldn't fail the build

    def run_model_impl(
        self, info: typing.Dict, dockerfile: str, run_details: RunDetails
    ) -> None:
        """Handler of running model

        Args:
            info: The model information.
            dockerfile: The docker file.
            run_details: The run details.
        """
        print("")
        print(f"Running model {info['name']} on container built from {dockerfile}")

        if "MAD_CONTAINER_IMAGE" not in self.context.ctx:
            # build docker image
            image_docker_name = (
                info["name"].replace("/", "_").lower() # replace / with _ for models in scripts/somedir/ from madengine discover
                + "_"
                + os.path.basename(dockerfile).replace(".Dockerfile", "")
            )
            run_details.docker_file = dockerfile

            # get docker context from dockerfile
            docker_context = self.get_context_path(info)

            run_build_arg = {}
            if "cred" in info and info["cred"] != "":
                if info["cred"] not in self.creds:
                    raise RuntimeError(
                        "Credentials("
                        + info["cred"]
                        + ") to run model not found in credential.json; Please contact the model owner, "
                        + info["owner"]
                        + "."
                    )
                # add cred to build args
                for key_cred, value_cred in self.creds[info["cred"]].items():
                    run_build_arg[info["cred"] + "_" + key_cred.upper()] = value_cred

            # get build args from context
            build_args = self.get_build_arg(run_build_arg)

            use_cache_str = ""
            if self.args.clean_docker_cache:
                use_cache_str = "--no-cache"

            # build docker container
            print(f"Building Docker image...")
            build_start_time = time.time()
            # get docker image name
            run_details.docker_image = "ci-" + image_docker_name
            # get container name
            container_name = "container_" + re.sub('.*:','', image_docker_name) # remove docker container hub details

            ## Note: --network=host added to fix issue on CentOS+FBK kernel, where iptables is not available
            self.console.sh(
                "docker build "
                + use_cache_str
                + " --network=host "
                + " -t "
                + run_details.docker_image
                + " --pull -f "
                + dockerfile
                + " "
                + build_args
                + " "
                + docker_context,
                timeout=None,
            )
            run_details.build_duration = time.time() - build_start_time
            print(f"Build Duration: {run_details.build_duration} seconds")

            print(f"MAD_CONTAINER_IMAGE is {run_details.docker_image}")

            # print base docker image info
            if (
                "docker_build_arg" in self.context.ctx
                and "BASE_DOCKER" in self.context.ctx["docker_build_arg"]
            ):
                run_details.base_docker = self.context.ctx["docker_build_arg"]["BASE_DOCKER"]
            else:
                run_details.base_docker = self.console.sh(
                    "grep '^ARG BASE_DOCKER=' "
                    + dockerfile
                    + " | sed -E 's/ARG BASE_DOCKER=//g'"
                )
            print(f"BASE DOCKER is {run_details.base_docker}")

            # print base docker image digest
            run_details.docker_sha = self.console.sh("docker manifest inspect " + run_details.base_docker + " | grep digest | head -n 1 | cut -d \\\" -f 4")
            print(f"BASE DOCKER SHA is {run_details.docker_sha}")

        else:
            container_name = "container_" + self.context.ctx["MAD_CONTAINER_IMAGE"].replace("/", "_").replace(":", "_")
            run_details.docker_image = self.context.ctx["MAD_CONTAINER_IMAGE"]

            print(f"MAD_CONTAINER_IMAGE is {run_details.docker_image}")
            print(f"Warning: User override MAD_CONTAINER_IMAGE. Model support on image not guaranteed.")

        # prepare docker run options
        gpu_vendor = self.context.ctx["gpu_vendor"]
        docker_options = ""

        if gpu_vendor.find("AMD") != -1:
            docker_options = "--network host -u root --group-add video \
            --cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined --ipc=host "
        elif gpu_vendor.find("NVIDIA") != -1:
            docker_options = "--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --cap-add SYS_NICE --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined  --network host -u root --ipc=host "
        else:
            raise RuntimeError("Unable to determine gpu vendor.")

        # initialize pre, encapsulate and post scripts
        pre_encapsulate_post_scripts = {"pre_scripts": [], "encapsulate_script": "", "post_scripts": []}

        if "pre_scripts" in self.context.ctx:
            pre_encapsulate_post_scripts["pre_scripts"] = self.context.ctx["pre_scripts"]

        if "post_scripts" in self.context.ctx:
            pre_encapsulate_post_scripts["post_scripts"] = self.context.ctx["post_scripts"]

        if "encapsulate_script" in self.context.ctx:
            pre_encapsulate_post_scripts["encapsulate_script"] = self.context.ctx["encapsulate_script"]

        # get docker run options
        docker_options += "--env MAD_MODEL_NAME='" + info["name"] + "' "
        # Since we are doing Jenkins level environment collection in the docker container, pass in the jenkins build number.
        docker_options += f"--env JENKINS_BUILD_NUMBER='{os.environ.get('BUILD_NUMBER','0')}' "

        # gather data
        # TODO: probably can use context.ctx instead of another dictionary like run_env here
        run_env = {}
        mount_datapaths = None

        if "data" in info and info["data"] != "":
            mount_datapaths = self.data.get_mountpaths(info["data"])
            model_dataenv = self.data.get_env(info["data"])

            if model_dataenv is not None:
                run_env.update(model_dataenv)

            run_env["MAD_DATANAME"] = info["data"]

        if "cred" in info and info["cred"] != "":
            if info["cred"] not in self.creds:
                raise RuntimeError(
                    "Credentials("
                    + info["cred"]
                    + ") to run model not found in credential.json; Please contact the model owner, "
                    + info["owner"]
                    + "."
                )
            # add cred to run_env
            for key_cred, value_cred in self.creds[info["cred"]].items():
                run_env[info["cred"] + "_" + key_cred.upper()] = value_cred

        self.apply_tools(pre_encapsulate_post_scripts, run_env)

        docker_options += self.get_gpu_arg(info["n_gpus"])
        docker_options += self.get_cpu_arg()

        # Must set env vars and mounts at the end
        docker_options += self.get_env_arg(run_env)
        docker_options += self.get_mount_arg(mount_datapaths)
        docker_options += f" {run_details.additional_docker_run_options}"

        # if --shm-size is set, remove --ipc=host
        if "SHM_SIZE" in self.context.ctx:
            docker_options = docker_options.replace("--ipc=host", "")

        print(docker_options)

        # get machine name
        run_details.machine_name = self.console.sh("hostname")
        print(f"MACHINE NAME is {run_details.machine_name}")

        # set timeout
        timeout = 7200  # default 2 hrs
        if "timeout" in info:
            timeout = info["timeout"]

        if self.args.timeout >= 0:
            timeout = self.args.timeout

        print(f"Setting timeout to {str(timeout)} seconds.")

        with Timeout(timeout):
            print(f"")
            model_docker = Docker(run_details.docker_image, container_name, docker_options, keep_alive=self.args.keep_alive, console=self.console)
            # check that user is root
            whoami = model_docker.sh("whoami")
            print( "USER is " + whoami )

            # echo gpu smi info
            if gpu_vendor.find("AMD") != -1:
                smi = model_docker.sh("/opt/rocm/bin/amd-smi || true")
            elif gpu_vendor.find("NVIDIA") != -1:
                smi = model_docker.sh("/usr/bin/nvidia-smi || true")
            else:
                raise RuntimeError("Unable to determine gpu vendor.")

            # clean up previous model run
            model_dir = "run_directory"
            if "url" in info and info["url"] != "":
                # model_dir is set to string after the last forwardslash in url field
                # adding for url field with and without trailing forwardslash (/)
                model_dir = info['url'].rstrip('/').split('/')[-1]

                # Validate model_dir to make sure there are no special characters
                special_char = r'[^a-zA-Z0-9\-\_]'  # allow hyphen and underscore
                if re.search(special_char, model_dir) is not None:
                    warnings.warn("Model url contains special character. Fix url.")

            model_docker.sh("rm -rf " + model_dir, timeout=240)

            # set safe.directory for workspace
            model_docker.sh("git config --global --add safe.directory /myworkspace")

            # clone model repo
            if "url" in info and info["url"] != "":
                if "cred" in info and info["cred"] != "":
                    print(f"Using cred for {info['cred']}")

                    if info["cred"] not in self.creds:
                        raise RuntimeError("Credentials(" + info["cred"] + ") to run model not found in credential.json; Please contact the model owner, " + info["owner"] + ".")

                    if info['url'].startswith('ssh://'):
                        model_docker.sh("git -c core.sshCommand='ssh -l  " + self.creds[ info["cred"] ]["username"] +
                                    " -i " + self.creds[ info["cred"] ]["ssh_key_file"] + " -o IdentitiesOnly=yes " +
                                    " -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no' " +
                                    " clone " + info['url'], timeout=240 )
                    else:   # http or https
                        model_docker.sh("git clone -c credential.helper='!f() { echo username=" + self.creds[ info["cred"] ]["username"] + \
                                    "; echo password=" + self.creds[ info["cred"] ]["password"] + "; };f' " + \
                                    info['url'], timeout=240, secret="git clone " + info['url'] )
                else:
                    model_docker.sh("git clone " + info["url"], timeout=240)

                # set safe.directory for model directory
                model_docker.sh("git config --global --add safe.directory /myworkspace/" + model_dir )

                # echo git commit
                run_details.git_commit = model_docker.sh("cd "+ model_dir + " && git rev-parse HEAD")
                print(f"MODEL GIT COMMIT is {run_details.git_commit}")

                # update submodule
                model_docker.sh("cd "+ model_dir + "; git submodule update --init --recursive")
            else:
                model_docker.sh("mkdir -p " + model_dir)

            # add system environment collection script to pre_scripts
            if self.args.generate_sys_env_details or self.context.ctx.get("gen_sys_env_details"):
                self.gather_system_env_details(pre_encapsulate_post_scripts, container_name)
            # run pre_scripts
            if pre_encapsulate_post_scripts["pre_scripts"]:
                self.run_pre_post_script(model_docker, model_dir, pre_encapsulate_post_scripts["pre_scripts"])

            scripts_arg = info['scripts']
            dir_path = None
            script_name = None
            if scripts_arg.endswith(".sh"):
                dir_path = os.path.dirname(scripts_arg)
                script_name = "bash " + os.path.basename(scripts_arg)
            else:
                dir_path = info['scripts']
                script_name = "bash run.sh"

            # add script_prepend_cmd
            script_name = pre_encapsulate_post_scripts["encapsulate_script"] + " " + script_name

            # print repo hash
            commit = model_docker.sh("cd "+ dir_path +"; git rev-parse HEAD || true  ")
            print("======================================================")
            print("MODEL REPO COMMIT: ", commit )
            print("======================================================")

            # copy scripts to model directory
            model_docker.sh("cp -vLR --preserve=all "+ dir_path +"/. "+ model_dir +"/")

            # prepare data inside container
            if 'data' in info and info['data'] != "":
                self.data.prepare_data( info['data'], model_docker )
                # Capture data provider information from selected_data_provider
                if hasattr(self.data, 'selected_data_provider') and self.data.selected_data_provider:
                    if 'dataname' in self.data.selected_data_provider:
                        run_details.dataname = self.data.selected_data_provider['dataname']
                    if 'data_provider_type' in self.data.selected_data_provider:
                        run_details.data_provider_type = self.data.selected_data_provider['data_provider_type']
                    if 'duration' in self.data.selected_data_provider:
                        run_details.data_download_duration = self.data.selected_data_provider['duration']
                    if 'size' in self.data.selected_data_provider:
                        run_details.data_size = self.data.selected_data_provider['size']
                    print(f"Data Provider Details: {run_details.dataname}, {run_details.data_provider_type}, {run_details.data_size}, {run_details.data_download_duration}s")

            selected_data_provider = {
                "node_name": run_details.machine_name,
                "build_number": os.environ.get('BUILD_NUMBER','0'),
                "model_name": info["name"] if "name" in info else ""
            }

            # Set build number in run_details
            run_details.build_number = os.environ.get('BUILD_NUMBER','0')

            print(f"Build Info::{selected_data_provider}")

            # keep model_dir as universally rw
            model_docker.sh("chmod -R a+rw " + model_dir)

            # run model
            test_start_time = time.time()
            if not self.args.skip_model_run:
                print("Running model...")
                if "model_args" in self.context.ctx:
                    model_docker.sh(
                        "cd "
                        + model_dir
                        + " && "
                        + script_name
                        + " "
                        + self.context.ctx["model_args"],
                        timeout=None,
                    )
                else:
                    model_docker.sh(
                        "cd " + model_dir + " && " + script_name + " " + info["args"],
                        timeout=None,
                    )
            else:
                print("Skipping model run")
                print(
                    "To run model: "
                    + "cd "
                    + model_dir
                    + " && "
                    + script_name
                    + " "
                    + info["args"]
                )

            run_details.test_duration = time.time() - test_start_time
            print("Test Duration: {} seconds".format(run_details.test_duration))

            # run post_scripts
            if pre_encapsulate_post_scripts["post_scripts"]:
                self.run_pre_post_script(model_docker, model_dir, pre_encapsulate_post_scripts["post_scripts"])

            # remove model directory
            if not self.args.keep_alive and not self.args.keep_model_dir:
                self._cleanup_model_directory(model_docker, model_dir)
            else:
                model_docker.sh("chmod -R a+rw " + model_dir)
                print("keep_alive is specified; model_dir(" + model_dir + ") is not removed")

        # explicitly delete model docker to stop the container, without waiting for the in-built garbage collector
        del model_docker

    def run_model(self, model_info: typing.Dict) -> bool:
        """Run model on container.

        Args:
            model_info: The model information.

        Returns:
            bool: The status of running model on container.

        Raises:
            Exception: An error occurred while running model on container.
        """
        print(f"Running model {model_info['name']} with {model_info}")

        # set default values if model run fails
        run_details = RunDetails()

        run_details.model = model_info["name"]
        run_details.n_gpus = model_info["n_gpus"]
        run_details.training_precision = model_info["training_precision"]
        run_details.args = model_info["args"]
        run_details.tags = model_info["tags"]
        run_details.additional_docker_run_options = model_info.get("additional_docker_run_options", "")
        # gets pipeline variable from jenkinsfile, default value is none
        run_details.pipeline = os.environ.get("pipeline")
        # Taking gpu arch from context assumes the host image and container have the same gpu arch.
        # Environment variable updates for MAD Public CI
        run_details.gpu_architecture = self.context.ctx["docker_env_vars"]["MAD_SYSTEM_GPU_ARCHITECTURE"]

        # Check the setting of shared memory size
        if "SHM_SIZE" in self.context.ctx:
            shm_size = self.context.ctx["SHM_SIZE"]
            if shm_size:
                run_details.additional_docker_run_options += f" --shm-size={shm_size}"
                print(f"Using SHM_SIZE from context: {shm_size}")

        # Check if model is deprecated
        if model_info.get("is_deprecated", False):
            print(f"WARNING: Model {model_info['name']} has been deprecated.")
            if self.args.ignore_deprecated_flag:
                print(f"WARNING: Running deprecated model {model_info['name']} due to --ignore-deprecated-flag.")
            else:
                print(f"WARNING: Skipping execution. No bypass flags mentioned.")
                return True  # exit early

        # check if model is supported on current gpu architecture, if not skip.
        list_skip_gpu_arch = []
        if (
            "skip_gpu_arch" in model_info
            and model_info["skip_gpu_arch"]
            and not self.args.disable_skip_gpu_arch
        ):
            list_skip_gpu_arch = model_info["skip_gpu_arch"].replace(" ", "").split(",")

        sys_gpu_arch = run_details.gpu_architecture
        if sys_gpu_arch and "NVIDIA" in sys_gpu_arch:
            sys_gpu_arch = sys_gpu_arch.split()[1]

        if list_skip_gpu_arch and sys_gpu_arch and sys_gpu_arch in list_skip_gpu_arch:
            print(
                f"Skipping model {run_details.model} as it is not supported on {run_details.gpu_architecture} architecture."
            )
            # add result to output
            self.return_status = True
            run_details.status = "SKIPPED"
            # generate exception for testing
            run_details.generate_json("perf_entry.json")
            update_perf_csv(exception_result="perf_entry.json", perf_csv=self.args.output)
        else:
            print(
                f"Running model {run_details.model} on {run_details.gpu_architecture} architecture."
            )

            try:
                # clean up docker
                self.clean_up_docker_container()

                # find dockerfiles, read their context and filter based on current context
                all_dockerfiles = self.console.sh(
                    "ls " + model_info["dockerfile"] + ".*"
                ).split("\n")

                dockerfiles = {}
                for cur_docker_file in all_dockerfiles:
                    # get context of dockerfile
                    dockerfiles[cur_docker_file] = self.console.sh(
                        "head -n5 "
                        + cur_docker_file
                        + " | grep '# CONTEXT ' | sed 's/# CONTEXT //g'"
                    )

                # filter dockerfiles based on context
                dockerfiles = self.context.filter(dockerfiles)
                print(f"FILTERED dockerfiles are {dockerfiles}")

                # check if dockerfiles are found, if not raise exception.
                if not dockerfiles:
                    raise Exception("No dockerfiles matching context found for model " + run_details.model)

                # run dockerfiles
                for cur_docker_file in dockerfiles.keys():
                    # reset build-specific run details for each dockerfile
                    run_details.docker_file = ""
                    run_details.base_docker = ""
                    run_details.docker_sha = ""
                    run_details.docker_image = ""
                    run_details.performance = ""
                    run_details.metric = ""
                    run_details.status = "FAILURE"
                    run_details.build_duration = ""
                    run_details.test_duration = ""

                    try:
                        # generate exception for testing
                        if model_info['args'] == "--exception":
                            raise Exception("Exception test!")

                        print(f"Processing Dockerfile: {cur_docker_file}")
                        # get base docker image
                        cur_docker_file_basename = os.path.basename(cur_docker_file)
                        # set log file path
                        log_file_path = (
                            run_details.model
                            + "_"
                            + cur_docker_file_basename.replace(".Dockerfile", "")
                            + ".live.log"
                        )
                        # Replace / with _ in log file path for models from discovery which use '/' as a separator
                        log_file_path = log_file_path.replace("/", "_")

                        with open(log_file_path, mode="w", buffering=1) as outlog:
                            with redirect_stdout(PythonicTee(outlog, self.args.live_output)), redirect_stderr(PythonicTee(outlog, self.args.live_output)):
                                self.run_model_impl(model_info, cur_docker_file, run_details)

                        if self.args.skip_model_run:
                            # move to next dockerfile
                            continue

                        # Check if we are looking for a single result or multiple.
                        multiple_results = (None if "multiple_results" not in model_info else model_info["multiple_results"])

                        # get performance metric from log
                        if multiple_results:
                            run_details.performance = multiple_results

                        else:
                            perf_regex = ".*performance:\\s*\\([+|-]\?[0-9]*[.]\\?[0-9]*\(e[+|-]\?[0-9]\+\)\?\\)\\s*.*\\s*"
                            run_details.performance = self.console.sh("cat " + log_file_path +
                                                        " | sed -n 's/" + perf_regex + "/\\1/p'")

                            metric_regex = ".*performance:\\s*[+|-]\?[0-9]*[.]\\?[0-9]*\(e[+|-]\?[0-9]\+\)\?\\s*\\(\\w*\\)\\s*"
                            run_details.metric = self.console.sh("cat " + log_file_path +
                                                    " | sed -n 's/" + metric_regex + "/\\2/p'")

                        # check if model passed or failed
                        run_details.status = 'SUCCESS' if run_details.performance else 'FAILURE'

                        # print stage perf results
                        run_details.print_perf()

                        # add result to output
                        if multiple_results:
                            run_details.generate_json("common_info.json", multiple_results=True)
                            update_perf_csv(
                                multiple_results=model_info['multiple_results'],
                                perf_csv=self.args.output,
                                model_name=run_details.model,
                                common_info="common_info.json",
                            )
                        else:
                            run_details.generate_json("perf_entry.json")
                            update_perf_csv(
                                single_result="perf_entry.json",
                                perf_csv=self.args.output,
                            )

                        self.return_status &= (run_details.status == 'SUCCESS')

                    except Exception as e:
                        self.return_status = False

                        print( "===== EXCEPTION =====")
                        print( "Exception: ", e )
                        traceback.print_exc()
                        print( "=============== =====")
                        run_details.status = "FAILURE"
                        run_details.generate_json("perf_entry.json")
                        update_perf_csv(
                            exception_result="perf_entry.json",
                            perf_csv=self.args.output,
                        )

            except Exception as e:
                self.return_status = False

                print( "===== EXCEPTION =====")
                print( "Exception: ", e )
                traceback.print_exc()
                print( "=============== =====")
                run_details.status = "FAILURE"
                run_details.generate_json("perf_entry.json")
                update_perf_csv(
                    exception_result="perf_entry.json",
                    perf_csv=self.args.output,
                )

        return self.return_status

    def run(self) -> bool:
        """Main flow of running model.

        Returns:
            bool: The status of running models on container.

        Raises:
            Exception: An error occurred while running models on container.
        """
        print(f"Running models with args {self.args}")

        self.console.sh("echo 'MAD Run Models'")
        # show node rocm info
        host_os = self.context.ctx["host_os"]

        if host_os.find("HOST_UBUNTU") != -1:
            print(self.console.sh("apt show rocm-libs -a", canFail=True))
        elif host_os.find("HOST_CENTOS") != -1:
            print(self.console.sh("yum info rocm-libs"))
        elif host_os.find("HOST_SLES") != -1:
            print(self.console.sh("zypper info rocm-libs"))
        elif host_os.find("HOST_AZURE") != -1:
            print(self.console.sh("tdnf info rocm-libs"))
        else:
            print("ERROR: Unable to detect host OS.")
            self.return_status = False
            return self.return_status

        # get credentials
        try:
            # MADEngine update
            credential_file = "credential.json"
            # read credentials
            with open(credential_file) as f:
                self.creds = json.load(f)

            print(f"Credentials: {self.creds}")

        except Exception as e:
            print(f"Exception encountered reading credential.json. {e}, ignoring ...")

        # copy scripts to model directory
        self.copy_scripts()

        discover_models = DiscoverModels(args=self.args)
        models = discover_models.run()

        # create performance csv
        if not os.path.exists(self.args.output):
            file_print(
                "model, n_gpus, training_precision, pipeline, args, tags, docker_file, base_docker, docker_sha, docker_image, git_commit, machine_name, gpu_architecture, performance, metric, relative_change, status, build_duration, test_duration, dataname, data_provider_type, data_size, data_download_duration, build_number, additional_docker_run_options",
                filename=self.args.output,
                mode="w",
            )

        for model_info in models:
            # Run model
            self.return_status &= self.run_model(model_info)

        # cleanup the model directory
        self.cleanup()
        # convert output csv to html
        print("Converting output csv to html...")
        convert_csv_to_html(file_path=self.args.output)

        if self.return_status:
            print("All models ran successfully.")
        else:
            print( "===== EXCEPTION =====")
            print("Some models failed to run.")

        return self.return_status

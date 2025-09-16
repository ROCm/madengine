#!/usr/bin/env python3
"""Module of context class.

This module contains the class to determine context.

Classes:
    Context: Class to determine context.

Functions:
    update_dict: Update dictionary.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import ast
import json
import collections.abc
import os
import re
import typing
# third-party modules
from madengine.core.console import Console


def update_dict(d: typing.Dict, u: typing.Dict) -> typing.Dict:
    """Update dictionary.
    
    Args:
        d: The dictionary.
        u: The update dictionary.
    
    Returns:
        dict: The updated dictionary.
    """
    # Update a dictionary with another dictionary, recursively.
    for k, v in u.items():
        # if the value is a dictionary, recursively update it, otherwise update the value.
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class Context:
    """Class to determine context.
    
    Attributes:
        console: The console.
        ctx: The context.
    
    Methods:
        get_ctx_test: Get context test.
        get_gpu_vendor: Get GPU vendor.
        get_host_os: Get host OS.
        get_numa_balancing: Get NUMA balancing.
        get_system_ngpus: Get system number of GPUs.
        get_system_gpu_architecture: Get system GPU architecture.
        get_docker_gpus: Get Docker GPUs.
        get_gpu_renderD_nodes: Get GPU renderD nodes.
        set_multi_node_runner: Sets multi-node runner context.
        filter: Filter.
    """
    def __init__(
            self, 
            additional_context: str=None, 
            additional_context_file: str=None
        ) -> None:
        """Constructor of the Context class.
        
        Args:
            additional_context: The additional context.
            additional_context_file: The additional context file.
            
        Raises:
            RuntimeError: If the GPU vendor is not detected.
            RuntimeError: If the GPU architecture is not detected.
        """
        # Initialize the console
        self.console = Console()

        # Initialize the context
        self.ctx = {}
        self.ctx["ctx_test"] = self.get_ctx_test()
        self.ctx["host_os"] = self.get_host_os()
        self.ctx["numa_balancing"] = self.get_numa_balancing()
        # Check if NUMA balancing is enabled or disabled.
        if self.ctx["numa_balancing"] == "1":
            print("Warning: numa balancing is ON ...")
        elif self.ctx["numa_balancing"] == "0":
            print("Warning: numa balancing is OFF ...")
        else:
            print("Warning: unknown numa balancing setup ...")

        # Keeping gpu_vendor for filterning purposes, if we filter using file names we can get rid of this attribute.
        self.ctx["gpu_vendor"] = self.get_gpu_vendor()

        # Initialize the docker context
        self.ctx["docker_env_vars"] = {}
        self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"] = self.ctx["gpu_vendor"]
        self.ctx["docker_env_vars"]["MAD_SYSTEM_NGPUS"] = self.get_system_ngpus()
        self.ctx["docker_env_vars"]["MAD_SYSTEM_GPU_ARCHITECTURE"] = self.get_system_gpu_architecture()
        self.ctx['docker_env_vars']['MAD_SYSTEM_HIP_VERSION'] = self.get_system_hip_version()
        self.ctx["docker_build_arg"] = {"MAD_SYSTEM_GPU_ARCHITECTURE": self.get_system_gpu_architecture()}
        self.ctx["docker_gpus"] = self.get_docker_gpus()
        self.ctx["gpu_renderDs"] = self.get_gpu_renderD_nodes()

        # Default multi-node configuration
        self.ctx['multi_node_args'] = {
            'RUNNER': 'torchrun',
            'MAD_RUNTIME_NGPUS': self.ctx['docker_env_vars']['MAD_SYSTEM_NGPUS'],  # Use system's GPU count
            'NNODES': 1,
            'NODE_RANK': 0,
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': 6006,
            'HOST_LIST': '',
            'NCCL_SOCKET_IFNAME': '', 
            'GLOO_SOCKET_IFNAME': ''
        }

        # Read and update MAD SECRETS env variable
        mad_secrets = {}
        for key in os.environ:
            if "MAD_SECRETS" in key:
                mad_secrets[key] = os.environ[key]
        if mad_secrets:
            update_dict(self.ctx['docker_build_arg'], mad_secrets)
            update_dict(self.ctx['docker_env_vars'], mad_secrets)  

        ## ADD MORE CONTEXTS HERE ##

        # additional contexts provided in file override detected contexts
        if additional_context_file:
            with open(additional_context_file) as f:
                update_dict(self.ctx, json.load(f))

        # additional contexts provided in command-line override detected contexts and contexts in file
        if additional_context:
            # Convert the string representation of python dictionary to a dictionary.
            dict_additional_context = ast.literal_eval(additional_context)

            update_dict(self.ctx, dict_additional_context)

        # Set multi-node runner after context update
        self.ctx['docker_env_vars']['MAD_MULTI_NODE_RUNNER'] = self.set_multi_node_runner()

    def get_ctx_test(self) -> str:
        """Get context test.
        
        Returns:
            str: The output of the shell command.

        Raises:
            RuntimeError: If the file 'ctx_test' is not found
        """
        # Check if the file 'ctx_test' exists, and if it does, print the contents of the file, otherwise print 'None'.
        return self.console.sh(
            "if [ -f 'ctx_test' ]; then cat ctx_test; else echo 'None'; fi || true"
        )

    def get_gpu_vendor(self) -> str:
        """Get GPU vendor.
        
        Returns:
            str: The output of the shell command.
        
        Raises:
            RuntimeError: If the GPU vendor is unable to detect.
        
        Note:
            What types of GPU vendors are supported?
            - NVIDIA
            - AMD
        """
        # Check if the GPU vendor is NVIDIA or AMD, and if it is unable to detect the GPU vendor.
        return self.console.sh(
            'bash -c \'if [[ -f /usr/bin/nvidia-smi ]] && $(/usr/bin/nvidia-smi > /dev/null 2>&1); then echo "NVIDIA"; elif [[ -f /opt/rocm/bin/amd-smi ]]; then echo "AMD"; elif [[ -f /usr/local/bin/amd-smi ]]; then echo "AMD"; else echo "Unable to detect GPU vendor"; fi || true\''
        )

    def get_host_os(self) -> str:
        """Get host OS.
        
        Returns:
            str: The output of the shell command.
        
        Raises:
            RuntimeError: If the host OS is unable to detect.

        Note:
            What types of host OS are supported?
            - Ubuntu
            - CentOS
            - SLES
        """
        # Check if the host OS is Ubuntu, CentOS, SLES, or if it is unable to detect the host OS.
        return self.console.sh(
            "if [ -f \"$(which apt)\" ]; then echo 'HOST_UBUNTU'; elif [ -f \"$(which yum)\" ]; then echo 'HOST_CENTOS'; elif [ -f \"$(which zypper)\" ]; then echo 'HOST_SLES'; elif [ -f \"$(which tdnf)\" ]; then echo 'HOST_AZURE'; else echo 'Unable to detect Host OS'; fi || true"
        )

    def get_numa_balancing(self) -> bool:
        """Get NUMA balancing.
        
        Returns:
            bool: The output of the shell command.

        Raises:
            RuntimeError: If the NUMA balancing is not enabled or disabled.

        Note:
            NUMA balancing is enabled if the output is '1', and disabled if the output is '0'.
            
            What is NUMA balancing?
            Non-Uniform Memory Access (NUMA) is a computer memory design used in multiprocessing, 
            where the memory access time depends on the memory location relative to the processor.
        """
        # Check if NUMA balancing is enabled or disabled.
        path = "/proc/sys/kernel/numa_balancing"
        if os.path.exists(path):
            return self.console.sh("cat /proc/sys/kernel/numa_balancing || true")
        else:
            return False

    def get_system_ngpus(self) -> int:
        """Get system number of GPUs.
        
        Returns:
            int: The number of GPUs.
        
        Raises:
            RuntimeError: If the GPU vendor is not detected.
        
        Note:
            What types of GPU vendors are supported?
            - NVIDIA
            - AMD
        """
        number_gpus = 0
        if self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"] == "AMD":
            number_gpus = int(self.console.sh("amd-smi list --csv | tail -n +3 | wc -l"))
        elif self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"] == "NVIDIA":
            number_gpus = int(self.console.sh("nvidia-smi -L | wc -l"))
        else:
            raise RuntimeError("Unable to determine gpu vendor.")

        return number_gpus

    def get_system_gpu_architecture(self) -> str:
        """Get system GPU architecture.
        
        Returns:
            str: The GPU architecture.
        
        Raises:
            RuntimeError: If the GPU vendor is not detected.
            RuntimeError: If the GPU architecture is unable to determine.
        
        Note:
            What types of GPU vendors are supported?
            - NVIDIA
            - AMD
        """
        if self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"] == "AMD":
            return self.console.sh("/opt/rocm/bin/rocminfo |grep -o -m 1 'gfx.*'")
        elif self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"] == "NVIDIA":
            return self.console.sh(
                "nvidia-smi -L | head -n1 | sed 's/(UUID: .*)//g' | sed 's/GPU 0: //g'"
            )
        else:
            raise RuntimeError("Unable to determine gpu architecture.")

    def get_system_hip_version(self):
        if self.ctx['docker_env_vars']['MAD_GPU_VENDOR']=='AMD':
            return self.console.sh("hipconfig --version | cut -d'.' -f1,2")
        elif self.ctx['docker_env_vars']['MAD_GPU_VENDOR']=='NVIDIA':
            return self.console.sh("nvcc --version | sed -n 's/^.*release \\([0-9]\\+\\.[0-9]\\+\\).*$/\\1/p'")
        else:
            raise RuntimeError("Unable to determine hip version.")

    def get_docker_gpus(self) -> typing.Optional[str]:
        """Get Docker GPUs.
        
        Returns:
            str: The range of GPUs.
        """
        if int(self.ctx["docker_env_vars"]["MAD_SYSTEM_NGPUS"]) > 0:
            return "0-{}".format(
                int(self.ctx["docker_env_vars"]["MAD_SYSTEM_NGPUS"]) - 1
            )
        return None

    def get_gpu_renderD_nodes(self) -> typing.Optional[typing.List[int]]:
        """Get GPU renderD nodes from KFD properties.
        
        Returns:
            list: The list of GPU renderD nodes.

        Raises:
            RuntimeError: If the ROCm version is not detected

        Note:
            What is renderD?
            - renderD is the device node used for GPU rendering.
            What is KFD?
            - Kernel Fusion Driver (KFD) is the driver used for Heterogeneous System Architecture (HSA).
            What types of GPU vendors are supported?
            - AMD
        """
        # Initialize the GPU renderD nodes.
        gpu_renderDs = None
        # Check if the GPU vendor is AMD.
        if self.ctx['docker_env_vars']['MAD_GPU_VENDOR']=='AMD':
            # get rocm version
            rocm_version = self.console.sh("cat /opt/rocm/.info/version | cut -d'-' -f1")
            
            # get renderDs from KFD properties
            kfd_properties = self.console.sh("grep -r drm_render_minor /sys/devices/virtual/kfd/kfd/topology/nodes").split("\n")
            kfd_properties = [line for line in kfd_properties if int(line.split()[-1])!=0] # CPUs are 0, skip them
            kfd_renderDs = [int(line.split()[-1]) for line in kfd_properties]

            # get list of GPUs
            output = self.console.sh("amd-smi list -e --json")
            if output:
                data = json.loads(output)
            else:
                raise ValueError("Failed to retrieve AMD GPU data")            

            # get gpu id - renderD mapping using unique id if ROCm < 6.1.2 and node id otherwise
            # node id is more robust but is only available from 6.1.2
            if tuple(map(int, rocm_version.split("."))) < (6,1,2):
                kfd_unique_ids = self.console.sh("grep -r unique_id /sys/devices/virtual/kfd/kfd/topology/nodes").split("\n")
                kfd_unique_ids = [hex(int(item.split()[-1])) for item in kfd_unique_ids] #get unique_id and convert it to hex

                # map unique ids to renderDs
                uniqueid_renderD_map = {unique_id:renderD for unique_id, renderD in zip(kfd_unique_ids, kfd_renderDs)}

                # get gpu id unique id map from amd-smi
                gpuid_uuid_map = {}
                for item in data:
                    gpuid_uuid_map[item["gpu"]] = hex(int(item["hip_uuid"].split("-")[1], 16))

                # sort gpu_renderDs based on gpu ids
                gpu_renderDs = [uniqueid_renderD_map[gpuid_uuid_map[gpuid]] for gpuid in sorted(gpuid_uuid_map.keys())]
            else:
                kfd_nodeids = [int(re.search(r"\d+",line.split()[0]).group()) for line in kfd_properties]

                # map node ids to renderDs
                nodeid_renderD_map = {nodeid: renderD for nodeid, renderD in zip(kfd_nodeids, kfd_renderDs)}

                # get gpu id node id map from amd-smi
                gpuid_nodeid_map = {}
                for item in data:
                    gpuid_nodeid_map[item["gpu"]] = item["node_id"]

                # sort gpu_renderDs based on gpu ids
                gpu_renderDs = [nodeid_renderD_map[gpuid_nodeid_map[gpuid]] for gpuid in sorted(gpuid_nodeid_map.keys())]

        return gpu_renderDs

    def set_multi_node_runner(self) -> str:
        """
        Sets the `MAD_MULTI_NODE_RUNNER` environment variable based on the selected multi-node
        runner (e.g., `torchrun`, `mpirun`, or fallback to `python3`). This method dynamically
        generates the appropriate command based on the provided multi-node configuration.

        Returns:
            str: The command string for the multi-node runner, including necessary arguments and
            environment variable settings.
        """
        # NOTE: mpirun is untested
        if self.ctx["multi_node_args"]["RUNNER"] == 'mpirun':
            if not self.ctx["multi_node_args"]["HOST_LIST"]:
                self.ctx["multi_node_args"]["HOST_LIST"] = f"localhost:{self.ctx['multi_node_args']['MAD_RUNTIME_NGPUS']}"
            multi_node_runner = (
                f"mpirun -np {self.ctx['multi_node_args']['NNODES'] * self.ctx['multi_node_args']['MAD_RUNTIME_NGPUS']} "
                f"--host {self.ctx['multi_node_args']['HOST_LIST']}"
            )
        else:
            distributed_args = (
                f"--nproc_per_node {self.ctx['multi_node_args']['MAD_RUNTIME_NGPUS']} "
                f"--nnodes {self.ctx['multi_node_args']['NNODES']} "
                f"--node_rank {self.ctx['multi_node_args']['NODE_RANK']} "
                f"--master_addr {self.ctx['multi_node_args']['MASTER_ADDR']} "
                f"--master_port {self.ctx['multi_node_args']['MASTER_PORT']}"
            )
            multi_node_runner = f"torchrun {distributed_args}"

        # Add NCCL and GLOO interface environment variables
        multi_node_runner = (
            f"NCCL_SOCKET_IFNAME={self.ctx['multi_node_args']['NCCL_SOCKET_IFNAME']} "
            f"GLOO_SOCKET_IFNAME={self.ctx['multi_node_args']['GLOO_SOCKET_IFNAME']} "
            f"{multi_node_runner}"
        )

        return multi_node_runner

    def filter(self, unfiltered: typing.Dict) -> typing.Dict:
        """Filter the unfiltered dictionary based on the context.
        
        Args:
            unfiltered: The unfiltered dictionary.
        
        Returns:
            dict: The filtered dictionary.
        """
        # Initialize the filtered dictionary.
        filtered = {}
        # Iterate over the unfiltered dictionary and filter based on the context
        for dockerfile in unfiltered.keys():
            # Convert the string representation of python dictionary to a dictionary.
            dockerctx = ast.literal_eval(unfiltered[dockerfile])
            # logic : if key is in the Dockerfile, it has to match current context
            # if context is empty in Dockerfile, it will match
            match = True
            # Iterate over the docker context and check if the context matches the current context.
            for dockerctx_key in dockerctx.keys():
                if (
                    dockerctx_key in self.ctx
                    and dockerctx[dockerctx_key] != self.ctx[dockerctx_key]
                ):
                    match = False
                    continue
            # If the context matches, add it to the filtered dictionary.
            if match:
                filtered[dockerfile] = unfiltered[dockerfile]
        return filtered

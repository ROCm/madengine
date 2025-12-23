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
from madengine.utils.gpu_validator import validate_rocm_installation, GPUInstallationError, GPUVendor
from madengine.utils.gpu_tool_factory import get_gpu_tool_manager
from madengine.utils.gpu_tool_manager import BaseGPUToolManager


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
        _gpu_context_initialized: Flag to track if GPU context is initialized.
        _system_context_initialized: Flag to track if system context is initialized.
        _build_only_mode: Flag to indicate if running in build-only mode.

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
        init_system_context: Initialize system-specific context.
        init_gpu_context: Initialize GPU-specific context for runtime.
        init_build_context: Initialize build-specific context.
        init_runtime_context: Initialize runtime-specific context.
        ensure_system_context: Ensure system context is initialized.
        ensure_runtime_context: Ensure runtime context is initialized.
        filter: Filter.
    """

    def __init__(
        self,
        additional_context: str = None,
        additional_context_file: str = None,
        build_only_mode: bool = False,
    ) -> None:
        """Constructor of the Context class.

        Args:
            additional_context: The additional context.
            additional_context_file: The additional context file.
            build_only_mode: Whether running in build-only mode (no GPU detection).

        Raises:
            RuntimeError: If GPU detection fails and not in build-only mode.
        """
        # Initialize the console
        self.console = Console()
        self._gpu_context_initialized = False
        self._build_only_mode = build_only_mode
        self._system_context_initialized = False
        self._gpu_tool_manager = None  # Lazy initialization

        # Initialize base context
        self.ctx = {}

        # Initialize docker contexts as empty - will be populated based on mode
        self.ctx["docker_build_arg"] = {}
        self.ctx["docker_env_vars"] = {}

        # Read and update MAD SECRETS env variable (can be used for both build and run)
        mad_secrets = {}
        for key in os.environ:
            if "MAD_SECRETS" in key:
                mad_secrets[key] = os.environ[key]
        if mad_secrets:
            update_dict(self.ctx["docker_build_arg"], mad_secrets)
            update_dict(self.ctx["docker_env_vars"], mad_secrets)

        # Additional contexts provided in file override detected contexts
        if additional_context_file:
            with open(additional_context_file) as f:
                update_dict(self.ctx, json.load(f))

        # Additional contexts provided in command-line override detected contexts and contexts in file
        if additional_context:
            # Convert the string representation of python dictionary to a dictionary.
            dict_additional_context = ast.literal_eval(additional_context)
            update_dict(self.ctx, dict_additional_context)

        # Initialize context based on mode
        # User-provided contexts will not be overridden by detection
        if not build_only_mode:
            # For full workflow mode, initialize everything (legacy behavior preserved)
            self.init_runtime_context()
        else:
            # For build-only mode, only initialize what's needed for building
            self.init_build_context()

        ## ADD MORE CONTEXTS HERE ##

    def init_build_context(self) -> None:
        """Initialize build-specific context.

        This method sets up only the context needed for Docker builds,
        avoiding GPU detection that would fail on build-only nodes.
        System-specific contexts (host_os, numa_balancing, etc.) should be
        provided via --additional-context for build-only nodes if needed.
        """
        print("Initializing build-only context...")

        # Initialize only essential system contexts if not provided via additional_context
        if "ctx_test" not in self.ctx:
            try:
                self.ctx["ctx_test"] = self.get_ctx_test()
            except Exception as e:
                print(f"Warning: Could not detect ctx_test on build node: {e}")

        if "host_os" not in self.ctx:
            try:
                self.ctx["host_os"] = self.get_host_os()
                print(f"Detected host OS: {self.ctx['host_os']}")
            except Exception as e:
                print(f"Warning: Could not detect host OS on build node: {e}")
                print(
                    "Consider providing host_os via --additional-context if needed for build"
                )

        # Don't detect GPU-specific contexts in build-only mode
        # These should be provided via additional_context if needed for build args
        if "MAD_SYSTEM_GPU_ARCHITECTURE" not in self.ctx.get("docker_build_arg", {}):
            print(
                "Info: MAD_SYSTEM_GPU_ARCHITECTURE not provided - should be set via --additional-context for GPU-specific builds"
            )

        # Don't initialize NUMA balancing check for build-only nodes
        # This is runtime-specific and should be handled on execution nodes

    def init_runtime_context(self) -> None:
        """Initialize runtime-specific context.

        This method sets up the full context including system and GPU detection
        for nodes that will run containers.
        """
        print("Initializing runtime context with system and GPU detection...")
        # Initialize system context first
        self.init_system_context()
        # Initialize GPU context
        self.init_gpu_context()

    def init_system_context(self) -> None:
        """Initialize system-specific context.

        This method detects system configuration like OS, NUMA balancing, etc.
        Should be called on runtime nodes to get actual execution environment context.
        """
        if self._system_context_initialized:
            return

        print("Detecting system configuration...")

        try:
            # Initialize system contexts if not already provided via additional_context
            if "ctx_test" not in self.ctx:
                self.ctx["ctx_test"] = self.get_ctx_test()

            if "host_os" not in self.ctx:
                self.ctx["host_os"] = self.get_host_os()
                print(f"Detected host OS: {self.ctx['host_os']}")

            if "numa_balancing" not in self.ctx:
                self.ctx["numa_balancing"] = self.get_numa_balancing()

                # Check if NUMA balancing is enabled or disabled.
                if self.ctx["numa_balancing"] == "1":
                    print("Warning: numa balancing is ON ...")
                elif self.ctx["numa_balancing"] == "0":
                    print("Warning: numa balancing is OFF ...")
                else:
                    print("Warning: unknown numa balancing setup ...")

            self._system_context_initialized = True

        except Exception as e:
            print(f"Warning: System context detection failed: {e}")
            if not self._build_only_mode:
                raise RuntimeError(
                    f"System context detection failed on runtime node: {e}"
                )

    def init_gpu_context(self) -> None:
        """Initialize GPU-specific context for runtime.

        This method detects GPU configuration and sets up environment variables
        needed for container execution. Should only be called on GPU nodes.
        User-provided GPU contexts will not be overridden.

        Raises:
            RuntimeError: If GPU detection fails.
        """
        if self._gpu_context_initialized:
            return

        print("Detecting GPU configuration...")

        try:
            # GPU vendor detection - only if not provided by user
            if "gpu_vendor" not in self.ctx:
                self.ctx["gpu_vendor"] = self.get_gpu_vendor()
                print(f"Detected GPU vendor: {self.ctx['gpu_vendor']}")
            else:
                print(f"Using provided GPU vendor: {self.ctx['gpu_vendor']}")

            # Initialize docker env vars for runtime - only if not already set
            if "MAD_GPU_VENDOR" not in self.ctx["docker_env_vars"]:
                self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"] = self.ctx["gpu_vendor"]

            if "MAD_SYSTEM_NGPUS" not in self.ctx["docker_env_vars"]:
                self.ctx["docker_env_vars"][
                    "MAD_SYSTEM_NGPUS"
                ] = self.get_system_ngpus()

            if "MAD_SYSTEM_GPU_ARCHITECTURE" not in self.ctx["docker_env_vars"]:
                self.ctx["docker_env_vars"][
                    "MAD_SYSTEM_GPU_ARCHITECTURE"
                ] = self.get_system_gpu_architecture()

            if "MAD_SYSTEM_HIP_VERSION" not in self.ctx["docker_env_vars"]:
                self.ctx["docker_env_vars"][
                    "MAD_SYSTEM_HIP_VERSION"
                ] = self.get_system_hip_version()

            if "MAD_SYSTEM_GPU_PRODUCT_NAME" not in self.ctx["docker_env_vars"]:
                self.ctx["docker_env_vars"][
                    "MAD_SYSTEM_GPU_PRODUCT_NAME"
                ] = self.get_system_gpu_product_name()

            # Also add to build args (for runtime builds) - only if not already set
            if "MAD_SYSTEM_GPU_ARCHITECTURE" not in self.ctx["docker_build_arg"]:
                self.ctx["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"] = self.ctx[
                    "docker_env_vars"
                ]["MAD_SYSTEM_GPU_ARCHITECTURE"]

            # Docker GPU configuration - only if not already set
            if "docker_gpus" not in self.ctx:
                self.ctx["docker_gpus"] = self.get_docker_gpus()

            if "gpu_renderDs" not in self.ctx:
                self.ctx["gpu_renderDs"] = self.get_gpu_renderD_nodes()

            self._gpu_context_initialized = True

        except Exception as e:
            if self._build_only_mode:
                print(
                    f"Warning: GPU detection failed in build-only mode (expected): {e}"
                )
            else:
                raise RuntimeError(f"GPU detection failed: {e}")

    def ensure_runtime_context(self) -> None:
        """Ensure runtime context is initialized.

        This method should be called before any runtime operations
        that require system and GPU context.
        """
        if not self._system_context_initialized and not self._build_only_mode:
            self.init_system_context()
        if not self._gpu_context_initialized and not self._build_only_mode:
            self.init_gpu_context()

    def ensure_system_context(self) -> None:
        """Ensure system context is initialized.

        This method should be called when system context is needed
        but may not be initialized (e.g., in build-only mode).
        """
        if not self._system_context_initialized:
            self.init_system_context()

    def _get_tool_manager(self) -> BaseGPUToolManager:
        """Get GPU tool manager for the current vendor (lazy initialization).
        
        Returns:
            GPU tool manager instance
            
        Raises:
            ValueError: If GPU vendor cannot be determined or is unsupported
        """
        if self._gpu_tool_manager is None:
            # Determine vendor from context or detect automatically
            if "MAD_GPU_VENDOR" in self.ctx.get("docker_env_vars", {}):
                vendor_str = self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"]
                if vendor_str == "AMD":
                    vendor = GPUVendor.AMD
                elif vendor_str == "NVIDIA":
                    vendor = GPUVendor.NVIDIA
                else:
                    vendor = None  # Auto-detect
            else:
                vendor = None  # Auto-detect
            
            self._gpu_tool_manager = get_gpu_tool_manager(vendor)
        
        return self._gpu_tool_manager

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
        """Get GPU vendor with fallback support (PR #54).

        Returns:
            str: The GPU vendor ("NVIDIA", "AMD", or error message).

        Raises:
            RuntimeError: If the GPU vendor is unable to detect.

        Note:
            What types of GPU vendors are supported?
            - NVIDIA
            - AMD
            
        PR #54 Enhancement:
            Added fallback to rocm-smi if amd-smi is missing.
        """
        # Check NVIDIA first (simplest check)
        if os.path.exists("/usr/bin/nvidia-smi"):
            try:
                result = self.console.sh("/usr/bin/nvidia-smi > /dev/null 2>&1 && echo 'NVIDIA' || echo ''", timeout=180)
                if result and result.strip() == "NVIDIA":
                    return "NVIDIA"
            except Exception as e:
                print(f"Warning: nvidia-smi check failed: {e}")
        
        # Check AMD - try amd-smi first, fallback to rocm-smi (PR #54)
        # Increased timeout to 180s for SLURM compute nodes where GPU initialization may be slow
        amd_smi_paths = ["/opt/rocm/bin/amd-smi", "/usr/local/bin/amd-smi"]
        for amd_smi_path in amd_smi_paths:
            if os.path.exists(amd_smi_path):
                try:
                    # Verify amd-smi actually works (180s timeout for slow GPU initialization)
                    result = self.console.sh(f"{amd_smi_path} list > /dev/null 2>&1 && echo 'AMD' || echo ''", timeout=180)
                    if result and result.strip() == "AMD":
                        return "AMD"
                except Exception as e:
                    print(f"Warning: amd-smi check failed for {amd_smi_path}: {e}")
        
        # Fallback to rocm-smi (PR #54)
        if os.path.exists("/opt/rocm/bin/rocm-smi"):
            try:
                result = self.console.sh("/opt/rocm/bin/rocm-smi --showid > /dev/null 2>&1 && echo 'AMD' || echo ''", timeout=180)
                if result and result.strip() == "AMD":
                    return "AMD"
            except Exception as e:
                print(f"Warning: rocm-smi check failed: {e}")
        
        return "Unable to detect GPU vendor"

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
        """Get system number of GPUs using tool manager.

        Returns:
            int: The number of GPUs.

        Raises:
            RuntimeError: If the GPU vendor is not detected or GPU count cannot be determined.

        Note:
            What types of GPU vendors are supported?
            - NVIDIA
            - AMD
            
        Enhancement:
            Uses version-aware tool manager with automatic fallback (PR #54).
        """
        vendor = self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"]
        
        if vendor == "AMD":
            try:
                tool_manager = self._get_tool_manager()
                return tool_manager.get_gpu_count()
            except Exception as e:
                raise RuntimeError(
                    f"Unable to determine number of AMD GPUs. "
                    f"Error: {e}"
                )
        elif vendor == "NVIDIA":
            try:
                tool_manager = self._get_tool_manager()
                return tool_manager.get_gpu_count()
            except Exception as e:
                # Fallback to direct command for NVIDIA (longer timeout for slow compute nodes)
                try:
                    number_gpus = int(self.console.sh("nvidia-smi -L | wc -l", timeout=180))
                    return number_gpus
                except Exception:
                    raise RuntimeError(
                        f"Unable to determine number of NVIDIA GPUs. "
                        f"Error: {e}"
                    )
        else:
            raise RuntimeError(f"Unable to determine gpu vendor: {vendor}")

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
            try:
                arch = self.console.sh("/opt/rocm/bin/rocminfo |grep -o -m 1 'gfx.*'")
                if not arch or arch.strip() == "":
                    raise RuntimeError("rocminfo returned empty architecture")
                return arch
            except Exception as e:
                raise RuntimeError(
                    f"Unable to determine AMD GPU architecture. "
                    f"Ensure ROCm is installed and rocminfo is accessible at /opt/rocm/bin/rocminfo. "
                    f"Error: {e}"
                )
        elif self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"] == "NVIDIA":
            return self.console.sh(
                "nvidia-smi -L | head -n1 | sed 's/(UUID: .*)//g' | sed 's/GPU 0: //g'"
            )
        else:
            raise RuntimeError("Unable to determine gpu architecture.")

    def get_system_gpu_product_name(self) -> str:
        """Get system GPU product name with fallback (PR #54).
        
        Returns:
            str: The GPU product name (e.g., AMD Instinct MI300X, NVIDIA H100 80GB HBM3).
        
        Raises:
            RuntimeError: If the GPU vendor is not detected.
            RuntimeError: If the GPU product name is unable to determine.
        
        Note:
            What types of GPU vendors are supported?
            - NVIDIA
            - AMD
            
        PR #54 Enhancement:
            Added rocm-smi fallback for AMD GPUs when amd-smi unavailable.
        """
        vendor = self.ctx["docker_env_vars"]["MAD_GPU_VENDOR"]
        
        if vendor == "AMD":
            try:
                tool_manager = self._get_tool_manager()
                return tool_manager.get_gpu_product_name(gpu_id=0)
            except Exception as e:
                raise RuntimeError(
                    f"Unable to determine AMD GPU product name. "
                    f"Error: {e}"
                )
        elif vendor == "NVIDIA":
            try:
                tool_manager = self._get_tool_manager()
                return tool_manager.get_gpu_product_name(gpu_id=0)
            except Exception as e:
                # Fallback to direct command for NVIDIA (longer timeout for slow compute nodes)
                try:
                    return self.console.sh("nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i 0", timeout=180)
                except Exception:
                    raise RuntimeError(
                        f"Unable to determine NVIDIA GPU product name. "
                        f"Error: {e}"
                    )
        else:
            raise RuntimeError(f"Unable to determine gpu product name for vendor: {vendor}")

    def get_system_hip_version(self):
        """Get HIP/CUDA version using tool manager.
        
        Returns:
            str: Version string (e.g., "6.4" for ROCm, "12.0" for CUDA)
            
        Raises:
            RuntimeError: If version cannot be determined
            
        Enhancement:
            Uses tool manager for robust version detection with multiple fallbacks.
        """
        vendor = self.ctx['docker_env_vars']['MAD_GPU_VENDOR']
        
        if vendor == 'AMD':
            try:
                tool_manager = self._get_tool_manager()
                version_str = tool_manager.get_version()
                if version_str:
                    # Return major.minor only (e.g., "6.4.1" -> "6.4")
                    parts = version_str.split('.')
                    if len(parts) >= 2:
                        return f"{parts[0]}.{parts[1]}"
                    return version_str
                
                # Fallback to hipconfig if tool manager fails
                version = self.console.sh("hipconfig --version | cut -d'.' -f1,2")
                if not version or version.strip() == "":
                    raise RuntimeError("hipconfig returned empty version")
                return version
                
            except Exception as e:
                raise RuntimeError(
                    f"Unable to determine HIP version. "
                    f"Ensure ROCm is installed and hipconfig is accessible. "
                    f"Error: {e}"
                )
        elif vendor == 'NVIDIA':
            try:
                tool_manager = self._get_tool_manager()
                return tool_manager.get_version() or self.console.sh("nvcc --version | sed -n 's/^.*release \\([0-9]\\+\\.[0-9]\\+\\).*$/\\1/p'")
            except Exception:
                return self.console.sh("nvcc --version | sed -n 's/^.*release \\([0-9]\\+\\.[0-9]\\+\\).*$/\\1/p'")
        else:
            raise RuntimeError(f"Unable to determine hip version for vendor: {vendor}")

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
            list: The list of GPU renderD nodes, or None if not AMD GPU.

        Raises:
            RuntimeError: If the ROCm version cannot be determined.
            RuntimeError: If KFD properties cannot be read.
            ValueError: If AMD GPU data cannot be retrieved.
            KeyError: If expected fields are missing from amd-smi output.

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
        if self.ctx['docker_env_vars']['MAD_GPU_VENDOR'] != 'AMD':
            return gpu_renderDs
            
        try:
            # Get ROCm version using tool manager for robust detection (PR #54)
            try:
                tool_manager = self._get_tool_manager()
                rocm_version = tool_manager.get_rocm_version()
                if not rocm_version:
                    raise RuntimeError("Tool manager returned None for ROCm version")
            except Exception as e:
                # Fallback to direct file read
                rocm_version_str = self.console.sh("cat /opt/rocm/.info/version | cut -d'-' -f1")
                if not rocm_version_str or rocm_version_str.strip() == "":
                    raise RuntimeError("Failed to retrieve ROCm version from /opt/rocm/.info/version")
                
                # Parse version safely
                try:
                    rocm_version = tuple(map(int, rocm_version_str.strip().split(".")))
                except (ValueError, AttributeError) as parse_err:
                    raise RuntimeError(f"Failed to parse ROCm version '{rocm_version_str}': {parse_err}")
            
            # Get renderDs from KFD properties
            # Try KFD topology first (preferred), but gracefully handle permission errors
            # On HPC/multi-user systems, KFD topology files may be restricted
            kfd_renderDs = None
            kfd_properties = []
            try:
                kfd_output = self.console.sh("grep -r drm_render_minor /sys/devices/virtual/kfd/kfd/topology/nodes")
                if kfd_output and kfd_output.strip():
                    kfd_properties = kfd_output.split("\n")
                    # Filter out empty lines and CPU entries (renderD value 0)
                    kfd_properties = [
                        line for line in kfd_properties 
                        if line.strip() and line.split() and int(line.split()[-1]) != 0
                    ]
                    if kfd_properties:
                        kfd_renderDs = [int(line.split()[-1]) for line in kfd_properties]
            except Exception as kfd_error:
                # KFD topology read failed (common on HPC clusters with restricted permissions)
                # Will use amd-smi/rocm-smi fallback which provides renderD info directly
                print(f"Note: KFD topology not accessible ({kfd_error}), using ROCm tools fallback")

            # Get gpu id - renderD mapping using unique id if ROCm < 6.4.1 and node id otherwise
            # node id is more robust but is only available from 6.4.1 (PR #54)
            if rocm_version < (6, 4, 1):
                # Legacy method using unique_id
                kfd_unique_output = self.console.sh("grep -r unique_id /sys/devices/virtual/kfd/kfd/topology/nodes")
                if not kfd_unique_output:
                    raise RuntimeError("Failed to retrieve unique_id from KFD properties")
                
                kfd_unique_ids_raw = kfd_unique_output.split("\n")
                # Convert unique_ids to hex, filtering empty lines
                kfd_unique_ids = []
                for item in kfd_unique_ids_raw:
                    if item.strip():
                        try:
                            unique_id_int = int(item.split()[-1])
                            kfd_unique_ids.append(hex(unique_id_int))
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Failed to parse unique_id from line '{item}': {e}")
                            continue

                if len(kfd_unique_ids) != len(kfd_renderDs):
                    raise RuntimeError(
                        f"Mismatch between unique_ids count ({len(kfd_unique_ids)}) "
                        f"and renderDs count ({len(kfd_renderDs)})"
                    )

                # Map unique ids to renderDs
                uniqueid_renderD_map = {
                    unique_id: renderD 
                    for unique_id, renderD in zip(kfd_unique_ids, kfd_renderDs)
                }

                # Get GPU ID to unique ID mapping from rocm-smi (longer timeout for slow compute nodes)
                rsmi_output = self.console.sh("rocm-smi --showuniqueid | grep 'Unique.*:'", timeout=180)
                if not rsmi_output or rsmi_output.strip() == "":
                    raise RuntimeError("Failed to retrieve unique IDs from rocm-smi")
                
                rsmi_lines = [line.strip() for line in rsmi_output.split("\n") if line.strip()]
                
                # Sort gpu_renderDs based on GPU IDs
                gpu_renderDs = []
                for line in rsmi_lines:
                    try:
                        unique_id = line.split()[-1]
                        if unique_id not in uniqueid_renderD_map:
                            raise KeyError(f"Unique ID '{unique_id}' from rocm-smi not found in KFD mapping")
                        gpu_renderDs.append(uniqueid_renderD_map[unique_id])
                    except (IndexError, KeyError) as e:
                        raise RuntimeError(f"Failed to map unique ID from line '{line}': {e}")
            else:
                # Modern method using amd-smi (ROCm >= 6.4.0)
                # Get list of GPUs from amd-smi (redirect stderr to filter warnings)
                # Longer timeout (180s) for slow GPU initialization on SLURM compute nodes
                output = self.console.sh("amd-smi list -e --json 2>/dev/null || amd-smi list -e --json 2>&1", timeout=180)
                if not output or output.strip() == "":
                    raise ValueError("Failed to retrieve AMD GPU data from amd-smi")
                
                # amd-smi may output warnings before JSON - extract only JSON part
                # Look for lines starting with '[' or '{' (JSON start)
                json_start = -1
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('[') or line.strip().startswith('{'):
                        json_start = i
                        break
                
                if json_start >= 0:
                    json_output = '\n'.join(lines[json_start:])
                else:
                    json_output = output
                
                try:
                    data = json.loads(json_output)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse amd-smi JSON output: {e}. Output was: {output[:200]}")
                
                if not data or not isinstance(data, list):
                    raise ValueError("amd-smi returned empty or invalid data")

                # Check if we successfully got KFD renderDs
                if kfd_renderDs:
                    # Original method: Map KFD renderDs via node_id from amd-smi
                    kfd_nodeids = []
                    for line in kfd_properties:
                        try:
                            match = re.search(r"\d+", line.split()[0])
                            if match:
                                kfd_nodeids.append(int(match.group()))
                            else:
                                print(f"Warning: Could not extract node ID from line: {line}")
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Failed to parse node ID from line '{line}': {e}")
                            continue

                    if len(kfd_nodeids) != len(kfd_renderDs):
                        raise RuntimeError(
                            f"Mismatch between node IDs count ({len(kfd_nodeids)}) "
                            f"and renderDs count ({len(kfd_renderDs)})"
                        )

                    # Map node ids to renderDs
                    nodeid_renderD_map = {
                        nodeid: renderD 
                        for nodeid, renderD in zip(kfd_nodeids, kfd_renderDs)
                    }

                    # Get gpu id to node id map from amd-smi
                    gpuid_nodeid_map = {}
                    for item in data:
                        try:
                            gpuid_nodeid_map[item["gpu"]] = item["node_id"]
                        except KeyError as e:
                            raise KeyError(f"Failed to parse node_id from amd-smi data: {e}. Item: {item}")

                    # Sort gpu_renderDs based on gpu ids
                    try:
                        gpu_renderDs = [
                            nodeid_renderD_map[gpuid_nodeid_map[gpuid]] 
                            for gpuid in sorted(gpuid_nodeid_map.keys())
                        ]
                    except KeyError as e:
                        raise RuntimeError(f"Failed to map GPU IDs to renderDs: {e}")
                else:
                    # Fallback method: Get renderD directly from amd-smi (ROCm >= 6.4.1)
                    # This is actually BETTER - no KFD topology parsing needed!
                    print("Using amd-smi renderD info directly (cleaner method)")
                    gpu_renderDs = []
                    for item in sorted(data, key=lambda x: x["gpu"]):
                        try:
                            render_str = item["render"]  # e.g., "renderD128"
                            render_num = int(render_str.replace("renderD", ""))
                            gpu_renderDs.append(render_num)
                        except (KeyError, ValueError) as e:
                            raise RuntimeError(f"Failed to parse renderD from amd-smi: {e}. Item: {item}")

        except (RuntimeError, ValueError, KeyError) as e:
            # Re-raise with context
            raise RuntimeError(f"Error in get_gpu_renderD_nodes: {e}") from e
        except Exception as e:
            # Catch unexpected errors
            raise RuntimeError(f"Unexpected error in get_gpu_renderD_nodes: {e}") from e

        return gpu_renderDs

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

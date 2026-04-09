#!/usr/bin/env python3
"""
NVIDIA Tool Manager

Basic NVIDIA CUDA tool manager wrapping nvidia-smi and nvcc.
Maintains current behavior without sophisticated version-aware logic.

This is a placeholder for future enhancement. Current implementation provides:
- Simple wrappers around nvidia-smi and nvcc
- Basic error handling
- Consistent interface with BaseGPUToolManager

Future enhancements could include:
- CUDA version-aware tool selection
- Fallback between different CUDA tool versions
- More sophisticated error handling

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from typing import Optional

from madengine.utils.gpu_tool_manager import BaseGPUToolManager


class NvidiaToolManager(BaseGPUToolManager):
    """NVIDIA CUDA tool manager with basic functionality.
    
    Provides simple wrappers around NVIDIA tools while maintaining
    compatibility with BaseGPUToolManager interface.
    
    Current implementation:
    - nvidia-smi for GPU queries
    - nvcc for CUDA version
    - Basic error handling
    
    No version-aware tool selection yet (deferred for future work).
    """
    
    # Tool paths
    NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"
    NVCC_PATH = "/usr/local/cuda/bin/nvcc"
    
    def __init__(self):
        """Initialize NVIDIA tool manager."""
        super().__init__()
        self._log_debug("Initialized NVIDIA tool manager")
    
    def get_version(self) -> Optional[str]:
        """Get CUDA version as string.
        
        Returns:
            CUDA version string or None if unable to detect
        """
        return self.get_cuda_version()
    
    def get_cuda_version(self) -> Optional[str]:
        """Get CUDA version from nvcc.
        
        Returns:
            CUDA version string (e.g., "12.0") or None if unable to detect
        """
        # Check cache first
        cached = self._get_cached_result("cuda_version")
        if cached is not None:
            return cached
        
        try:
            # Try nvcc --version
            if self.is_tool_available(self.NVCC_PATH):
                command = f"{self.NVCC_PATH} --version | sed -n 's/^.*release \\([0-9]\\+\\.[0-9]\\+\\).*$/\\1/p'"
                success, stdout, stderr = self._execute_shell_command(command)
                
                if success and stdout:
                    version = stdout.strip()
                    self._cache_result("cuda_version", version)
                    self._log_info(f"CUDA version: {version}")
                    return version
            
            # Fallback: Try nvidia-smi to get driver version
            if self.is_tool_available(self.NVIDIA_SMI_PATH):
                command = f"{self.NVIDIA_SMI_PATH} --query | grep 'CUDA Version' | awk '{{print $4}}'"
                success, stdout, stderr = self._execute_shell_command(command)
                
                if success and stdout:
                    version = stdout.strip()
                    self._cache_result("cuda_version", version)
                    self._log_info(f"CUDA version (from nvidia-smi): {version}")
                    return version
            
            self._log_warning("Unable to detect CUDA version")
            return None
            
        except Exception as e:
            self._log_error(f"Error detecting CUDA version: {e}")
            return None
    
    def get_driver_version(self) -> Optional[str]:
        """Get NVIDIA driver version.
        
        Returns:
            Driver version string or None if unable to detect
        """
        # Check cache
        cached = self._get_cached_result("driver_version")
        if cached is not None:
            return cached
        
        try:
            if self.is_tool_available(self.NVIDIA_SMI_PATH):
                command = f"{self.NVIDIA_SMI_PATH} --query-gpu=driver_version --format=csv,noheader | head -n1"
                success, stdout, stderr = self._execute_shell_command(command)
                
                if success and stdout:
                    version = stdout.strip()
                    self._cache_result("driver_version", version)
                    self._log_info(f"NVIDIA driver version: {version}")
                    return version
            
            self._log_warning("Unable to detect NVIDIA driver version")
            return None
            
        except Exception as e:
            self._log_error(f"Error detecting driver version: {e}")
            return None
    
    def execute_command(
        self,
        command: str,
        fallback_command: Optional[str] = None,
        timeout: int = 30
    ) -> str:
        """Execute command with optional fallback.
        
        Args:
            command: Primary command to execute
            fallback_command: Optional fallback command (currently not used for NVIDIA)
            timeout: Command timeout in seconds
            
        Returns:
            Command output as string
            
        Raises:
            RuntimeError: If command fails
        """
        success, stdout, stderr = self._execute_shell_command(command, timeout)
        
        if success:
            return stdout
        
        # Try fallback if provided
        if fallback_command:
            self._log_warning(f"Primary command failed, trying fallback: {fallback_command[:50]}...")
            success, stdout, stderr = self._execute_shell_command(fallback_command, timeout)
            
            if success:
                return stdout
            else:
                raise RuntimeError(
                    f"Both primary and fallback commands failed.\n"
                    f"Primary: {command}\n"
                    f"Fallback: {fallback_command}\n"
                    f"Error: {stderr}"
                )
        else:
            raise RuntimeError(f"Command failed: {command}\nError: {stderr}")
    
    def execute_nvidia_smi(self, args: str, timeout: int = 30) -> str:
        """Execute nvidia-smi with specified arguments.
        
        Args:
            args: Arguments to pass to nvidia-smi
            timeout: Command timeout in seconds
            
        Returns:
            Command output as string
            
        Raises:
            RuntimeError: If nvidia-smi is not available or command fails
        """
        if not self.is_tool_available(self.NVIDIA_SMI_PATH):
            raise RuntimeError(
                f"nvidia-smi not found at {self.NVIDIA_SMI_PATH}\n"
                f"Ensure NVIDIA drivers are installed."
            )
        
        command = f"{self.NVIDIA_SMI_PATH} {args}"
        return self.execute_command(command, timeout=timeout)
    
    def execute_nvcc(self, args: str, timeout: int = 30) -> str:
        """Execute nvcc with specified arguments.
        
        Args:
            args: Arguments to pass to nvcc
            timeout: Command timeout in seconds
            
        Returns:
            Command output as string
            
        Raises:
            RuntimeError: If nvcc is not available or command fails
        """
        if not self.is_tool_available(self.NVCC_PATH):
            raise RuntimeError(
                f"nvcc not found at {self.NVCC_PATH}\n"
                f"Ensure CUDA toolkit is installed."
            )
        
        command = f"{self.NVCC_PATH} {args}"
        return self.execute_command(command, timeout=timeout)
    
    def get_gpu_count(self) -> int:
        """Get number of NVIDIA GPUs in the system.
        
        Returns:
            Number of GPUs detected
            
        Raises:
            RuntimeError: If unable to detect GPUs
        """
        # Check cache
        cached = self._get_cached_result("gpu_count")
        if cached is not None:
            return cached
        
        try:
            output = self.execute_nvidia_smi("-L | wc -l")
            count = int(output.strip())
            
            self._cache_result("gpu_count", count)
            self._log_info(f"Detected {count} NVIDIA GPU(s)")
            
            return count
            
        except Exception as e:
            raise RuntimeError(
                f"Unable to determine number of NVIDIA GPUs.\n"
                f"Error: {e}\n"
                f"Suggestions:\n"
                f"- Verify NVIDIA drivers: nvidia-smi\n"
                f"- Check GPU accessibility: ls -la /dev/nvidia*"
            )
    
    def get_gpu_product_name(self, gpu_id: int = 0) -> str:
        """Get GPU product name.
        
        Args:
            gpu_id: GPU index (0-based)
            
        Returns:
            GPU product name (e.g., "NVIDIA H100 80GB HBM3")
            
        Raises:
            RuntimeError: If unable to get product name
        """
        cache_key = f"gpu_product_name:{gpu_id}"
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        try:
            output = self.execute_nvidia_smi(
                f"--query-gpu=name --format=csv,noheader,nounits -i {gpu_id}"
            )
            product_name = output.strip()
            
            self._cache_result(cache_key, product_name)
            self._log_debug(f"GPU {gpu_id} product name: {product_name}")
            
            return product_name
            
        except Exception as e:
            raise RuntimeError(
                f"Unable to get GPU product name for GPU {gpu_id}.\n"
                f"Error: {e}\n"
                f"Ensure GPU {gpu_id} exists: nvidia-smi -L"
            )
    
    def get_gpu_architecture(self, gpu_id: int = 0) -> str:
        """Get GPU architecture/compute capability.
        
        Args:
            gpu_id: GPU index (0-based)
            
        Returns:
            GPU architecture string
            
        Raises:
            RuntimeError: If unable to detect GPU architecture
        """
        cache_key = f"gpu_architecture:{gpu_id}"
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        try:
            # Get full GPU name which includes architecture info
            output = self.execute_nvidia_smi(
                f"-L | head -n{gpu_id + 1} | tail -n1 | sed 's/(UUID: .*)//g' | sed 's/GPU {gpu_id}: //g'"
            )
            arch = output.strip()
            
            self._cache_result(cache_key, arch)
            self._log_debug(f"GPU {gpu_id} architecture: {arch}")
            
            return arch
            
        except Exception as e:
            raise RuntimeError(
                f"Unable to determine GPU architecture for GPU {gpu_id}.\n"
                f"Error: {e}"
            )


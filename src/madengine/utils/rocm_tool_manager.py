#!/usr/bin/env python3
"""
ROCm Tool Manager

Version-aware AMD ROCm tool manager with automatic fallback between amd-smi and
rocm-smi based on ROCm version and tool availability.

Based on PR #54: https://github.com/ROCm/madengine/pull/54
- ROCm version threshold: 6.4.1 (use amd-smi for >= 6.4.1, rocm-smi for < 6.4.1)
- Automatic fallback to rocm-smi when amd-smi is unavailable
- Robust error handling with actionable suggestions

References:
- ROCm best practices: https://github.com/ROCm/TheRock
- ROCm systems: https://github.com/ROCm/rocm-systems

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from madengine.utils.gpu_tool_manager import BaseGPUToolManager


# ROCm version threshold for amd-smi vs rocm-smi (from PR #54)
ROCM_VERSION_THRESHOLD = (6, 4, 1)


class ROCmToolManager(BaseGPUToolManager):
    """AMD ROCm tool manager with version-aware tool selection.
    
    Features:
    - Automatic ROCm version detection from multiple sources
    - Version-aware tool selection (amd-smi >= 6.4.1, rocm-smi < 6.4.1)
    - Automatic fallback with warnings when preferred tool unavailable
    - Comprehensive error messages with troubleshooting suggestions
    
    Tool Selection Logic:
    - ROCm >= 6.4.1: Prefer amd-smi, fallback to rocm-smi with warning
    - ROCm < 6.4.1: Use rocm-smi
    - If both tools fail: Raise error with debugging information
    """
    
    # Tool paths
    AMD_SMI_PATH = "/opt/rocm/bin/amd-smi"
    ROCM_SMI_PATH = "/opt/rocm/bin/rocm-smi"
    HIPCONFIG_PATH = "/opt/rocm/bin/hipconfig"
    ROCMINFO_PATH = "/opt/rocm/bin/rocminfo"
    ROCM_VERSION_FILE = "/opt/rocm/.info/version"
    
    def __init__(self):
        """Initialize ROCm tool manager."""
        super().__init__()
        self._log_debug("Initialized ROCm tool manager")
    
    def get_version(self) -> Optional[str]:
        """Get ROCm version as string.
        
        Returns:
            ROCm version string (e.g., "6.4.1") or None if unable to detect
        """
        version_tuple = self.get_rocm_version()
        if version_tuple:
            return ".".join(map(str, version_tuple))
        return None
    
    def get_rocm_version(self) -> Optional[Tuple[int, int, int]]:
        """Get ROCm version as tuple.
        
        Tries multiple detection methods in order:
        1. hipconfig --version
        2. /opt/rocm/.info/version file
        3. rocminfo parsing
        
        Results are cached for performance.
        
        Returns:
            ROCm version as tuple (major, minor, patch) or None if unable to detect
            
        Example:
            >>> manager = ROCmToolManager()
            >>> manager.get_rocm_version()
            (6, 4, 1)
        """
        # Check cache first
        cached = self._get_cached_result("rocm_version")
        if cached is not None:
            return cached
        
        version = None
        
        # Method 1: Try hipconfig --version
        if self.is_tool_available(self.HIPCONFIG_PATH):
            success, stdout, stderr = self._execute_shell_command(
                f"{self.HIPCONFIG_PATH} --version",
                timeout=10
            )
            if success and stdout:
                # Parse version like "6.4.1-12345" -> (6, 4, 1)
                try:
                    version_str = stdout.split('-')[0].strip()
                    parts = version_str.split('.')
                    if len(parts) >= 3:
                        version = (int(parts[0]), int(parts[1]), int(parts[2]))
                        self._log_debug(f"Detected ROCm version from hipconfig: {version}")
                except (ValueError, IndexError) as e:
                    self._log_warning(f"Failed to parse hipconfig version '{stdout}': {e}")
        
        # Method 2: Try version file
        if version is None and os.path.exists(self.ROCM_VERSION_FILE):
            try:
                with open(self.ROCM_VERSION_FILE, 'r') as f:
                    version_str = f.read().strip().split('-')[0]
                    parts = version_str.split('.')
                    if len(parts) >= 3:
                        version = (int(parts[0]), int(parts[1]), int(parts[2]))
                        self._log_debug(f"Detected ROCm version from file: {version}")
            except (IOError, ValueError, IndexError) as e:
                self._log_warning(f"Failed to read version file: {e}")
        
        # Method 3: Try rocminfo (less reliable, last resort)
        if version is None and self.is_tool_available(self.ROCMINFO_PATH):
            success, stdout, stderr = self._execute_shell_command(
                f"{self.ROCMINFO_PATH} | grep -i 'ROCm Version' | head -n1",
                timeout=10
            )
            if success and stdout:
                try:
                    # Parse output like "ROCm Version: 6.4.1"
                    match = re.search(r'(\d+)\.(\d+)\.(\d+)', stdout)
                    if match:
                        version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                        self._log_debug(f"Detected ROCm version from rocminfo: {version}")
                except (ValueError, AttributeError) as e:
                    self._log_warning(f"Failed to parse rocminfo output: {e}")
        
        # Cache the result (even if None)
        self._cache_result("rocm_version", version)
        
        if version:
            self._log_info(f"ROCm version: {'.'.join(map(str, version))}")
        else:
            self._log_warning("Unable to detect ROCm version from any source")
        
        return version
    
    def get_preferred_smi_tool(self) -> str:
        """Get the preferred SMI tool based on ROCm version.
        
        Returns:
            Tool name: 'amd-smi' or 'rocm-smi'
            
        Logic:
            - ROCm >= 6.4.1: Prefer amd-smi
            - ROCm < 6.4.1: Use rocm-smi
            - Unknown version: Try amd-smi first (conservative choice)
        """
        version = self.get_rocm_version()
        
        if version is None:
            self._log_warning("ROCm version unknown, defaulting to amd-smi")
            return "amd-smi"
        
        if version >= ROCM_VERSION_THRESHOLD:
            return "amd-smi"
        else:
            return "rocm-smi"
    
    def execute_command(
        self,
        command: str,
        fallback_command: Optional[str] = None,
        timeout: int = 30
    ) -> str:
        """Execute command with optional fallback.
        
        Args:
            command: Primary command to execute
            fallback_command: Optional fallback command if primary fails
            timeout: Command timeout in seconds
            
        Returns:
            Command output as string
            
        Raises:
            RuntimeError: If both primary and fallback commands fail
        """
        # Try primary command
        success, stdout, stderr = self._execute_shell_command(command, timeout)
        
        if success:
            self._log_debug(f"Command succeeded: {command[:50]}...")
            return stdout
        
        # Log primary failure
        self._log_warning(f"Primary command failed: {command[:50]}... Error: {stderr}")
        
        # Try fallback if provided
        if fallback_command:
            self._log_info(f"Trying fallback command: {fallback_command[:50]}...")
            success, stdout, stderr = self._execute_shell_command(fallback_command, timeout)
            
            if success:
                self._log_warning("Fallback command succeeded (primary tool may be missing or misconfigured)")
                return stdout
            else:
                # Both failed
                raise RuntimeError(
                    f"Both primary and fallback commands failed.\n"
                    f"Primary: {command}\n"
                    f"Primary error: {stderr}\n"
                    f"Fallback: {fallback_command}\n"
                    f"Fallback error: {stderr}"
                )
        else:
            # No fallback, raise error
            raise RuntimeError(f"Command failed: {command}\nError: {stderr}")
    
    def execute_smi_command(self, command_template: str, use_amd_smi: bool = True, **kwargs) -> str:
        """Execute SMI command with automatic tool selection and fallback.
        
        Args:
            command_template: Command template with {tool} placeholder
            use_amd_smi: If True, use amd-smi syntax; if False, use rocm-smi syntax
            **kwargs: Additional format parameters for command template
            
        Returns:
            Command output as string
            
        Example:
            >>> manager = ROCmToolManager()
            >>> # Will try amd-smi, fallback to rocm-smi if needed
            >>> output = manager.execute_smi_command("{tool} list --csv")
        """
        preferred_tool = self.get_preferred_smi_tool()
        
        # Format command with preferred tool
        if preferred_tool == "amd-smi":
            tool_path = self.AMD_SMI_PATH
            fallback_path = self.ROCM_SMI_PATH
        else:
            tool_path = self.ROCM_SMI_PATH
            fallback_path = self.AMD_SMI_PATH
        
        command = command_template.format(tool=tool_path, **kwargs)
        
        # Create fallback command if fallback tool is available
        fallback_command = None
        if self.is_tool_available(fallback_path):
            fallback_command = command_template.format(tool=fallback_path, **kwargs)
        
        return self.execute_command(command, fallback_command)
    
    def get_gpu_count(self) -> int:
        """Get number of AMD GPUs in the system.
        
        Returns:
            Number of GPUs detected
            
        Raises:
            RuntimeError: If unable to detect GPUs with any tool
        """
        # Check cache
        cached = self._get_cached_result("gpu_count")
        if cached is not None:
            return cached
        
        preferred_tool = self.get_preferred_smi_tool()
        
        try:
            if preferred_tool == "amd-smi":
                # Try amd-smi list --csv
                command = f"{self.AMD_SMI_PATH} list --csv | tail -n +3 | wc -l"
                fallback = f"{self.ROCM_SMI_PATH} --showid --csv | tail -n +2 | wc -l"
            else:
                # Use rocm-smi
                command = f"{self.ROCM_SMI_PATH} --showid --csv | tail -n +2 | wc -l"
                fallback = f"{self.AMD_SMI_PATH} list --csv | tail -n +3 | wc -l" if self.is_tool_available(self.AMD_SMI_PATH) else None
            
            output = self.execute_command(command, fallback)
            count = int(output.strip())
            
            # Cache result
            self._cache_result("gpu_count", count)
            self._log_info(f"Detected {count} AMD GPU(s)")
            
            return count
            
        except Exception as e:
            raise RuntimeError(
                f"Unable to determine number of AMD GPUs.\n"
                f"Error: {e}\n"
                f"Suggestions:\n"
                f"- Verify ROCm installation: ls -la /opt/rocm/bin/\n"
                f"- Check GPU accessibility: ls -la /dev/kfd /dev/dri\n"
                f"- Ensure user is in 'video' and 'render' groups\n"
                f"- See: https://github.com/ROCm/TheRock"
            )
    
    def get_gpu_product_name(self, gpu_id: int = 0) -> str:
        """Get GPU product name with fallback (from PR #54).
        
        Args:
            gpu_id: GPU index (0-based)
            
        Returns:
            GPU product name (e.g., "AMD Instinct MI300X")
            
        Raises:
            RuntimeError: If unable to get product name with any tool
        """
        cache_key = f"gpu_product_name:{gpu_id}"
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        preferred_tool = self.get_preferred_smi_tool()
        
        try:
            if preferred_tool == "amd-smi":
                # Try amd-smi static command
                command = f"{self.AMD_SMI_PATH} static -g {gpu_id} | grep MARKET_NAME: | cut -d ':' -f 2"
                # Fallback to rocm-smi with different syntax (PR #54)
                fallback = f"{self.ROCM_SMI_PATH} --showproductname | grep 'GPU\\[{gpu_id}\\]' | awk '{{print $NF}}'"
            else:
                # Use rocm-smi
                command = f"{self.ROCM_SMI_PATH} --showproductname | grep 'GPU\\[{gpu_id}\\]' | awk '{{print $NF}}'"
                # Fallback to amd-smi if available
                fallback = f"{self.AMD_SMI_PATH} static -g {gpu_id} | grep MARKET_NAME: | cut -d ':' -f 2" if self.is_tool_available(self.AMD_SMI_PATH) else None
            
            output = self.execute_command(command, fallback)
            product_name = output.strip()
            
            # Cache result
            self._cache_result(cache_key, product_name)
            self._log_debug(f"GPU {gpu_id} product name: {product_name}")
            
            return product_name
            
        except Exception as e:
            raise RuntimeError(
                f"Unable to get GPU product name for GPU {gpu_id}.\n"
                f"Error: {e}\n"
                f"Suggestions:\n"
                f"- Verify GPU {gpu_id} exists: {self.ROCM_SMI_PATH} --showid\n"
                f"- Check ROCm version: cat /opt/rocm/.info/version\n"
                f"- For ROCm >= 6.4.1, ensure amd-smi is installed"
            )
    
    def get_gpu_architecture(self) -> str:
        """Get GPU architecture (e.g., gfx908, gfx90a, gfx942).
        
        Returns:
            GPU architecture string
            
        Raises:
            RuntimeError: If unable to detect GPU architecture
        """
        # Check cache
        cached = self._get_cached_result("gpu_architecture")
        if cached:
            return cached
        
        try:
            # Use rocminfo to get architecture (most reliable)
            command = f"{self.ROCMINFO_PATH} | grep -o -m 1 'gfx.*'"
            success, stdout, stderr = self._execute_shell_command(command)
            
            if success and stdout:
                arch = stdout.strip()
                self._cache_result("gpu_architecture", arch)
                self._log_info(f"GPU architecture: {arch}")
                return arch
            else:
                raise RuntimeError(f"rocminfo failed or returned empty: {stderr}")
                
        except Exception as e:
            raise RuntimeError(
                f"Unable to determine GPU architecture.\n"
                f"Error: {e}\n"
                f"Suggestions:\n"
                f"- Verify rocminfo is accessible: {self.ROCMINFO_PATH} --version\n"
                f"- Check GPU is visible: {self.ROCM_SMI_PATH} --showid\n"
                f"- Ensure ROCm is properly installed"
            )
    
    def get_gpu_vendor_check(self) -> str:
        """Check GPU vendor with fallback (from PR #54).
        
        Returns:
            "AMD" if AMD GPU detected, error message otherwise
            
        Note:
            This checks if AMD SMI tools can detect GPUs, confirming AMD vendor.
        """
        try:
            # Try to get GPU count - if successful, AMD GPUs are present
            count = self.get_gpu_count()
            if count > 0:
                return "AMD"
            else:
                return "No AMD GPUs detected"
        except Exception as e:
            return f"Unable to detect AMD GPU vendor: {e}"
    
    def list_gpus_json(self) -> List[Dict]:
        """List all GPUs with detailed information in JSON format.
        
        Returns:
            List of GPU information dictionaries
            
        Raises:
            RuntimeError: If unable to list GPUs
        """
        preferred_tool = self.get_preferred_smi_tool()
        
        try:
            if preferred_tool == "amd-smi" and self.is_tool_available(self.AMD_SMI_PATH):
                # Try amd-smi list with JSON output
                command = f"{self.AMD_SMI_PATH} list --json"
                success, stdout, stderr = self._execute_shell_command(command)
                
                if success and stdout:
                    try:
                        return json.loads(stdout)
                    except json.JSONDecodeError as e:
                        self._log_warning(f"Failed to parse amd-smi JSON: {e}")
            
            # Fallback: parse rocm-smi output
            command = f"{self.ROCM_SMI_PATH} --showid"
            output = self.execute_command(command)
            
            # Parse rocm-smi output to JSON-like structure
            gpus = []
            for line in output.split('\n'):
                if 'GPU[' in line:
                    try:
                        gpu_id = int(line.split('[')[1].split(']')[0])
                        gpus.append({"gpu": gpu_id, "node_id": gpu_id})
                    except (IndexError, ValueError):
                        continue
            
            return gpus
            
        except Exception as e:
            raise RuntimeError(f"Unable to list GPUs: {e}")


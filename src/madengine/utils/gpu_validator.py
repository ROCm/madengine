#!/usr/bin/env python3
"""GPU Installation Validation Utility

This module provides comprehensive validation for GPU installations on nodes.
It supports both AMD ROCm and NVIDIA CUDA, checking for essential components
and providing detailed error messages to help diagnose installation issues.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import subprocess
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GPUVendor(Enum):
    """Supported GPU vendors"""
    AMD = "AMD"
    NVIDIA = "NVIDIA"
    UNKNOWN = "UNKNOWN"


@dataclass
class GPUValidationResult:
    """Result of GPU validation check"""
    is_valid: bool
    vendor: GPUVendor
    version: Optional[str] = None  # ROCm version or CUDA version
    issues: List[str] = None
    warnings: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []


class ROCmValidator:
    """Validator for AMD ROCm installation"""
    
    # Essential ROCm components to check
    ESSENTIAL_PATHS = {
        'rocm_root': '/opt/rocm',
        'hip_path': '/opt/rocm/bin/hipconfig',
        'rocminfo': '/opt/rocm/bin/rocminfo',
    }
    
    # Optional but recommended components
    RECOMMENDED_PATHS = {
        'amd_smi': '/opt/rocm/bin/amd-smi',
        'rocm_smi': '/opt/rocm/bin/rocm-smi',
    }
    
    # KFD (Kernel Fusion Driver) paths
    KFD_PATHS = {
        'kfd_device': '/dev/kfd',
        'kfd_topology': '/sys/devices/virtual/kfd/kfd/topology/nodes',
    }
    
    def __init__(self, verbose: bool = False):
        """Initialize ROCm validator
        
        Args:
            verbose: If True, print detailed validation progress
        """
        self.verbose = verbose
        
    def _run_command(self, cmd: List[str], timeout: int = 10) -> Tuple[bool, str, str]:
        """Run a command and return success status and output
        
        Args:
            cmd: Command to run as list of strings
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except FileNotFoundError:
            return False, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, "", str(e)
    
    def _check_path_exists(self, path: str) -> bool:
        """Check if a path exists"""
        return os.path.exists(path)
    
    def _get_rocm_version(self) -> Optional[str]:
        """Get ROCm version from system
        
        Returns:
            ROCm version string or None if not found
        """
        # Try hipconfig first
        success, stdout, _ = self._run_command(['hipconfig', '--version'])
        if success and stdout:
            return stdout.split('-')[0]  # Remove build suffix
        
        # Try version file
        version_file = '/opt/rocm/.info/version'
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    version = f.read().strip().split('-')[0]
                    return version
            except Exception:
                pass
        
        return None
    
    def _check_gpu_accessible(self) -> Tuple[bool, str]:
        """Check if GPUs are accessible
        
        Returns:
            Tuple of (accessible, message)
        """
        # Try rocminfo first
        success, stdout, stderr = self._run_command(['rocminfo'])
        if success:
            # Check if any GPU agents are listed
            if 'Agent' in stdout and 'gfx' in stdout.lower():
                return True, "GPUs accessible via rocminfo"
            else:
                return False, "rocminfo ran but no GPU agents detected"
        
        # Try amd-smi
        success, stdout, stderr = self._run_command(['amd-smi', 'list'])
        if success and stdout:
            return True, "GPUs accessible via amd-smi"
        
        # Try rocm-smi
        success, stdout, stderr = self._run_command(['rocm-smi'])
        if success and stdout:
            return True, "GPUs accessible via rocm-smi"
        
        return False, "No GPU management tool could detect GPUs"
    
    def _check_kfd_driver(self) -> Tuple[bool, List[str], List[str]]:
        """Check if KFD driver is loaded
        
        Returns:
            Tuple of (loaded, critical_issues, warnings)
        """
        critical_issues = []
        warnings = []
        
        # Check /dev/kfd - this is critical
        if not self._check_path_exists('/dev/kfd'):
            critical_issues.append("/dev/kfd device not found - KFD driver may not be loaded")
        
        # Check KFD topology - this is critical
        if not self._check_path_exists('/sys/devices/virtual/kfd/kfd/topology/nodes'):
            critical_issues.append("KFD topology not found - GPU topology may not be available")
        
        # Check dmesg for amdgpu module - this is just a warning if other checks pass
        success, stdout, _ = self._run_command(['dmesg'], timeout=5)
        if success:
            if 'amdgpu' not in stdout.lower():
                warnings.append("amdgpu driver messages not found in dmesg")
        
        return len(critical_issues) == 0, critical_issues, warnings
    
    def validate(self) -> GPUValidationResult:
        """Perform comprehensive ROCm validation
        
        Returns:
            GPUValidationResult with validation results
        """
        result = GPUValidationResult(is_valid=True, vendor=GPUVendor.AMD)
        
        if self.verbose:
            print("=" * 70)
            print("ROCm Installation Validation")
            print("=" * 70)
        
        # 1. Check essential paths
        if self.verbose:
            print("\n[1/6] Checking essential ROCm paths...")
        
        for name, path in self.ESSENTIAL_PATHS.items():
            if not self._check_path_exists(path):
                result.is_valid = False
                result.issues.append(f"Essential path missing: {path} ({name})")
                if self.verbose:
                    print(f"  ✗ {name}: NOT FOUND at {path}")
            else:
                if self.verbose:
                    print(f"  ✓ {name}: Found at {path}")
        
        # 2. Get ROCm version
        if self.verbose:
            print("\n[2/6] Detecting ROCm version...")
        
        version = self._get_rocm_version()
        if version:
            result.version = version
            if self.verbose:
                print(f"  ✓ ROCm version: {version}")
        else:
            result.is_valid = False
            result.issues.append("Unable to detect ROCm version")
            if self.verbose:
                print(f"  ✗ ROCm version: NOT DETECTED")
        
        # 3. Check recommended tools
        if self.verbose:
            print("\n[3/6] Checking recommended ROCm tools...")
        
        has_smi = False
        for name, path in self.RECOMMENDED_PATHS.items():
            if self._check_path_exists(path):
                has_smi = True
                if self.verbose:
                    print(f"  ✓ {name}: Found at {path}")
            else:
                if self.verbose:
                    print(f"  ⚠ {name}: NOT FOUND at {path}")
        
        if not has_smi:
            result.warnings.append("No GPU management tool (amd-smi/rocm-smi) found")
            result.suggestions.append("Install ROCm SMI tools for GPU monitoring")
        
        # 4. Check KFD driver
        if self.verbose:
            print("\n[4/6] Checking KFD driver...")
        
        kfd_ok, kfd_critical_issues, kfd_warnings = self._check_kfd_driver()
        
        # 5. Check GPU accessibility
        if self.verbose:
            print("\n[5/6] Checking GPU accessibility...")
        
        gpu_accessible, gpu_msg = self._check_gpu_accessible()
        if gpu_accessible:
            if self.verbose:
                print(f"  ✓ {gpu_msg}")
        else:
            result.is_valid = False
            result.issues.append(gpu_msg)
            if self.verbose:
                print(f"  ✗ {gpu_msg}")
        
        # Now decide how to handle KFD issues based on GPU accessibility
        # If GPUs are accessible, treat KFD dmesg warnings as non-critical
        if not kfd_ok:
            if gpu_accessible:
                # GPUs work despite KFD warnings - downgrade to warnings
                result.warnings.extend(kfd_critical_issues)
                result.warnings.extend(kfd_warnings)
                if self.verbose:
                    for issue in kfd_critical_issues:
                        print(f"  ⚠ {issue}")
                    for warning in kfd_warnings:
                        print(f"  ⚠ {warning}")
            else:
                # GPUs not accessible and KFD has issues - critical
                result.is_valid = False
                result.issues.extend(kfd_critical_issues)
                result.warnings.extend(kfd_warnings)
                if self.verbose:
                    for issue in kfd_critical_issues:
                        print(f"  ✗ {issue}")
                    for warning in kfd_warnings:
                        print(f"  ⚠ {warning}")
        else:
            # KFD is OK, but check for warnings
            result.warnings.extend(kfd_warnings)
            if self.verbose:
                print(f"  ✓ KFD driver loaded")
                for warning in kfd_warnings:
                    print(f"  ⚠ {warning}")
        
        # 6. Check permissions
        if self.verbose:
            print("\n[6/6] Checking permissions...")
        
        if os.path.exists('/dev/kfd'):
            try:
                # Try to access /dev/kfd
                if os.access('/dev/kfd', os.R_OK | os.W_OK):
                    if self.verbose:
                        print(f"  ✓ /dev/kfd is accessible")
                else:
                    result.warnings.append("Current user may not have permission to access /dev/kfd")
                    result.suggestions.append("Add user to 'video' or 'render' group: sudo usermod -aG video,render $USER")
                    if self.verbose:
                        print(f"  ⚠ /dev/kfd exists but may not be accessible by current user")
            except Exception as e:
                result.warnings.append(f"Unable to check /dev/kfd permissions: {e}")
        
        # Generate suggestions based on issues
        if result.issues:
            if not self._check_path_exists('/opt/rocm'):
                result.suggestions.append(
                    "ROCm does not appear to be installed. Install ROCm: "
                    "https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
                )
            
            if "KFD driver" in str(result.issues):
                result.suggestions.append(
                    "Load amdgpu kernel module: sudo modprobe amdgpu"
                )
                result.suggestions.append(
                    "Reboot the system after ROCm installation to ensure kernel drivers are loaded"
                )
        
        # Print summary
        if self.verbose:
            print("\n" + "=" * 70)
            print("Validation Summary")
            print("=" * 70)
            if result.is_valid:
                print("✓ ROCm installation is VALID")
            else:
                print("✗ ROCm installation has ISSUES")
            
            if result.version:
                print(f"\nROCm Version: {result.version}")
            
            if result.issues:
                print(f"\nIssues Found ({len(result.issues)}):")
                for i, issue in enumerate(result.issues, 1):
                    print(f"  {i}. {issue}")
            
            if result.warnings:
                print(f"\nWarnings ({len(result.warnings)}):")
                for i, warning in enumerate(result.warnings, 1):
                    print(f"  {i}. {warning}")
            
            if result.suggestions:
                print(f"\nSuggestions ({len(result.suggestions)}):")
                for i, suggestion in enumerate(result.suggestions, 1):
                    print(f"  {i}. {suggestion}")
            
            print("=" * 70)
        
        return result
    
    def get_error_message(self, result: GPUValidationResult) -> str:
        """Generate a detailed error message from validation result
        
        Args:
            result: ROCmValidationResult from validate()
            
        Returns:
            Formatted error message string
        """
        if result.is_valid:
            return ""
        
        lines = ["ROCm installation validation FAILED:"]
        lines.append("")
        
        if result.issues:
            lines.append("Critical Issues:")
            for issue in result.issues:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        if result.suggestions:
            lines.append("Suggested Actions:")
            for suggestion in result.suggestions:
                lines.append(f"  • {suggestion}")
            lines.append("")
        
        lines.append("Please ensure ROCm is properly installed before running madengine.")
        lines.append("Installation guide: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html")
        
        return "\n".join(lines)


class NVIDIAValidator:
    """Validator for NVIDIA CUDA installation"""
    
    def __init__(self, verbose: bool = False):
        """Initialize NVIDIA validator
        
        Args:
            verbose: If True, print detailed validation progress
        """
        self.verbose = verbose
    
    def _run_command(self, cmd: List[str], timeout: int = 10) -> Tuple[bool, str, str]:
        """Run a command and return success status and output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except FileNotFoundError:
            return False, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, "", str(e)
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version from nvidia-smi or nvcc"""
        # Try nvidia-smi first
        success, stdout, _ = self._run_command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])
        if success and stdout:
            return stdout.split('\n')[0].strip()
        
        # Try nvcc as fallback
        success, stdout, _ = self._run_command(['nvcc', '--version'])
        if success and 'release' in stdout.lower():
            # Extract version from output like "release 11.8, V11.8.89"
            import re
            match = re.search(r'release (\d+\.\d+)', stdout)
            if match:
                return match.group(1)
        
        return None
    
    def validate(self) -> GPUValidationResult:
        """Perform NVIDIA CUDA validation
        
        Returns:
            GPUValidationResult with validation results
        """
        result = GPUValidationResult(is_valid=True, vendor=GPUVendor.NVIDIA)
        
        if self.verbose:
            print("=" * 70)
            print("NVIDIA GPU (CUDA) Validation")
            print("=" * 70)
            print()
        
        # 1. Check nvidia-smi
        if self.verbose:
            print("[1/4] Checking nvidia-smi availability...")
        
        if not os.path.exists("/usr/bin/nvidia-smi"):
            result.is_valid = False
            result.issues.append("nvidia-smi not found at /usr/bin/nvidia-smi")
            if self.verbose:
                print("  ✗ nvidia-smi: NOT FOUND")
        else:
            if self.verbose:
                print("  ✓ nvidia-smi: Found")
        
        # 2. Test nvidia-smi execution
        if self.verbose:
            print("\n[2/4] Testing nvidia-smi execution...")
        
        success, stdout, stderr = self._run_command(['nvidia-smi', '--list-gpus'])
        if not success:
            result.is_valid = False
            result.issues.append(f"nvidia-smi failed to execute: {stderr}")
            if self.verbose:
                print(f"  ✗ nvidia-smi execution failed: {stderr}")
        else:
            if self.verbose:
                print("  ✓ nvidia-smi executed successfully")
        
        # 3. Get CUDA version
        if self.verbose:
            print("\n[3/4] Detecting CUDA version...")
        
        version = self._get_cuda_version()
        if version:
            result.version = version
            if self.verbose:
                print(f"  ✓ CUDA/Driver version: {version}")
        else:
            result.warnings.append("Unable to detect CUDA version")
            if self.verbose:
                print("  ⚠ Could not detect CUDA version")
        
        # 4. Count GPUs
        if self.verbose:
            print("\n[4/4] Counting available GPUs...")
        
        success, stdout, _ = self._run_command(['nvidia-smi', '--list-gpus'])
        if success and stdout:
            gpu_count = len([line for line in stdout.split('\n') if line.strip()])
            if gpu_count > 0:
                if self.verbose:
                    print(f"  ✓ Found {gpu_count} GPU(s)")
                    for line in stdout.split('\n'):
                        if line.strip():
                            print(f"     - {line.strip()}")
            else:
                result.warnings.append("No GPUs detected")
                if self.verbose:
                    print("  ⚠ No GPUs detected")
        
        # Generate suggestions
        if result.issues:
            if "nvidia-smi not found" in str(result.issues):
                result.suggestions.append(
                    "Install NVIDIA drivers and CUDA toolkit: "
                    "https://developer.nvidia.com/cuda-downloads"
                )
            if "failed to execute" in str(result.issues):
                result.suggestions.append("Check if NVIDIA drivers are properly loaded: lsmod | grep nvidia")
                result.suggestions.append("Try reinstalling NVIDIA drivers")
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("NVIDIA Validation Summary")
            print("=" * 70)
            if result.is_valid:
                print("✓ NVIDIA GPU installation is VALID")
            else:
                print("✗ NVIDIA GPU installation has ISSUES")
            
            if result.version:
                print(f"\nDriver/CUDA Version: {result.version}")
            
            if result.issues:
                print(f"\nIssues Found ({len(result.issues)}):")
                for i, issue in enumerate(result.issues, 1):
                    print(f"  {i}. {issue}")
            
            if result.warnings:
                print(f"\nWarnings ({len(result.warnings)}):")
                for i, warning in enumerate(result.warnings, 1):
                    print(f"  {i}. {warning}")
            
            if result.suggestions:
                print(f"\nSuggestions ({len(result.suggestions)}):")
                for i, suggestion in enumerate(result.suggestions, 1):
                    print(f"  {i}. {suggestion}")
            
            print("=" * 70)
        
        return result


def detect_gpu_vendor() -> GPUVendor:
    """Detect which GPU vendor is present on the system
    
    Returns:
        GPUVendor enum value
    """
    if os.path.exists("/usr/bin/nvidia-smi"):
        return GPUVendor.NVIDIA
    elif os.path.exists("/opt/rocm/bin/rocm-smi") or os.path.exists("/opt/rocm/bin/amd-smi"):
        return GPUVendor.AMD
    else:
        return GPUVendor.UNKNOWN


def validate_gpu_installation(vendor: Optional[GPUVendor] = None, verbose: bool = False, raise_on_error: bool = True) -> GPUValidationResult:
    """Validate GPU installation on the current node
    
    Args:
        vendor: GPU vendor to validate (auto-detected if None)
        verbose: Print detailed validation progress
        raise_on_error: Raise GPUInstallationError if validation fails
        
    Returns:
        GPUValidationResult
        
    Raises:
        GPUInstallationError: If validation fails and raise_on_error is True
    """
    if vendor is None:
        vendor = detect_gpu_vendor()
    
    if vendor == GPUVendor.AMD:
        validator = ROCmValidator(verbose=verbose)
        rocm_result = validator.validate()
        # Convert ROCmValidationResult to GPUValidationResult
        result = GPUValidationResult(
            is_valid=rocm_result.is_valid,
            vendor=GPUVendor.AMD,
            version=rocm_result.version,
            issues=rocm_result.issues,
            warnings=rocm_result.warnings,
            suggestions=rocm_result.suggestions
        )
    elif vendor == GPUVendor.NVIDIA:
        validator = NVIDIAValidator(verbose=verbose)
        result = validator.validate()
    else:
        result = GPUValidationResult(is_valid=False, vendor=GPUVendor.UNKNOWN)
        result.issues.append("No GPU vendor detected")
        result.suggestions.append("Install NVIDIA drivers (https://developer.nvidia.com/cuda-downloads)")
        result.suggestions.append("Or install AMD ROCm (https://rocm.docs.amd.com)")
    
    if not result.is_valid and raise_on_error:
        raise GPUInstallationError(result)
    
    return result


class GPUInstallationError(RuntimeError):
    """Exception raised when GPU installation validation fails"""
    
    def __init__(self, validation_result: GPUValidationResult):
        """Initialize with validation result
        
        Args:
            validation_result: GPUValidationResult from validation
        """
        self.validation_result = validation_result
        message = self._format_error_message(validation_result)
        super().__init__(message)
    
    def _format_error_message(self, result: GPUValidationResult) -> str:
        """Generate a detailed error message from validation result"""
        if result.is_valid:
            return ""
        
        lines = [f"{result.vendor.value} GPU installation validation FAILED:"]
        lines.append("")
        
        if result.issues:
            lines.append("Critical Issues:")
            for issue in result.issues:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        if result.suggestions:
            lines.append("Suggested Actions:")
            for suggestion in result.suggestions:
                lines.append(f"  • {suggestion}")
            lines.append("")
        
        vendor_docs = {
            GPUVendor.AMD: "https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html",
            GPUVendor.NVIDIA: "https://developer.nvidia.com/cuda-downloads",
        }
        
        lines.append(f"Please ensure {result.vendor.value} GPU drivers and tools are properly installed.")
        if result.vendor in vendor_docs:
            lines.append(f"Installation guide: {vendor_docs[result.vendor]}")
        
        return "\n".join(lines)


# Backwards compatibility aliases
ROCmValidationResult = GPUValidationResult  # For backwards compatibility
ROCmInstallationError = GPUInstallationError  # For backwards compatibility


def validate_rocm_installation(verbose: bool = False, raise_on_error: bool = True) -> GPUValidationResult:
    """Validate ROCm installation on the current node (backwards compatibility wrapper)
    
    Args:
        verbose: Print detailed validation progress
        raise_on_error: Raise GPUInstallationError if validation fails
        
    Returns:
        GPUValidationResult
        
    Raises:
        GPUInstallationError: If validation fails and raise_on_error is True
    """
    return validate_gpu_installation(vendor=GPUVendor.AMD, verbose=verbose, raise_on_error=raise_on_error)


if __name__ == "__main__":
    # Command-line usage
    import sys
    
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    result = validate_gpu_installation(vendor=None, verbose=verbose, raise_on_error=False)
    
    if result.is_valid:
        print(f"\n✓ {result.vendor.value} GPU installation validated successfully")
        if result.version:
            print(f"Version: {result.version}")
    
    sys.exit(0 if result.is_valid else 1)

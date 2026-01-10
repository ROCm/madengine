#!/usr/bin/env python3
"""
GPU Passthrough Configuration for KVM VMs.

Supports multiple GPU passthrough modes:
- SR-IOV (Single Root I/O Virtualization)
- VFIO (Virtual Function I/O) - full GPU passthrough
- vGPU (Virtual GPU) - for NVIDIA GRID/AMD MxGPU

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from enum import Enum


class GPUPassthroughMode(Enum):
    """GPU passthrough modes."""
    SRIOV = "sriov"           # SR-IOV Virtual Functions
    VFIO = "vfio"             # Full GPU passthrough
    VGPU = "vgpu"             # Virtual GPU (NVIDIA GRID/AMD MxGPU)
    NONE = "none"             # No GPU passthrough


class GPUPassthroughManager:
    """
    Manages GPU passthrough configuration for VMs.
    
    Handles:
    - GPU PCI device discovery
    - SR-IOV Virtual Function creation
    - VFIO driver binding
    - IOMMU group validation
    - Resource cleanup
    """
    
    def __init__(self):
        """Initialize GPU passthrough manager."""
        self.active_vfs: List[str] = []  # Track active Virtual Functions
        self.bound_devices: List[str] = []  # Track VFIO-bound devices
    
    def validate_iommu_enabled(self) -> bool:
        """
        Check if IOMMU is enabled (required for GPU passthrough).
        
        Returns:
            True if IOMMU is enabled
        """
        try:
            result = subprocess.run(
                ["dmesg"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return "IOMMU enabled" in result.stdout or "AMD-Vi" in result.stdout or "DMAR" in result.stdout
        except:
            return False
    
    def find_gpu_devices(self, vendor: str = "AMD") -> List[Dict[str, str]]:
        """
        Find GPU devices on the system.
        
        Args:
            vendor: GPU vendor ("AMD" or "NVIDIA")
            
        Returns:
            List of GPU device info dicts
        """
        devices = []
        
        # AMD PCI vendor ID: 1002, NVIDIA: 10de
        vendor_id = "1002" if vendor.upper() == "AMD" else "10de"
        
        try:
            result = subprocess.run(
                ["lspci", "-D", "-nn", "-d", f"{vendor_id}:"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                
                # Parse PCI address and device info
                match = re.match(r"^([0-9a-f:\.]+)\s+(.+?)\s+\[([0-9a-f]{4}):([0-9a-f]{4})\]", line)
                if match:
                    pci_addr = match.group(1)
                    device_name = match.group(2)
                    vendor_id = match.group(3)
                    device_id = match.group(4)
                    
                    # Filter out non-GPU devices (audio controllers, etc.)
                    if "VGA" in device_name or "Display" in device_name or "3D" in device_name:
                        devices.append({
                            "pci_address": pci_addr,
                            "name": device_name,
                            "vendor_id": vendor_id,
                            "device_id": device_id
                        })
        except Exception as e:
            print(f"Warning: Could not enumerate GPU devices: {e}")
        
        return devices
    
    def get_iommu_group(self, pci_address: str) -> Optional[str]:
        """
        Get IOMMU group for a PCI device.
        
        Args:
            pci_address: PCI address (e.g., "0000:01:00.0")
            
        Returns:
            IOMMU group number or None
        """
        iommu_path = f"/sys/bus/pci/devices/{pci_address}/iommu_group"
        
        if os.path.exists(iommu_path):
            # Read the symlink to get group number
            group_link = os.readlink(iommu_path)
            group_num = os.path.basename(group_link)
            return group_num
        
        return None
    
    def check_sriov_capable(self, pci_address: str) -> Tuple[bool, int]:
        """
        Check if a GPU supports SR-IOV and max VFs.
        
        Args:
            pci_address: PCI address
            
        Returns:
            (is_capable, max_vfs)
        """
        sriov_totalvfs_path = f"/sys/bus/pci/devices/{pci_address}/sriov_totalvfs"
        
        if os.path.exists(sriov_totalvfs_path):
            try:
                with open(sriov_totalvfs_path, 'r') as f:
                    max_vfs = int(f.read().strip())
                    return (max_vfs > 0, max_vfs)
            except:
                pass
        
        return (False, 0)
    
    def enable_sriov(self, pci_address: str, num_vfs: int = 1) -> List[str]:
        """
        Enable SR-IOV on a GPU and create Virtual Functions.
        
        Args:
            pci_address: Physical Function PCI address
            num_vfs: Number of Virtual Functions to create
            
        Returns:
            List of VF PCI addresses
        """
        # Check if SR-IOV is supported
        is_capable, max_vfs = self.check_sriov_capable(pci_address)
        if not is_capable:
            raise RuntimeError(f"GPU {pci_address} does not support SR-IOV")
        
        if num_vfs > max_vfs:
            raise ValueError(
                f"Requested {num_vfs} VFs but GPU only supports {max_vfs}"
            )
        
        # Enable VFs via sysfs
        sriov_numvfs_path = f"/sys/bus/pci/devices/{pci_address}/sriov_numvfs"
        
        try:
            # First disable any existing VFs
            subprocess.run(
                ["sudo", "sh", "-c", f"echo 0 > {sriov_numvfs_path}"],
                check=True,
                timeout=10
            )
            
            # Enable requested number of VFs
            subprocess.run(
                ["sudo", "sh", "-c", f"echo {num_vfs} > {sriov_numvfs_path}"],
                check=True,
                timeout=10
            )
            
            # Discover VF addresses
            vf_addresses = self._discover_vf_addresses(pci_address, num_vfs)
            self.active_vfs.extend(vf_addresses)
            
            return vf_addresses
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to enable SR-IOV on {pci_address}: {e}")
    
    def disable_sriov(self, pci_address: str):
        """
        Disable SR-IOV on a GPU.
        
        Args:
            pci_address: Physical Function PCI address
        """
        sriov_numvfs_path = f"/sys/bus/pci/devices/{pci_address}/sriov_numvfs"
        
        if os.path.exists(sriov_numvfs_path):
            try:
                subprocess.run(
                    ["sudo", "sh", "-c", f"echo 0 > {sriov_numvfs_path}"],
                    check=True,
                    timeout=10
                )
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to disable SR-IOV on {pci_address}: {e}")
    
    def _discover_vf_addresses(self, pf_address: str, num_vfs: int) -> List[str]:
        """
        Discover PCI addresses of Virtual Functions.
        
        Args:
            pf_address: Physical Function address
            num_vfs: Expected number of VFs
            
        Returns:
            List of VF PCI addresses
        """
        vf_addresses = []
        
        # VFs are listed in sysfs under the PF
        virtfn_dir = f"/sys/bus/pci/devices/{pf_address}"
        
        for i in range(num_vfs):
            virtfn_link = os.path.join(virtfn_dir, f"virtfn{i}")
            if os.path.exists(virtfn_link):
                # Read symlink to get VF address
                vf_path = os.readlink(virtfn_link)
                vf_addr = os.path.basename(vf_path)
                vf_addresses.append(vf_addr)
        
        return vf_addresses
    
    def bind_to_vfio(self, pci_address: str):
        """
        Bind a GPU to VFIO driver for passthrough.
        
        Args:
            pci_address: PCI address of GPU
        """
        try:
            # Get current driver
            driver_path = f"/sys/bus/pci/devices/{pci_address}/driver"
            current_driver = None
            if os.path.exists(driver_path):
                current_driver = os.path.basename(os.readlink(driver_path))
            
            # Unbind from current driver
            if current_driver:
                unbind_path = f"/sys/bus/pci/drivers/{current_driver}/unbind"
                subprocess.run(
                    ["sudo", "sh", "-c", f"echo {pci_address} > {unbind_path}"],
                    check=False  # May fail if already unbound
                )
            
            # Get vendor and device IDs
            vendor_id = self._read_sysfs(f"/sys/bus/pci/devices/{pci_address}/vendor")
            device_id = self._read_sysfs(f"/sys/bus/pci/devices/{pci_address}/device")
            
            if vendor_id and device_id:
                # Remove 0x prefix
                vendor_id = vendor_id.replace("0x", "")
                device_id = device_id.replace("0x", "")
                
                # Bind to vfio-pci
                subprocess.run(
                    ["sudo", "modprobe", "vfio-pci"],
                    check=True
                )
                
                subprocess.run(
                    ["sudo", "sh", "-c", 
                     f"echo {vendor_id} {device_id} > /sys/bus/pci/drivers/vfio-pci/new_id"],
                    check=False  # May already be registered
                )
                
                subprocess.run(
                    ["sudo", "sh", "-c",
                     f"echo {pci_address} > /sys/bus/pci/drivers/vfio-pci/bind"],
                    check=True
                )
                
                self.bound_devices.append(pci_address)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to bind {pci_address} to VFIO: {e}")
    
    def unbind_from_vfio(self, pci_address: str):
        """
        Unbind a GPU from VFIO driver.
        
        Args:
            pci_address: PCI address of GPU
        """
        try:
            unbind_path = "/sys/bus/pci/drivers/vfio-pci/unbind"
            subprocess.run(
                ["sudo", "sh", "-c", f"echo {pci_address} > {unbind_path}"],
                check=False  # May fail if not bound
            )
            
            if pci_address in self.bound_devices:
                self.bound_devices.remove(pci_address)
        except:
            pass
    
    def _read_sysfs(self, path: str) -> Optional[str]:
        """Read a sysfs file safely."""
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except:
            return None
    
    def configure_passthrough(self, mode: GPUPassthroughMode, 
                             pci_addresses: List[str],
                             num_vfs: int = 1) -> List[str]:
        """
        Configure GPU passthrough based on mode.
        
        Args:
            mode: Passthrough mode (SRIOV, VFIO, VGPU)
            pci_addresses: List of GPU PCI addresses
            num_vfs: Number of VFs for SR-IOV mode
            
        Returns:
            List of PCI addresses to pass to VM
        """
        if mode == GPUPassthroughMode.NONE:
            return []
        
        vm_gpu_addresses = []
        
        for pci_addr in pci_addresses:
            if mode == GPUPassthroughMode.SRIOV:
                # Enable SR-IOV and use VF
                vf_addresses = self.enable_sriov(pci_addr, num_vfs)
                # Use first VF for VM
                if vf_addresses:
                    vm_gpu_addresses.append(vf_addresses[0])
            
            elif mode == GPUPassthroughMode.VFIO:
                # Bind GPU to VFIO for full passthrough
                self.bind_to_vfio(pci_addr)
                vm_gpu_addresses.append(pci_addr)
            
            elif mode == GPUPassthroughMode.VGPU:
                # vGPU configuration (vendor-specific)
                # For now, just pass through the address
                vm_gpu_addresses.append(pci_addr)
        
        return vm_gpu_addresses
    
    def cleanup_passthrough(self, mode: GPUPassthroughMode, 
                           pci_addresses: List[str]):
        """
        Clean up GPU passthrough configuration.
        
        Args:
            mode: Passthrough mode
            pci_addresses: List of GPU PCI addresses (Physical Functions)
        """
        if mode == GPUPassthroughMode.SRIOV:
            # Disable SR-IOV
            for pci_addr in pci_addresses:
                self.disable_sriov(pci_addr)
            self.active_vfs.clear()
        
        elif mode == GPUPassthroughMode.VFIO:
            # Unbind from VFIO
            for pci_addr in self.bound_devices[:]:
                self.unbind_from_vfio(pci_addr)
    
    def verify_passthrough_ready(self) -> Tuple[bool, List[str]]:
        """
        Verify system is ready for GPU passthrough.
        
        Returns:
            (is_ready, list_of_issues)
        """
        issues = []
        
        # Check IOMMU enabled
        if not self.validate_iommu_enabled():
            issues.append("IOMMU not enabled in kernel (add intel_iommu=on or amd_iommu=on to boot params)")
        
        # Check vfio-pci module available
        result = subprocess.run(
            ["modinfo", "vfio-pci"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            issues.append("vfio-pci kernel module not available")
        
        # Check for GPUs
        amd_gpus = self.find_gpu_devices("AMD")
        nvidia_gpus = self.find_gpu_devices("NVIDIA")
        
        if not amd_gpus and not nvidia_gpus:
            issues.append("No GPUs detected")
        
        return (len(issues) == 0, issues)
    
    def get_gpu_info(self, pci_address: str) -> Dict[str, str]:
        """
        Get detailed information about a GPU.
        
        Args:
            pci_address: PCI address
            
        Returns:
            Dict with GPU info
        """
        info = {
            "pci_address": pci_address,
            "iommu_group": self.get_iommu_group(pci_address) or "N/A",
        }
        
        # Check SR-IOV capability
        is_sriov, max_vfs = self.check_sriov_capable(pci_address)
        info["sriov_capable"] = str(is_sriov)
        info["max_vfs"] = str(max_vfs)
        
        # Get current driver
        driver_path = f"/sys/bus/pci/devices/{pci_address}/driver"
        if os.path.exists(driver_path):
            info["driver"] = os.path.basename(os.readlink(driver_path))
        else:
            info["driver"] = "none"
        
        return info

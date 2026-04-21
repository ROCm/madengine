#!/usr/bin/env python3
"""Module to get GPU information using amd-smi

This module contains the class ProfUtils to get GPU information using amd-smi.
This script maintains API consistency across GPU vendor utilities.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
import sys
import logging
from typing import List, Optional, Dict, Any

sys.path.append("/opt/rocm/libexec/amdsmi_cli/")
try:
    from amdsmi_init import amdsmi_interface
    from amdsmi_init import amdsmi_cli_init, amdsmi_cli_shutdown
except ImportError:
    raise ImportError("Could not import /opt/rocm/libexec/amdsmi_cli/amdsmi_init.py")


class ProfUtils:
    """Class to get GPU information using AMD amd-smi utility.
    
    Attributes:
        amdsmi_initialized: Whether amdsmi interface is initialized.
        processor_handles: List of GPU processor handles.
    """
    
    def __init__(self, mode) -> None:
        """Initialize the amd-smi utils class
        
        @param mode: Mode parameter for compatibility (not used in amd-smi)
        """
        self.amdsmi_initialized = False
        self.processor_handles = []
        
        try:
            # Initialize amdsmi using the amdsmi_cli_init function
            amdsmi_cli_init()
            self.amdsmi_initialized = True
            logging.debug("amdsmi_cli_init() successful")
        except Exception as e:
            raise ImportError(f"Failed to initialize amd-smi interface: {e}")
        
        try:
            # Get processor handles (GPU devices)
            self.processor_handles = amdsmi_interface.amdsmi_get_processor_handles()
            if not self.processor_handles:
                raise ImportError("No GPU devices found via amdsmi")
            logging.debug(f"Found {len(self.processor_handles)} GPU devices")
        except Exception as e:
            raise ImportError(f"Failed to get GPU processor handles: {e}")

    def get_power(self, device: int) -> str:
        """Get current socket power of a given device.
        
        Args:
            device: GPU device index.
            
        Returns:
            Power consumption in watts as string, or 'N/A' if unavailable.
        """
        try:
            if device >= len(self.processor_handles):
                return 'N/A'
            
            processor_handle = self.processor_handles[device]
            power_info = amdsmi_interface.amdsmi_get_power_info(processor_handle)
            
            # power_info is a dict with keys like 'current_socket_power', 'average_socket_power', etc.
            # Values are in milliwatts, convert to watts
            if 'current_socket_power' in power_info:
                power_mw = power_info['current_socket_power']
                return str(float(power_mw) / 1000.0)
            elif 'average_socket_power' in power_info:
                power_mw = power_info['average_socket_power']
                return str(float(power_mw) / 1000.0)
            
            return 'N/A'
        except Exception as e:
            logging.debug(f"Failed to get power for device {device}: {e}")
            return 'N/A'

    def list_devices(self) -> List[int]:
        """Get list of GPU device indices.
        
        Returns:
            List of device indices.
        """
        # Return indices based on the number of processor handles we got
        return list(range(len(self.processor_handles)))

    def get_mem_info(self, device: int) -> float:
        """Get memory usage percentage for a device.
        
        Args:
            device: GPU device index.
            
        Returns:
            Memory usage percentage as float.
        """
        try:
            if device >= len(self.processor_handles):
                return 0.0
            
            processor_handle = self.processor_handles[device]
            
            # Try to get VRAM usage directly
            vram_info = amdsmi_interface.amdsmi_get_gpu_vram_usage(processor_handle)
            
            # vram_info is a dict with 'vram_used' and 'vram_total' in bytes
            if isinstance(vram_info, dict) and 'vram_used' in vram_info and 'vram_total' in vram_info:
                used = float(vram_info['vram_used'])
                total = float(vram_info['vram_total'])
                if total > 0:
                    return round((used / total) * 100, 2)
            
            return 0.0
        except Exception as e:
            logging.debug(f"Failed to get memory info for device {device}: {e}")
            return 0.0

    def check_if_secondary_die(self, device: int) -> bool:
        """Check if GPU device is the secondary die in a MCM.
        
        MI200 device specific feature check.
        The secondary dies lack power management features.
        
        Args:
            device: The device to check.
            
        Returns:
            True if secondary die, False otherwise.
        """
        try:
            if device >= len(self.processor_handles):
                return False
            
            processor_handle = self.processor_handles[device]
            
            # Check if power management is enabled - secondary dies typically don't have it
            is_power_mgmt_enabled = amdsmi_interface.amdsmi_is_gpu_power_management_enabled(processor_handle)
            if not is_power_mgmt_enabled:
                return True
            
            # Alternative check: get power info and see if it's zero/unavailable
            try:
                power_info = amdsmi_interface.amdsmi_get_power_info(processor_handle)
                if isinstance(power_info, dict):
                    # If both current and average power are 0, it's likely a secondary die
                    current_power = power_info.get('current_socket_power', -1)
                    avg_power = power_info.get('average_socket_power', -1)
                    if current_power == 0 and avg_power == 0:
                        return True
            except:
                # If we can't get power info, might be secondary die
                return True
            
            return False
        except Exception as e:
            logging.debug(f"Failed to check secondary die for device {device}: {e}")
            return False
#!/usr/bin/env python3
"""Module to get GPU information using pynvml

This module contains the class ProfUtils to get GPU information using pynvml.
This script maintains API consistency across GPU vendor utilities.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import typing
import logging
from typing import Optional, List

# third-party modules
import pynvml


class ProfUtils:
    """Class to get GPU information using NVIDIA pynvml library.
    
    Attributes:
        device_count: Number of NVIDIA GPUs detected.
        handles: List of NVML device handles.
        device_list: List of device indices.
    """

    def __init__(self, mode: str) -> None:
        """Initialize the NVIDIA profiler utility.
        
        Args:
            mode: Mode parameter for API compatibility (not used for NVIDIA).
            
        Raises:
            RuntimeError: If NVML initialization fails.
        """
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}")
        
        try:
            self.device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to get device count: {e}")
        
        self.handles: List = []
        self.device_list: List[int] = []
        
        for i in range(self.device_count):
            try:
                self.device_list.append(i)
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            except pynvml.NVMLError as e:
                logging.warning(f"Failed to get handle for device {i}: {e}")

    def get_power(self, device: int) -> str:
        """Get current power consumption of a GPU device.
        
        Args:
            device: GPU device index.
            
        Returns:
            Power consumption in watts as string, or 'N/A' if unavailable.
        """
        if device < 0 or device >= len(self.handles):
            logging.error(f"Invalid device index: {device}")
            return 'N/A'
        
        try:
            # nvmlDeviceGetPowerUsage returns milliwatts
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handles[device])
            power_watts = float(power_mw) / 1000.0
            return str(round(power_watts, 2))
        except pynvml.NVMLError as e:
            logging.debug(f"Failed to get power for device {device}: {e}")
            return 'N/A'

    def list_devices(self) -> List[int]:
        """Get list of available GPU device indices.
        
        Returns:
            List of device indices.
        """
        return self.device_list.copy()

    def get_mem_info(self, device: int) -> float:
        """Get memory usage percentage for a GPU device.
        
        Args:
            device: GPU device index.
            
        Returns:
            Memory usage percentage as float (0-100).
        """
        if device < 0 or device >= len(self.handles):
            logging.error(f"Invalid device index: {device}")
            return 0.0
        
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handles[device])
            if info.total > 0:
                usage_percent = (float(info.used) / float(info.total)) * 100.0
                return round(usage_percent, 2)
            return 0.0
        except pynvml.NVMLError as e:
            logging.debug(f"Failed to get memory info for device {device}: {e}")
            return 0.0

    def check_if_secondary_die(self, device: int) -> bool:
        """Check if device is a secondary die.
        
        This method is provided for API compatibility with AMD utils.
        NVIDIA GPUs do not have the concept of secondary dies like AMD MCM GPUs.
        
        Args:
            device: GPU device index.
            
        Returns:
            Always False for NVIDIA GPUs.
        """
        return False

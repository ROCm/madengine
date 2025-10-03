#!/usr/bin/env python3
"""Module to get GPU information using amd-smi

This module contains the class ProfUtils to get GPU information using amd-smi.
This script maintains API consistency across GPU vendor utilities.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
import subprocess
import json
import re
import logging
from typing import List, Union, Optional, Dict, Any


class ProfUtils:
    """Class to get GPU information using AMD amd-smi utility.
    
    Attributes:
        amd_smi_available: Whether amd-smi command is available.
    """
    
    def __init__(self, mode) -> None:
        """Initialize the amd-smi utils class
        
        @param mode: Mode parameter for compatibility (not used in amd-smi)
        """
        self.amd_smi_available = self._check_amd_smi_available()
        if not self.amd_smi_available:
            raise ImportError("amd-smi command not found or not accessible")
        
        # Test if we can get device list to verify amd-smi is working
        try:
            self._run_amd_smi_command(['list'])
        except Exception as e:
            raise ImportError(f"amd-smi is not working properly: {e}")

    def _check_amd_smi_available(self) -> bool:
        """Check if amd-smi command is available"""
        try:
            result = subprocess.run(['amd-smi', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def _run_amd_smi_command(self, args: List[str], json_output: bool = True) -> Union[dict, str]:
        """Run amd-smi command and return the result
        
        @param args: List of arguments for amd-smi command
        @param json_output: Whether to expect JSON output
        @return: Parsed JSON dict or raw string output
        """
        cmd = ['amd-smi'] + args
        if json_output and '--json' not in args:
            cmd.append('--json')
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"amd-smi command failed: {result.stderr}")
            
            if json_output:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return raw output
                    return result.stdout
            else:
                return result.stdout
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("amd-smi command timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to run amd-smi command: {e}")

    def get_power(self, device: int) -> str:
        """Get current socket power of a given device.
        
        Args:
            device: GPU device index.
            
        Returns:
            Power consumption in watts as string, or 'N/A' if unavailable.
        """
        try:
            # Get power information for specific device
            result = self._run_amd_smi_command(['metric', '-d', str(device), '-P'])
            
            if isinstance(result, dict):
                # Navigate the JSON structure to find power data
                gpu_key = f"gpu_{device}"
                if gpu_key in result:
                    power_data = result[gpu_key].get('power', {})
                    if 'socket_power' in power_data:
                        power_str = power_data['socket_power']
                        # Extract numeric value from string like "150.0 W"
                        power_match = re.search(r'(\d+\.?\d*)', str(power_str))
                        if power_match:
                            return power_match.group(1)
                    elif 'average_socket_power' in power_data:
                        power_str = power_data['average_socket_power']
                        power_match = re.search(r'(\d+\.?\d*)', str(power_str))
                        if power_match:
                            return power_match.group(1)
            
            return 'N/A'
            
        except Exception:
            return 'N/A'

    def list_devices(self) -> List[int]:
        """Get list of GPU device indices.
        
        Returns:
            List of device indices.
        """
        try:
            result = self._run_amd_smi_command(['list'])
            
            if isinstance(result, dict):
                # Extract device indices from the JSON response
                devices = []
                for key in result.keys():
                    if key.startswith('gpu_'):
                        try:
                            device_id = int(key.split('_')[1])
                            devices.append(device_id)
                        except (ValueError, IndexError):
                            continue
                return sorted(devices)
            else:
                # Parse text output if JSON parsing failed
                devices = []
                lines = result.split('\n')
                for line in lines:
                    # Look for GPU device patterns
                    match = re.search(r'GPU\s*(\d+)', line, re.IGNORECASE)
                    if match:
                        devices.append(int(match.group(1)))
                return sorted(list(set(devices)))  # Remove duplicates and sort
                
        except Exception:
            # Return empty list if we can't get devices
            return []

    def get_mem_info(self, device: int) -> float:
        """Get memory usage percentage for a device.
        
        Args:
            device: GPU device index.
            
        Returns:
            Memory usage percentage as float.
        """
        try:
            # Get memory information for specific device
            result = self._run_amd_smi_command(['metric', '-d', str(device), '-m'])
            
            if isinstance(result, dict):
                gpu_key = f"gpu_{device}"
                if gpu_key in result:
                    memory_data = result[gpu_key].get('memory', {})
                    
                    # Try to get memory usage percentage directly
                    if 'vram_usage_percent' in memory_data:
                        usage_str = memory_data['vram_usage_percent']
                        usage_match = re.search(r'(\d+\.?\d*)', str(usage_str))
                        if usage_match:
                            return float(usage_match.group(1))
                    
                    # Calculate from used/total if available
                    if 'vram_used' in memory_data and 'vram_total' in memory_data:
                        used_str = memory_data['vram_used']
                        total_str = memory_data['vram_total']
                        
                        # Extract numeric values (removing units like MB, GB)
                        used_match = re.search(r'(\d+\.?\d*)', str(used_str))
                        total_match = re.search(r'(\d+\.?\d*)', str(total_str))
                        
                        if used_match and total_match:
                            used = float(used_match.group(1))
                            total = float(total_match.group(1))
                            
                            # Convert units if needed (assuming same units for both)
                            if total > 0:
                                return round((used / total) * 100, 2)
            
            return 0.0
            
        except Exception:
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
            # Get energy/power information to check if it's a secondary die
            result = self._run_amd_smi_command(['metric', '-d', str(device), '-P'])
            
            if isinstance(result, dict):
                gpu_key = f"gpu_{device}"
                if gpu_key in result:
                    power_data = result[gpu_key].get('power', {})
                    
                    # Check if energy counter is available and non-zero
                    if 'energy_counter' in power_data:
                        energy_str = power_data['energy_counter']
                        energy_match = re.search(r'(\d+\.?\d*)', str(energy_str))
                        if energy_match:
                            energy_value = float(energy_match.group(1))
                            # Secondary die typically has energy counter == 0
                            return energy_value == 0.0
                    
                    # Alternative check: if socket power is not available or 0
                    if 'socket_power' in power_data:
                        power_str = power_data['socket_power']
                        power_match = re.search(r'(\d+\.?\d*)', str(power_str))
                        if power_match:
                            power_value = float(power_match.group(1))
                            # Secondary die typically has no socket power reading
                            return power_value == 0.0
                    
                    # If no power metrics are available, might be secondary die
                    if not power_data:
                        return True
            
            return False
            
        except Exception:
            # Default to False if we can't determine
            return False

    def get_device_info(self, device: int) -> Dict[str, Any]:
        """Get comprehensive device information.
        
        Args:
            device: GPU device index.
            
        Returns:
            Dictionary with device information.
        """
        try:
            result = self._run_amd_smi_command(['metric', '-d', str(device)])
            
            if isinstance(result, dict):
                gpu_key = f"gpu_{device}"
                if gpu_key in result:
                    return result[gpu_key]
            
            return {}
            
        except Exception:
            return {}

    def get_temperature(self, device: int) -> Optional[float]:
        """Get GPU temperature.
        
        Args:
            device: GPU device index.
            
        Returns:
            Temperature in Celsius, or None if unavailable.
        """
        try:
            result = self._run_amd_smi_command(['metric', '-d', str(device), '-t'])
            
            if isinstance(result, dict):
                gpu_key = f"gpu_{device}"
                if gpu_key in result:
                    temp_data = result[gpu_key].get('temperature', {})
                    if 'edge_temperature' in temp_data:
                        temp_str = temp_data['edge_temperature']
                        temp_match = re.search(r'(\d+\.?\d*)', str(temp_str))
                        if temp_match:
                            return float(temp_match.group(1))
            
            return None
            
        except Exception:
            return None

    def get_clock_frequencies(self, device: int) -> Dict[str, Any]:
        """Get clock frequencies for GPU and memory.
        
        Args:
            device: GPU device index.
            
        Returns:
            Dictionary with clock frequencies.
        """
        try:
            result = self._run_amd_smi_command(['metric', '-d', str(device), '-c'])
            
            if isinstance(result, dict):
                gpu_key = f"gpu_{device}"
                if gpu_key in result:
                    return result[gpu_key].get('clock', {})
            
            return {}
            
        except Exception:
            return {}
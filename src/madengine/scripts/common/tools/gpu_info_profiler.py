#!/usr/bin/env python3
"""madengine CLI tool for profiling GPU usage of running LLMs and Deep Learning models.

This script provides a command-line interface to profile GPU usage of running LLMs and Deep Learning models.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import multiprocessing
import threading
import time
import datetime
import subprocess
import sys
import csv
import os
import logging
import typing
from typing import Optional, List, Dict, Any


def check_amd_smi_available() -> bool:
    """Check if amd-smi command or Python bindings are available.
    
    Returns:
        bool: True if amd-smi is available, False otherwise.
    """
    # First check for Python bindings (more reliable for programmatic access)
    try:
        sys.path.append("/opt/rocm/libexec/amdsmi_cli/")
        from amdsmi_init import amdsmi_interface
        logging.debug("amd-smi Python bindings found at /opt/rocm/libexec/amdsmi_cli/")
        return True
    except ImportError:
        logging.debug("amd-smi Python bindings not found")
    
    # Fallback to checking command-line tool
    try:
        result = subprocess.run(
            ['amd-smi', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logging.debug("amd-smi command-line tool found")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        logging.debug(f"amd-smi command not available: {e}")
    
    return False

def get_rocm_version() -> Optional[float]:
    """Get ROCm version from system.
    
    Returns:
        Optional[float]: ROCm version as major.minor (e.g., 6.1), or None if not detected.
    """
    try:
        # Try hipconfig --version first (more reliable)
        result = subprocess.run(
            ['hipconfig', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # example output: 6.1.40092-038397aaa
            version_str = result.stdout.strip()
            version_parts = version_str.split('.')[:2]  # Get major.minor
            return float('.'.join(version_parts))
    except (subprocess.SubprocessError, ValueError, IndexError) as e:
        logging.debug(f"hipconfig check failed: {e}")
    
    try:
        # Fallback to /opt/rocm/.info/version
        if os.path.exists("/opt/rocm/.info/version"):
            result = subprocess.run(['cat', '/opt/rocm/.info/version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_str = result.stdout.strip().split('-')[0]  # Remove build suffix
                version_parts = version_str.split('.')[:2]  # Get major.minor
                return float('.'.join(version_parts))
    except (IOError, ValueError, IndexError) as e:
        logging.debug(f"ROCm version file check failed: {e}")
    
    return None

def detect_gpu_vendor() -> tuple[bool, bool]:
    """Detect GPU vendor (NVIDIA or AMD/ROCm).
    
    Returns:
        tuple[bool, bool]: (is_nvidia, is_rocm)
        
    Raises:
        ValueError: If no GPU management tools are found.
    """
    if os.path.exists("/usr/bin/nvidia-smi"):
        return True, False
    elif os.path.exists("/opt/rocm/bin/rocm-smi") or check_amd_smi_available():
        return False, True
    else:
        error_msg = (
            "Unable to detect GPU vendor. No GPU management tools found.\n"
            "For NVIDIA: /usr/bin/nvidia-smi not found\n"
            "For AMD: /opt/rocm/bin/rocm-smi and amd-smi not found\n\n"
            "Please ensure:\n"
            "  1. GPU drivers are installed\n"
            "  2. For AMD GPUs: ROCm is properly installed (https://rocm.docs.amd.com)\n"
            "  3. For NVIDIA GPUs: CUDA toolkit is installed\n"
            "  4. GPU management tools are in system PATH"
        )
        raise ValueError(error_msg)


def initialize_profiler_utils(is_nvidia: bool, is_rocm: bool) -> Any:
    """Initialize the appropriate profiler utility based on GPU vendor.
    
    Args:
        is_nvidia: Whether NVIDIA GPU is detected.
        is_rocm: Whether AMD ROCm GPU is detected.
        
    Returns:
        Any: The ProfUtils class for the detected GPU vendor.
        
    Raises:
        ImportError: If the required profiler utility cannot be imported.
    """
    if is_nvidia:
        try:
            from pynvml_utils import ProfUtils
            return ProfUtils
        except ImportError as e:
            raise ImportError(f"Could not import pynvml_utils.py: {e}")
    
    # ROCm path: choose between rocm-smi and amd-smi based on version
    rocm_version = get_rocm_version()
    use_amd_smi = False
    
    logging.info(f"Detected ROCm version: {rocm_version}")
    logging.info(f"amd-smi available: {check_amd_smi_available()}")
    
    if rocm_version is not None and rocm_version >= 6.4:
        # ROCm >= 6.4: prefer amd-smi if available
        if check_amd_smi_available():
            use_amd_smi = True
            logging.info(f"Using amd-smi for ROCm {rocm_version}")
        else:
            logging.warning(f"ROCm {rocm_version} detected but amd-smi not available, using rocm-smi")
    else:
        logging.info(f"ROCm {rocm_version} < 6.4, using rocm-smi")
    
    if use_amd_smi:
        try:
            from amd_smi_utils import ProfUtils
            logging.info("Successfully imported amd_smi_utils")
            return ProfUtils
        except ImportError as import_err:
            # Fallback to rocm-smi if amd-smi import fails
            logging.warning(f"amd-smi import failed: {import_err}, falling back to rocm-smi")
            try:
                from rocm_smi_utils import ProfUtils
                return ProfUtils
            except ImportError as e:
                raise ImportError(f"Could not import amd_smi_utils.py or rocm_smi_utils.py: {e}")
        except Exception as init_err:
            # Catch initialization errors from amd_smi_utils.__init__
            logging.warning(f"amd-smi initialization failed: {init_err}, falling back to rocm-smi")
            try:
                from rocm_smi_utils import ProfUtils
                return ProfUtils
            except ImportError as e:
                raise ImportError(f"Could not import rocm_smi_utils.py after amd-smi init failed: {e}")
    else:
        # ROCm < 6.4 or amd-smi not available: use rocm-smi
        try:
            from rocm_smi_utils import ProfUtils
            return ProfUtils
        except ImportError as e:
            raise ImportError(f"Could not import rocm_smi_utils.py: {e}")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Module-level GPU detection (performed once at import, but deferred if used as library)
IS_NVIDIA: bool = False
IS_ROCM: bool = False
_GPU_DETECTED: bool = False

def _ensure_gpu_detected() -> None:
    """Ensure GPU vendor detection has been performed."""
    global IS_NVIDIA, IS_ROCM, _GPU_DETECTED
    if not _GPU_DETECTED:
        IS_NVIDIA, IS_ROCM = detect_gpu_vendor()
        _GPU_DETECTED = True


def run_command(commandstring: str) -> None:
    """Run the command string.
    
    This function runs the command string.
    
    Args:
        commandstring (str): The command string to run.
    
    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    logging.debug(commandstring)
    subprocess.run(commandstring, shell=True, check=True, executable="/bin/bash")


def run_command0(commandstring: str) -> None:
    """Run command on GPU device 0.
    
    Args:
        commandstring: The command string to run.
    
    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    env_var = "HIP_VISIBLE_DEVICES" if IS_ROCM else "CUDA_VISIBLE_DEVICES"
    command = f"{env_var}=0 {commandstring}"
    logging.debug(f"Running command on device 0: {command}")
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")


def run_command1(commandstring: str) -> None:
    """Run command on GPU device 1.
    
    Args:
        commandstring: The command string to run.
    
    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    env_var = "HIP_VISIBLE_DEVICES" if IS_ROCM else "CUDA_VISIBLE_DEVICES"
    command = f"{env_var}=1 {commandstring}"
    logging.debug(f"Running command on device 1: {command}")
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")


class EventController(threading.Thread):
    """Thread to control workload execution and synchronize with profilers.
    
    Attributes:
        event: Threading event to signal profiler start/stop.
        commandstring: The command string to execute.
        dual_gcd: Whether to run workload on dual GCDs (AMD-specific).
    """
    def __init__(self, event: threading.Event, commandstring: str, dual_gcd: str, profiler: Any) -> None:
        """Initialize the event controller.
        
        Args:
            event: Threading event for synchronization.
            commandstring: Command to execute.
            dual_gcd: "true" to run on dual GCDs, otherwise "false".
            profiler: GPU profiler utility instance.
        """
        super().__init__()
        self.event = event
        self.commandstring = commandstring
        self.dual_gcd = dual_gcd
        self.profiler = profiler

    def run(self) -> None:
        """Execute workload and control profiler lifecycle.
        
        Raises:
            EnvironmentError: If dual GCD mode is requested but not available.
        """
        # Signal profiler to start
        self.event.set()
        time.sleep(1)  # Allow profiler to initialize

        n_devices = len(self.profiler.list_devices())
        
        # Dual GCD mode (AMD-specific)
        if IS_ROCM and n_devices == 2 and self.dual_gcd == "true":
            logging.info("Running workload on both GCDs")
            p0 = multiprocessing.Process(target=run_command0, args=[self.commandstring])
            p1 = multiprocessing.Process(target=run_command1, args=[self.commandstring])
            
            logging.info("Workload starting...")
            p0.start()
            p1.start()
            p0.join()
            p1.join()
            logging.info("Workload completed")
            
        elif IS_ROCM and n_devices != 2 and self.dual_gcd == "true":
            self.event.clear()
            raise EnvironmentError(
                f"Dual GCD mode requested but system has {n_devices} GCD(s), expected 2"
            )
        else:
            # Single device mode
            p0 = multiprocessing.Process(target=run_command, args=[self.commandstring])
            logging.info("Workload starting...")
            p0.start()
            p0.join()
            logging.info("Workload completed")

        time.sleep(1)  # Allow profiler to capture final samples
        self.event.clear()  # Signal profiler to stop


class ProfilerThread(threading.Thread):
    """Base thread class for GPU profiling.
    
    Attributes:
        data: List of profiling samples collected.
        devices: List of GPU device IDs to profile.
        sampling_rate: Time interval between samples (seconds).
        event: Threading event for synchronization with workload.
    """
    def __init__(self, devices: List[int], sampling_rate: float, event: threading.Event) -> None:
        """Initialize the profiler thread.
        
        Args:
            devices: List of GPU device IDs to profile.
            sampling_rate: Sampling interval in seconds.
            event: Threading event for synchronization.
        """
        super().__init__()
        self.data: List[Dict[str, Any]] = []
        self.devices = devices
        self.sampling_rate = sampling_rate
        self.event = event

    def run(self, prof_fun: Any, header_string: str) -> None:
        """Execute profiling loop.
        
        Args:
            prof_fun: Function to call for getting metric value for a device.
            header_string: Column header prefix for CSV output.
        """
        self.event.wait()  # Wait for workload to start
        logging.info("Profiler started")
        
        while self.event.is_set():
            now = datetime.datetime.now()
            row: Dict[str, Any] = {"time": now.strftime("%Y-%m-%d %H:%M:%S.%f")}
            
            for device_id in self.devices:
                current_val = prof_fun(device_id)
                row[f"{header_string}{device_id}"] = current_val
            
            logging.debug(f"Sample: {row}")
            self.data.append(row)
            time.sleep(self.sampling_rate)
        
        logging.info(f"Profiler stopped. Collected {len(self.data)} samples")


class PowerProfiler(ProfilerThread):
    """Thread for profiling GPU power consumption.
    
    Attributes:
        prof_fun: Function to get power metric.
        header_string: CSV column header prefix.
    """
    def __init__(self, devices: List[int], sampling_rate: float, event: threading.Event, 
                 profiler: Any, device_filter: str) -> None:
        """Initialize the power profiler.
        
        Args:
            devices: List of GPU device IDs to profile.
            sampling_rate: Sampling interval in seconds.
            event: Threading event for synchronization.
            profiler: GPU profiler utility instance.
            device_filter: Device filter string ("all" or specific device).
        
        Raises:
            ValueError: If a specified device is a secondary die (AMD-specific).
        """
        super().__init__(devices, sampling_rate, event)
        
        # AMD-specific: Filter out secondary dies
        if IS_ROCM and device_filter != "all":
            for device_id in self.devices:
                if profiler.check_if_secondary_die(device_id):
                    raise ValueError(f"Device {device_id} is a secondary die")
        elif IS_ROCM and device_filter == "all":
            self.devices = [d for d in self.devices if not profiler.check_if_secondary_die(d)]

        self.prof_fun = profiler.get_power
        self.header_string = "Power(Watt) GPU"

    def run(self) -> None:
        """Execute power profiling."""
        super().run(prof_fun=self.prof_fun, header_string=self.header_string)


class VRAMProfiler(ProfilerThread):
    """Thread for profiling GPU VRAM/memory usage.
    
    Attributes:
        prof_fun: Function to get memory metric.
        header_string: CSV column header prefix.
    """
    def __init__(self, devices: List[int], sampling_rate: float, event: threading.Event, 
                 profiler: Any) -> None:
        """Initialize the VRAM profiler.
        
        Args:
            devices: List of GPU device IDs to profile.
            sampling_rate: Sampling interval in seconds.
            event: Threading event for synchronization.
            profiler: GPU profiler utility instance.
        """
        super().__init__(devices, sampling_rate, event)
        self.prof_fun = profiler.get_mem_info
        self.header_string = "vram(%) GPU"

    def run(self) -> None:
        """Execute VRAM profiling."""
        super().run(prof_fun=self.prof_fun, header_string=self.header_string)


def main() -> None:
    """Profile GPU usage during workload execution.
    
    Reads configuration from environment variables:
        MODE: "power" or "vram"
        DEVICE: Comma-separated device IDs or "all"
        SAMPLING_RATE: Sampling interval in seconds
        DUAL_GCD: "true" to enable dual GCD mode (AMD-specific)
    
    Raises:
        ValueError: If MODE is invalid or required env vars are missing.
        EnvironmentError: If dual GCD mode is incompatible with system.
    """
    # Ensure GPU vendor has been detected
    _ensure_gpu_detected()
    # Reconstruct command string from arguments
    commandstring = ""
    for arg in sys.argv[1:]:
        if " " in arg:
            commandstring += f'"{arg}" '
        else:
            commandstring += f"{arg} "
    
    # Get required environment variables
    mode = os.environ.get("MODE")
    device = os.environ.get("DEVICE")
    sampling_rate_str = os.environ.get("SAMPLING_RATE")
    dual_gcd = os.environ.get("DUAL_GCD", "false")
    
    # Validate environment variables
    if not mode:
        raise ValueError("MODE environment variable is required")
    if not device:
        raise ValueError("DEVICE environment variable is required")
    if not sampling_rate_str:
        raise ValueError("SAMPLING_RATE environment variable is required")
    
    try:
        sampling_rate = float(sampling_rate_str)
    except ValueError:
        raise ValueError(f"Invalid SAMPLING_RATE: {sampling_rate_str}")
    
    if mode not in ["power", "vram"]:
        raise ValueError(f"Invalid MODE: {mode}. Must be 'power' or 'vram'")
    
    # Initialize profiler utility
    prof_utils_class = initialize_profiler_utils(IS_NVIDIA, IS_ROCM)
    try:
        profiler = prof_utils_class(mode)
    except ImportError as e:
        # If amd-smi initialization fails, try falling back to rocm-smi
        logging.error(f"Failed to initialize profiler: {e}")
        if IS_ROCM:
            logging.warning("Attempting fallback to rocm-smi")
            try:
                from rocm_smi_utils import ProfUtils as RocmSmiProfUtils
                profiler = RocmSmiProfUtils(mode)
                logging.info("Successfully fell back to rocm-smi")
            except Exception as fallback_err:
                raise RuntimeError(f"Failed to initialize both amd-smi and rocm-smi: {e}, {fallback_err}")
        else:
            raise
    
    # Create synchronization event
    event = threading.Event()

    # Parse device list
    device_list = device.split(",")
    
    if len(device_list) == 1 and device_list[0] == "all":
        device_list = profiler.list_devices()
    elif len(device_list) == 1 and device_list[0].isdigit():
        device_list = [int(device_list[0])]
    else:
        device_list = [int(d) for d in device_list]
    
    logging.info(f"Profiling mode: {mode}, devices: {device_list}, sampling rate: {sampling_rate}s")

    # Create threads
    workload_thread = EventController(
        event=event,
        commandstring=commandstring,
        dual_gcd=dual_gcd,
        profiler=profiler
    )
    
    if mode == "power":
        profiler_thread = PowerProfiler(
            devices=device_list,
            sampling_rate=sampling_rate,
            event=event,
            profiler=profiler,
            device_filter=device
        )
    else:  # mode == "vram"
        profiler_thread = VRAMProfiler(
            devices=device_list,
            sampling_rate=sampling_rate,
            event=event,
            profiler=profiler
        )

    # Execute profiling
    workload_thread.start()
    profiler_thread.start()
    workload_thread.join()
    profiler_thread.join()

    # Write results to CSV
    output_file = os.environ.get("OUTPUT_FILE", "prof.csv")
    
    if not profiler_thread.data:
        logging.error("No profiling data collected")
        sys.exit(1)
    else:
        try:
            with open(output_file, "w", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=profiler_thread.data[0].keys())
                writer.writeheader()
                writer.writerows(profiler_thread.data)
            logging.info(f"Profiling data written to {output_file}")
        except IOError as e:
            logging.error(f"Failed to write output file: {e}")
            sys.exit(1)
    

if __name__ == "__main__":
    main()

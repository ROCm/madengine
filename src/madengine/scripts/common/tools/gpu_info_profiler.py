#!/usr/bin/env python3
"""Mad Engine CLI tool for profiling GPU usage of running LLMs and Deep Learning models.

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
# import pandas


def check_amd_smi_available():
    """Check if amd-smi command is available"""
    import subprocess
    try:
        result = subprocess.run(['amd-smi', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def get_rocm_version():
    """Get ROCm version from system"""
    import subprocess
    try:
        # Try hipconfig --version first (more reliable)
        result = subprocess.run(['hipconfig', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # example output: 6.1.40092-038397aaa
            version_str = result.stdout.strip()
            version_parts = version_str.split('.')[:2]  # Get major.minor
            return float('.'.join(version_parts))
    except:
        pass
    
    try:
        # Fallback to /opt/rocm/.info/version
        if os.path.exists("/opt/rocm/.info/version"):
            result = subprocess.run(['cat', '/opt/rocm/.info/version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_str = result.stdout.strip().split('-')[0]  # Remove build suffix
                version_parts = version_str.split('.')[:2]  # Get major.minor
                return float('.'.join(version_parts))
    except:
        pass
    
    return None

# Detect GPU vendor: NVIDIA vs ROCm
if os.path.exists("/usr/bin/nvidia-smi"):
    is_nvidia = True
    is_rocm = False
elif os.path.exists("/opt/rocm/bin/rocm-smi") or check_amd_smi_available():
    is_nvidia = False
    is_rocm = True
else:
    raise ValueError("Unable to detect GPU vendor")

# For ROCm systems, choose between rocm-smi and amd-smi based on version
if is_nvidia:
    from pynvml_utils import prof_utils
else:  # is_rocm
    rocm_version = get_rocm_version()
    use_amd_smi = False
    
    if rocm_version is not None and rocm_version >= 6.1:
        # ROCm >= 6.1: prefer amd-smi if available
        if check_amd_smi_available():
            use_amd_smi = True
    
    if use_amd_smi:
        try:
            from amd_smi_utils import prof_utils
        except ImportError:
            # Fallback to rocm-smi if amd-smi import fails
            try:
                from rocm_smi_utils import prof_utils
            except ImportError:
                raise ImportError("Could not import amd_smi_utils.py or rocm_smi_utils.py")
    else:
        # ROCm < 6.1 or amd-smi not available: use rocm-smi
        try:
            from rocm_smi_utils import prof_utils
        except ImportError:
            raise ImportError("Could not import rocm_smi_utils.py")

logging.basicConfig(level=logging.INFO)

# Get envs as global variables
mode = os.environ.get("MODE")
device = os.environ.get("DEVICE")
profiler = None


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
    """Run the command string on device 0.
    
    This function runs the command string on device 0.
    
    Args:
        commandstring (str): The command string to run.
    
    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    command = (
        "HIP_VISIBLE_DEVICES=0 " + commandstring
        if is_rocm
        else "CUDA_VISIBLE_DEVICES=0 " + commandstring
    )
    logging.debug(command)
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")


def run_command1(commandstring: str) -> None:
    """Run the command string on device 1.
    
    This function runs the command string on device 1.
    
    Args:
        commandstring (str): The command string to run.
    
    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    command = (
        "HIP_VISIBLE_DEVICES=1 " + commandstring
        if is_rocm
        else "CUDA_VISIBLE_DEVICES=1 " + commandstring
    )
    logging.debug(command)
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")


class event_ctl(threading.Thread):
    """Thread class to control the event of power profiler.
    
    This class is a thread class to control the event of power profiler.
    
    Attributes:
        event (threading.Event): The event to control the power profiler.
        commandstring (str): The command string to run.
        dual_gcd (str): The dual GCD flag.
    """
    def __init__(self, **kwargs) -> None:
        """Initialize the event control thread.
        
        This function initializes the event control thread.
        
        Args:
            kwargs (dict): The keyword arguments.
            
        Raises:
            EnvironmentError: If the docker container has different number of GCDs than required.
        """
        super().__init__()
        self.event = kwargs["event"]
        self.commandstring = kwargs["commandstring"]
        self.dual_gcd = kwargs["dual_gcd"]

    def run(self) -> None:
        """Run the event control thread.
        
        This function runs the event control thread.
        
        Raises:
            EnvironmentError: If the docker container has different number of GCDs than required.
        """
        # signal power profiler starting
        self.event.set()
        # wait for 1 sec
        time.sleep(1)

        # check the number of GCDs
        n_devices = len(profiler.listDevices())
        # if there are two GCDs and dual_gcd is true, run two processes/workloads on two GCDs, otherwise run one process/workload on one GCD.
        if is_rocm and n_devices == 2 and self.dual_gcd == "true":
            logging.debug(
                "================= two GCDs detected, same workload running on both GCDs ================="
            )
            # start running two processes/workloads on two GCDs
            p0 = multiprocessing.Process(target=run_command0, args=[self.commandstring])
            p1 = multiprocessing.Process(target=run_command1, args=[self.commandstring])
            logging.debug("================= workload starts =================")
            p0.start()
            p1.start()
            # wait for processes complete
            p0.join()
            p1.join()
            logging.debug("================= workload ends =================")
        elif is_rocm and n_devices != 2 and self.dual_gcd == "true":
            # if there are not two GCDs and dual_gcd is true, raise an error, since the docker container has different number of GCDs than required.
            self.event.clear()
            raise EnvironmentError(
                "The docker container has "
                + str(n_devices)
                + " GCD(s), but got '2-GCD' required."
            )
        else:
            # start running one process/workload on one GCD
            p0 = multiprocessing.Process(target=run_command, args=[self.commandstring])
            logging.debug("================= workload starts =================")
            p0.start()
            # wait for processes complete
            p0.join()
            logging.debug("================= workload ends =================")

        # wait for 1 sec
        time.sleep(1)
        # signal power profiler stopping
        self.event.clear()


class prof_thread(threading.Thread):
    """Thread class to profile GPU usage.
    
    This class is a thread class to profile GPU usage.
    
    Attributes:
        data (list): The data list.
        devices (list): The device list.
        sampling_rate (float): The sampling rate.
        event (threading.Event): The event to control the power profiler.
    """
    def __init__(self, **kwargs) -> None:
        """Initialize the profiler thread.
        
        This function initializes the profiler thread.
        
        Args:
            kwargs (dict): The keyword arguments.
            
        Raises:
            ValueError: If the device is a secondary die.
        """
        super().__init__()
        self.data = []
        self.devices = kwargs["devices"]
        self.sampling_rate = kwargs["sampling_rate"]
        self.event = kwargs["event"]

    def run(
            self, 
            prof_fun: typing.Any,
            header_string: str
        ):
        """Run the profiler thread.
        
        This function runs the profiler thread.
        
        Args:
            prof_fun (typing.Any): The profiler function.
            header_string (str): The header string.
            
        Raises:
            ValueError: If the device is a secondary die.
        """
        # wait for the event to be set
        self.event.wait()
        logging.debug("profiler started")
        # start profiling
        while self.event.isSet():
            # get current value, and append to data, sleep for sampling rate.
            now = datetime.datetime.now()
            row = {"time": now.strftime("%Y-%m-%d %H:%M:%S.%f")}
            for d in self.devices:
                current_val = prof_fun(d)
                row[header_string + str(d)] = current_val
            logging.debug(row)
            self.data.append(row)
            # sampling rate
            time.sleep(self.sampling_rate)


class pwr_prof(prof_thread):
    """Thread class to profile GPU power usage.
    
    This class is a thread class to profile GPU power usage.
    
    Attributes:
        prof_fun (typing.Any): The profiler function.
        header_string (str): The header string.
    """
    def __init__(self, **kwargs) -> None:
        """Initialize the power profiler thread.
        
        This function initializes the power profiler thread.
        
        Args:
            kwargs (dict): The keyword arguments.
        
        Raises:
            ValueError: If the device is a secondary die.
        """
        super().__init__(**kwargs)
        
        if is_rocm and device != "all":
            for d in self.devices:
                if profiler.checkIfSecondaryDie(d):
                    raise ValueError("Device " + str(d) + " is a secondary die.")
        elif is_rocm and device == "all":
            self.devices = [
                d for d in self.devices if not profiler.checkIfSecondaryDie(d)
            ]

        self.prof_fun = profiler.getPower
        self.header_string = "Power(Watt) GPU"

    def run(self) -> None:
        """Run the power profiler thread.

        This function runs the power profiler thread.
        """
        super().run(prof_fun=self.prof_fun, header_string=self.header_string)


class vram_prof(prof_thread):
    """Thread class to profile GPU VRAM usage.
    
    This class is a thread class to profile GPU VRAM usage.
    
    Attributes:
        prof_fun (typing.Any): The profiler function.
        header_string (str): The header string.
    """
    def __init__(self, **kwargs) -> None:
        """Initialize the VRAM profiler thread.
        
        This function initializes the VRAM profiler thread.
        
        Args:
            kwargs (dict): The keyword arguments.

        Raises:
            ValueError: If the device is a secondary die.
        """
        super().__init__(**kwargs)
        self.prof_fun = profiler.getMemInfo
        self.header_string = "vram(%) GPU"

    def run(self) -> None:
        """Run the VRAM profiler thread.
        
        This function runs the VRAM profiler thread.
        
        Raises:
            ValueError: If the device is a secondary die.
        """
        super().run(prof_fun=self.prof_fun, header_string=self.header_string)


def main() -> None:
    """Main function to profile GPU usage of running LLMs and Deep Learning models.
    
    This function profiles GPU usage of running LLMs and Deep Learning models.
    
    Raises:
        ValueError: If the mode is invalid.
    """
    # Reconstruct the command string
    commandstring = ""

    # Put the quotes back in if there are spaces in the arguments.
    for arg in sys.argv[1:]:  # skip sys.argv[0] since the question didn't ask for it
        if " " in arg:
            commandstring += '"{}" '.format(arg)  # Put the quotes back in
        else:
            commandstring += "{} ".format(arg)
    
    # Get env 
    sampling_rate = float(os.environ.get("SAMPLING_RATE"))
    dual_gcd = os.environ.get("DUAL-GCD")

    # Create event
    event = threading.Event()

    # Get device list
    device_list = device.split(",")

    # Create profiler
    global profiler
    profiler = prof_utils(mode)

    if len(device_list) == 1 and device_list[0] == "all":
        device_list = profiler.listDevices()
    elif len(device_list) == 1 and device_list[0].isdigit():
        device_list = [int(device_list[0])]
    else:
        device_list = [int(d) for d in device_list]

    t1 = event_ctl(event=event, commandstring=commandstring, dual_gcd=dual_gcd)
    if mode == "power":
        t2 = pwr_prof(sampling_rate=sampling_rate, devices=device_list, event=event)
    elif mode == "vram":
        t2 = vram_prof(sampling_rate=sampling_rate, devices=device_list, event=event)
    else:
        raise ValueError(mode + " is an invalid mode")

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    file_name = "/myworkspace/prof.csv"
    
    if not t2.data:
        logging.error("No data to write to csv file.")
    else:
        # write t2.data to a csv file
        with open(file_name, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=t2.data[0].keys())
            writer.writeheader()
            for row in t2.data:
                writer.writerow(row)

    # df = pandas.DataFrame(t2.data)
    # df.to_csv(file_name, index=False)
    

if __name__ == "__main__":
    main()

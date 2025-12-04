#!/usr/bin/env python3
"""Module to get library trace information of ROCm libraries

This module contains the class GetLibraryTrace to get library trace information of ROCm libraries.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import sys
import io
import os
import re
from datetime import datetime
import csv
import subprocess
from contextlib import redirect_stdout, redirect_stderr
import typing


# Global variables of the trace mode
mode = os.environ.get("TRACE_MODE", "").replace(" ", "").split(",")
# Add the trace mode to the global variable
if os.environ.get("ROCBLAS_TRACE"):
    mode.append("rocblas_trace")

if os.environ.get("HIPBLASLT_TRACE"):
    mode.append("hipblaslt_trace")    

if os.environ.get("TENSILE_TRACE"):
    mode.append("tensile_trace")

if os.environ.get("MIOPEN_TRACE"):
    mode.append("miopen_trace")

if os.environ.get("RCCL_TRACE"):
    mode.append("rccl_trace")

# Global data storage
filtered_configs = {}


def process_rocblas_trace(output_lines: list) -> bool:
    """Process the rocBLAS trace information

    This function processes the rocBLAS trace information from the output lines

    Args:
        output_lines: List of output lines

    Returns:
        matched: Boolean value
    """
    # Patterns of regex to match the rocBLAS trace information.
    patterns = [r"./rocblas-bench .*$", r"^.*rocblas_function: .*$"]
    regexes = {p: re.compile(p) for p in patterns}
    # Boolean value to check if the trace information is matched
    matched = False
    # iterate through the patterns and match
    for pattern, regex in regexes.items():
        config_cnt = {}
        for line in output_lines:
            match = regex.search(line)
            if match:
                matched = True
                if match.group(0) in config_cnt:
                    config_cnt[match.group(0)] += 1
                else:
                    config_cnt[match.group(0)] = 1
        # key to store the config count
        key = "rocblas" if "rocblas-bench" in pattern else "rocblas_function"
        # iterate through the config count
        for config in config_cnt:
            if config in filtered_configs[key]:
                filtered_configs[key][config] += config_cnt[config]
            else:
                filtered_configs[key][config] = config_cnt[config]
    return matched


def process_hipblaslt_trace(output_lines: list) -> bool:
    """Process the hipBLASLT trace information
    This function processes the hipBLASLT trace information from the output lines
    Args:
        output_lines: List of output lines
    Returns:
        matched: Boolean value
    """
    patterns = [r"hipblaslt-bench .*$"]
    regexes = {p: re.compile(p) for p in patterns}
    matched = False
    for pattern, regex in regexes.items():
        config_cnt = {}
        for line in output_lines:
            match = regex.search(line)
            if match:
                matched = True
                if match.group(0) in config_cnt:
                    config_cnt[match.group(0)] += 1
                else:
                    config_cnt[match.group(0)] = 1
        key = "hipblaslt" if "hipblaslt-bench" in pattern else "hipblaslt_function"
        for config in config_cnt:
            if config in filtered_configs[key]:
                filtered_configs[key][config] += config_cnt[config]
            else:
                filtered_configs[key][config] = config_cnt[config]
    return matched


def process_tensile_trace(output_lines: list) -> bool:
    """Process the Tensile trace information

    This function processes the Tensile trace information from the output lines

    Args:
        output_lines: List of output lines

    Returns:
        matched: Boolean value
    """
    # Regex pattern to match the Tensile trace information
    RE_MATCH = re.compile(r"^Running kernel: (.*)$")
    # Boolean value to check if the trace information is
    matched = False

    config_cnt = {}
    # iterate through the output lines
    for line in output_lines:
        match = RE_MATCH.search(line)
        if match:
            matched = True
            if match.group(1) in config_cnt:
                config_cnt[match.group(1)] += 1
            else:
                config_cnt[match.group(1)] = 1

    for config in config_cnt:
        if config in filtered_configs["tensile"]:
            filtered_configs["tensile"][config] += config_cnt[config]
        else:
            filtered_configs["tensile"][config] = config_cnt[config]
    return matched


def process_miopen_trace(output_lines: list) -> bool:
    """Process the MIOpen trace information

    This function processes the MIOpen trace information from the output lines

    Args:
        output_lines: List of output lines

    Returns:
        matched: Boolean value
    """
    RE_MATCH = re.compile(r"MIOpen\(HIP\): Command \[.*\] (./bin/MIOpenDriver .*)$")
    matched = False
    config_cnt = {}
    for line in output_lines:
        match = RE_MATCH.search(line)
        if match:
            matched = True
            if match.group(0) in config_cnt:
                config_cnt[match.group(1)] += 1
            else:
                config_cnt[match.group(1)] = 1

    for config in config_cnt:
        if config in filtered_configs["miopen"]:
            filtered_configs["miopen"][config] += config_cnt[config]
        else:
            filtered_configs["miopen"][config] = config_cnt[config]
    return matched


class LibraryFilter(object):
    """Class to filter the library trace information
    
    This class filters the library trace information based on the mode
    
    Args:
        mode: Mode of the trace
        liveOutput: Boolean value
        printConfigs: Boolean value
    """
    def __init__(
            self, 
            mode: str, 
            liveOutput: bool=False, 
            printConfigs: bool=False
        ) -> None:
        """Initialize the LibraryFilter class
        
        Args:
            mode: Mode of the trace
            liveOutput: Boolean value
            printConfigs: Boolean value

        Returns:
            None
        """
        self.mode = mode
        self.stdio = None
        if liveOutput:
            self.stdio = sys.__stdout__  # actual stdout for printing

        self.printConfigs = printConfigs

    def write(
            self, 
            data: str
        ) -> None:
        """Write the data
        
        This function writes the data
        
        Args:
            data: Data to write
        
        Returns:
            None
        """
        matched = False
        r_match = False
        t_match = False
        m_match = False

        if "rocblas_trace" in mode:
            r_match = process_rocblas_trace(data.splitlines())
            matched |= r_match

        if "hipblaslt_trace" in mode:
            r_match = process_hipblaslt_trace(data.splitlines() )
            matched |= r_match 

        if "tensile_trace" in mode:
            t_match = process_tensile_trace(data.splitlines())
            matched |= t_match

        if "miopen_trace" in mode:
            m_match = process_miopen_trace(data.splitlines())
            matched |= m_match

        if self.stdio and (self.printConfigs or (not matched)):
            self.stdio.write(data)
        # else: #debug
        #    self.stdio.write( "$(%s,%s,%s) " % (r_match, t_match, m_match) + data )

    def flush(self):
        if self.stdio:
            self.stdio.flush()


def run_command(
        commandstring: str, 
        request_env: typing.Dict[str, str],
        outlog: typing.Any
    ):
    """Run the command
    
    This function runs the command
    
    Args:
        commandstring: Command string
        request_env: Request environment
        outlog: Output log
    
    Returns:
        None
    """
    modified_env = os.environ.copy()
    modified_env.update(request_env)

    with redirect_stdout(outlog), redirect_stderr(outlog):
        process = subprocess.Popen(commandstring, shell=True, env=modified_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        outlog.write(stdout.decode())
        outlog.write(stderr.decode())


def main():
    """Main function to get library trace information"""
    commandstring = ""
    for arg in sys.argv[1:]:  # skip sys.argv[0] since the question didn't ask for it
        if " " in arg:
            commandstring += '"{}" '.format(arg)  # Put the quotes back in
        else:
            commandstring += "{} ".format(arg)

    # WORKAROUND: This command does not stack
    # calling multiple get_library_trace calls in a chain is equivalent to calling it once
    commandstring = re.sub("([~ ]|^).*get_library_trace ", "", commandstring)

    request_env = {}
    if "rocblas_trace" in mode:
        request_env["ROCBLAS_LAYER"] = "6"
    if "miopen_trace" in mode:
        request_env["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
    if "tensile_trace" in mode:
        request_env["TENSILE_DB"] = "0x8000"
    if "rccl_trace" in mode:
        request_env["NCCL_DEBUG"] = "INFO"
        request_env["RCCL_KERNEL_COLL_TRACE_ENABLE"] = "1"
        request_env["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"

    # Initialize the filtered_configs
    filtered_configs["rocblas"] = {}
    filtered_configs["hipblaslt"] = {}
    filtered_configs["rocblas_function"] = {}
    filtered_configs["miopen"] = {}
    filtered_configs["tensile"] = {}

    # Initialize the LibraryFilter
    outlog = LibraryFilter(mode, liveOutput=True)
    run_command(commandstring, request_env, outlog)
    job_id = os.environ.get("JENKINS_BUILD_NUMBER", "0")
    model_name = os.environ.get("MAD_MODEL_NAME", "")
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write the library trace information to the CSV file
    filename = os.environ.get("OUTPUT_FILE", "library_trace.csv")
    fields = ["jobid", "created_date", "model", "library", "config", "calls"]
    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for library in filtered_configs:
            for config in filtered_configs[library]:
                csvwriter.writerow(
                    [
                        job_id,
                        date,
                        model_name,
                        library,
                        config,
                        filtered_configs[library][config],
                    ]
                )


if __name__ == "__main__":
    main()

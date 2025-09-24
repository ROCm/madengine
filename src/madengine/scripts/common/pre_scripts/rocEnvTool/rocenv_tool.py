"""Tool to collect system environment information.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
import os
import sys
import argparse
from console import Console
from csv_parser import CSVParser
import json
import shutil

rocm_version = None
pkgtype = None
env_map = {}

class CommandInfo:
    '''
        section_info (str): Name of the section.
        cmds (list) : command list for a particular section.
    '''
    def __init__(self, section_info, cmds):
        self.section_info = section_info
        self.cmds = cmds

def get_cmd(cmd, known_paths):
    '''
        A function to get the full path to the command.

        Args:
            cmd (str): command name.
            known_paths (list): list of known paths to search for the command.

        Returns:
            full path to the command if found, else throws an exception.
    '''

    cmd_path = shutil.which(cmd)
    if cmd_path is not None:
        return cmd_path

    for path in known_paths:
        if not os.path.isdir(path):
            continue

        cmd_path = os.path.join(path, cmd)
        if os.path.isfile(cmd_path) and os.access(cmd_path, os.X_OK):
            return cmd_path

    # throw exception if command not found.
    raise FileNotFoundError(f'{cmd} not found.')

## utility functions.
def parse_env_tags_json(json_file):
    env_tags = None
    with open(json_file) as f:
        env_tags = json.load(f)
    configs = env_tags["env_tags"]
    return configs

## Hardware information.
def print_hardware_information():
    cmd = "lshw"
    try:
        cmd = get_cmd(cmd, ["/usr/bin", "/usr/sbin", "/sbin"])
        cmd_info = CommandInfo("HardwareInformation", [cmd])
        return cmd_info
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return None

## CPU Hardware Information
def print_cpu_hardware_information():
    cmd ="/usr/bin/lscpu"
    cmd_info = CommandInfo("CPU Information", [cmd])
    return cmd_info

## GPU Hardware information.
def print_gpu_hardware_information(gpu_device_type):
    cmd = None
    if gpu_device_type == "AMD":
        try:
            cmd = "rocminfo"
            cmd = get_cmd(cmd, ["/opt/rocm/bin"])
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            return None
    elif gpu_device_type == "NVIDIA":
        try:
            cmd = "nvidia-smi"
            cmd = get_cmd(cmd, ["/usr/bin", "/usr/local/cuda/bin"])
            cmd = f'{cmd} -L'
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            return None
    else:
        print ("WARNING: Unknown GPU device detected")
    cmd_info = CommandInfo("GPU Information", [cmd])
    return cmd_info

## BIOS Information.
def print_bios_settings():
    cmd = "dmidecode"
    try:
        cmd = get_cmd(cmd, ["/usr/sbin", "/sbin"])
        cmd_info = CommandInfo("dmidecode Information", [cmd])
        return cmd_info
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return None

## OS information.
def print_os_information():
    cmd1 = "/bin/uname -a"
    cmd2 = "/bin/cat /etc/os-release"
    cmd_info = CommandInfo("OS Distribution", [cmd1, cmd2])
    return cmd_info

## Memory Information.
def print_memory_information():
    cmd = "lsmem"
    try:
        cmd = get_cmd(cmd, ["/usr/bin", "/bin"])
        cmd_info = CommandInfo("Memory Information", [cmd])
        return cmd_info
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return None

## ROCm version data
def print_rocm_version_information():
    cmd1 = "/bin/ls -v -d /opt/rocm*"
    global rocm_version
    rocm_major = 0
    rocm_minor = 0
    rocm_patch = 0
    if (not os.environ.get('ROCM_VERSION')):
        rocm_version_header = "/opt/rocm/include/rocm-core/rocm_version.h"
        if os.path.isfile(rocm_version_header):
            fs = open("/opt/rocm/include/rocm-core/rocm_version.h", 'r')
            lines = fs.readlines()
            fs.close()
        for line in lines:
            if "#define ROCM_VERSION_MAJOR" in line:
                rocm_major = line.split("#define ROCM_VERSION_MAJOR")[1].strip()
            if "#define ROCM_VERSION_MINOR" in line:
                rocm_minor = line.split("#define ROCM_VERSION_MINOR")[1].strip()
            if "#define ROCM_VERSION_PATCH" in line:
                rocm_patch = line.split("#define ROCM_VERSION_PATCH")[1].strip()
        rocm_version = "rocm-" + str(rocm_major) + "." + str(rocm_minor) + "." + str(rocm_patch)

    cmd2 = "echo '==== Using " + rocm_version + " to collect ROCm information.==== '"
    cmd_info = CommandInfo("Available ROCm versions", [cmd1, cmd2])
    return cmd_info

def print_rocm_repo_setup():
    #cmd = "/bin/grep -i -E 'rocm|amdgpu' /etc/apt/sources.list.d/* /etc/zypp/repos.d/* /etc/yum.repos.d/*"
    cmd = None
    if os.path.exists("/etc/zypp/repos.d"):
        cmd = "/bin/grep -i -E 'rocm|amdgpu' /etc/zypp/repos.d/*"
    elif os.path.exists("/etc/apt/sources.list.d"):
        cmd = "/bin/grep -i -E 'rocm|amdgpu' /etc/apt/sources.list.d/*"
    elif os.path.exists("/etc/yum.repos.d/"):
        cmd = "/bin/grep -i -E 'rocm|amdgpu' /etc/yum.repos.d/*"

    cmd_info = CommandInfo("ROCm Repo Setup", [cmd])
    return cmd_info

def print_rocm_packages_installed():
    d  = {}
    with open("/etc/os-release") as fs:
        for line in fs:
            if "=" in line:
                k,v = line.rstrip().split("=")
                d[k] = v.strip('"')
    pkgtype = d['ID_LIKE']
    cmd1 = "echo ' Pkg type: '" + pkgtype
    cmd2 = None
    if pkgtype == "debian":
        cmd2 = "/usr/bin/dpkg -l | /bin/grep -i -E 'ocl-icd|kfdtest|llvm-amd|miopen|half|^ii  hip|hcc|hsa|rocm|atmi|^ii  comgr|composa|amd-smi|aomp|amdgpu|rock|mivision|migraph|rocprofiler|roctracer|rocbl|hipify|rocsol|rocthr|rocff|rocalu|rocprim|rocrand|rccl|rocspar|rdc|rocwmma|rpp|openmp|amdfwflash|ocl |opencl' | /usr/bin/sort"
    else:
        cmd2 = "/usr/bin/rpm -qa | /bin/grep -i -E 'ocl-icd|kfdtest|llvm-amd|miopen|half|hip|hcc|hsa|rocm|atmi|comgr|composa|amd-smi|aomp|amdgpu|rock|mivision|migraph|rocprofiler|roctracer|rocblas|hipify|rocsol|rocthr|rocff|rocalu|rocprim|rocrand|rccl|rocspar|rdc|rocwmma|rpp|openmp|amdfwflash|ocl|opencl' | /usr/bin/sort"
    cmd_info = CommandInfo("ROCm Packages Installed", [cmd1, cmd2])
    return cmd_info

def print_rocm_environment_variables():
    cmd = "env | /bin/grep -i -E 'rocm|hsa|hip|mpi|openmp|ucx|miopen'"
    cmd_info = CommandInfo("ROCm environment variables", [cmd])
    return cmd_info

def print_rocm_smi_details(smi_config):
    cmd_info = None
    cmd = "rocm-smi"
    try:
        cmd = get_cmd(cmd, ["/opt/rocm/bin"])
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return None
    if (smi_config == "rocm_smi"):
        cmd_info = CommandInfo("ROCm SMI", [cmd])
    elif (smi_config == "ifwi_version"):
        ifwi_cmd = cmd + " -v"
        cmd_info = CommandInfo("IFWI version", [ifwi_cmd])
    elif (smi_config == "rocm_smi_showhw"):
        showhw_cmd = cmd + " --showhw"
        cmd_info = CommandInfo("ROCm SMI showhw", [showhw_cmd])
    elif (smi_config == "rocm_smi_pcie"):
        pcie_cmd = cmd + " -c | /bin/grep -i -E 'pcie'"
        cmd_info = CommandInfo("ROCm SMI pcieclk clock", [pcie_cmd])
    elif (smi_config == "rocm_smi_pids"):
        pids_cmd1 = "ls /sys/class/kfd/kfd/proc/"
        pids_cmd2 = cmd + " --showpids"
        cmd_info = CommandInfo("KFD PIDs sysfs kfd proc", [pids_cmd1, pids_cmd2])
    elif (smi_config == "rocm_smi_topology"):
        showtops_cmd = cmd + " --showtopo"
        cmd_info = CommandInfo("showtop topology", [showtops_cmd])
    elif (smi_config == "rocm_smi_showserial"):
        serial_cmd = cmd + " --showserial"
        cmd_info = CommandInfo("showserial", [serial_cmd])
    elif (smi_config == "rocm_smi_showperflevel"):
        perf_cmd = cmd + " --showperflevel"
        cmd_info = CommandInfo("showperflevel", [perf_cmd])
    elif (smi_config == "rocm_smi_showrasinfo"):
        showrasinfo_cmd = cmd + " --showrasinfo all"
        cmd_info = CommandInfo("ROCm SMI showrasinfo all", [showrasinfo_cmd])
    elif (smi_config == "rocm_smi_showxgmierr"):
        showxgmierr_cmd = cmd + " --showxgmierr"
        cmd_info = CommandInfo("ROCm SMI showxgmierr", [showxgmierr_cmd])
    elif (smi_config == "rocm_smi_clocks"):
        clock_cmd = cmd + " -cga"
        cmd_info = CommandInfo("ROCm SMI clocks", [clock_cmd])
    elif (smi_config == "rocm_smi_showcompute_partition"):
        compute_cmd = cmd + " --showcomputepartition"
        cmd_info = CommandInfo("ROCm Show computepartition", [compute_cmd])
    elif (smi_config == "rocm_smi_nodesbw"):
        nodesbw_cmd = cmd + " --shownodesbw"
        cmd_info = CommandInfo("ROCm Show Nodebsion", [nodesbw_cmd])
    elif (smi_config == "rocm_smi_gpudeviceid"):
        gpudeviceid_cmd = cmd + " -i -d 0"
        cmd_info = CommandInfo("ROCM Show GPU Device ID", [gpudeviceid_cmd])
    else:
        cmd_info = None
    return cmd_info

def print_rocm_info_details():
    cmd = "rocminfo"
    try:
        cmd = get_cmd(cmd, ["/opt/rocm/bin"])
        cmd_info = CommandInfo("rocminfo", [cmd])
        return cmd_info
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return None

## dmesg boot logs - GPU/ATOM/DRM/BIOS
def print_dmesg_logs(ignore_prev_boot_logs=True):
    cmds = []
    if os.path.exists("/var/log/journal"):
        cmds.append("echo 'Persistent logging enabled.'")
    else:
        cmd1_str = "WARNING: Persistent logging possibly disabled.\n"
        cmd1_str = cmd1_str + "WARNING: Please run: \n"
        cmd1_str = cmd1_str + "       sudo mkdir -p /var/log/journal\n"
        cmd1_str = cmd1_str + "       sudo systemctl restart systemd-journald.service \n"
        cmd1_str = cmd1_str + "WARNING: to enable persistent boot logs for collection and analysis.\n"
        cmd1_str = "echo " + cmd1_str
        cmds.append(cmd1_str)

    cmds.append("echo 'Section: dmesg boot logs'")
    cmds.append("/bin/dmesg -T | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|watchdog|bug|nmi|dazed|too|mce|edac|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER|amdfwflash'")
    if not ignore_prev_boot_logs:
        cmd_exec = None
        if os.path.exists("/bin/journalctl"):
            cmd_exec = "/bin/journalctl"
        elif os.path.exists("/usr/bin/journalctl"):
            cmd_exec = "/usr/bin/journalctl"
        else:
            cmd_exec = None

        if cmd_exec is not None:
            cmds.append("echo 'Section: Current boot logs'")
            boot_exec = "/bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|watchdog|bug|nmi|dazed|too|mce|edac|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER|amdfwflash'"
            cmds.append(cmd_exec + " -b | " + boot_exec)
            cmds.append("echo 'Section: Previous boot logs'")
            cmds.append(cmd_exec + " -b 1 | " + boot_exec)
            cmds.append("echo 'Section: Second boot logs'")
            cmds.append(cmd_exec + " -b 2 | " + boot_exec)

    cmd_info = CommandInfo("dmesg GPU/DRM/ATOM/BIOS", cmds)
    return cmd_info

## print amdgpu modinfo
def print_amdgpu_modinfo():
    cmd = "modinfo"
    try:
        cmd = get_cmd(cmd, ["/sbin", "/usr/sbin"])
        cmd = f"{cmd} amdgpu"
        cmd_info = CommandInfo("amdgpu modinfo", [cmd])
        return cmd_info
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return None

## print pip list
def print_pip_list_details():
    cmd = "pip3 list --disable-pip-version-check"
    cmd_info = CommandInfo("Pip3 package list ", [cmd])
    return cmd_info

def print_check_numa_balancing():
    cmd = "cat /proc/sys/kernel/numa_balancing"
    cmd_info = CommandInfo("Numa balancing Info", [cmd])
    return cmd_info

## print cuda version information.
def print_cuda_version_information():
    cmd = "nvcc"
    try:
        cmd = get_cmd(cmd, ["/usr/local/cuda/bin", "/usr/bin"])
        cmd = f"{cmd} --version"
        cmd_info = CommandInfo("CUDA information", [cmd])
        return cmd_info
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        return None

def print_cuda_env_variables():
    cmd = "env | /bin/grep -i -E 'cuda|nvidia|pytorch|mpi|openmp|ucx|cu'"
    cmd_info = CommandInfo("CUDA Env Variables", [cmd])
    return cmd_info

def print_cuda_packages_installed():
    d  = {}
    with open("/etc/os-release") as fs:
        for line in fs:
            if "=" in line:
                k,v = line.rstrip().split("=")
                d[k] = v.strip('"')
    pkgtype = d['ID_LIKE']
    cmd1 = "echo ' Pkg type: '" + pkgtype
    cmd2 = None
    if pkgtype == "debian":
        cmd2 = "/usr/bin/dpkg -l | /bin/grep -i -E 'cuda|cu|atlas|hdf5|nccl|nvinfer|nvjpeg|onnx'"
    else:
        cmd2 = "/usr/bin/rpm -qa | /bin/grep -i -E 'cuda|cu|atlas|hdf5|nccl|nvinfer|nvjpeg|onnx'"
    cmd_info = CommandInfo("ROCm Packages Installed", [cmd1, cmd2])
    return cmd_info

def dump_system_env_information(configs, output_name):
    out_dir = "." + output_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        os.system("rm -rf " + out_dir)
        os.makedirs(out_dir)

    for config in configs:
        keys = env_map.keys()
        if config in keys:
            out_path = os.path.join(out_dir, config)
            os.makedirs(out_path)
            log_file = out_path + "/" + config + ".txt"
            fs = open(log_file, 'w')

            cmd_info = env_map[config]
            if cmd_info is not None:
                cmd = "echo ------- Section: " + config + "    ----------"
                out = console.sh(cmd)
                fs.write(out)
                fs.write("\n")

                cmds = cmd_info.cmds
                for cmd in cmds:
                    if config in ["rocm_env_variables", "dmsg_gpu_drm_atom_logs", "rocm_smi_pcie"]:
                        out = console.sh(cmd, canFail=True)
                    else:
                        out = console.sh(cmd)
                    fs.write(out)
                    fs.write("\n")
            fs.close()

def determine_gpu_device_type():
    gpu_device_type = ""
    rocm_smi_out = console.sh("/opt/rocm/bin/rocm-smi || true")
    nv_smi_out = console.sh("nvidia-smi -L || true")
    if not "not found" in rocm_smi_out:
        gpu_device_type = "AMD"
    if not "not found" in nv_smi_out:
        gpu_device_type = "NVIDIA"
    return gpu_device_type

def generate_env_info(gpu_device_type):
    global env_map
    env_map["hardware_information"] = print_hardware_information()
    env_map["cpu_information"] = print_cpu_hardware_information()
    env_map["gpu_information"] = print_gpu_hardware_information(gpu_device_type)
    env_map["bios_settings"] = print_bios_settings()
    env_map["os_information"] = print_os_information()
    env_map["dmsg_gpu_drm_atom_logs"] = print_dmesg_logs(ignore_prev_boot_logs=True)
    env_map["amdgpu_modinfo"] = print_amdgpu_modinfo()
    env_map["memory_information"] = print_memory_information()
    print ("GPU Device type detected is: {}".format(gpu_device_type))
    if gpu_device_type == "AMD":
        env_map["rocm_information"] = print_rocm_version_information()
        env_map["rocm_repo_setup"] = print_rocm_repo_setup()
        env_map["rocm_packages_installed"] = print_rocm_packages_installed()
        env_map["rocm_env_variables"] = print_rocm_environment_variables()
        env_map["rocm_smi"] = print_rocm_smi_details("rocm_smi")
        env_map["ifwi_version"] = print_rocm_smi_details("ifwi_version")
        env_map["rocm_smi_showhw"] = print_rocm_smi_details("rocm_smi_showhw")
        env_map["rocm_smi_pcie"] = print_rocm_smi_details("rocm_smi_pcie")
        env_map["rocm_smi_pids"] = print_rocm_smi_details("rocm_smi_pids")
        env_map["rocm_smi_topology"] = print_rocm_smi_details("rocm_smi_topology")
        env_map["rocm_smi_showserial"] = print_rocm_smi_details("rocm_smi_showserial")
        env_map["rocm_smi_showperflevel"] = print_rocm_smi_details("rocm_smi_showperflevel")
        env_map["rocm_smi_showrasinfo"] = print_rocm_smi_details("rocm_smi_showrasinfo")
        env_map["rocm_smi_showxgmierr"] = print_rocm_smi_details("rocm_smi_showxgmierr")
        env_map["rocm_smi_clocks"] = print_rocm_smi_details("rocm_smi_clocks")
        env_map["rocm_smi_showcompute_partition"] = print_rocm_smi_details("rocm_smi_showcompute_partition")
        env_map["rocm_smi_nodesbwi"] = print_rocm_smi_details("rocm_smi_nodesbwi")
        env_map["rocm_smi_gpudeviceid"] = print_rocm_smi_details("rocm_smi_gpudeviceid")
        env_map["rocm_info"] = print_rocm_info_details()
    elif gpu_device_type == "NVIDIA":
        env_map["cuda_information"] = print_cuda_version_information()
        env_map["cuda_env_variables"] = print_cuda_env_variables()
        env_map["cuda_packages_installed"] = print_cuda_packages_installed()
    env_map["pip_list"] = print_pip_list_details()

    if os.path.exists("/proc/sys/kernel/numa_balancing"):
        env_map["numa_balancing"] = print_check_numa_balancing()

def main():
    gpu_device_type = determine_gpu_device_type()
    generate_env_info(gpu_device_type)
    configs = env_map.keys()
    if args.lite:
        configs = parse_env_tags_json("env_tags.json")
    dump_system_env_information(configs, args.output_name)
    print ("OK: finished dumping the system env details in .{} folder".format(args.output_name))
    if args.dump_csv or args.print_csv:
        csv_file = args.output_name + ".csv"
        out_dir = "." + args.output_name
        csv_parser = CSVParser(csv_file, out_dir, configs)
        csv_parser.dump_csv_output()
        if args.print_csv:
            csv_parser.print_csv_output()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lite", action="store_true", help="System environment data lite version taken from env_tags.json")
    parser.add_argument("--dump-csv", action="store_true", help="Dump system config info in CSV file")
    parser.add_argument("--print-csv", action="store_true", help="Print system config info data")
    parser.add_argument("--output-name", required=False, default="sys_config_info", help="Output file or directory name")
    args = parser.parse_args()
    console = Console(shellVerbose=False, live_output=False)

    main()

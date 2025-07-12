"""CSV Parser - parses various sys config log files and dumps into CSV.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
import os
from console import Console

'''
CSV Parser - parses various sys config log files and dumps into CSV.
Only the below tags are supported.
Enable dumping it via adding --dump-csv in rocEnvTool

os_information
cpu_information
gpu_information
memory_information
rocm_information
rocm_packages_installed
rocm_env_variables
pip_list
numa_balancing
'''
class CSVParser:
    def __init__(self, filename, sys_config_files_path, tags):
        self.filename = filename
        self.path = sys_config_files_path
        self.tags = tags
        self.sys_config_info_list = []
        self.gpu_device_type = self.determine_gpu_device_type()

    def determine_gpu_device_type(self):
        console = Console()
        gpu_device_type = ""
        rocm_smi_out = console.sh("/opt/rocm/bin/rocm-smi || true")
        nv_smi_out = console.sh("nvidia-smi -L || true")
        if not "not found" in rocm_smi_out:
            gpu_device_type = "AMD"
        if not "not found" in nv_smi_out:
            gpu_device_type = "NVIDIA"
        return gpu_device_type

    def get_log_file_data(self, log_file_path):
        fs = open(log_file_path, 'r')
        lines = fs.readlines()
        fs.close()

        return lines

    def dump_os_information_in_csv(self, os_info_path):
        lines = self.get_log_file_data(os_info_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        for j in range(1, len(lines)):
            line = lines[j].rstrip()
            if j == 1:
                values = line.split(" ")
                info_list.append("Distribution|" + values[0])
                info_list.append("Node name|" + values[1])
                info_list.append("Kernel version| " + values[2])
            if "PRETTY_NAME" in line:
                info_list.append("OS version|" + line.split("=")[1].replace('"', ''))
        return info_list

    def dump_cpu_information_in_csv(self, cpu_log_path):
        lines = self.get_log_file_data(cpu_log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        for j in range(1, len(lines)):
            values = lines[j].rstrip().split(":")
            info_list.append(values[0] + "|" + values[1].lstrip())
        return info_list

    def dump_gpu_information_in_csv(self, gpu_log_path, device_type):
        lines = self.get_log_file_data(gpu_log_path)
        info_list = []
        info_list.append(lines[0].rstrip())

        if device_type == "AMD":
            name = ""
            uuid = ""
            marketing_name = ""
            vendor_name = ""
            num_gpu = 0
            for j in range(1, len(lines)):
                line = lines[j].rstrip()
                if ("Name:" in line and "gfx" in line):
                    name = line.split(":")[1].lstrip()
                if ("Uuid:" in line):
                    uuid = line.split(":")[1].lstrip()
                if ("Marketing Name:" in line):
                    marketing_name = line.split(":")[1].lstrip()
                if ("Vendor Name:" in line):
                    vendor_name = line.split(":")[1].lstrip()
                if ("Device Type:" in line):
                    device_type = line.split(":")[1].lstrip()
                    if device_type == "GPU":
                        break

            for j in range(1, len(lines)):
                line = lines[j].rstrip()
                if ("Device Type:" in line):
                    device_type = line.split(":")[1].lstrip()
                    if (device_type == "GPU"):
                        num_gpu += 1
            info_list.append("Name|" + name)
            info_list.append("Uuid|" + uuid)
            info_list.append("Marketing Name|" + marketing_name)
            info_list.append("Vendor Name|" + vendor_name)
            info_list.append("Num GPU|" + str(num_gpu))
        else:
            num_gpu = 0
            name = ""
            uuid = ""
            for j in range(1, len(lines)):
                line = lines[j].rstrip()
                if "GPU" in line:
                    num_gpu += 1
                    values = line.split(":")
                    name = values[1].split("(UUID")[0]
                    uuid = values[2]
            info_list.append("Name|" + name)
            info_list.append("Uuid|" + uuid)
            info_list.append("Num GPU|" + str(num_gpu))

        return info_list

    def dump_rocm_smi_gpudeviceid_in_csv(self, rocm_smi_log_path):
        lines = self.get_log_file_data(rocm_smi_log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        for j in range(1, len(lines)):
            line = lines[j].rstrip()
            if "Device ID" in line:
                values = line.rstrip().split(":")
                info_list.append("GPU" + values[1] + "|" + values[2].lstrip())
                return info_list
        return info_list

    def dump_memory_information_in_csv(self, memory_log):
        lines = self.get_log_file_data(memory_log)
        info_list = []
        info_list.append(lines[0].rstrip())
        for j in range(1, len(lines)):
            line = lines[j].rstrip()
            if "Memory block size:" in line:
                info_list.append("Memory block size|" + line.split(":")[1].lstrip())
            if "Total online memory:" in line:
                info_list.append("Total online memory|" + line.split(":")[1].lstrip())
            if "Total offline memory:" in line:
                info_list.append("Total offline memory|" + line.split(":")[1].lstrip())
        return info_list

    def dump_rocm_information_in_csv(self, rocm_info_path):
        lines = self.get_log_file_data(rocm_info_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        version_path = os.path.join("/opt/rocm", ".info", "version")
        rocm_version = open(version_path).read().rstrip()
        info_list.append("ROCm version|" + rocm_version)
        return info_list

    def dump_rocm_packages_installed_in_csv(self, log_path):
        lines = self.get_log_file_data(log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        pkg_type = lines[1].rstrip().split("Pkg type:")[1].lstrip()
        if pkg_type == "debian":
            for j in range(2, len(lines)):
                line = lines[j].rstrip()
                if "ii" in line:
                    values = line.split()
                    info_list.append(values[1] + "|" + values[2])
        else:
            for j in range(2, len(lines)):
                line = lines[j].rstrip()
                info_list.append(line)
        return info_list

    def dump_rocm_env_variables_in_csv(self, log_path):
        lines = self.get_log_file_data(log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        for j in range(1, len(lines)):
            env_values = lines[j].rstrip().split("=")
            if (env_values[0]):
                info_list.append(env_values[0] + "|" + env_values[1])
        return info_list

    def dump_pip_list_in_csv(self, log_path):
        lines = self.get_log_file_data(log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        for j in range(3, len(lines)):
            line = lines[j].rstrip()
            values = line.split()
            info_list.append(values[0] + "|" + values[1])
        return info_list

    def dump_numa_balancing_in_csv(self, log_path):
        lines = self.get_log_file_data(log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        info_list.append("Numa Balacing|" + lines[1].rstrip())
        return info_list

    def dump_cuda_information_in_csv(self, log_path):
        lines = self.get_log_file_data(log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        for j in range(1, len(lines)):
            line = lines[j].rstrip()
            if "Build" in line:
                cuda_version = line.split("Build")[1].lstrip()
                info_list.append("CUDA Version|" + cuda_version)
        return info_list

    def dump_cuda_packages_installed_in_csv(self, log_path):
        lines = self.get_log_file_data(log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        pkg_type = lines[1].rstrip().split("Pkg type:")[1].lstrip()
        if pkg_type == "debian":
            for j in range(2, len(lines)):
                line = lines[j].rstrip()
                if "ii" in line:
                    values = line.split()
                    info_list.append(values[1] + "|" + values[2])
        else:
            for j in range(2, len(lines)):
                line = lines[j].rstrip()
                info_list.append(line)
        return info_list

    def dump_cuda_env_variables_in_csv(self, log_path):
        lines = self.get_log_file_data(log_path)
        info_list = []
        info_list.append(lines[0].rstrip())
        for j in range(1, len(lines)):
            env_values = lines[j].rstrip().split("=")
            info_list.append(env_values[0] + "|" + env_values[1])
        return info_list

    def dump_csv_output(self):
        gpu_device_type = self.gpu_device_type
        fs = open(self.filename, 'w')
        fs.write("sep=|")
        fs.write("\n")
        sys_config_info = []
        for tag in self.tags:
            log_path = os.path.abspath(self.path + "/" + tag + "/" + tag + ".txt")
            if not os.path.isfile(log_path):
                continue
            else:
                if tag == "os_information":
                    sys_config_info.extend(self.dump_os_information_in_csv(log_path))
                if tag == "cpu_information":
                    sys_config_info.extend(self.dump_cpu_information_in_csv(log_path))
                if tag == "gpu_information":
                    sys_config_info.extend(self.dump_gpu_information_in_csv(log_path, gpu_device_type))
                if tag == "rocm_smi_gpudeviceid":
                    sys_config_info.extend(self.dump_rocm_smi_gpudeviceid_in_csv(log_path))
                if tag == "memory_information":
                    sys_config_info.extend(self.dump_memory_information_in_csv(log_path))
                if tag == "rocm_information":
                    sys_config_info.extend(self.dump_rocm_information_in_csv(log_path))
                if tag == "rocm_packages_installed":
                    sys_config_info.extend(self.dump_rocm_packages_installed_in_csv(log_path))
                if tag == "rocm_env_variables":
                    sys_config_info.extend(self.dump_rocm_env_variables_in_csv(log_path))
                if tag == "cuda_information":
                    sys_config_info.extend(self.dump_cuda_information_in_csv(log_path))
                if tag == "cuda_packages_installed":
                    sys_config_info.extend(self.dump_cuda_packages_installed_in_csv(log_path))
                if tag == "cuda_env_variables":
                    sys_config_info.extend(self.dump_cuda_env_variables_in_csv(log_path))
                if tag == "pip_list":
                    sys_config_info.extend(self.dump_pip_list_in_csv(log_path))
                if tag == "numa_balancing":
                    sys_config_info.extend(self.dump_numa_balancing_in_csv(log_path))

        self.sys_config_info_list = sys_config_info

        for j in range(len(sys_config_info)):
            fs.write(sys_config_info[j])
            fs.write("\n")
        fs.close()
        print("\n" + "="*60)
        print(f"✅ SUCCESS: System config data dumped to {self.filename}")
        print("="*60 + "\n")

    def print_csv_output(self):
        print("\n" + "="*80)
        print("📋 SYSTEM CONFIG INFO - ENVIRONMENT VARIABLES")
        print("="*80)
        if self.sys_config_info_list:
            for j in range(len(self.sys_config_info_list)):
                line = self.sys_config_info_list[j]
                # Add some formatting for key-value pairs
                if "|" in line and not line.startswith("Tag"):
                    key, value = line.split("|", 1)
                    print(f"🔹 {key:<30}: {value}")
                else:
                    print(f"📌 {line}")
        else:
            print("❌ No system config information available")
        print("="*80 + "\n")

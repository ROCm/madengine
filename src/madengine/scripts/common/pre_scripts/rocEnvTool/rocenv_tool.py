"""Tool to collect system environment information (TheRock + Traditional ROCm compatible).

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
import os
import sys
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from console import Console
from csv_parser import CSVParser

rocm_version = None
pkgtype = None
env_map = {}
installation_type = None  # 'therock' or 'traditional' or 'unknown'
rocm_paths = {}  # Dynamic paths for ROCm components


class CommandInfo:
    '''
        section_info (str): Name of the section.
        cmds (list) : command list for a particular section.
    '''
    def __init__(self, section_info, cmds):
        self.section_info = section_info
        self.cmds = cmds


class RocmPathResolver:
    """
    Detects and resolves ROCm installation paths for both TheRock and traditional installations.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.installation_type = 'unknown'
        self.rocm_root = None
        self.paths = {
            'rocminfo': None,
            'rocm_smi': None,
            'hipcc': None,
            'amdclang': None,
            'version_file': None,
            'manifest_file': None,
        }
        self.therock_details = {}
        self.detect()
    
    def log(self, message: str):
        """Print verbose log messages."""
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def detect(self):
        """Detect ROCm installation type and locate components."""
        # Method 1: Check for TheRock via rocm-sdk command
        if self._detect_therock_python_package():
            return
        
        # Method 2: Check environment variables for TheRock
        if self._detect_therock_from_env():
            return
        
        # Method 3: Check for TheRock in common paths
        if self._detect_therock_tarball():
            return
        
        # Method 4: Fallback to traditional ROCm
        if self._detect_traditional_rocm():
            return
        
        # Method 5: Try to find binaries in PATH
        self._detect_from_path()
    
    def _is_therock_installation(self, path: Path) -> bool:
        """Check if a path contains TheRock installation markers."""
        if not path.exists():
            return False
        
        # Check for TheRock manifest
        manifest_path = path / "share" / "therock" / "therock_manifest.json"
        if manifest_path.exists():
            self.log(f"Found TheRock manifest at {manifest_path}")
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    self.therock_details['manifest'] = manifest
            except Exception as e:
                self.log(f"Error reading manifest: {e}")
            return True
        
        # Check for dist_info.json
        dist_info_path = path / "share" / "therock" / "dist_info.json"
        if dist_info_path.exists():
            self.log(f"Found TheRock dist_info at {dist_info_path}")
            return True
        
        return False
    
    def _detect_therock_python_package(self) -> bool:
        """Detect TheRock via Python package installation."""
        self.log("Checking for rocm-sdk command...")
        
        rocm_sdk_path = shutil.which("rocm-sdk")
        if rocm_sdk_path:
            self.log(f"Found rocm-sdk at {rocm_sdk_path}")
            
            try:
                # Get root path from rocm-sdk
                result = subprocess.run(
                    ["rocm-sdk", "path", "--root"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    root_path = Path(result.stdout.strip())
                    if self._is_therock_installation(root_path):
                        self.installation_type = 'therock'
                        self.rocm_root = str(root_path)
                        self._populate_therock_paths(root_path)
                        return True
            except Exception as e:
                self.log(f"Error getting rocm-sdk path: {e}")
        
        return False
    
    def _detect_therock_from_env(self) -> bool:
        """Detect TheRock from environment variables."""
        self.log("Checking environment variables...")
        
        for var in ['ROCM_PATH', 'ROCM_HOME', 'HIP_PATH']:
            value = os.environ.get(var)
            if value:
                path = Path(value)
                if self._is_therock_installation(path):
                    self.log(f"Found TheRock via ${var}={value}")
                    self.installation_type = 'therock'
                    self.rocm_root = str(path)
                    self._populate_therock_paths(path)
                    return True
        
        return False
    
    def _detect_therock_tarball(self) -> bool:
        """Detect TheRock tarball installations in common paths."""
        self.log("Checking common TheRock installation paths...")
        
        common_paths = [
            Path("/opt/rocm"),
            Path.home() / "rocm",
            Path.home() / "therock",
            Path("/usr/local/rocm"),
            Path.home() / ".local" / "rocm",
        ]
        
        for path in common_paths:
            if self._is_therock_installation(path):
                self.log(f"Found TheRock at {path}")
                self.installation_type = 'therock'
                self.rocm_root = str(path)
                self._populate_therock_paths(path)
                return True
        
        return False
    
    def _detect_traditional_rocm(self) -> bool:
        """Detect traditional ROCm installation."""
        self.log("Checking for traditional ROCm installation...")
        
        # Check for traditional ROCm marker
        version_file = Path("/opt/rocm/.info/version")
        if version_file.exists():
            self.log("Found traditional ROCm at /opt/rocm")
            self.installation_type = 'traditional'
            self.rocm_root = "/opt/rocm"
            self._populate_traditional_paths()
            return True
        
        return False
    
    def _detect_from_path(self):
        """Try to find ROCm binaries in PATH."""
        self.log("Searching for ROCm binaries in PATH...")
        
        # Try to find rocminfo
        rocminfo = shutil.which("rocminfo")
        if rocminfo:
            self.paths['rocminfo'] = rocminfo
            # Try to infer root from binary location
            rocminfo_path = Path(rocminfo)
            if rocminfo_path.exists():
                potential_root = rocminfo_path.parent.parent
                if self._is_therock_installation(potential_root):
                    self.installation_type = 'therock'
                    self.rocm_root = str(potential_root)
                    self._populate_therock_paths(potential_root)
                else:
                    self.installation_type = 'unknown'
                    self.rocm_root = str(potential_root)
        
        # Try to find other binaries
        self.paths['rocm_smi'] = shutil.which("rocm-smi")
        self.paths['hipcc'] = shutil.which("hipcc")
        self.paths['amdclang'] = shutil.which("amdclang")
    
    def _populate_therock_paths(self, root: Path):
        """Populate paths for TheRock installation."""
        bin_dir = root / "bin"
        
        self.paths['rocminfo'] = str(bin_dir / "rocminfo") if (bin_dir / "rocminfo").exists() else None
        self.paths['rocm_smi'] = str(bin_dir / "rocm-smi") if (bin_dir / "rocm-smi").exists() else None
        self.paths['hipcc'] = str(bin_dir / "hipcc") if (bin_dir / "hipcc").exists() else None
        self.paths['amdclang'] = str(bin_dir / "amdclang") if (bin_dir / "amdclang").exists() else None
        
        # Check for manifest
        manifest = root / "share" / "therock" / "therock_manifest.json"
        if manifest.exists():
            self.paths['manifest_file'] = str(manifest)
    
    def _populate_traditional_paths(self):
        """Populate paths for traditional ROCm installation."""
        self.paths['rocminfo'] = "/opt/rocm/bin/rocminfo"
        self.paths['rocm_smi'] = "/opt/rocm/bin/rocm-smi"
        self.paths['hipcc'] = "/opt/rocm/bin/hipcc"
        self.paths['version_file'] = "/opt/rocm/.info/version"
    
    def get_version(self) -> str:
        """Get ROCm version string."""
        if self.installation_type == 'therock':
            return self._get_therock_version()
        elif self.installation_type == 'traditional':
            return self._get_traditional_version()
        else:
            return "unknown"
    
    def _get_therock_version(self) -> str:
        """Get TheRock version from manifest or rocm-sdk."""
        # Try rocm-sdk command
        if shutil.which("rocm-sdk"):
            try:
                result = subprocess.run(
                    ["rocm-sdk", "version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass
        
        # Try manifest file
        if self.therock_details.get('manifest'):
            commit = self.therock_details['manifest'].get('the_rock_commit', 'unknown')
            return f"TheRock (commit: {commit[:8]})"
        
        return "TheRock (version unknown)"
    
    def _get_traditional_version(self) -> str:
        """Get traditional ROCm version from version file or header."""
        # Try version file
        version_file = Path("/opt/rocm/.info/version")
        if version_file.exists():
            try:
                return version_file.read_text().strip()
            except Exception:
                pass
        
        # Try version header
        version_header = Path("/opt/rocm/include/rocm-core/rocm_version.h")
        if version_header.exists():
            try:
                content = version_header.read_text()
                major = minor = patch = 0
                for line in content.split('\n'):
                    if "#define ROCM_VERSION_MAJOR" in line:
                        major = line.split()[-1]
                    if "#define ROCM_VERSION_MINOR" in line:
                        minor = line.split()[-1]
                    if "#define ROCM_VERSION_PATCH" in line:
                        patch = line.split()[-1]
                return f"rocm-{major}.{minor}.{patch}"
            except Exception:
                pass
        
        return "unknown"


## Utility functions
def parse_env_tags_json(json_file):
    env_tags = None
    with open(json_file) as f:
        env_tags = json.load(f)
    configs = env_tags["env_tags"]
    return configs


## Hardware information
def print_hardware_information():
    cmd = None
    possible_paths = ["/usr/bin/lshw", "/usr/sbin/lshw", "/sbin/lshw"]
    for path in possible_paths:
        if os.path.isfile(path):
            cmd = path
            break
    
    if cmd is None:
        print("WARNING: Install lshw to get hardware information")
        print("         (TheRock images may not include this by default)")
    
    if cmd is not None:
        cmd_info = CommandInfo("HardwareInformation", [cmd])
        return cmd_info
    else:
        return None


## CPU Hardware Information
def print_cpu_hardware_information():
    cmd = "/usr/bin/lscpu"
    if not os.path.exists(cmd):
        cmd = "lscpu"  # Try PATH
    cmd_info = CommandInfo("CPU Information", [cmd])
    return cmd_info


## GPU Hardware information
def print_gpu_hardware_information(gpu_device_type, path_resolver):
    if gpu_device_type == "AMD":
        # Use dynamic path from resolver
        cmd = path_resolver.paths.get('rocminfo') or "rocminfo"
    elif gpu_device_type == "NVIDIA":
        cmd = "nvidia-smi -L"
    else:
        print("WARNING: Unknown GPU device detected")
        cmd = "echo 'Unknown GPU device'"
    
    cmd_info = CommandInfo("GPU Information", [cmd])
    return cmd_info


## BIOS Information
def print_bios_settings():
    cmd = "/usr/sbin/dmidecode"
    if not os.path.exists(cmd):
        cmd = "dmidecode"  # Try PATH
    cmd_info = CommandInfo("dmidecode Information", [cmd])
    return cmd_info


## OS information
def print_os_information():
    cmd1 = "uname -a"
    cmd2 = "cat /etc/os-release"
    cmd_info = CommandInfo("OS Distribution", [cmd1, cmd2])
    return cmd_info


## Memory Information
def print_memory_information():
    cmd = "/usr/bin/lsmem"
    if not os.path.exists(cmd):
        cmd = "lsmem"  # Try PATH
    cmd_info = CommandInfo("Memory Information", [cmd])
    return cmd_info


## ROCm version data
def print_rocm_version_information(path_resolver):
    global rocm_version
    
    # List all ROCm-like directories
    cmd1 = "ls -v -d /opt/rocm* 2>/dev/null || echo 'No /opt/rocm* directories found'"
    
    # Get version from resolver
    rocm_version = path_resolver.get_version()
    
    cmd2 = f"echo '==== Installation Type: {path_resolver.installation_type} ===='"
    rocm_root_display = path_resolver.rocm_root or "Not found"
    cmd3 = f"echo '==== ROCm Root: {rocm_root_display} ===='"
    cmd4 = f"echo '==== Using {rocm_version} to collect ROCm information ===='"
    
    cmds = [cmd1, cmd2, cmd3, cmd4]
    
    # Add TheRock-specific info
    if path_resolver.installation_type == 'therock':
        manifest_file = path_resolver.paths.get('manifest_file')
        if manifest_file:
            cmd5 = f"echo '==== TheRock Manifest: {manifest_file} ===='"
            cmd6 = f"cat {manifest_file}"
            cmds.extend([cmd5, cmd6])
    
    cmd_info = CommandInfo("Available ROCm versions", cmds)
    return cmd_info


def print_rocm_repo_setup(path_resolver):
    """Print repo setup - only for traditional ROCm installations."""
    cmds = []
    
    if path_resolver.installation_type == 'therock':
        cmds.append("echo 'TheRock does not use traditional package repositories'")
        cmds.append("echo 'TheRock is installed via Python pip packages or tarballs'")
        
        # Try to get pip package info
        if shutil.which("rocm-sdk"):
            cmds.append("echo 'Checking rocm-sdk Python package...'")
            cmds.append("rocm-sdk version || true")
            cmds.append("rocm-sdk path --root || true")
        
        # Check if we're in a venv
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            cmds.append(f"echo 'Virtual environment: {venv_path}'")
            cmds.append("pip list | grep -i rocm || true")
    else:
        # Traditional ROCm repo check
        cmd = None
        if os.path.exists("/etc/zypp/repos.d"):
            cmd = "/bin/grep -i -E 'rocm|amdgpu' /etc/zypp/repos.d/* || echo 'No ROCm repos found'"
        elif os.path.exists("/etc/apt/sources.list.d"):
            cmd = "/bin/grep -i -E 'rocm|amdgpu' /etc/apt/sources.list.d/* || echo 'No ROCm repos found'"
        elif os.path.exists("/etc/yum.repos.d/"):
            cmd = "/bin/grep -i -E 'rocm|amdgpu' /etc/yum.repos.d/* || echo 'No ROCm repos found'"
        
        if cmd:
            cmds.append(cmd)
    
    cmd_info = CommandInfo("ROCm Repo Setup", cmds)
    return cmd_info


def print_rocm_packages_installed(path_resolver):
    """Print installed ROCm packages - adapted for TheRock."""
    cmds = []
    
    if path_resolver.installation_type == 'therock':
        cmds.append("echo 'Installation Type: TheRock (no system packages)'")
        cmds.append("echo ''")
        
        # Check Python packages
        cmds.append("echo '=== Python ROCm Packages ==='")
        cmds.append("pip list 2>/dev/null | grep -i -E 'rocm|hip|torch' || echo 'No Python ROCm packages found'")
        
        # List files in TheRock installation
        if path_resolver.rocm_root:
            cmds.append("echo ''")
            cmds.append(f"echo '=== TheRock Installation Contents ({path_resolver.rocm_root}) ==='")
            cmds.append(f"ls -lh {path_resolver.rocm_root}/bin/ 2>/dev/null || true")
            cmds.append(f"ls -lh {path_resolver.rocm_root}/lib/ 2>/dev/null | head -20 || true")
        
        # Check for dist_info
        if path_resolver.rocm_root:
            dist_info = Path(path_resolver.rocm_root) / "share" / "therock" / "dist_info.json"
            if dist_info.exists():
                cmds.append("echo ''")
                cmds.append("echo '=== TheRock Distribution Info ==='")
                cmds.append(f"cat {dist_info}")
    else:
        # Traditional package listing
        d = {}
        try:
            with open("/etc/os-release") as fs:
                for line in fs:
                    if "=" in line:
                        k, v = line.rstrip().split("=", 1)
                        d[k] = v.strip('"')
        except Exception:
            d = {'ID_LIKE': 'unknown'}
        
        pkgtype = d.get('ID_LIKE', d.get('ID', 'unknown'))
        cmds.append(f"echo 'Package type: {pkgtype}'")
        
        if 'debian' in pkgtype.lower():
            cmd = "/usr/bin/dpkg -l 2>/dev/null | /bin/grep -i -E 'ocl-icd|kfdtest|llvm-amd|miopen|half|^ii  hip|hcc|hsa|rocm|atmi|^ii  comgr|composa|amd-smi|aomp|amdgpu|rock|mivision|migraph|rocprofiler|roctracer|rocbl|hipify|rocsol|rocthr|rocff|rocalu|rocprim|rocrand|rccl|rocspar|rdc|rocwmma|rpp|openmp|amdfwflash|ocl |opencl' | /usr/bin/sort || echo 'No packages found'"
        else:
            cmd = "/usr/bin/rpm -qa 2>/dev/null | /bin/grep -i -E 'ocl-icd|kfdtest|llvm-amd|miopen|half|hip|hcc|hsa|rocm|atmi|comgr|composa|amd-smi|aomp|amdgpu|rock|mivision|migraph|rocprofiler|roctracer|rocblas|hipify|rocsol|rocthr|rocff|rocalu|rocprim|rocrand|rccl|rocspar|rdc|rocwmma|rpp|openmp|amdfwflash|ocl|opencl' | /usr/bin/sort || echo 'No packages found'"
        
        cmds.append(cmd)
    
    cmd_info = CommandInfo("ROCm Packages Installed", cmds)
    return cmd_info


def print_rocm_environment_variables():
    cmd = "env | /bin/grep -i -E 'rocm|hsa|hip|mpi|openmp|ucx|miopen|virtual_env|conda' || echo 'No relevant env vars found'"
    cmd_info = CommandInfo("ROCm environment variables", [cmd])
    return cmd_info


def print_rocm_smi_details(smi_config, path_resolver):
    cmd_info = None
    
    # Use dynamic path
    rocm_smi_cmd = path_resolver.paths.get('rocm_smi') or "rocm-smi"
    
    if smi_config == "rocm_smi":
        cmd_info = CommandInfo("ROCm SMI", [f"{rocm_smi_cmd} || echo 'rocm-smi not available'"])
    elif smi_config == "ifwi_version":
        ifwi_cmd = f"{rocm_smi_cmd} -v || echo 'IFWI version not available'"
        cmd_info = CommandInfo("IFWI version", [ifwi_cmd])
    elif smi_config == "rocm_smi_showhw":
        showhw_cmd = f"{rocm_smi_cmd} --showhw || echo 'rocm-smi --showhw not available'"
        cmd_info = CommandInfo("ROCm SMI showhw", [showhw_cmd])
    elif smi_config == "rocm_smi_pcie":
        pcie_cmd = f"{rocm_smi_cmd} -c 2>/dev/null | /bin/grep -i -E 'pcie' || echo 'PCIe info not available'"
        cmd_info = CommandInfo("ROCm SMI pcieclk clock", [pcie_cmd])
    elif smi_config == "rocm_smi_pids":
        pids_cmd1 = "ls /sys/class/kfd/kfd/proc/ 2>/dev/null || echo 'KFD proc not available'"
        pids_cmd2 = f"{rocm_smi_cmd} --showpids || echo 'showpids not available'"
        cmd_info = CommandInfo("KFD PIDs sysfs kfd proc", [pids_cmd1, pids_cmd2])
    elif smi_config == "rocm_smi_topology":
        showtops_cmd = f"{rocm_smi_cmd} --showtopo || echo 'showtopo not available'"
        cmd_info = CommandInfo("showtop topology", [showtops_cmd])
    elif smi_config == "rocm_smi_showserial":
        serial_cmd = f"{rocm_smi_cmd} --showserial || echo 'showserial not available'"
        cmd_info = CommandInfo("showserial", [serial_cmd])
    elif smi_config == "rocm_smi_showperflevel":
        perf_cmd = f"{rocm_smi_cmd} --showperflevel || echo 'showperflevel not available'"
        cmd_info = CommandInfo("showperflevel", [perf_cmd])
    elif smi_config == "rocm_smi_showrasinfo":
        showrasinfo_cmd = f"{rocm_smi_cmd} --showrasinfo all || echo 'showrasinfo not available'"
        cmd_info = CommandInfo("ROCm SMI showrasinfo all", [showrasinfo_cmd])
    elif smi_config == "rocm_smi_showxgmierr":
        showxgmierr_cmd = f"{rocm_smi_cmd} --showxgmierr || echo 'showxgmierr not available'"
        cmd_info = CommandInfo("ROCm SMI showxgmierr", [showxgmierr_cmd])
    elif smi_config == "rocm_smi_clocks":
        clock_cmd = f"{rocm_smi_cmd} -cga || echo 'clock info not available'"
        cmd_info = CommandInfo("ROCm SMI clocks", [clock_cmd])
    elif smi_config == "rocm_smi_showcompute_partition":
        compute_cmd = f"{rocm_smi_cmd} --showcomputepartition || echo 'showcomputepartition not available'"
        cmd_info = CommandInfo("ROCm Show computepartition", [compute_cmd])
    elif smi_config == "rocm_smi_nodesbw":
        nodesbw_cmd = f"{rocm_smi_cmd} --shownodesbw || echo 'shownodesbw not available'"
        cmd_info = CommandInfo("ROCm Show Nodebsion", [nodesbw_cmd])
    elif smi_config == "rocm_smi_gpudeviceid":
        gpudeviceid_cmd = f"{rocm_smi_cmd} -i -d 0 || echo 'GPU device ID not available'"
        cmd_info = CommandInfo("ROCM Show GPU Device ID", [gpudeviceid_cmd])
    else:
        cmd_info = None
    
    return cmd_info


def print_rocm_info_details(path_resolver):
    rocminfo_cmd = path_resolver.paths.get('rocminfo') or "rocminfo"
    cmd = f"{rocminfo_cmd} || echo 'rocminfo not available'"
    cmd_info = CommandInfo("rocminfo", [cmd])
    return cmd_info


## dmesg boot logs - GPU/ATOM/DRM/BIOS
def print_dmesg_logs(ignore_prev_boot_logs=True):
    cmds = []
    if os.path.exists("/var/log/journal"):
        cmds.append("echo 'Persistent logging enabled.'")
    else:
        cmd1_str = "WARNING: Persistent logging possibly disabled.\\n"
        cmd1_str = cmd1_str + "WARNING: Please run: \\n"
        cmd1_str = cmd1_str + "       sudo mkdir -p /var/log/journal\\n"
        cmd1_str = cmd1_str + "       sudo systemctl restart systemd-journald.service \\n"
        cmd1_str = cmd1_str + "WARNING: to enable persistent boot logs for collection and analysis.\\n"
        cmd1_str = "echo '" + cmd1_str + "'"
        cmds.append(cmd1_str)

    cmds.append("echo 'Section: dmesg boot logs'")
    cmds.append("/bin/dmesg -T 2>/dev/null | /bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|watchdog|bug|nmi|dazed|too|mce|edac|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER|amdfwflash' || echo 'dmesg not available'")
    
    if not ignore_prev_boot_logs:
        cmd_exec = shutil.which("journalctl")
        
        if cmd_exec is not None:
            cmds.append("echo 'Section: Current boot logs'")
            boot_exec = "/bin/grep -i -E ' Linux v| Command line|power|pnp|pci|gpu|drm|error|xgmi|panic|watchdog|bug|nmi|dazed|too|mce|edac|oop|fail|fault|atom|bios|kfd|vfio|iommu|ras_mask|ECC|smpboot.*CPU|pcieport.*AER|amdfwflash'"
            cmds.append(f"{cmd_exec} -b 2>/dev/null | {boot_exec} || echo 'journalctl not available'")
            cmds.append("echo 'Section: Previous boot logs'")
            cmds.append(f"{cmd_exec} -b 1 2>/dev/null | {boot_exec} || echo 'Previous boot logs not available'")
            cmds.append("echo 'Section: Second boot logs'")
            cmds.append(f"{cmd_exec} -b 2 2>/dev/null | {boot_exec} || echo 'Second boot logs not available'")

    cmd_info = CommandInfo("dmesg GPU/DRM/ATOM/BIOS", cmds)
    return cmd_info


## print amdgpu modinfo
def print_amdgpu_modinfo():
    cmd = "/sbin/modinfo amdgpu 2>/dev/null || modinfo amdgpu 2>/dev/null || echo 'amdgpu module not loaded/available'"
    cmd_info = CommandInfo("amdgpu modinfo", [cmd])
    return cmd_info


## print pip list
def print_pip_list_details():
    cmd = "pip3 list --disable-pip-version-check 2>/dev/null || pip list --disable-pip-version-check 2>/dev/null || echo 'pip not available'"
    cmd_info = CommandInfo("Pip3 package list", [cmd])
    return cmd_info


def print_check_numa_balancing():
    cmd = "cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo 'NUMA balancing info not available'"
    cmd_info = CommandInfo("Numa balancing Info", [cmd])
    return cmd_info


## print cuda version information
def print_cuda_version_information():
    cmd = "nvcc --version 2>/dev/null || echo 'CUDA not available'"
    cmd_info = CommandInfo("CUDA information", [cmd])
    return cmd_info


def print_cuda_env_variables():
    cmd = "env | /bin/grep -i -E 'cuda|nvidia|pytorch|mpi|openmp|ucx|cu' || echo 'No CUDA env vars found'"
    cmd_info = CommandInfo("CUDA Env Variables", [cmd])
    return cmd_info


def print_cuda_packages_installed():
    d = {}
    try:
        with open("/etc/os-release") as fs:
            for line in fs:
                if "=" in line:
                    k, v = line.rstrip().split("=", 1)
                    d[k] = v.strip('"')
        
        pkgtype = d.get('ID_LIKE', d.get('ID', 'unknown'))
        cmd1 = f"echo 'Pkg type: {pkgtype}'"
        cmd2 = None
        
        if 'debian' in pkgtype.lower():
            cmd2 = "/usr/bin/dpkg -l 2>/dev/null | /bin/grep -i -E 'cuda|cu|atlas|hdf5|nccl|nvinfer|nvjpeg|onnx' || echo 'No CUDA packages found'"
        else:
            cmd2 = "/usr/bin/rpm -qa 2>/dev/null | /bin/grep -i -E 'cuda|cu|atlas|hdf5|nccl|nvinfer|nvjpeg|onnx' || echo 'No CUDA packages found'"
        
        cmd_info = CommandInfo("CUDA Packages Installed", [cmd1, cmd2])
    except Exception as e:
        cmd_info = CommandInfo("CUDA Packages Installed", [f"echo 'Error checking packages: {e}'"])
    
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
                    # Changed to canFail=True for robustness with TheRock
                    out = console.sh(cmd, canFail=True)
                    fs.write(out)
                    fs.write("\n")
            fs.close()


def determine_gpu_device_type(path_resolver):
    gpu_device_type = ""
    
    # Try rocm-smi
    rocm_smi_cmd = path_resolver.paths.get('rocm_smi') or "rocm-smi"
    rocm_smi_out = console.sh(f"{rocm_smi_cmd} 2>/dev/null || true", canFail=True)
    
    # Try nvidia-smi
    nv_smi_out = console.sh("nvidia-smi -L 2>/dev/null || true", canFail=True)
    
    if rocm_smi_out and "not found" not in rocm_smi_out and len(rocm_smi_out) > 10:
        gpu_device_type = "AMD"
    elif nv_smi_out and "not found" not in nv_smi_out and len(nv_smi_out) > 10:
        gpu_device_type = "NVIDIA"
    
    return gpu_device_type


def generate_env_info(gpu_device_type, path_resolver):
    global env_map
    
    print(f"Installation Type: {path_resolver.installation_type}")
    print(f"ROCm Root: {path_resolver.rocm_root or 'Not found'}")
    print(f"GPU Device Type: {gpu_device_type or 'Unknown'}")
    
    env_map["hardware_information"] = print_hardware_information()
    env_map["cpu_information"] = print_cpu_hardware_information()
    env_map["gpu_information"] = print_gpu_hardware_information(gpu_device_type, path_resolver)
    env_map["bios_settings"] = print_bios_settings()
    env_map["os_information"] = print_os_information()
    env_map["dmsg_gpu_drm_atom_logs"] = print_dmesg_logs(ignore_prev_boot_logs=True)
    env_map["amdgpu_modinfo"] = print_amdgpu_modinfo()
    env_map["memory_information"] = print_memory_information()
    
    if gpu_device_type == "AMD":
        env_map["rocm_information"] = print_rocm_version_information(path_resolver)
        env_map["rocm_repo_setup"] = print_rocm_repo_setup(path_resolver)
        env_map["rocm_packages_installed"] = print_rocm_packages_installed(path_resolver)
        env_map["rocm_env_variables"] = print_rocm_environment_variables()
        env_map["rocm_smi"] = print_rocm_smi_details("rocm_smi", path_resolver)
        env_map["ifwi_version"] = print_rocm_smi_details("ifwi_version", path_resolver)
        env_map["rocm_smi_showhw"] = print_rocm_smi_details("rocm_smi_showhw", path_resolver)
        env_map["rocm_smi_pcie"] = print_rocm_smi_details("rocm_smi_pcie", path_resolver)
        env_map["rocm_smi_pids"] = print_rocm_smi_details("rocm_smi_pids", path_resolver)
        env_map["rocm_smi_topology"] = print_rocm_smi_details("rocm_smi_topology", path_resolver)
        env_map["rocm_smi_showserial"] = print_rocm_smi_details("rocm_smi_showserial", path_resolver)
        env_map["rocm_smi_showperflevel"] = print_rocm_smi_details("rocm_smi_showperflevel", path_resolver)
        env_map["rocm_smi_showrasinfo"] = print_rocm_smi_details("rocm_smi_showrasinfo", path_resolver)
        env_map["rocm_smi_showxgmierr"] = print_rocm_smi_details("rocm_smi_showxgmierr", path_resolver)
        env_map["rocm_smi_clocks"] = print_rocm_smi_details("rocm_smi_clocks", path_resolver)
        env_map["rocm_smi_showcompute_partition"] = print_rocm_smi_details("rocm_smi_showcompute_partition", path_resolver)
        env_map["rocm_smi_nodesbwi"] = print_rocm_smi_details("rocm_smi_nodesbw", path_resolver)
        env_map["rocm_smi_gpudeviceid"] = print_rocm_smi_details("rocm_smi_gpudeviceid", path_resolver)
        env_map["rocm_info"] = print_rocm_info_details(path_resolver)
    elif gpu_device_type == "NVIDIA":
        env_map["cuda_information"] = print_cuda_version_information()
        env_map["cuda_env_variables"] = print_cuda_env_variables()
        env_map["cuda_packages_installed"] = print_cuda_packages_installed()
    
    env_map["pip_list"] = print_pip_list_details()

    if os.path.exists("/proc/sys/kernel/numa_balancing"):
        env_map["numa_balancing"] = print_check_numa_balancing()


def main():
    # Initialize path resolver
    path_resolver = RocmPathResolver(verbose=args.verbose)
    
    # Detect GPU type with resolver
    gpu_device_type = determine_gpu_device_type(path_resolver)
    
    # Generate environment info
    generate_env_info(gpu_device_type, path_resolver)
    
    # Get configs
    configs = env_map.keys()
    if args.lite:
        configs = parse_env_tags_json("env_tags.json")
    
    # Dump system environment information
    dump_system_env_information(configs, args.output_name)
    print(f"OK: finished dumping the system env details in .{args.output_name} folder")
    
    # CSV output
    if args.dump_csv or args.print_csv:
        csv_file = args.output_name + ".csv"
        out_dir = "." + args.output_name
        csv_parser = CSVParser(csv_file, out_dir, configs)
        csv_parser.dump_csv_output()
        if args.print_csv:
            csv_parser.print_csv_output()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="System environment data collection tool (TheRock + Traditional ROCm compatible)"
    )
    parser.add_argument("--lite", action="store_true", 
                       help="System environment data lite version taken from env_tags.json")
    parser.add_argument("--dump-csv", action="store_true", 
                       help="Dump system config info in CSV file")
    parser.add_argument("--print-csv", action="store_true", 
                       help="Print system config info data")
    parser.add_argument("--output-name", required=False, default="sys_config_info", 
                       help="Output file or directory name")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose detection output")
    
    args = parser.parse_args()
    console = Console(shellVerbose=False, live_output=False)

    main()

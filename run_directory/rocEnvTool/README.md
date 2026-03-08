# ROCm Environment Tool - TheRock Compatible

## Overview

`rocenv_tool.py` is a comprehensive ROCm environment collection tool that works with **both TheRock and traditional ROCm installations**. This tool automatically detects the installation type and adapts its behavior accordingly, collecting important system configuration details that are crucial for debugging and system analysis.

**Note:** This tool requires sudo privileges for collecting some system information.

## Key Features

### 1. **Automatic Installation Detection**
- Detects TheRock installations (Python packages, tarballs, local builds)
- Detects traditional ROCm installations (apt/yum packages)
- Falls back to PATH-based detection if neither is found

### 2. **Dynamic Path Resolution**
- No hardcoded paths to `/opt/rocm`
- Automatically locates `rocminfo`, `rocm-smi`, `hipcc`, etc.
- Works with custom installation directories

### 3. **Robust Error Handling**
- Commands don't fail if tools are missing
- Graceful fallbacks for unavailable features
- Works in minimal container environments

### 4. **TheRock-Specific Features**
- Displays TheRock manifest information
- Shows Python package installations
- Reports virtual environment details
- Lists installation contents

### 5. **Backward Compatibility**
- All original functionality preserved
- Works with existing CSV parser
- Compatible with env_tags.json

## Differences from Original Version

| Aspect | Original (v1) | Current |
|--------|--------------|----------|
| Path detection | Hardcoded `/opt/rocm` | Dynamic detection |
| Installation types | Traditional ROCm only | TheRock + Traditional |
| Package listing | `dpkg -l` / `rpm -qa` | Adaptive (pip for TheRock) |
| Error handling | Fails on missing tools | Graceful fallbacks |
| Version detection | `/opt/rocm/.info/version` | Multi-method detection |
| Repo checking | apt/yum repos | Detects TheRock vs traditional |

## Usage

### Basic Usage

```bash
# Run with automatic detection
python3 rocenv_tool.py

# Verbose mode to see detection details
python3 rocenv_tool.py --verbose

# Custom output name
python3 rocenv_tool.py --output-name my_system_info

# Lite mode (uses env_tags.json)
python3 rocenv_tool.py --lite

# Generate CSV output
python3 rocenv_tool.py --dump-csv

# Generate and print CSV
python3 rocenv_tool.py --dump-csv --print-csv

# Run with sudo for full system information
sudo python3 rocenv_tool.py
```

### Command-Line Options

```
--lite              Use lite version from env_tags.json
--dump-csv          Generate CSV file with system info
--print-csv         Print CSV data to console
--output-name NAME  Output directory name (default: sys_config_info)
-v, --verbose       Enable verbose detection output
```

## How Detection Works

### Detection Methods (in order)

1. **Python Package Detection**
   - Checks for `rocm-sdk` command in PATH
   - Uses `rocm-sdk path --root` to find installation
   - Verifies TheRock markers (manifest.json)

2. **Environment Variable Detection**
   - Checks `ROCM_PATH`, `ROCM_HOME`, `HIP_PATH`
   - Verifies paths for TheRock markers

3. **Common Path Detection**
   - Searches `/opt/rocm`, `~/rocm`, `~/therock`, etc.
   - Checks for `share/therock/therock_manifest.json`

4. **Traditional ROCm Detection**
   - Checks `/opt/rocm/.info/version`
   - Uses traditional package manager paths

5. **PATH-based Detection**
   - Searches for `rocminfo`, `rocm-smi` in PATH
   - Infers installation root from binary location

### TheRock Installation Markers

TheRock installations are identified by:
- `share/therock/therock_manifest.json` (primary marker)
- `share/therock/dist_info.json` (secondary marker)
- Unique directory structure (`lib/llvm/`)
- `rocm-sdk` command availability

## Details Collected

### Tags Available for Lite Mode:

* `hardware_information` - System hardware details
* `cpu_information` - CPU specifications and info
* `gpu_information` - GPU hardware details
* `bios_settings` - BIOS configuration
* `os_information` - Operating system details
* `dmsg_gpu_drm_atom_logs` - GPU kernel logs
* `amdgpu_modinfo` - AMD GPU module information
* `memory_information` - System memory details
* `rocm_information` - ROCm installation details
* `rocm_repo_setup` - Repository configuration
* `rocm_packages_installed` - Installed ROCm packages
* `rocm_env_variables` - ROCm environment variables
* `rocm_smi` - ROCm System Management Interface output
* `ifwi_version` - Integrated Firmware Image version
* `rocm_smi_showhw` - Hardware topology
* `rocm_smi_pcie` - PCIe information
* `rocm_smi_pids` - Process information
* `rocm_smi_topology` - System topology
* `rocm_smi_showserial` - Serial numbers
* `rocm_smi_showperflevel` - Performance levels
* `rocm_smi_showrasinfo` - RAS information
* `rocm_smi_showxgmierr` - XGMI errors
* `rocm_smi_clocks` - Clock information
* `rocm_smi_showcompute_partition` - Compute partitions
* `rocm_smi_nodesbwi` - Node bandwidth
* `rocm_info` - ROCm information utility output
* `pip_list` - Python packages installed
* `numa_balancing` - NUMA balancing status

## Output Structure

The tool generates a directory (default: `.sys_config_info/`) with subdirectories for each category:

```
.sys_config_info/
├── os_information/
│   └── os_information.txt
├── cpu_information/
│   └── cpu_information.txt
├── gpu_information/
│   └── gpu_information.txt
├── rocm_information/
│   └── rocm_information.txt
├── rocm_packages_installed/
│   └── rocm_packages_installed.txt
├── rocm_env_variables/
│   └── rocm_env_variables.txt
├── rocm_smi/
│   └── rocm_smi.txt
├── pip_list/
│   └── pip_list.txt
└── ... (more sections)
```

## TheRock-Specific Output

When TheRock is detected, the output includes:

### rocm_information section
- Installation type: `therock`
- ROCm root path
- TheRock manifest content (commit hash, submodules)
- Version information from `rocm-sdk version`

### rocm_repo_setup section
- Message indicating TheRock doesn't use traditional repos
- `rocm-sdk` command output
- Virtual environment information (if applicable)
- Python package list

### rocm_packages_installed section
- Python ROCm packages (`pip list | grep rocm`)
- TheRock installation directory contents
- `dist_info.json` content (GPU targets, etc.)

## Examples

### Example 1: TheRock in Docker Container

```bash
# In a container built from TheRock
$ python3 rocenv_tool.py --verbose

[DEBUG] Checking for rocm-sdk command...
[DEBUG] Found rocm-sdk at /usr/local/bin/rocm-sdk
[DEBUG] Found TheRock manifest at /opt/rocm/share/therock/therock_manifest.json
Installation Type: therock
ROCm Root: /opt/rocm
GPU Device Type: AMD
OK: finished dumping the system env details in .sys_config_info folder
```

### Example 2: Traditional ROCm System

```bash
# On a system with apt-installed ROCm
$ python3 rocenv_tool.py

Installation Type: traditional
ROCm Root: /opt/rocm
GPU Device Type: AMD
OK: finished dumping the system env details in .sys_config_info folder
```

### Example 3: TheRock Python Virtual Environment

```bash
# In a venv with TheRock pip packages
$ source .venv/bin/activate
$ python3 rocenv_tool.py --verbose

[DEBUG] Checking for rocm-sdk command...
[DEBUG] Found rocm-sdk at /home/user/.venv/bin/rocm-sdk
[DEBUG] Found TheRock at /home/user/.venv/lib/python3.10/site-packages/_rocm_sdk_core
Installation Type: therock
ROCm Root: /home/user/.venv/lib/python3.10/site-packages/_rocm_sdk_core
GPU Device Type: AMD
OK: finished dumping the system env details in .sys_config_info folder
```

## Troubleshooting

### Issue: No ROCm installation detected

**Solution:**
1. Run with `--verbose` to see detection details
2. Ensure ROCm binaries are in PATH: `export PATH=/path/to/rocm/bin:$PATH`
3. Set environment variable: `export ROCM_PATH=/path/to/rocm`
4. For Python packages: activate your virtual environment first

### Issue: rocm-smi not found

**For TheRock:**
- TheRock installations may not include all tools
- Output will show "rocm-smi not available" (not an error)
- Script continues with other available tools

**For Traditional ROCm:**
- Ensure ROCm is properly installed
- Check PATH includes `/opt/rocm/bin`

### Issue: Permission denied errors

**Solution:**
- Some commands require sudo (dmidecode, lshw)
- Run as root for full system information: `sudo python3 rocenv_tool.py`
- Or skip privileged commands (they're non-essential)

### Issue: Commands timing out

**Solution:**
- Check if GPU is accessible
- Verify driver installation
- Some commands may hang if hardware isn't responding

## Integration with Existing Tools

### CSV Parser Compatibility

The tool maintains compatibility with the existing `csv_parser.py`:

```python
# CSV parsing still works
csv_parser = CSVParser(csv_file, out_dir, configs)
csv_parser.dump_csv_output()
csv_parser.print_csv_output()
```

**Note:** TheRock installations may produce different CSV formats for:
- Package listings (pip packages vs dpkg/rpm)
- Repository information (Python packages vs apt repos)

### env_tags.json Support

Lite mode works with `env_tags.json`:

```bash
python3 rocenv_tool.py --lite
```

Only collects information for tags specified in `env_tags.json`.

## Best Practices

1. **Use verbose mode for debugging:**
   ```bash
   python3 rocenv_tool.py --verbose
   ```

2. **Set ROCM_PATH for custom installations:**
   ```bash
   export ROCM_PATH=/custom/path/to/rocm
   python3 rocenv_tool.py
   ```

3. **Activate venv for Python package detection:**
   ```bash
   source .venv/bin/activate
   python3 rocenv_tool.py
   ```

4. **Run as root for complete information:**
   ```bash
   sudo python3 rocenv_tool.py
   ```

5. **Use lite mode for quick checks:**
   ```bash
   python3 rocenv_tool.py --lite
   ```

## Known Limitations

1. **Multi-installation detection:**
   - Tool detects first valid installation found
   - Priority: Python package > env vars > common paths > traditional

2. **Partial installations:**
   - Some TheRock installations may lack certain tools
   - Output will note "not available" for missing tools

3. **Custom build directories:**
   - Local builds may not be auto-detected
   - Use ROCM_PATH environment variable

4. **CSV format variations:**
   - TheRock package listings differ from traditional
   - May affect CSV parser output format

## Technical Details

### RocmPathResolver Class

The core detection logic is in the `RocmPathResolver` class:

```python
resolver = RocmPathResolver(verbose=True)

# Access installation info
print(resolver.installation_type)  # 'therock', 'traditional', or 'unknown'
print(resolver.rocm_root)          # Installation root path
print(resolver.paths['rocminfo'])  # Path to rocminfo binary
print(resolver.get_version())      # ROCm version string
```

### Command Generation

All commands are generated dynamically:

```python
# Dynamic path resolution
cmd = f"{path_resolver.paths.get('rocminfo') or 'rocminfo'} || echo 'rocminfo not available'"
```

This ensures:
- Commands work regardless of installation location
- Graceful failure if tools are missing
- Informative error messages

## Support

For issues or questions:
1. Run with `--verbose` to see detection details
2. Check output for specific error messages
3. Verify ROCm installation is functional
4. Review the test script: `test_rocenv.sh`

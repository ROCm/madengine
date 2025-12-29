#!/usr/bin/env python3
"""
TheRock ROCm Distribution Detection Script

This script detects if TheRock (The HIP Environment and ROCm Kit) is installed
on the system. TheRock uses Python pip packages or standalone tarballs instead
of traditional apt/system package managers.

Detection methods:
1. Python package installation (via pip in venvs or site-packages)
2. Tarball installation (custom directories)
3. Local build directories
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class TherockDetector:
    """Detects TheRock ROCm installations on the system."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.installations: List[Dict] = []

    def log(self, message: str):
        """Print verbose log messages."""
        if self.verbose:
            print(f"[DEBUG] {message}")

    def detect_all(self) -> List[Dict]:
        """Run all detection methods and return list of found installations."""
        self.log("Starting TheRock detection...")

        # Method 1: Check for rocm-sdk command in PATH
        self._detect_rocm_sdk_command()

        # Method 2: Check Python site-packages
        self._detect_python_packages()

        # Method 3: Check common installation paths
        self._detect_tarball_installations()

        # Method 4: Check environment variables
        self._detect_from_env_vars()

        # Method 5: Check for local build directories
        self._detect_build_directories()

        return self.installations

    def _add_installation(self, install_type: str, path: Path, details: Dict):
        """Add a detected installation to the list."""
        installation = {
            "type": install_type,
            "path": str(path.resolve()),
            "details": details,
        }
        
        # Avoid duplicates
        if not any(inst["path"] == installation["path"] for inst in self.installations):
            self.installations.append(installation)
            self.log(f"Found {install_type} installation at: {path}")

    def _is_therock_installation(self, path: Path) -> Optional[Dict]:
        """
        Check if a path contains TheRock installation markers.
        
        Returns dict with installation details if TheRock is detected, None otherwise.
        """
        if not path.exists():
            return None

        details = {}

        # Marker 1: therock_manifest.json
        manifest_path = path / "share" / "therock" / "therock_manifest.json"
        if manifest_path.exists():
            self.log(f"Found therock_manifest.json at {manifest_path}")
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    details["manifest"] = {
                        "commit": manifest.get("the_rock_commit", "unknown"),
                        "submodules_count": len(manifest.get("submodules", [])),
                    }
            except Exception as e:
                self.log(f"Error reading manifest: {e}")

        # Marker 2: dist_info.json
        dist_info_path = path / "share" / "therock" / "dist_info.json"
        if dist_info_path.exists():
            self.log(f"Found dist_info.json at {dist_info_path}")
            try:
                with open(dist_info_path, "r") as f:
                    dist_info = json.load(f)
                    details["dist_info"] = {
                        "amdgpu_targets": dist_info.get("dist_amdgpu_targets", "unknown"),
                    }
            except Exception as e:
                self.log(f"Error reading dist_info: {e}")

        # Marker 3: Unique directory structure (lib/llvm symlink)
        llvm_symlink = path / "llvm"
        if llvm_symlink.exists() and llvm_symlink.is_symlink():
            target = os.readlink(llvm_symlink)
            if target == "lib/llvm":
                self.log(f"Found TheRock-specific llvm symlink at {llvm_symlink}")
                details["llvm_symlink"] = True

        # Marker 4: Check for TheRock-specific binaries
        bin_dir = path / "bin"
        if bin_dir.exists():
            therock_binaries = []
            for binary in ["amdclang", "amdclang++", "amdflang", "hipcc"]:
                if (bin_dir / binary).exists():
                    therock_binaries.append(binary)
            if therock_binaries:
                details["binaries"] = therock_binaries

        # If we found any TheRock markers, return details
        if details:
            return details
        
        return None

    def _detect_rocm_sdk_command(self):
        """Detect rocm-sdk command in PATH (indicates pip installation)."""
        self.log("Checking for rocm-sdk command...")
        
        rocm_sdk_path = shutil.which("rocm-sdk")
        if rocm_sdk_path:
            self.log(f"Found rocm-sdk at: {rocm_sdk_path}")
            
            # Try to get installation details
            details = {"command_path": rocm_sdk_path}
            
            # Get version
            try:
                result = subprocess.run(
                    ["rocm-sdk", "version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    details["version"] = result.stdout.strip()
            except Exception as e:
                self.log(f"Error getting version: {e}")

            # Get root path
            try:
                result = subprocess.run(
                    ["rocm-sdk", "path", "--root"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    root_path = Path(result.stdout.strip())
                    therock_details = self._is_therock_installation(root_path)
                    if therock_details:
                        details.update(therock_details)
                        self._add_installation("python_package", root_path, details)
                        return
            except Exception as e:
                self.log(f"Error getting root path: {e}")

    def _detect_python_packages(self):
        """Detect TheRock Python packages in site-packages."""
        self.log("Checking Python site-packages...")
        
        try:
            import site
            import importlib.util
            
            # Check for rocm_sdk package
            spec = importlib.util.find_spec("rocm_sdk")
            if spec and spec.origin:
                package_path = Path(spec.origin).parent
                self.log(f"Found rocm_sdk package at: {package_path}")
                
                # Try to import and get details
                try:
                    import rocm_sdk
                    details = {
                        "package_path": str(package_path),
                        "version": getattr(rocm_sdk, "__version__", "unknown"),
                    }
                    
                    # Try to get rocm_sdk_core path for TheRock markers
                    core_spec = importlib.util.find_spec("_rocm_sdk_core")
                    if core_spec and core_spec.origin:
                        core_path = Path(core_spec.origin).parent
                        therock_details = self._is_therock_installation(core_path)
                        if therock_details:
                            details.update(therock_details)
                            self._add_installation("python_package", core_path, details)
                except Exception as e:
                    self.log(f"Error importing rocm_sdk: {e}")
                    
        except Exception as e:
            self.log(f"Error checking Python packages: {e}")

    def _detect_tarball_installations(self):
        """Detect tarball installations in common paths."""
        self.log("Checking common installation paths...")
        
        # Common installation directories for tarballs
        common_paths = [
            Path.home() / "rocm",
            Path.home() / "therock",
            Path("/opt/rocm"),
            Path("/usr/local/rocm"),
            Path.home() / ".local" / "rocm",
        ]
        
        for path in common_paths:
            if path.exists():
                details = self._is_therock_installation(path)
                if details:
                    self._add_installation("tarball", path, details)

    def _detect_from_env_vars(self):
        """Detect TheRock from environment variables."""
        self.log("Checking environment variables...")
        
        env_vars = [
            "ROCM_PATH",
            "ROCM_HOME",
            "HIP_PATH",
        ]
        
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                path = Path(value)
                if path.exists():
                    self.log(f"Checking {var}={value}")
                    details = self._is_therock_installation(path)
                    if details:
                        details["detected_via"] = var
                        self._add_installation("environment_variable", path, details)

    def _detect_build_directories(self):
        """Detect local TheRock build directories."""
        self.log("Checking for local build directories...")
        
        # Check current directory and parent directories
        current = Path.cwd()
        for _ in range(5):  # Check up to 5 levels up
            # Check for TheRock source indicators
            if (current / "CMakeLists.txt").exists() and (current / "version.json").exists():
                try:
                    with open(current / "version.json", "r") as f:
                        version_data = json.load(f)
                        if "rocm-version" in version_data:
                            self.log(f"Found TheRock source at: {current}")
                            
                            # Check build directory
                            build_dir = current / "build"
                            if build_dir.exists():
                                dist_dir = build_dir / "dist"
                                if dist_dir.exists():
                                    for dist_subdir in dist_dir.iterdir():
                                        if dist_subdir.is_dir():
                                            details = self._is_therock_installation(dist_subdir)
                                            if details:
                                                details["source_path"] = str(current)
                                                details["rocm_version"] = version_data.get("rocm-version")
                                                self._add_installation("local_build", dist_subdir, details)
                except Exception as e:
                    self.log(f"Error checking build directory: {e}")
            
            parent = current.parent
            if parent == current:
                break
            current = parent


def format_installation_info(installation: Dict) -> str:
    """Format installation information for display."""
    lines = []
    lines.append(f"\nType: {installation['type']}")
    lines.append(f"Path: {installation['path']}")
    
    details = installation.get("details", {})
    
    if "version" in details:
        lines.append(f"Version: {details['version']}")
    
    if "rocm_version" in details:
        lines.append(f"ROCm Version: {details['rocm_version']}")
    
    if "manifest" in details:
        manifest = details["manifest"]
        lines.append(f"TheRock Commit: {manifest.get('commit', 'unknown')}")
        lines.append(f"Submodules: {manifest.get('submodules_count', 0)}")
    
    if "dist_info" in details:
        dist_info = details["dist_info"]
        lines.append(f"GPU Targets: {dist_info.get('amdgpu_targets', 'unknown')}")
    
    if "binaries" in details:
        lines.append(f"Compilers: {', '.join(details['binaries'])}")
    
    if "command_path" in details:
        lines.append(f"Command: {details['command_path']}")
    
    if "detected_via" in details:
        lines.append(f"Detected via: ${details['detected_via']}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Detect TheRock ROCm installations on the system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Detect all installations
  %(prog)s -v                 # Verbose output
  %(prog)s --json             # Output as JSON
  %(prog)s --path /opt/rocm   # Check specific path
        """,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Check specific path for TheRock installation",
    )
    
    args = parser.parse_args()
    
    detector = TherockDetector(verbose=args.verbose)
    
    # If specific path provided, check only that
    if args.path:
        details = detector._is_therock_installation(args.path)
        if details:
            installation = {
                "type": "manual_check",
                "path": str(args.path.resolve()),
                "details": details,
            }
            installations = [installation]
        else:
            print(f"No TheRock installation detected at: {args.path}")
            sys.exit(1)
    else:
        # Run full detection
        installations = detector.detect_all()
    
    # Output results
    if not installations:
        print("No TheRock installations detected.")
        print("\nTheRock uses Python pip packages or tarballs, not apt.")
        print("See: https://github.com/ROCm/TheRock/blob/main/RELEASES.md")
        sys.exit(1)
    
    if args.json:
        print(json.dumps(installations, indent=2))
    else:
        print(f"Found {len(installations)} TheRock installation(s):")
        for i, installation in enumerate(installations, 1):
            print(f"\n{'=' * 60}")
            print(f"Installation #{i}")
            print('=' * 60)
            print(format_installation_info(installation))
        
        print(f"\n{'=' * 60}")
        print("\nTheRock Installation Info:")
        print("- TheRock does NOT use apt/system packages")
        print("- It installs via Python pip OR standalone tarballs")
        print("- Python packages install to venv site-packages")
        print("- Tarballs extract to custom directories")
        print("\nFor more info: https://github.com/ROCm/TheRock")
    
    sys.exit(0)


if __name__ == "__main__":
    main()


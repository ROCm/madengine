#!/usr/bin/env python3
"""
Simplified setup.py for madengine

This setup.py provides compatibility with environments that require traditional 
setup.py installations while reading configuration from pyproject.toml.

For modern installations, prefer:
    pip install .
    python -m build
    pip install -e .[dev]

For legacy compatibility:
    python setup.py install
    python setup.py develop

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import sys
from pathlib import Path

try:
    from setuptools import setup, find_packages
except ImportError:
    print("setuptools is required for setup.py")
    print("Install it using: pip install setuptools")
    sys.exit(1)

def read_readme(readme_file="README.md"):
    """Read README.md file for long description."""
    readme_path = Path(__file__).parent / readme_file
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    
    # Fallback to README.md if specified file doesn't exist
    fallback_path = Path(__file__).parent / "README.md"
    if fallback_path.exists() and readme_file != "README.md":
        with open(fallback_path, "r", encoding="utf-8") as f:
            return f.read()
    
    return ""

def get_config_from_pyproject():
    """Read configuration from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            try:
                import toml as tomllib_alt
                def load(f):
                    if hasattr(f, 'read'):
                        content = f.read()
                        if isinstance(content, bytes):
                            content = content.decode('utf-8')
                        return tomllib_alt.loads(content)
                    else:
                        return tomllib_alt.load(f)
                tomllib.load = load
            except ImportError:
                print("Warning: No TOML library found. Using fallback configuration.")
                return get_fallback_config()
    
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if not pyproject_path.exists():
        print("Warning: pyproject.toml not found. Using fallback configuration.")
        return get_fallback_config()
    
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        
        project = data.get("project", {})
        
        # Extract configuration
        config = {
            "name": project.get("name", "madengine"),
            "description": project.get("description", "MAD Engine"),
            "authors": project.get("authors", []),
            "dependencies": project.get("dependencies", []),
            "optional_dependencies": project.get("optional-dependencies", {}),
            "requires_python": project.get("requires-python", ">=3.8"),
            "classifiers": project.get("classifiers", []),
            "urls": project.get("urls", {}),
            "scripts": project.get("scripts", {}),
            "readme": project.get("readme", "README.md"),
        }
        
        return config
        
    except Exception as e:
        print(f"Warning: Could not read pyproject.toml: {e}")
        return get_fallback_config()

def get_fallback_config():
    """Fallback configuration if pyproject.toml cannot be read."""
    return {
        "name": "madengine",
        "description": "MAD Engine is a set of interfaces to run various AI models from public MAD.",
        "authors": [{"name": "Advanced Micro Devices", "email": "mad.support@amd.com"}],
        "dependencies": [
            "pandas", "GitPython", "jsondiff", "sqlalchemy", "setuptools-rust",
            "paramiko", "mysql-connector-python", "pymysql", "tqdm", "pytest",
            "typing-extensions", "pymongo", "toml",
        ],
        "optional_dependencies": {
            "dev": [
                "pytest", "pytest-cov", "pytest-xdist", "pytest-timeout",
                "pytest-mock", "pytest-asyncio",
            ]
        },
        "requires_python": ">=3.8",
        "classifiers": [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        "urls": {
            "Homepage": "https://github.com/ROCm/madengine",
            "Issues": "https://github.com/ROCm/madengine/issues",
        },
        "scripts": {
            "madengine": "madengine.mad:main"
        },
    }

def get_version():
    """Get version from git tags or fallback to a default."""
    try:
        import subprocess
        import re
        
        # Try to get version from git describe first (more accurate)
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--dirty", "--always", "--long"],
                capture_output=True, text=True, timeout=10, cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                version_str = result.stdout.strip()
                
                # Handle case where there are no tags yet
                if not version_str or len(version_str.split('-')) < 3:
                    # Try to get just the commit hash
                    result = subprocess.run(
                        ["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, timeout=10, cwd=Path(__file__).parent
                    )
                    if result.returncode == 0:
                        commit = result.stdout.strip()
                        # Check if dirty
                        dirty_result = subprocess.run(
                            ["git", "diff-index", "--quiet", "HEAD", "--"],
                            capture_output=True, cwd=Path(__file__).parent
                        )
                        is_dirty = dirty_result.returncode != 0
                        if is_dirty:
                            return f"1.0.0.dev0+g{commit}.dirty"
                        else:
                            return f"1.0.0.dev0+g{commit}"
                
                # Clean up the version string to be PEP 440 compliant
                if version_str.startswith('v'):
                    version_str = version_str[1:]
                
                # Handle patterns like "1.0.0-5-g1234567" or "1.0.0-5-g1234567-dirty"
                match = re.match(r'^([^-]+)-(\d+)-g([a-f0-9]+)(-dirty)?$', version_str)
                if match:
                    base_version, distance, commit, dirty = match.groups()
                    if distance == "0":
                        # Exact tag match
                        if dirty:
                            return f"{base_version}+dirty"
                        else:
                            return base_version
                    else:
                        # Post-release version
                        version_str = f"{base_version}.post{distance}+g{commit}"
                        if dirty:
                            version_str += ".dirty"
                        return version_str
                
                # Handle case where we just have a commit hash (no tags)
                if re.match(r'^[a-f0-9]+(-dirty)?$', version_str):
                    clean_hash = version_str.replace('-dirty', '')
                    if '-dirty' in version_str:
                        return f"1.0.0.dev0+g{clean_hash}.dirty"
                    else:
                        return f"1.0.0.dev0+g{clean_hash}"
                
                return version_str
                
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Fallback to short commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            return f"1.0.0.dev0+g{commit}"
            
    except Exception:
        pass
    
    # Final fallback
    return "1.0.0.dev0"

def main():
    """Main setup function."""
    try:
        config = get_config_from_pyproject()
        
        # Extract author information
        authors = config.get("authors", [])
        if authors:
            author_name = authors[0].get("name", "Advanced Micro Devices")
            author_email = authors[0].get("email", "mad.support@amd.com")
        else:
            author_name = "Advanced Micro Devices"
            author_email = "mad.support@amd.com"
        
        # Extract scripts/entry points
        scripts = config.get("scripts", {})
        entry_points = {"console_scripts": []}
        for script_name, module_path in scripts.items():
            entry_points["console_scripts"].append(f"{script_name}={module_path}")
        
        # Find all packages
        packages = find_packages(where="src")
        if not packages:
            print("Warning: No packages found in src/ directory")
            # Fallback: look for madengine package specifically
            import os
            src_path = Path(__file__).parent / "src"
            if (src_path / "madengine").exists():
                packages = ["madengine"] + [
                    f"madengine.{name}" for name in find_packages(where="src/madengine")
                ]
        
        # Setup package data to include scripts
        package_data = {"madengine": ["scripts/**/*"]}
        
        # Check if scripts directory exists and add patterns accordingly
        scripts_path = Path(__file__).parent / "src" / "madengine" / "scripts"
        if scripts_path.exists():
            # Add more specific patterns to ensure all script files are included
            package_data["madengine"].extend([
                "scripts/*",
                "scripts/*/*",
                "scripts/*/*/*",
                "scripts/*/*/*/*",
            ])
        
        # Get version
        version = get_version()
        
        # Setup configuration
        setup_kwargs = {
            "name": config["name"],
            "version": version,
            "author": author_name,
            "author_email": author_email,
            "description": config["description"],
            "long_description": read_readme(config.get("readme", "README.md")),
            "long_description_content_type": "text/markdown",
            "url": config["urls"].get("Homepage", "https://github.com/ROCm/madengine"),
            "project_urls": config["urls"],
            "package_dir": {"": "src"},
            "packages": packages,
            "install_requires": config["dependencies"],
            "extras_require": config["optional_dependencies"],
            "python_requires": config["requires_python"],
            "entry_points": entry_points if entry_points["console_scripts"] else None,
            "classifiers": config["classifiers"],
            "include_package_data": True,
            "package_data": package_data,
            "zip_safe": False,
            "platforms": ["any"],
        }
        
        # Remove None values to avoid setuptools warnings
        setup_kwargs = {k: v for k, v in setup_kwargs.items() if v is not None}
        
        # Print some info for debugging
        if len(sys.argv) > 1 and any(arg in sys.argv for arg in ["--version", "--help", "--help-commands"]):
            print(f"madengine version: {version}")
            print(f"Found {len(packages)} packages")
            if entry_points and entry_points["console_scripts"]:
                print(f"Console scripts: {', '.join(entry_points['console_scripts'])}")
        
        setup(**setup_kwargs)
        
    except Exception as e:
        print(f"Error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

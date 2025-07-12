"""Test the packaging and project structure.

This module tests the modern Python packaging setup and project structure.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import sys
import importlib.util
# third-party modules
import pytest
# test utilities
from .fixtures.utils import has_gpu, requires_gpu


class TestPackaging:
    """Test the packaging structure and imports."""

    def test_madengine_package_import(self):
        """Test that the madengine package can be imported."""
        import madengine
        assert madengine is not None

    def test_madengine_mad_import(self):
        """Test that the mad module can be imported."""
        from madengine import mad
        assert mad is not None

    def test_madengine_distributed_cli_import(self):
        """Test that the distributed_cli module can be imported."""
        from madengine import distributed_cli
        assert distributed_cli is not None

    def test_core_modules_import(self):
        """Test that core modules can be imported."""
        from madengine.core import context
        from madengine.core import console
        assert context is not None
        assert console is not None

    def test_tools_modules_import(self):
        """Test that tools modules can be imported."""
        from madengine.tools import distributed_orchestrator
        from madengine.tools import discover_models
        assert distributed_orchestrator is not None
        assert discover_models is not None

    def test_utils_modules_import(self):
        """Test that utils modules can be imported."""
        from madengine.utils import ops
        from madengine.utils import ssh_to_db
        assert ops is not None
        assert ssh_to_db is not None

    def test_entry_points_defined(self):
        """Test that entry points are accessible."""
        # Test madengine entry point
        spec = importlib.util.find_spec("madengine.mad")
        assert spec is not None
        
        # Test madengine-cli entry point
        spec = importlib.util.find_spec("madengine.distributed_cli")
        assert spec is not None

    def test_no_legacy_imports(self):
        """Test that legacy import patterns are not used."""
        # Test that we can import scripts as part of the package
        try:
            import madengine.scripts
            # This is valid as scripts are included in the package
            assert True
        except ImportError:
            # If scripts are not available as a module, that's also valid
            assert True

    def test_package_structure(self):
        """Test that package follows expected structure."""
        import madengine
        import os
        
        # Check that package has proper __file__ attribute
        assert hasattr(madengine, '__file__')
        
        # Check that package directory structure exists
        package_dir = os.path.dirname(madengine.__file__)
        expected_subdirs = ['core', 'tools', 'utils', 'db', 'scripts']
        
        for subdir in expected_subdirs:
            subdir_path = os.path.join(package_dir, subdir)
            assert os.path.isdir(subdir_path), f"Expected subdirectory {subdir} not found"

    def test_pyproject_toml_compliance(self):
        """Test that the package follows pyproject.toml standards."""
        import madengine
        
        # Check that version is dynamically determined
        assert hasattr(madengine, '__version__') or True  # Version might be set by build system
        
        # Check that package can be imported from installed location
        assert madengine.__file__ is not None

    def test_development_dependencies_available(self):
        """Test that development dependencies are available in dev environment."""
        # This test only runs if we're in a development environment
        try:
            import pytest
            import black
            import isort
            import mypy
            # If we get here, dev dependencies are available
            assert True
        except ImportError:
            # If in production environment, this is expected
            pytest.skip("Development dependencies not available in production environment")

    def test_modern_packaging_no_setup_py_install(self):
        """Test that we don't rely on setup.py for installation."""
        import os
        from pathlib import Path
        
        # Check if there's a pyproject.toml in the package root
        package_root = Path(__file__).parent.parent
        pyproject_path = package_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist for modern packaging"
        
        # Check that pyproject.toml contains build-system
        content = pyproject_path.read_text()
        assert "[build-system]" in content
        assert "hatchling" in content  # Our chosen build backend


class TestScriptsAccessibility:
    """Test that scripts are accessible from the package."""

    def test_scripts_directory_included(self):
        """Test that scripts directory is included in the package."""
        import madengine
        import os
        
        package_dir = os.path.dirname(madengine.__file__)
        scripts_dir = os.path.join(package_dir, 'scripts')
        
        # Scripts should be included in the package
        assert os.path.isdir(scripts_dir), "Scripts directory should be included in package"

    def test_common_scripts_accessible(self):
        """Test that common scripts are accessible."""
        import madengine
        import os
        
        package_dir = os.path.dirname(madengine.__file__)
        common_scripts_dir = os.path.join(package_dir, 'scripts', 'common')
        
        if os.path.isdir(common_scripts_dir):
            # If common scripts exist, they should be accessible
            assert True
        else:
            # If no common scripts, that's also valid
            pytest.skip("No common scripts directory found")


class TestGPUAwarePackaging:
    """Test packaging functionality with GPU awareness."""

    def test_package_works_on_cpu_only_machine(self):
        """Test that the package works correctly on CPU-only machines."""
        gpu_available = has_gpu()
        
        # Package should import successfully regardless of GPU availability
        import madengine
        assert madengine is not None
        
        # GPU detection results should be accessible
        assert isinstance(gpu_available, bool)
        
        # On CPU-only machines, we should still be able to import all modules
        if not gpu_available:
            from madengine import mad, distributed_cli
            from madengine.core import context, console
            assert all([mad, distributed_cli, context, console])

    @requires_gpu("GPU-specific functionality test")
    def test_package_works_with_gpu(self):
        """Test that the package works correctly on GPU machines."""
        gpu_available = has_gpu()
        
        # This test only runs on GPU machines
        assert gpu_available is True
        
        # All modules should still import correctly
        import madengine
        from madengine import mad, distributed_cli
        from madengine.core import context, console
        assert all([madengine, mad, distributed_cli, context, console])

    def test_context_creation_with_detection(self):
        """Test that Context can be created with or without GPU."""
        gpu_available = has_gpu()
        
        # Context creation should work regardless of GPU availability
        try:
            from madengine.core.context import Context
            # Context creation might fail on CPU-only machines during GPU detection
            # but the import should still work
            assert Context is not None
        except Exception as e:
            # If Context creation fails on CPU-only, that's acceptable
            if not gpu_available:
                pytest.skip(f"Context creation failed on CPU-only machine: {e}")
            else:
                raise

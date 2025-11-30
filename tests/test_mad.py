"""DEPRECATED: Test the legacy mad.py module (argparse-based CLI).

⚠️ DEPRECATED - Tests superseded by test_mad_cli.py ⚠️

This test file is DEPRECATED in favor of comprehensive test_mad_cli.py.
While mad.py itself remains functional for backward compatibility, 
testing focus has shifted to the modern mad_cli.py interface.

See test_mad.DEPRECATED.txt for details.

Replacement: Use test_mad_cli.py for comprehensive CLI testing.

NOTE: 
- mad.py (legacy) - Still works, tests deprecated
- mad_cli.py (modern) - Recommended, comprehensive tests in test_mad_cli.py

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# Skip all tests in this file - superseded by test_mad_cli.py
import pytest
pytestmark = pytest.mark.skip(reason="DEPRECATED: Use test_mad_cli.py for CLI tests")

# built-in modules
import os
import sys
import subprocess
import typing

# third-party modules
import pytest

# project modules
from madengine import mad


class TestLegacyMad:
    """Test the legacy mad.py module (argparse-based).
    
    These tests ensure backward compatibility with the original
    argparse-based CLI. All tests run the script directly via subprocess
    to verify the entry point works correctly.
    """

    def test_mad_cli(self):
        """Test legacy mad.py --help command."""
        # Construct the path to the script
        script_path = os.path.join(
            os.path.dirname(__file__), "../src/madengine", "mad.py"
        )
        # Run the script with arguments using subprocess.run
        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode("utf-8")
        print(output)
        assert result.returncode == 0
        assert "Models automation and dashboarding" in output or "command-line tool" in output

    def test_mad_run_cli(self):
        """Test legacy mad.py run --help command."""
        script_path = os.path.join(
            os.path.dirname(__file__), "../src/madengine", "mad.py"
        )
        result = subprocess.run(
            [sys.executable, script_path, "run", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode("utf-8")
        print(output)
        assert result.returncode == 0
        assert "--tags" in output  # Verify run command has expected options

    def test_mad_report_cli(self):
        """Test legacy mad.py report --help command."""
        script_path = os.path.join(
            os.path.dirname(__file__), "../src/madengine", "mad.py"
        )
        result = subprocess.run(
            [sys.executable, script_path, "report", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode("utf-8")
        print(output)
        assert result.returncode == 0

    def test_mad_database_cli(self):
        """Test legacy mad.py database --help command."""
        script_path = os.path.join(
            os.path.dirname(__file__), "../src/madengine", "mad.py"
        )
        result = subprocess.run(
            [sys.executable, script_path, "database", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode("utf-8")
        print(output)
        assert result.returncode == 0

    def test_mad_discover_cli(self):
        """Test legacy mad.py discover --help command."""
        script_path = os.path.join(
            os.path.dirname(__file__), "../src/madengine", "mad.py"
        )
        result = subprocess.run(
            [sys.executable, script_path, "discover", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode("utf-8")
        print(output)
        assert result.returncode == 0

    def test_mad_version_cli(self):
        """Test legacy mad.py --version command."""
        script_path = os.path.join(
            os.path.dirname(__file__), "../src/madengine", "mad.py"
        )
        result = subprocess.run(
            [sys.executable, script_path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode("utf-8")
        print(output)
        assert result.returncode == 0
        # Version should be printed (could be "dev" or actual version)
        assert len(output.strip()) > 0

    def test_legacy_and_modern_cli_both_work(self):
        """Integration test: Verify both CLI interfaces are accessible."""
        # Test legacy can be imported
        from madengine import mad
        assert hasattr(mad, 'main')
        
        # Test modern can be imported  
        from madengine import mad_cli
        assert hasattr(mad_cli, 'app')
        assert hasattr(mad_cli, 'cli_main')

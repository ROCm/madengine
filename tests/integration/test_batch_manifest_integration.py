"""Integration tests for batch manifest build workflow.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import tempfile

import pytest
from typer.testing import CliRunner

from madengine.cli import app


class TestBatchManifestBuildIntegration:
    """Integration tests for batch manifest build functionality."""

    def test_batch_manifest_mutually_exclusive_with_tags(self):
        """Test that --batch-manifest and --tags are mutually exclusive."""
        runner = CliRunner()
        
        # Create a simple batch manifest
        batch_data = [{"model_name": "dummy", "build_new": True}]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            batch_file = f.name
        
        try:
            # Test that using both options is rejected
            result = runner.invoke(
                app,
                [
                    "build",
                    "--batch-manifest", batch_file,
                    "--tags", "dummy",
                    "--additional-context", '{"gpu_vendor": "AMD", "guest_os": "UBUNTU"}'
                ]
            )
            
            # Should fail with mutual exclusivity error
            assert result.exit_code != 0
            assert "Cannot specify both --batch-manifest and --tags" in result.output
        finally:
            os.unlink(batch_file)


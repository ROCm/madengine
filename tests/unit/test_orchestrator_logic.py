"""
Orchestrator logic unit tests.

Pure unit tests for orchestrator initialization and logic without external dependencies.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
from unittest.mock import MagicMock, mock_open, patch

from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.orchestration.run_orchestrator import RunOrchestrator
from madengine.core.errors import ConfigurationError


@pytest.mark.unit
class TestBuildOrchestratorInit:
    """Test Build Orchestrator initialization."""
    
    @patch("madengine.orchestration.build_orchestrator.Context")
    @patch("os.path.exists", return_value=False)
    def test_initializes_with_minimal_args(self, mock_exists, mock_context):
        """Should initialize with minimal arguments."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True
        
        orchestrator = BuildOrchestrator(mock_args)
        
        assert orchestrator.args == mock_args
        assert orchestrator.additional_context == {}
        assert orchestrator.credentials is None
    
    @patch("madengine.orchestration.build_orchestrator.Context")
    @patch("os.path.exists", return_value=False)
    def test_parses_additional_context_json(self, mock_exists, mock_context):
        """Should parse JSON additional context."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"key": "value"}'
        mock_args.live_output = True
        
        orchestrator = BuildOrchestrator(mock_args)
        
        assert orchestrator.additional_context == {"key": "value"}


@pytest.mark.unit
class TestRunOrchestratorInit:
    """Test Run Orchestrator initialization."""
    
    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_initializes_with_args(self, mock_context):
        """Should initialize with provided arguments."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True
        
        orchestrator = RunOrchestrator(mock_args)
        
        assert orchestrator.args == mock_args
        assert orchestrator.additional_context == {}
    
    def test_parses_deploy_type_from_context(self):
        """Should extract deploy type from additional context."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"deploy": "slurm"}'
        mock_args.live_output = True
        
        orchestrator = RunOrchestrator(mock_args)
        
        assert orchestrator.additional_context["deploy"] == "slurm"


@pytest.mark.unit
class TestManifestValidation:
    """Test manifest validation logic."""
    
    @patch("os.path.exists", return_value=False)
    def test_run_without_manifest_or_tags_raises_error(self, mock_exists):
        """Should raise ConfigurationError without manifest or tags."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True
        
        orchestrator = RunOrchestrator(mock_args)
        
        with pytest.raises(ConfigurationError):
            orchestrator.execute(manifest_file=None, tags=None)


# Total: 5 unit tests

#!/usr/bin/env python3
"""
Unit tests for MADEngine CLI error handling integration.

Tests the integration of unified error handling in mad_cli.py and
distributed_orchestrator.py components.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from rich.console import Console

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from madengine.core.errors import (
    ErrorHandler,
    ConfigurationError,
    set_error_handler,
    get_error_handler,
    create_error_context
)


class TestMadCLIErrorIntegration:
    """Test mad_cli.py error handling integration."""
    
    @patch('madengine.cli.utils.Console')
    def test_setup_logging_creates_error_handler(self, mock_console_class):
        """Test that setup_logging initializes the unified error handler."""
        from madengine.cli import setup_logging
        
        mock_console = Mock(spec=Console)
        mock_console_class.return_value = mock_console
        
        # Clear any existing global error handler
        set_error_handler(None)
        
        # Call setup_logging
        setup_logging(verbose=True)
        
        # Verify error handler was set
        handler = get_error_handler()
        assert handler is not None
        assert isinstance(handler, ErrorHandler)
        assert handler.verbose is True
    
    def test_setup_logging_verbose_flag(self):
        """Test that verbose flag is properly passed to error handler."""
        from madengine.cli import setup_logging
        
        # Test with verbose=False
        setup_logging(verbose=False)
        handler = get_error_handler()
        assert handler.verbose is False
        
        # Test with verbose=True
        setup_logging(verbose=True)
        handler = get_error_handler()
        assert handler.verbose is True
    
    def test_build_command_error_handling(self):
        """Test that build command imports and can use unified error handling."""
        from madengine.cli import ExitCode
        
        # Test that the import works and error handling is available
        try:
            # This tests the actual import in mad_cli.py
            from madengine.cli import setup_logging
            
            # Verify error handler can be set up
            setup_logging(verbose=False)
            
            # Verify handle_error can be imported in the context where it's used
            from madengine.core.errors import handle_error, create_error_context
            
            # Create a test error to ensure the system works
            error = Exception("Test build error")
            context = create_error_context(
                operation="build", 
                phase="build",
                component="CLI"
            )
            
            # This should not raise an exception
            handle_error(error, context=context)
            
        except ImportError as e:
            pytest.fail(f"Error handling integration failed: {e}")
    
    @patch('madengine.cli.utils.console')
    def test_cli_error_display_consistency(self, mock_console):
        """Test that CLI errors are displayed consistently through unified handler."""
        from madengine.cli import setup_logging
        
        # Setup logging to initialize error handler
        setup_logging(verbose=False)
        
        # Get the initialized error handler
        handler = get_error_handler()
        
        # Create a test error
        error = ConfigurationError(
            "Invalid configuration",
            context=create_error_context(
                operation="cli_command",
                component="CLI",
                phase="validation"
            )
        )
        
        # Handle the error through the unified system
        handler.handle_error(error)
        
        # The error should be displayed through Rich console
        # (Note: The actual console calls depend on the handler implementation)
        assert handler.console is not None


@pytest.mark.skip(reason="DEPRECATED: DistributedOrchestrator removed, use test_orchestration.py instead")
class TestDistributedOrchestratorErrorIntegration:
    """Test distributed_orchestrator.py error handling integration.
    
    DEPRECATED: distributed_orchestrator.py was removed in favor of 
    orchestration/build_orchestrator.py and orchestration/run_orchestrator.py.
    """
    
    def test_orchestrator_imports_error_handling(self):
        """Test that distributed_orchestrator imports unified error handling."""
        try:
            from madengine.tools.distributed_orchestrator import (
                handle_error, create_error_context, ConfigurationError
            )
            # If import succeeds, the integration is working
            assert handle_error is not None
            assert create_error_context is not None
            assert ConfigurationError is not None
        except ImportError as e:
            pytest.fail(f"Error handling imports failed in distributed_orchestrator: {e}")
    
    @patch('madengine.tools.distributed_orchestrator.handle_error')
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    @patch('os.path.exists', return_value=True)
    def test_orchestrator_credential_loading_error_handling(self, mock_exists, mock_open, mock_handle_error):
        """Test that credential loading uses unified error handling."""
        from madengine.tools.distributed_orchestrator import DistributedOrchestrator
        
        # Mock args object
        mock_args = Mock()
        mock_args.tags = ["test"]
        mock_args.registry = None
        mock_args.additional_context = "{}"
        mock_args.additional_context_file = None
        mock_args.clean_docker_cache = False
        mock_args.manifest_output = "test.json"
        mock_args.live_output = False
        mock_args.output = "test.csv"
        mock_args.ignore_deprecated_flag = False
        mock_args.data_config_file_name = "data.json"
        mock_args.tools_json_file_name = "tools.json"
        mock_args.generate_sys_env_details = True
        mock_args.force_mirror_local = None
        mock_args.disable_skip_gpu_arch = False
        mock_args.verbose = False
        mock_args._separate_phases = True
        
        # Create orchestrator (should trigger credential loading)
        with patch('madengine.tools.distributed_orchestrator.Context'):
            with patch('madengine.tools.distributed_orchestrator.Data'):
                try:
                    orchestrator = DistributedOrchestrator(mock_args)
                except Exception:
                    # Expected to fail due to mocking, but error handling should be called
                    pass
        
        # Verify that handle_error was called for credential loading failure
        assert mock_handle_error.called
    
    def test_orchestrator_error_context_creation(self):
        """Test that orchestrator creates proper error contexts."""
        from madengine.tools.distributed_orchestrator import create_error_context
        
        context = create_error_context(
            operation="load_credentials",
            component="DistributedOrchestrator",
            file_path="credential.json"
        )
        
        assert context.operation == "load_credentials"
        assert context.component == "DistributedOrchestrator"
        assert context.file_path == "credential.json"
    
    @patch('madengine.tools.distributed_orchestrator.handle_error')
    def test_orchestrator_configuration_error_handling(self, mock_handle_error):
        """Test that configuration errors are properly handled with context."""
        from madengine.tools.distributed_orchestrator import (
            ConfigurationError, create_error_context
        )
        
        # Simulate configuration error handling in orchestrator
        error_context = create_error_context(
            operation="load_credentials",
            component="DistributedOrchestrator",
            file_path="credential.json"
        )
        
        config_error = ConfigurationError(
            "Could not load credentials: File not found",
            context=error_context,
            suggestions=["Check if credential.json exists and has valid JSON format"]
        )
        
        # Handle the error
        mock_handle_error(config_error)
        
        # Verify the error was handled
        mock_handle_error.assert_called_once_with(config_error)
        
        # Verify error structure
        called_error = mock_handle_error.call_args[0][0]
        assert isinstance(called_error, ConfigurationError)
        assert called_error.context.operation == "load_credentials"
        assert called_error.context.component == "DistributedOrchestrator"
        assert called_error.suggestions[0] == "Check if credential.json exists and has valid JSON format"


class TestErrorHandlingWorkflow:
    """Test complete error handling workflow across components."""
    
    @patch('madengine.cli.utils.console')
    def test_end_to_end_error_flow(self, mock_console):
        """Test complete error flow from CLI through orchestrator."""
        from madengine.cli import setup_logging
        from madengine.core.errors import ValidationError
        
        # Setup unified error handling
        setup_logging(verbose=True)
        handler = get_error_handler()
        
        # Create an error that might occur in the orchestrator
        orchestrator_error = ValidationError(
            "Invalid model tag format",
            context=create_error_context(
                operation="model_discovery",
                component="DistributedOrchestrator",
                phase="validation",
                model_name="invalid::tag"
            ),
            suggestions=[
                "Use format: model_name:version",
                "Check model name contains only alphanumeric characters"
            ]
        )
        
        # Handle the error through the unified system
        handler.handle_error(orchestrator_error)
        
        # Verify the error was processed
        assert handler.console is not None
        assert orchestrator_error.context.operation == "model_discovery"
        assert orchestrator_error.context.component == "DistributedOrchestrator"
        assert len(orchestrator_error.suggestions) == 2
    
    def test_error_logging_integration(self):
        """Test that errors are properly logged with structured data."""
        from madengine.cli import setup_logging
        from madengine.core.errors import BuildError
        
        # Setup logging
        setup_logging(verbose=False)
        handler = get_error_handler()
        
        # Create a build error with rich context
        build_error = BuildError(
            "Docker build failed",
            context=create_error_context(
                operation="docker_build",
                component="DockerBuilder",
                phase="build",
                model_name="test_model",
                additional_info={"dockerfile": "Dockerfile.ubuntu.amd"}
            ),
            suggestions=["Check Dockerfile syntax", "Verify base image availability"]
        )
        
        # Mock the logger to capture log calls
        with patch.object(handler, 'logger') as mock_logger:
            handler.handle_error(build_error)
            
            # Verify logging was called with structured data
            mock_logger.error.assert_called_once()
            log_call_args = mock_logger.error.call_args
            
            # Check the log message
            assert "build: Docker build failed" in log_call_args[0][0]
            
            # Check the extra structured data
            extra_data = log_call_args[1]['extra']
            assert extra_data['context']['operation'] == "docker_build"
            assert extra_data['context']['component'] == "DockerBuilder"
            assert extra_data['recoverable'] is False  # BuildError is not recoverable
            assert len(extra_data['suggestions']) == 2
    
    def test_error_context_serialization(self):
        """Test that error contexts can be serialized for logging and debugging."""
        from madengine.core.errors import RuntimeError
        
        context = create_error_context(
            operation="model_execution",
            component="ContainerRunner",
            phase="runtime",
            model_name="llama2",
            node_id="worker-node-01",
            file_path="/models/llama2/run.sh",
            additional_info={
                "container_id": "abc123",
                "gpu_count": 2,
                "timeout": 3600
            }
        )
        
        error = RuntimeError(
            "Model execution failed with exit code 1",
            context=context
        )
        
        # Test that context can be serialized
        context_dict = error.context.__dict__
        json_str = json.dumps(context_dict, default=str)
        
        # Verify all context information is in the serialized form
        assert "model_execution" in json_str
        assert "ContainerRunner" in json_str
        assert "runtime" in json_str
        assert "llama2" in json_str
        assert "worker-node-01" in json_str
        assert "abc123" in json_str


class TestErrorHandlingPerformance:
    """Test performance aspects of error handling."""
    
    def test_error_handler_initialization_performance(self):
        """Test that error handler initialization is fast."""
        import time
        from madengine.core.errors import ErrorHandler
        from rich.console import Console
        
        start_time = time.time()
        
        # Create multiple error handlers
        for _ in range(100):
            console = Console()
            handler = ErrorHandler(console=console, verbose=False)
        
        end_time = time.time()
        
        # Should be able to create 100 handlers in under 1 second
        assert end_time - start_time < 1.0
    
    def test_error_context_creation_performance(self):
        """Test that error context creation is efficient."""
        import time
        
        start_time = time.time()
        
        # Create many error contexts
        for i in range(1000):
            context = create_error_context(
                operation=f"operation_{i}",
                component=f"Component_{i}",
                phase="test",
                model_name=f"model_{i}",
                additional_info={"iteration": i}
            )
        
        end_time = time.time()
        
        # Should be able to create 1000 contexts in under 0.1 seconds
        assert end_time - start_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
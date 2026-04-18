"""Integration tests for error handling: CLI integration, workflow, unified system, backward compat.

Merged from test_cli_error_integration and test_error_system_integration.
Deduplicated: single setup_logging/handler test, one context serialization test.
"""

import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock

import pytest

# Ensure src on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from madengine.core.errors import (
    ErrorHandler,
    MADEngineError,
    ValidationError,
    ConfigurationError,
    RunnerError,
    set_error_handler,
    get_error_handler,
    create_error_context,
)


# ---- CLI error integration ----

class TestCLIErrorIntegration:
    """CLI error handling setup and display."""

    @patch("madengine.cli.utils.Console")
    def test_setup_logging_creates_error_handler(self, mock_console_class):
        """setup_logging initializes the unified error handler; verbose flag is respected."""
        from madengine.cli import setup_logging
        from rich.console import Console

        mock_console = Mock(spec=Console)
        mock_console_class.return_value = mock_console
        set_error_handler(None)
        setup_logging(verbose=True)
        handler = get_error_handler()
        assert handler is not None and isinstance(handler, ErrorHandler)
        assert handler.verbose is True
        setup_logging(verbose=False)
        assert get_error_handler().verbose is False

    def test_build_command_error_handling(self):
        """Build command can use unified error handling (imports and handle_error)."""
        from madengine.cli import setup_logging
        from madengine.core.errors import handle_error

        setup_logging(verbose=False)
        error = Exception("Test build error")
        context = create_error_context(operation="build", phase="build", component="CLI")
        handle_error(error, context=context)

    @patch("madengine.cli.utils.console")
    def test_cli_error_display_consistency(self, mock_console):
        """CLI errors go through unified handler and handler has console."""
        from madengine.cli import setup_logging

        setup_logging(verbose=False)
        handler = get_error_handler()
        error = ConfigurationError(
            "Invalid configuration",
            context=create_error_context(operation="cli_command", component="CLI", phase="validation"),
        )
        handler.handle_error(error)
        assert handler.console is not None


# ---- Error workflow ----

class TestErrorWorkflow:
    """End-to-end error flow and logging."""

    @patch("madengine.cli.utils.console")
    def test_end_to_end_error_flow(self, mock_console):
        """Error flow from CLI through orchestrator-style error."""
        from madengine.cli import setup_logging

        setup_logging(verbose=True)
        handler = get_error_handler()
        err = ValidationError(
            "Invalid model tag format",
            context=create_error_context(
                operation="model_discovery",
                component="DistributedOrchestrator",
                phase="validation",
                model_name="invalid::tag",
            ),
            suggestions=["Use format: model_name:version", "Check model name"],
        )
        handler.handle_error(err)
        assert err.context.operation == "model_discovery"
        assert len(err.suggestions) == 2

    def test_error_logging_integration(self):
        """Errors are logged with structured data."""
        from madengine.cli import setup_logging
        from madengine.core.errors import BuildError

        setup_logging(verbose=False)
        handler = get_error_handler()
        build_error = BuildError(
            "Docker build failed",
            context=create_error_context(
                operation="docker_build",
                component="DockerBuilder",
                phase="build",
                model_name="test_model",
                additional_info={"dockerfile": "Dockerfile.ubuntu.amd"},
            ),
            suggestions=["Check Dockerfile syntax", "Verify base image availability"],
        )
        with patch.object(handler, "logger") as mock_logger:
            handler.handle_error(build_error)
            mock_logger.error.assert_called_once()
            extra = mock_logger.error.call_args[1]["extra"]
            assert extra["context"]["operation"] == "docker_build"
            assert extra["recoverable"] is False
            assert len(extra["suggestions"]) == 2

    def test_error_context_serialization(self):
        """Error context can be serialized for logging."""
        from madengine.core.errors import ExecutionError

        context = create_error_context(
            operation="model_execution",
            component="ContainerRunner",
            phase="runtime",
            model_name="llama2",
            node_id="worker-node-01",
            additional_info={"container_id": "abc123", "gpu_count": 2},
        )
        error = ExecutionError("Model execution failed", context=context)
        data = json.dumps(error.context.__dict__, default=str)
        assert "model_execution" in data and "ContainerRunner" in data and "abc123" in data


# ---- Unified error system ----

class TestUnifiedErrorSystem:
    """Unified error handling system."""

    def test_error_system_basic_functionality(self):
        """Handler handles ValidationError and context is preserved."""
        mock_console = Mock()
        handler = ErrorHandler(console=mock_console, verbose=False)
        context = create_error_context(
            operation="test_operation", component="TestComponent", model_name="test_model"
        )
        error = ValidationError("Test validation error", context=context)
        handler.handle_error(error)
        mock_console.print.assert_called_once()
        assert error.context.operation == "test_operation"
        assert error.recoverable is True

    def test_runner_error_base_class(self):
        """RunnerError is MADEngineError with recoverable and context."""
        context = create_error_context(operation="runner_test", component="TestRunner")
        error = RunnerError("Test runner error", context=context)
        assert isinstance(error, MADEngineError)
        assert error.recoverable is True
        assert error.context.operation == "runner_test"

    def test_error_context_serialization_unified(self):
        """Context with many fields serializes for logging."""
        context = create_error_context(
            operation="serialization_test",
            component="TestComponent",
            model_name="test_model",
            node_id="test_node",
            additional_info={"key": "value", "number": 42},
        )
        error = ValidationError("Test error", context=context)
        data = json.dumps(error.context.__dict__, default=str)
        assert "serialization_test" in data and "test_node" in data and "42" in data

    def test_error_hierarchy_consistency(self):
        """All error types inherit MADEngineError and have context/category/recoverable."""
        from madengine.core.errors import (
            ValidationError,
            NetworkError,
            AuthenticationError,
            ExecutionError,
            BuildError,
            DiscoveryError,
            OrchestrationError,
            RunnerError,
            ConfigurationError,
            DeploymentTimeoutError,
        )

        for error_class in [
            ValidationError,
            NetworkError,
            AuthenticationError,
            ExecutionError,
            BuildError,
            DiscoveryError,
            OrchestrationError,
            RunnerError,
            ConfigurationError,
            DeploymentTimeoutError,
        ]:
            err = error_class("Test error message")
            assert isinstance(err, MADEngineError)
            assert err.context is not None
            assert err.category is not None
            assert isinstance(err.recoverable, bool)

    def test_global_error_handler_workflow(self):
        """handle_error uses global handler when set."""
        from madengine.core.errors import handle_error

        mock_console = Mock()
        handler = ErrorHandler(console=mock_console, verbose=False)
        set_error_handler(handler)
        error = ValidationError(
            "Global handler test",
            context=create_error_context(operation="global_test", component="TestGlobalHandler"),
        )
        handle_error(error)
        mock_console.print.assert_called_once()

    def test_error_suggestions_and_recovery(self):
        """Suggestions and context are stored and displayed."""
        suggestions = ["Check config", "Verify network", "Try --verbose"]
        error = ConfigurationError(
            "Configuration validation failed",
            context=create_error_context(
                operation="config_validation", file_path="/path/to/config.json"
            ),
            suggestions=suggestions,
        )
        assert error.suggestions == suggestions
        assert error.context.file_path == "/path/to/config.json"
        mock_console = Mock()
        ErrorHandler(console=mock_console).handle_error(error)
        mock_console.print.assert_called_once()

    def test_nested_error_handling(self):
        """Nested errors with cause chain are handled."""
        from madengine.core.errors import ExecutionError as MADRuntimeError, OrchestrationError, NetworkError

        orig = NetworkError("Network timeout")
        runtime = MADRuntimeError("Operation failed", cause=orig)
        final = OrchestrationError("Orchestration failed", cause=runtime)
        assert final.cause == runtime and runtime.cause == orig
        mock_console = Mock()
        ErrorHandler(console=mock_console, verbose=True).handle_error(final, show_traceback=True)
        assert mock_console.print.call_count >= 1

    def test_error_performance(self):
        """Handle 100 errors in under 1 second."""
        import time

        mock_console = Mock()
        handler = ErrorHandler(console=mock_console)
        start = time.time()
        for i in range(100):
            err = ValidationError(
                f"Test error {i}",
                context=create_error_context(operation=f"test_op_{i}", component="PerformanceTest"),
            )
            handler.handle_error(err)
        assert time.time() - start < 1.0
        assert mock_console.print.call_count == 100


# ---- Performance (lightweight) ----

class TestErrorHandlingPerformance:
    """Error handler and context creation performance."""

    def test_error_handler_initialization_performance(self):
        """Create 100 handlers in under 1 second."""
        import time
        from rich.console import Console

        start = time.time()
        for _ in range(100):
            ErrorHandler(console=Console(), verbose=False)
        assert time.time() - start < 1.0

    def test_error_context_creation_performance(self):
        """Create 1000 contexts in under 0.1 seconds."""
        import time

        start = time.time()
        for i in range(1000):
            create_error_context(
                operation=f"op_{i}", component=f"C_{i}", phase="test", model_name=f"m_{i}"
            )
        assert time.time() - start < 0.1


# ---- Backward compatibility ----

class TestErrorSystemBackwardCompatibility:
    """Backward compatibility of the error system."""

    def test_legacy_exception_handling_still_works(self):
        """Legacy ValueError can be handled via handle_error with context."""
        try:
            raise ValueError("Legacy error")
        except Exception as e:
            mock_console = Mock()
            handler = ErrorHandler(console=mock_console)
            context = create_error_context(operation="legacy_handling", component="LegacyTest")
            handler.handle_error(e, context=context)
            mock_console.print.assert_called_once()

    def test_error_system_without_rich(self):
        """Errors can be created and have str/recoverable when Rich is unavailable."""
        with patch("madengine.core.errors.Console", side_effect=ImportError):
            error = ValidationError("Test without Rich")
            assert "Test without Rich" in str(error)
            assert error.recoverable is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Unit tests for madengine unified error handling system.

Tests the core error handling functionality including error types,
context management, Rich console integration, and error propagation.
"""

import re
from unittest.mock import Mock

import pytest
from rich.console import Console

from madengine.core.errors import (
    AuthenticationError,
    BuildError,
    ConfigurationError,
    DeploymentTimeoutError,
    DiscoveryError,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ExecutionError,
    MADEngineError,
    NetworkError,
    OrchestrationError,
    RunnerError,
    ValidationError,
    create_error_context,
    get_error_handler,
    handle_error,
    set_error_handler,
)


class TestErrorContext:
    """Test error context data structure."""

    def test_error_context_creation(self):
        """Test error context creation with all fields."""
        additional_info = {"key": "value"}
        context = ErrorContext(
            operation="test_operation",
            phase="execution",
            component="TestComponent",
            model_name="test_model",
            additional_info=additional_info,
        )

        assert context.operation == "test_operation"
        assert context.phase == "execution"
        assert context.component == "TestComponent"
        assert context.model_name == "test_model"
        assert context.additional_info == additional_info


class TestMADEngineErrorHierarchy:
    """Test madengine error class hierarchy."""

    def test_base_madengine_error(self):
        """Test base madengine error functionality."""
        context = ErrorContext(operation="test")
        error = MADEngineError(
            message="Test error",
            category=ErrorCategory.RUNTIME,
            context=context,
            recoverable=True,
            suggestions=["Try again", "Check logs"],
        )

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.category == ErrorCategory.RUNTIME
        assert error.context == context
        assert error.recoverable is True
        assert error.suggestions == ["Try again", "Check logs"]
        assert error.cause is None

    @pytest.mark.parametrize(
        "error_class,category,recoverable,message",
        [
            (ValidationError, ErrorCategory.VALIDATION, True, "Invalid input"),
            (NetworkError, ErrorCategory.CONNECTION, True, "Connection failed"),
            (BuildError, ErrorCategory.BUILD, False, "Build failed"),
            (RunnerError, ErrorCategory.RUNNER, True, "Runner execution failed"),
            (AuthenticationError, ErrorCategory.AUTHENTICATION, True, "Auth failed"),
            (ConfigurationError, ErrorCategory.CONFIGURATION, True, "Config error"),
        ],
    )
    def test_error_types(self, error_class, category, recoverable, message):
        """Test all error types with parametrized test."""
        error = error_class(message)

        assert isinstance(error, MADEngineError)
        assert error.category == category
        assert error.recoverable is recoverable
        assert str(error) == message

    def test_error_with_cause(self):
        """Test error with underlying cause."""
        original_error = ValueError("Original error")
        mad_error = ExecutionError("Runtime failure", cause=original_error)

        assert mad_error.cause == original_error
        assert str(mad_error) == "Runtime failure"

    def test_execution_error_is_mad_engine_error(self):
        """Test that ExecutionError is a MADEngineError."""
        error = ExecutionError("Test error")
        assert isinstance(error, ExecutionError)
        assert isinstance(error, MADEngineError)


class TestErrorHandler:
    """Test ErrorHandler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_console = Mock(spec=Console)
        self.error_handler = ErrorHandler(console=self.mock_console, verbose=False)

    def test_handle_madengine_error(self):
        """Test handling of madengine structured errors."""
        context = create_error_context(
            operation="test_operation", component="TestComponent"
        )
        error = ValidationError(
            "Test validation error", context=context, suggestions=["Check input"]
        )

        self.error_handler.handle_error(error)

        # Verify console.print was called
        self.mock_console.print.assert_called()
        call_args = self.mock_console.print.call_args[0]
        panel = call_args[0]
        assert "Validation Error" in panel.title

    def test_handle_generic_error(self):
        """Test handling of generic Python exceptions."""
        error = ValueError("Generic Python error")
        context = create_error_context(operation="test_op")

        self.error_handler.handle_error(error, context=context)
        self.mock_console.print.assert_called()


class TestGlobalErrorHandler:
    """Test global error handler functionality."""

    def test_set_and_get_error_handler(self):
        """Test setting and getting global error handler."""
        mock_console = Mock(spec=Console)
        handler = ErrorHandler(console=mock_console)

        set_error_handler(handler)
        retrieved_handler = get_error_handler()

        assert retrieved_handler == handler


class TestErrorRecoveryAndSuggestions:
    """Test error recovery indicators and suggestions."""

    def test_recoverable_errors(self):
        """Test that validation errors are marked as recoverable."""
        error = ValidationError("Validation error")
        assert error.recoverable is True

    def test_non_recoverable_errors(self):
        """Test that execution/build errors are marked as non-recoverable."""
        assert ExecutionError("Runtime error").recoverable is False
        assert BuildError("Build error").recoverable is False


class TestErrorPatternMatching:
    """Test error pattern matching for log analysis.

    These tests validate the error pattern fixes for GPT2 training,
    ensuring ROCProf logs are correctly excluded while real errors are caught.
    """

    @pytest.fixture
    def benign_patterns(self):
        """Benign patterns that should be excluded from error detection."""
        return [
            r"^E[0-9]{8}.*generateRocpd\.cpp",
            r"^W[0-9]{8}.*simple_timer\.cpp",
            r"^W[0-9]{8}.*generateRocpd\.cpp",
            r"^E[0-9]{8}.*tool\.cpp",
            "Opened result file:",
            "SQLite3 generation ::",
            r"\[rocprofv3\]",
            "rocpd_op:",
            "rpd_tracer:",
        ]

    @pytest.fixture
    def error_patterns(self):
        """Error patterns that should be detected in logs."""
        return [
            "OutOfMemoryError",
            "HIP out of memory",
            "CUDA out of memory",
            "RuntimeError:",
            "AssertionError:",
            "ValueError:",
            "SystemExit",
            r"failed \(exitcode:",
            r"Traceback \(most recent call last\)",
            "FAILED",
            "Exception:",
            "ImportError:",
            "ModuleNotFoundError:",
        ]

    def test_benign_patterns_match_rocprof_logs(self, benign_patterns):
        """Test that benign patterns correctly match ROCProf logs."""
        # Test cases that should be excluded (false positives)
        rocprof_messages = [
            "E20251230 16:43:09.797714 140310524069632 generateRocpd.cpp:605] Opened result file: /myworkspace/transformers/banff-cyxtera-s83-5/1004_results.db",
            "W20251230 16:43:09.852161 140310524069632 simple_timer.cpp:55] SQLite3 generation :: rocpd_string",
            "W20251230 16:43:09.896980 140310524069632 simple_timer.cpp:55] [rocprofv3] output generation ::     0.121982 sec",
            "E20251230 16:43:12.684603 140140898293696 tool.cpp:2420] HIP (runtime) version 7.1.0 initialized",
            "rocpd_op: 0",
            "rpd_tracer: finalized in 50.142105 ms",
        ]

        for test_line in rocprof_messages:
            matched = any(re.search(pattern, test_line) for pattern in benign_patterns)
            assert matched, f"Failed to match ROCProf log: {test_line[:80]}"

    def test_error_patterns_catch_real_errors(self, error_patterns):
        """Test that error patterns correctly catch real errors."""
        # Test cases that should be caught (real errors)
        real_errors = [
            "RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB",
            "ImportError: cannot import name 'AutoModel' from 'transformers'",
            "ModuleNotFoundError: No module named 'torch'",
            "Traceback (most recent call last):",
            "ValueError: invalid literal for int() with base 10: 'abc'",
            "AssertionError: Expected shape (2, 3) but got (3, 2)",
            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: FAILED",
        ]

        for test_line in real_errors:
            matched = any(re.search(pattern, test_line) for pattern in error_patterns)
            assert matched, f"Failed to catch error: {test_line[:80]}"

    def test_rocprof_messages_dont_trigger_errors(self, error_patterns):
        """Test that ROCProf messages don't trigger error patterns."""
        # ROCProf messages that should NOT trigger errors
        rocprof_messages = [
            "E20251230 16:43:09.797714 140310524069632 generateRocpd.cpp:605] Opened result file",
            "W20251230 16:43:09.852161 140310524069632 simple_timer.cpp:55] SQLite3 generation",
            "rocpd_op: 0",
            "rpd_tracer: finalized in 50.142105 ms",
        ]

        for test_line in rocprof_messages:
            matched = any(re.search(pattern, test_line) for pattern in error_patterns)
            assert (
                not matched
            ), f"False positive: {test_line[:80]} matched error pattern"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

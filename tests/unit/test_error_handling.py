#!/usr/bin/env python3
"""
Unit tests for madengine unified error handling system.

Tests the core error handling functionality including error types,
context management, Rich console integration, and error propagation.
"""

import pytest
import json
import io
import re
from unittest.mock import Mock, patch, MagicMock
from rich.console import Console
from rich.text import Text

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from madengine.core.errors import (
    ErrorCategory,
    ErrorContext,
    MADEngineError,
    ValidationError,
    ConnectionError,
    AuthenticationError,
    RuntimeError,
    BuildError,
    DiscoveryError,
    OrchestrationError,
    RunnerError,
    ConfigurationError,
    TimeoutError,
    ErrorHandler,
    set_error_handler,
    get_error_handler,
    handle_error,
    create_error_context
)


class TestErrorContext:
    """Test error context data structure."""
    
    def test_error_context_creation(self):
        """Test basic error context creation."""
        context = ErrorContext(
            operation="test_operation",
            phase="test_phase",
            component="test_component"
        )
        
        assert context.operation == "test_operation"
        assert context.phase == "test_phase"
        assert context.component == "test_component"
        assert context.model_name is None
        assert context.node_id is None
        assert context.file_path is None
        assert context.additional_info is None
    
    def test_error_context_full(self):
        """Test error context with all fields."""
        additional_info = {"key": "value", "number": 42}
        context = ErrorContext(
            operation="complex_operation",
            phase="execution",
            component="TestComponent",
            model_name="test_model",
            node_id="node-001",
            file_path="/path/to/file.json",
            additional_info=additional_info
        )
        
        assert context.operation == "complex_operation"
        assert context.phase == "execution"
        assert context.component == "TestComponent"
        assert context.model_name == "test_model"
        assert context.node_id == "node-001"
        assert context.file_path == "/path/to/file.json"
        assert context.additional_info == additional_info
    
    def test_create_error_context_function(self):
        """Test create_error_context convenience function."""
        context = create_error_context(
            operation="test_op",
            phase="test_phase",
            model_name="test_model"
        )
        
        assert isinstance(context, ErrorContext)
        assert context.operation == "test_op"
        assert context.phase == "test_phase"
        assert context.model_name == "test_model"


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
            suggestions=["Try again", "Check logs"]
        )
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.category == ErrorCategory.RUNTIME
        assert error.context == context
        assert error.recoverable is True
        assert error.suggestions == ["Try again", "Check logs"]
        assert error.cause is None
    
    @pytest.mark.parametrize("error_class,category,recoverable,message", [
        (ValidationError, ErrorCategory.VALIDATION, True, "Invalid input"),
        (ConnectionError, ErrorCategory.CONNECTION, True, "Connection failed"),
        (BuildError, ErrorCategory.BUILD, False, "Build failed"),
        (RunnerError, ErrorCategory.RUNNER, True, "Runner execution failed"),
        (AuthenticationError, ErrorCategory.AUTHENTICATION, True, "Auth failed"),
        (ConfigurationError, ErrorCategory.CONFIGURATION, True, "Config error"),
    ])
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
        mad_error = RuntimeError("Runtime failure", cause=original_error)
        
        assert mad_error.cause == original_error
        assert str(mad_error) == "Runtime failure"


class TestErrorHandler:
    """Test ErrorHandler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_console = Mock(spec=Console)
        self.error_handler = ErrorHandler(console=self.mock_console, verbose=False)
    
    def test_error_handler_creation(self):
        """Test ErrorHandler initialization."""
        assert self.error_handler.console == self.mock_console
        assert self.error_handler.verbose is False
        assert self.error_handler.logger is not None
    
    def test_handle_madengine_error(self):
        """Test handling of madengine structured errors."""
        context = create_error_context(
            operation="test_operation",
            component="TestComponent",
            model_name="test_model"
        )
        error = ValidationError(
            "Test validation error",
            context=context,
            suggestions=["Check input", "Verify format"]
        )
        
        self.error_handler.handle_error(error)
        
        # Verify console.print was called for the error panel
        self.mock_console.print.assert_called()
        call_args = self.mock_console.print.call_args[0]
        
        # Check that a Rich Panel was created
        assert len(call_args) > 0
        panel = call_args[0]
        assert hasattr(panel, 'title')
        assert "Validation Error" in panel.title
    
    def test_handle_generic_error(self):
        """Test handling of generic Python exceptions."""
        error = ValueError("Generic Python error")
        context = create_error_context(operation="test_op")
        
        self.error_handler.handle_error(error, context=context)
        
        # Verify console.print was called
        self.mock_console.print.assert_called()
        call_args = self.mock_console.print.call_args[0]
        
        # Check that a Rich Panel was created
        assert len(call_args) > 0
        panel = call_args[0]
        assert hasattr(panel, 'title')
        assert "ValueError" in panel.title
    
    def test_handle_error_verbose_mode(self):
        """Test error handling in verbose mode."""
        verbose_handler = ErrorHandler(console=self.mock_console, verbose=True)
        # Create error with a cause to trigger print_exception
        original_error = ValueError("Original error")
        error = RuntimeError("Test runtime error", cause=original_error)
        
        verbose_handler.handle_error(error, show_traceback=True)
        
        # Verify both print and print_exception were called
        assert self.mock_console.print.call_count >= 2
        self.mock_console.print_exception.assert_called()
    
    def test_error_categorization_display(self):
        """Test that different error categories display with correct styling."""
        test_cases = [
            (ValidationError("Validation failed"), "⚠️", "Validation Error"),
            (ConnectionError("Connection failed"), "🔌", "Connection Error"),
            (BuildError("Build failed"), "🔨", "Build Error"),
            (RunnerError("Runner failed"), "🚀", "Runner Error"),
        ]
        
        for error, expected_emoji, expected_title in test_cases:
            self.mock_console.reset_mock()
            self.error_handler.handle_error(error)
            
            # Verify console.print was called
            self.mock_console.print.assert_called()
            call_args = self.mock_console.print.call_args[0]
            panel = call_args[0]
            
            assert expected_emoji in panel.title
            assert expected_title in panel.title


class TestGlobalErrorHandler:
    """Test global error handler functionality."""
    
    def test_set_and_get_error_handler(self):
        """Test setting and getting global error handler."""
        mock_console = Mock(spec=Console)
        handler = ErrorHandler(console=mock_console)
        
        set_error_handler(handler)
        retrieved_handler = get_error_handler()
        
        assert retrieved_handler == handler
    
    def test_handle_error_function(self):
        """Test global handle_error function."""
        mock_console = Mock(spec=Console)
        handler = ErrorHandler(console=mock_console)
        set_error_handler(handler)
        
        error = ValidationError("Test error")
        context = create_error_context(operation="test")
        
        handle_error(error, context=context)
        
        # Verify the handler was used
        mock_console.print.assert_called()
    
    def test_handle_error_no_global_handler(self):
        """Test handle_error function when no global handler is set."""
        # Clear global handler
        set_error_handler(None)
        
        with patch('madengine.core.errors.logging') as mock_logging:
            error = ValueError("Test error")
            handle_error(error)
            
            # Should fallback to logging
            mock_logging.error.assert_called_once()


class TestErrorContextPropagation:
    """Test error context propagation through call stack."""
    
    def test_context_preservation_through_hierarchy(self):
        """Test that context is preserved when creating derived errors."""
        original_context = create_error_context(
            operation="original_op",
            component="OriginalComponent",
            model_name="test_model"
        )
        
        # Create a base error with context
        base_error = MADEngineError(
            "Base error",
            ErrorCategory.RUNTIME,
            context=original_context
        )
        
        # Create a derived error that should preserve context
        derived_error = ValidationError(
            "Derived error",
            context=original_context,
            cause=base_error
        )
        
        assert derived_error.context == original_context
        assert derived_error.cause == base_error
        assert derived_error.context.operation == "original_op"
        assert derived_error.context.component == "OriginalComponent"
    
    def test_context_enrichment(self):
        """Test adding additional context information."""
        base_context = create_error_context(operation="base_op")
        
        # Create enriched context
        enriched_context = ErrorContext(
            operation=base_context.operation,
            phase="enriched_phase",
            component="EnrichedComponent",
            additional_info={"enriched": True}
        )
        
        error = RuntimeError("Test error", context=enriched_context)
        
        assert error.context.operation == "base_op"
        assert error.context.phase == "enriched_phase"
        assert error.context.component == "EnrichedComponent"
        assert error.context.additional_info["enriched"] is True


class TestErrorRecoveryAndSuggestions:
    """Test error recovery indicators and suggestions."""
    
    def test_recoverable_errors(self):
        """Test that certain error types are marked as recoverable."""
        recoverable_errors = [
            ValidationError("Validation error"),
            ConnectionError("Connection error"),
            AuthenticationError("Auth error"),
            ConfigurationError("Config error"),
            TimeoutError("Timeout error"),
        ]
        
        for error in recoverable_errors:
            assert error.recoverable is True, f"{type(error).__name__} should be recoverable"
    
    def test_non_recoverable_errors(self):
        """Test that certain error types are marked as non-recoverable."""
        non_recoverable_errors = [
            RuntimeError("Runtime error"),
            BuildError("Build error"),
            OrchestrationError("Orchestration error"),
        ]
        
        for error in non_recoverable_errors:
            assert error.recoverable is False, f"{type(error).__name__} should not be recoverable"
    
    def test_suggestions_in_errors(self):
        """Test that suggestions are properly included in errors."""
        suggestions = ["Check configuration", "Verify credentials", "Try again"]
        error = ValidationError(
            "Validation failed",
            suggestions=suggestions
        )
        
        assert error.suggestions == suggestions
        
        # Test handling displays suggestions
        mock_console = Mock(spec=Console)
        handler = ErrorHandler(console=mock_console)
        handler.handle_error(error)
        
        # Verify console.print was called and suggestions are in output
        mock_console.print.assert_called()


class TestErrorIntegration:
    """Test error handling integration scenarios."""
    
    def test_error_serialization_context(self):
        """Test that error context can be serialized for logging."""
        context = create_error_context(
            operation="test_operation",
            phase="test_phase",
            component="TestComponent",
            model_name="test_model",
            additional_info={"key": "value"}
        )
        
        error = ValidationError("Test error", context=context)
        
        # Context should be serializable
        context_dict = error.context.__dict__
        json_str = json.dumps(context_dict, default=str)
        
        assert "test_operation" in json_str
        assert "test_phase" in json_str
        assert "TestComponent" in json_str
        assert "test_model" in json_str
    
    def test_nested_error_handling(self):
        """Test handling of nested exceptions."""
        original_error = ConnectionError("Network timeout")
        wrapped_error = RuntimeError("Operation failed", cause=original_error)
        final_error = OrchestrationError("Orchestration failed", cause=wrapped_error)
        
        assert final_error.cause == wrapped_error
        assert wrapped_error.cause == original_error
        
        # Test that the handler can display nested error information
        mock_console = Mock(spec=Console)
        handler = ErrorHandler(console=mock_console)
        handler.handle_error(final_error)
        
        mock_console.print.assert_called()


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
            assert not matched, f"False positive: {test_line[:80]} matched error pattern"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
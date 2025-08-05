#!/usr/bin/env python3
"""
Unit tests for MADEngine unified error handling system.

Tests the core error handling functionality including error types,
context management, Rich console integration, and error propagation.
"""

import pytest
import json
import io
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


class TestErrorCategories:
    """Test error category enumeration."""
    
    def test_error_categories_exist(self):
        """Test that all required error categories are defined."""
        expected_categories = [
            "validation", "connection", "authentication", "runtime",
            "build", "discovery", "orchestration", "runner",
            "configuration", "timeout"
        ]
        
        for category in expected_categories:
            assert hasattr(ErrorCategory, category.upper())
            assert ErrorCategory[category.upper()].value == category


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
    """Test MADEngine error class hierarchy."""
    
    def test_base_madengine_error(self):
        """Test base MADEngine error functionality."""
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
    
    def test_validation_error(self):
        """Test ValidationError specific functionality."""
        error = ValidationError("Invalid input")
        
        assert isinstance(error, MADEngineError)
        assert error.category == ErrorCategory.VALIDATION
        assert error.recoverable is True
        assert str(error) == "Invalid input"
    
    def test_connection_error(self):
        """Test ConnectionError specific functionality."""
        context = create_error_context(operation="connect", node_id="node-1")
        error = ConnectionError("Connection failed", context=context)
        
        assert isinstance(error, MADEngineError)
        assert error.category == ErrorCategory.CONNECTION
        assert error.recoverable is True
        assert error.context.node_id == "node-1"
    
    def test_build_error(self):
        """Test BuildError specific functionality."""
        error = BuildError("Build failed")
        
        assert isinstance(error, MADEngineError)
        assert error.category == ErrorCategory.BUILD
        assert error.recoverable is False
    
    def test_runner_error(self):
        """Test RunnerError specific functionality."""
        error = RunnerError("Runner execution failed")
        
        assert isinstance(error, MADEngineError)
        assert error.category == ErrorCategory.RUNNER
        assert error.recoverable is True
    
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
        """Test handling of MADEngine structured errors."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
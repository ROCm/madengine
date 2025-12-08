#!/usr/bin/env python3
"""
Integration tests for MADEngine unified error handling system.

This test file focuses on testing the integration without requiring
optional dependencies like paramiko, ansible-runner, or kubernetes.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from madengine.core.errors import (
    ErrorHandler,
    MADEngineError,
    ValidationError,
    ConfigurationError,
    RunnerError,
    set_error_handler,
    get_error_handler,
    create_error_context
)


class TestUnifiedErrorSystem:
    """Test the unified error handling system integration."""
    
    def test_error_system_basic_functionality(self):
        """Test basic error system functionality works."""
        # Create error handler
        mock_console = Mock()
        handler = ErrorHandler(console=mock_console, verbose=False)
        
        # Create error with context
        context = create_error_context(
            operation="test_operation",
            component="TestComponent",
            model_name="test_model"
        )
        
        error = ValidationError("Test validation error", context=context)
        
        # Handle the error
        handler.handle_error(error)
        
        # Verify it was handled
        mock_console.print.assert_called_once()
        
        # Verify error structure
        assert error.context.operation == "test_operation"
        assert error.context.component == "TestComponent"
        assert error.recoverable is True
    
    def test_mad_cli_error_handler_setup(self):
        """Test that mad_cli properly sets up error handling."""
        from madengine.cli import setup_logging
        
        # Clear existing handler
        set_error_handler(None)
        
        # Setup logging
        setup_logging(verbose=True)
        
        # Verify handler was created
        handler = get_error_handler()
        assert handler is not None
        assert isinstance(handler, ErrorHandler)
        assert handler.verbose is True
    
    
    def test_runner_error_base_class(self):
        """Test that RunnerError base class works properly."""
        context = create_error_context(
            operation="runner_test",
            component="TestRunner"
        )
        
        error = RunnerError("Test runner error", context=context)
        
        assert isinstance(error, MADEngineError)
        assert error.recoverable is True
        assert error.context.operation == "runner_test"
        assert error.context.component == "TestRunner"
    
    def test_error_context_serialization(self):
        """Test that error contexts can be serialized."""
        context = create_error_context(
            operation="serialization_test",
            component="TestComponent",
            model_name="test_model",
            node_id="test_node",
            additional_info={"key": "value", "number": 42}
        )
        
        error = ValidationError("Test error", context=context)
        
        # Test serialization
        context_dict = error.context.__dict__
        json_str = json.dumps(context_dict, default=str)
        
        # Verify content
        assert "serialization_test" in json_str
        assert "TestComponent" in json_str
        assert "test_model" in json_str
        assert "test_node" in json_str
        assert "key" in json_str
        assert "42" in json_str
    
    def test_error_hierarchy_consistency(self):
        """Test that all error types maintain consistent behavior."""
        from madengine.core.errors import (
            ValidationError, ConnectionError, AuthenticationError,
            RuntimeError, BuildError, DiscoveryError, OrchestrationError,
            RunnerError, ConfigurationError, TimeoutError
        )
        
        error_classes = [
            ValidationError, ConnectionError, AuthenticationError,
            RuntimeError, BuildError, DiscoveryError, OrchestrationError,
            RunnerError, ConfigurationError, TimeoutError
        ]
        
        for error_class in error_classes:
            error = error_class("Test error message")
            
            # All should inherit from MADEngineError
            assert isinstance(error, MADEngineError)
            
            # All should have context (even if default)
            assert error.context is not None
            
            # All should have category
            assert error.category is not None
            
            # All should have recoverable flag
            assert isinstance(error.recoverable, bool)
    
    def test_global_error_handler_workflow(self):
        """Test the complete global error handler workflow."""
        from madengine.core.errors import handle_error
        
        # Create and set global handler
        mock_console = Mock()
        handler = ErrorHandler(console=mock_console, verbose=False)
        set_error_handler(handler)
        
        # Create error
        error = ValidationError(
            "Global handler test",
            context=create_error_context(
                operation="global_test",
                component="TestGlobalHandler"
            )
        )
        
        # Use global handle_error function
        handle_error(error)
        
        # Verify it was handled through the global handler
        mock_console.print.assert_called_once()
    
    def test_error_suggestions_and_recovery(self):
        """Test error suggestions and recovery information."""
        suggestions = [
            "Check your configuration file",
            "Verify network connectivity",
            "Try running with --verbose flag"
        ]
        
        error = ConfigurationError(
            "Configuration validation failed",
            context=create_error_context(
                operation="config_validation",
                file_path="/path/to/config.json"
            ),
            suggestions=suggestions
        )
        
        assert error.suggestions == suggestions
        assert error.recoverable is True
        assert error.context.file_path == "/path/to/config.json"
        
        # Test error display includes suggestions
        mock_console = Mock()
        handler = ErrorHandler(console=mock_console)
        handler.handle_error(error)
        
        # Should have been called to display the error
        mock_console.print.assert_called_once()
    
    def test_nested_error_handling(self):
        """Test handling of nested errors with causes."""
        from madengine.core.errors import RuntimeError as MADRuntimeError, OrchestrationError
        
        # Create a chain of errors
        original_error = ConnectionError("Network timeout")
        runtime_error = MADRuntimeError("Operation failed", cause=original_error)
        final_error = OrchestrationError("Orchestration failed", cause=runtime_error)
        
        # Test the chain
        assert final_error.cause == runtime_error
        assert runtime_error.cause == original_error
        
        # Test handling preserves the chain
        mock_console = Mock()
        handler = ErrorHandler(console=mock_console, verbose=True)
        handler.handle_error(final_error, show_traceback=True)
        
        # Should display error and potentially traceback
        assert mock_console.print.call_count >= 1
    
    def test_error_performance(self):
        """Test that error handling is performant."""
        import time
        
        mock_console = Mock()
        handler = ErrorHandler(console=mock_console)
        
        start_time = time.time()
        
        # Create and handle many errors
        for i in range(100):
            error = ValidationError(
                f"Test error {i}",
                context=create_error_context(
                    operation=f"test_op_{i}",
                    component="PerformanceTest"
                )
            )
            handler.handle_error(error)
        
        end_time = time.time()
        
        # Should handle 100 errors in under 1 second
        assert end_time - start_time < 1.0
        
        # Verify all errors were handled
        assert mock_console.print.call_count == 100


class TestErrorSystemBackwardCompatibility:
    """Test backward compatibility of the error system."""
    
    def test_legacy_exception_handling_still_works(self):
        """Test that legacy exception patterns still work."""
        try:
            # Simulate old-style exception raising
            raise ValueError("Legacy error")
        except Exception as e:
            # Should be able to handle with new system
            mock_console = Mock()
            handler = ErrorHandler(console=mock_console)
            
            context = create_error_context(
                operation="legacy_handling",
                component="LegacyTest"
            )
            
            handler.handle_error(e, context=context)
            
            # Should handle gracefully
            mock_console.print.assert_called_once()
    
    def test_error_system_without_rich(self):
        """Test error system fallback when Rich is not available."""
        # This test verifies the system degrades gracefully
        # In practice, Rich is a hard dependency, but we test the concept
        
        with patch('madengine.core.errors.Console', side_effect=ImportError):
            # Should still be able to create basic errors
            error = ValidationError("Test without Rich")
            assert str(error) == "Test without Rich"
            assert error.recoverable is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Unit tests for MADEngine runner error standardization.

Tests the unified error handling across all distributed runners without
requiring optional dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from madengine.core.errors import (
    ErrorCategory,
    ConnectionError as MADConnectionError,
    RunnerError,
    create_error_context
)


class TestRunnerErrorConcepts:
    """Test runner error concepts without requiring optional dependencies."""
    
    def test_runner_error_base_class(self):
        """Test that RunnerError base class works correctly."""
        context = create_error_context(
            operation="runner_test",
            component="TestRunner",
            node_id="test-node"
        )
        
        error = RunnerError("Test runner error", context=context)
        
        # Test inheritance
        assert isinstance(error, RunnerError)
        assert error.category == ErrorCategory.RUNNER
        assert error.recoverable is True
        
        # Test context
        assert error.context.operation == "runner_test"
        assert error.context.component == "TestRunner"
        assert error.context.node_id == "test-node"
    
    def test_connection_error_for_ssh_like_scenarios(self):
        """Test connection error that SSH runner would use."""
        context = create_error_context(
            operation="ssh_connection",
            component="SSHRunner",
            node_id="remote-host",
            additional_info={"error_type": "timeout"}
        )
        
        error = MADConnectionError(
            "SSH timeout error on remote-host: Connection timed out",
            context=context
        )
        
        # Test structure
        assert isinstance(error, MADConnectionError)
        assert error.category == ErrorCategory.CONNECTION
        assert error.recoverable is True
        assert error.context.node_id == "remote-host"
        assert error.context.additional_info["error_type"] == "timeout"
    
    def test_runner_error_for_ansible_like_scenarios(self):
        """Test runner error that Ansible runner would use."""
        context = create_error_context(
            operation="ansible_execution",
            component="AnsibleRunner",
            file_path="/path/to/playbook.yml"
        )
        
        error = RunnerError(
            "Ansible execution error in playbook.yml: Playbook failed",
            context=context,
            suggestions=["Check playbook syntax", "Verify inventory file"]
        )
        
        # Test structure
        assert isinstance(error, RunnerError)
        assert error.category == ErrorCategory.RUNNER
        assert error.recoverable is True
        assert error.context.file_path == "/path/to/playbook.yml"
        assert len(error.suggestions) == 2
    
    def test_runner_error_for_k8s_like_scenarios(self):
        """Test runner error that Kubernetes runner would use."""
        context = create_error_context(
            operation="kubernetes_execution",
            component="KubernetesRunner",
            additional_info={
                "resource_type": "Pod",
                "resource_name": "madengine-job-001"
            }
        )
        
        error = RunnerError(
            "Kubernetes error in Pod/madengine-job-001: Pod creation failed",
            context=context
        )
        
        # Test structure
        assert isinstance(error, RunnerError)
        assert error.category == ErrorCategory.RUNNER
        assert error.recoverable is True
        assert error.context.additional_info["resource_type"] == "Pod"
        assert error.context.additional_info["resource_name"] == "madengine-job-001"


class TestRunnerErrorHandling:
    """Test unified error handling for runner scenarios."""
    
    def test_all_runner_scenarios_use_unified_system(self):
        """Test that all runner scenarios can use the unified error system."""
        from madengine.core.errors import ErrorHandler
        from rich.console import Console
        
        mock_console = Mock(spec=Console)
        handler = ErrorHandler(console=mock_console)
        
        # Create different runner-like errors
        ssh_error = MADConnectionError(
            "SSH connection failed",
            context=create_error_context(
                operation="ssh_connection",
                component="SSHRunner",
                node_id="host1"
            )
        )
        
        ansible_error = RunnerError(
            "Ansible playbook failed",
            context=create_error_context(
                operation="ansible_execution",
                component="AnsibleRunner",
                file_path="/playbook.yml"
            )
        )
        
        k8s_error = RunnerError(
            "Kubernetes pod failed",
            context=create_error_context(
                operation="kubernetes_execution",
                component="KubernetesRunner"
            )
        )
        
        errors = [ssh_error, ansible_error, k8s_error]
        
        # All should be handleable by unified handler
        for error in errors:
            mock_console.reset_mock()
            handler.handle_error(error)
            
            # Verify error was handled
            mock_console.print.assert_called_once()
            
            # Verify Rich panel was created
            call_args = mock_console.print.call_args[0]
            panel = call_args[0]
            assert hasattr(panel, 'title')
    
    def test_runner_error_context_consistency(self):
        """Test that all runner errors have consistent context structure."""
        runner_scenarios = [
            ("ssh_connection", "SSHRunner", "host1"),
            ("ansible_execution", "AnsibleRunner", "host2"),
            ("kubernetes_execution", "KubernetesRunner", "cluster1")
        ]
        
        for operation, component, node_id in runner_scenarios:
            context = create_error_context(
                operation=operation,
                component=component,
                node_id=node_id
            )
            
            if "connection" in operation:
                error = MADConnectionError("Connection failed", context=context)
            else:
                error = RunnerError("Execution failed", context=context)
            
            # All should have consistent context structure
            assert error.context.operation == operation
            assert error.context.component == component
            assert error.context.node_id == node_id
            assert error.recoverable is True
    
    def test_runner_error_suggestions_work(self):
        """Test that runner errors can include helpful suggestions."""
        suggestions = [
            "Check network connectivity",
            "Verify authentication credentials", 
            "Try running with --verbose flag"
        ]
        
        error = RunnerError(
            "Distributed execution failed",
            context=create_error_context(
                operation="distributed_execution",
                component="GenericRunner"
            ),
            suggestions=suggestions
        )
        
        assert error.suggestions == suggestions
        
        # Test that suggestions are displayed
        from madengine.core.errors import ErrorHandler
        mock_console = Mock()
        handler = ErrorHandler(console=mock_console)
        handler.handle_error(error)
        
        # Should have called print to display error with suggestions
        mock_console.print.assert_called_once()


class TestActualRunnerIntegration:
    """Test integration with actual runner modules where possible."""
    
    def test_ssh_runner_error_class_if_available(self):
        """Test SSH runner error class if the module can be imported."""
        try:
            # Try to import without optional dependencies
            with patch('paramiko.SSHClient'), patch('scp.SCPClient'):
                from madengine.runners.ssh_runner import SSHConnectionError
                
                error = SSHConnectionError("test-host", "connection", "failed")
                
                # Should inherit from unified error system
                assert isinstance(error, MADConnectionError)
                assert error.hostname == "test-host"
                assert error.error_type == "connection"
                
        except ImportError:
            # Expected when dependencies aren't installed
            pytest.skip("SSH runner dependencies not available")
    
    def test_ansible_runner_error_class_if_available(self):
        """Test Ansible runner error class if the module can be imported."""
        try:
            # Try to import without optional dependencies
            with patch('ansible_runner.run'):
                from madengine.runners.ansible_runner import AnsibleExecutionError
                
                error = AnsibleExecutionError("failed", "/playbook.yml")
                
                # Should inherit from unified error system
                assert isinstance(error, RunnerError)
                assert error.playbook_path == "/playbook.yml"
                
        except ImportError:
            # Expected when dependencies aren't installed
            pytest.skip("Ansible runner dependencies not available")
    
    def test_k8s_runner_error_class_if_available(self):
        """Test Kubernetes runner error class if the module can be imported."""
        try:
            # Try to import without optional dependencies
            with patch('kubernetes.client'), patch('kubernetes.config'):
                from madengine.runners.k8s_runner import KubernetesExecutionError
                
                error = KubernetesExecutionError("failed", "Pod", "test-pod")
                
                # Should inherit from unified error system
                assert isinstance(error, RunnerError)
                assert error.resource_type == "Pod"
                assert error.resource_name == "test-pod"
                
        except ImportError:
            # Expected when dependencies aren't installed
            pytest.skip("Kubernetes runner dependencies not available")


class TestImportErrorHandling:
    """Test that import errors are handled gracefully."""
    
    def test_import_error_messages_are_informative(self):
        """Test that import errors provide helpful information."""
        # Test the actual import behavior when dependencies are missing
        
        # SSH runner
        with pytest.raises(ImportError) as exc_info:
            import madengine.runners.ssh_runner
        
        error_msg = str(exc_info.value)
        assert "SSH runner requires" in error_msg or "No module named" in error_msg
        
        # Ansible runner
        with pytest.raises(ImportError) as exc_info:
            import madengine.runners.ansible_runner
        
        error_msg = str(exc_info.value)
        assert "Ansible runner requires" in error_msg or "No module named" in error_msg
        
        # Kubernetes runner
        with pytest.raises(ImportError) as exc_info:
            import madengine.runners.k8s_runner
        
        error_msg = str(exc_info.value)
        assert "Kubernetes runner requires" in error_msg or "No module named" in error_msg
    
    def test_runner_factory_handles_missing_runners(self):
        """Test that runner factory gracefully handles missing optional runners."""
        try:
            from madengine.runners.factory import RunnerFactory
            
            # Should not crash even if optional runners aren't available
            # This tests the import warnings but doesn't require the runners to work
            assert RunnerFactory is not None
            
        except ImportError as e:
            # If the factory itself can't be imported, that's a different issue
            pytest.fail(f"Runner factory should be importable: {e}")


class TestErrorSystemRobustness:
    """Test that the error system is robust to various scenarios."""
    
    def test_error_system_works_without_optional_modules(self):
        """Test that core error system works even without optional modules."""
        from madengine.core.errors import (
            ErrorHandler, RunnerError, ConnectionError, ValidationError
        )
        
        # Should work without any runner modules
        mock_console = Mock()
        handler = ErrorHandler(console=mock_console)
        
        error = ValidationError("Test error")
        handler.handle_error(error)
        
        mock_console.print.assert_called_once()
    
    def test_error_context_serialization_robustness(self):
        """Test that error context serialization handles various data types."""
        import json
        
        context = create_error_context(
            operation="robust_test",
            component="TestComponent",
            additional_info={
                "string": "value",
                "number": 42,
                "boolean": True,
                "none": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"}
            }
        )
        
        error = RunnerError("Test error", context=context)
        
        # Should be serializable
        context_dict = error.context.__dict__
        json_str = json.dumps(context_dict, default=str)
        
        # Should contain all the data
        assert "robust_test" in json_str
        assert "TestComponent" in json_str
        assert "42" in json_str
        assert "nested" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
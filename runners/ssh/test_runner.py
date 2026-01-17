#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for SSH Multi-Node Runner

This script provides comprehensive unit tests and validation for the SSH runner.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from config_manager import MultiNodeConfig, SSHConfig, ClusterConfig, TrainingConfig
    from ssh_client_manager import SSHClientManager
    from run import SSHMultiNodeRunner
except ImportError as e:
    print(f"Error: Could not import modules: {e}")
    print("Make sure all modules are in the same directory.")
    sys.exit(1)


class TestConfigManager(unittest.TestCase):
    """Test cases for configuration management."""
    
    def test_ssh_config_validation(self):
        """Test SSH configuration validation."""
        # Valid configuration with password
        config = SSHConfig(user="testuser", password="testpass")
        self.assertEqual(config.user, "testuser")
        self.assertEqual(config.password, "testpass")
        
        # Valid configuration with key file (mock file existence)
        with patch('os.path.exists', return_value=True):
            config = SSHConfig(user="testuser", key_file="/path/to/key")
            self.assertEqual(config.key_file, "/path/to/key")
        
        # Invalid configuration - no auth method
        with self.assertRaises(ValueError):
            SSHConfig(user="testuser")
        
        # Invalid configuration - non-existent key file
        with patch('os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                SSHConfig(user="testuser", key_file="/nonexistent/key")
    
    def test_cluster_config_validation(self):
        """Test cluster configuration validation."""
        # Valid configuration
        config = ClusterConfig(nodes=["node1", "node2"])
        self.assertEqual(config.nodes, ["node1", "node2"])
        self.assertEqual(config.master_addr, "node1")  # Should default to first node
        
        # Valid configuration with master_addr specified
        config = ClusterConfig(nodes=["node1", "node2"], master_addr="node1")
        self.assertEqual(config.master_addr, "node1")
        
        # Invalid configuration - empty nodes
        with self.assertRaises(ValueError):
            ClusterConfig(nodes=[])
        
        # Invalid configuration - whitespace-only nodes
        with self.assertRaises(ValueError):
            ClusterConfig(nodes=["", "  ", "\t"])
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Valid configuration
        config = TrainingConfig(model="test_model")
        self.assertEqual(config.model, "test_model")
        self.assertEqual(config.shared_data_path, "/nfs/data")  # Default
        
        # Invalid configuration - empty model
        with self.assertRaises(ValueError):
            TrainingConfig(model="")
    
    def test_config_from_args(self):
        """Test configuration creation from arguments."""
        mock_args = MagicMock()
        mock_args.model = 'test_model'
        mock_args.nodes = 'node1,node2,node3'
        mock_args.ssh_user = 'testuser'
        mock_args.ssh_password = 'testpass'
        mock_args.ssh_key = None
        
        # Set default values for optional attributes
        for attr, default in [
            ('master_addr', None),
            ('master_port', 4000),
            ('shared_data_path', '/nfs/data'),
            ('nccl_interface', 'ens14np0'),
            ('gloo_interface', 'ens14np0'),
            ('timeout', 3600),
            ('additional_args', ''),
            ('madengine_path', 'madengine'),
            ('working_directory', 'MAD'),
            ('ssh_timeout', 30),
            ('ssh_max_retries', 3)
        ]:
            setattr(mock_args, attr, getattr(mock_args, attr, default))
        
        config = MultiNodeConfig.from_args(mock_args)
        
        self.assertEqual(config.training.model, 'test_model')
        self.assertEqual(config.cluster.nodes, ['node1', 'node2', 'node3'])
        self.assertEqual(config.ssh.user, 'testuser')
        self.assertEqual(config.ssh.password, 'testpass')
    
    def test_config_from_file(self):
        """Test configuration loading from file."""
        config_content = """
[cluster]
nodes = node1,node2,node3
master_addr = node1
master_port = 5000

[ssh]
user = testuser
password = testpass

[training]
model = test_model
shared_data_path = /shared/data
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name
        
        try:
            config = MultiNodeConfig.from_config_file(temp_config_path)
            
            self.assertEqual(config.cluster.nodes, ['node1', 'node2', 'node3'])
            self.assertEqual(config.cluster.master_addr, 'node1')
            self.assertEqual(config.cluster.master_port, 5000)
            self.assertEqual(config.ssh.user, 'testuser')
            self.assertEqual(config.ssh.password, 'testpass')
            self.assertEqual(config.training.model, 'test_model')
            self.assertEqual(config.training.shared_data_path, '/shared/data')
            
        finally:
            os.unlink(temp_config_path)


class TestSSHClientManager(unittest.TestCase):
    """Test cases for SSH client management."""
    
    @patch('paramiko.SSHClient')
    def test_connectivity_test_success(self, mock_ssh_client_class):
        """Test successful connectivity test."""
        # Mock SSH client
        mock_client = MagicMock()
        mock_ssh_client_class.return_value = mock_client
        
        # Mock successful execution
        mock_client.exec_command.return_value = (None, MagicMock(), MagicMock())
        mock_client.exec_command.return_value[1].channel.recv_exit_status.return_value = 0
        mock_client.exec_command.return_value[1].read.return_value = b'connectivity_test'
        mock_client.exec_command.return_value[2].read.return_value = b''
        
        ssh_manager = SSHClientManager(
            hostname="testhost",
            username="testuser",
            password="testpass"
        )
        
        result = ssh_manager.test_connectivity()
        self.assertTrue(result)
    
    @patch('paramiko.SSHClient')
    def test_connectivity_test_failure(self, mock_ssh_client_class):
        """Test failed connectivity test."""
        # Mock SSH client that raises exception
        mock_client = MagicMock()
        mock_ssh_client_class.return_value = mock_client
        mock_client.connect.side_effect = Exception("Connection failed")
        
        ssh_manager = SSHClientManager(
            hostname="testhost",
            username="testuser",
            password="testpass"
        )
        
        result = ssh_manager.test_connectivity()
        self.assertFalse(result)
    
    def test_invalid_authentication(self):
        """Test invalid authentication configuration."""
        with self.assertRaises(ValueError):
            SSHClientManager(
                hostname="testhost",
                username="testuser"
                # No password or key_filename
            )


class TestSSHMultiNodeRunner(unittest.TestCase):
    """Test cases for SSH Multi-Node Runner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MultiNodeConfig(
            ssh=SSHConfig(user="testuser", password="testpass"),
            cluster=ClusterConfig(nodes=["node1", "node2"]),
            training=TrainingConfig(model="test_model")
        )
    
    def test_initialization(self):
        """Test runner initialization."""
        runner = SSHMultiNodeRunner(self.config)
        
        self.assertEqual(len(runner.ssh_managers), 2)
        self.assertIn("node1", runner.ssh_managers)
        self.assertIn("node2", runner.ssh_managers)
    
    def test_command_generation(self):
        """Test madengine command generation."""
        runner = SSHMultiNodeRunner(self.config)
        
        # Test command for node rank 0
        cmd_0 = runner._build_madengine_command(0)
        self.assertIn('madengine run', cmd_0)
        self.assertIn('test_model', cmd_0)
        self.assertIn('"NODE_RANK": "0"', cmd_0)
        self.assertIn('"NNODES": "2"', cmd_0)
        
        # Test command for node rank 1
        cmd_1 = runner._build_madengine_command(1)
        self.assertIn('"NODE_RANK": "1"', cmd_1)
    
    @patch.object(SSHClientManager, 'test_connectivity')
    def test_connectivity_check_success(self, mock_connectivity):
        """Test successful connectivity checking."""
        mock_connectivity.return_value = True
        
        runner = SSHMultiNodeRunner(self.config)
        reachable_nodes = runner._check_node_connectivity()
        
        self.assertEqual(len(reachable_nodes), 2)
        self.assertIn("node1", reachable_nodes)
        self.assertIn("node2", reachable_nodes)
    
    @patch.object(SSHClientManager, 'test_connectivity')
    def test_connectivity_check_partial_failure(self, mock_connectivity):
        """Test connectivity checking with partial failures."""
        # Mock node1 success, node2 failure
        mock_connectivity.side_effect = [True, False]
        
        runner = SSHMultiNodeRunner(self.config)
        reachable_nodes = runner._check_node_connectivity()
        
        self.assertEqual(len(reachable_nodes), 1)
        self.assertIn("node1", reachable_nodes)
        self.assertNotIn("node2", reachable_nodes)
    
    @patch.object(SSHClientManager, 'execute_command')
    def test_prerequisites_validation_success(self, mock_execute):
        """Test successful prerequisites validation."""
        # Mock successful responses for all checks
        mock_execute.side_effect = [
            (0, "exists", ""),      # Working directory exists
            (0, "found", ""),       # madengine found
            (0, "/home/user/MAD", "")  # Directory accessible
        ]
        
        runner = SSHMultiNodeRunner(self.config)
        success, error_msg = runner._validate_remote_prerequisites("node1")
        
        self.assertTrue(success)
        self.assertEqual(error_msg, "")
    
    @patch.object(SSHClientManager, 'execute_command')
    def test_prerequisites_validation_missing_directory(self, mock_execute):
        """Test prerequisites validation with missing directory."""
        mock_execute.return_value = (0, "missing", "")
        
        runner = SSHMultiNodeRunner(self.config)
        success, error_msg = runner._validate_remote_prerequisites("node1")
        
        self.assertFalse(success)
        self.assertIn("MAD folder not found", error_msg)


class TestIntegration(unittest.TestCase):
    """Integration tests with mocked components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MultiNodeConfig(
            ssh=SSHConfig(user="testuser", password="testpass"),
            cluster=ClusterConfig(nodes=["node1", "node2"]),
            training=TrainingConfig(model="test_model")
        )
    
    @patch.object(SSHMultiNodeRunner, '_execute_on_node')
    @patch.object(SSHMultiNodeRunner, '_check_all_prerequisites')
    @patch.object(SSHMultiNodeRunner, '_check_node_connectivity')
    def test_successful_run(self, mock_connectivity, mock_prerequisites, mock_execute):
        """Test successful end-to-end run."""
        # Mock all checks and execution as successful
        mock_connectivity.return_value = ["node1", "node2"]
        mock_prerequisites.return_value = True
        mock_execute.side_effect = [
            ("node1", True, "Training completed"),
            ("node2", True, "Training completed")
        ]
        
        runner = SSHMultiNodeRunner(self.config)
        result = runner.run()
        
        self.assertTrue(result)
        self.assertEqual(mock_execute.call_count, 2)
    
    @patch.object(SSHMultiNodeRunner, '_execute_on_node')
    @patch.object(SSHMultiNodeRunner, '_check_all_prerequisites')
    @patch.object(SSHMultiNodeRunner, '_check_node_connectivity')
    def test_failed_connectivity(self, mock_connectivity, mock_prerequisites, mock_execute):
        """Test run with connectivity failure."""
        # Mock partial connectivity failure
        mock_connectivity.return_value = ["node1"]  # node2 unreachable
        
        runner = SSHMultiNodeRunner(self.config)
        result = runner.run()
        
        self.assertFalse(result)
        mock_prerequisites.assert_not_called()
        mock_execute.assert_not_called()
    
    @patch.object(SSHMultiNodeRunner, '_execute_on_node')
    @patch.object(SSHMultiNodeRunner, '_check_all_prerequisites')
    @patch.object(SSHMultiNodeRunner, '_check_node_connectivity')
    def test_failed_prerequisites(self, mock_connectivity, mock_prerequisites, mock_execute):
        """Test run with prerequisites failure."""
        mock_connectivity.return_value = ["node1", "node2"]
        mock_prerequisites.return_value = False
        
        runner = SSHMultiNodeRunner(self.config)
        result = runner.run()
        
        self.assertFalse(result)
        mock_execute.assert_not_called()


def run_tests():
    """Run all tests."""
    print("🧪 Running SSH Multi-Node Runner Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSSHClientManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSSHMultiNodeRunner))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


def validate_environment():
    """Validate that the environment is set up correctly."""
    print("🔍 Validating environment...")
    
    issues = []
    
    # Check if paramiko is available
    try:
        import paramiko
        print("✅ paramiko is available")
    except ImportError:
        issues.append("❌ paramiko is not installed. Run: pip install paramiko")
    
    # Check if required files exist
    required_files = ['run.py', 'config_manager.py', 'ssh_client_manager.py', 'requirements.txt']
    current_dir = os.path.dirname(__file__)
    
    for filename in required_files:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            print(f"✅ {filename} found")
        else:
            issues.append(f"❌ {filename} not found in current directory")
    
    if issues:
        print("\n🚨 Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ Environment validation passed!")
        return True


if __name__ == "__main__":
    print("SSH Multi-Node Runner Test Suite")
    print("=" * 40)
    
    # Validate environment first
    if not validate_environment():
        print("\n💥 Environment validation failed. Please fix the issues above.")
        sys.exit(1)
    
    # Run tests
    if run_tests():
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

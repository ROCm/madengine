#!/usr/bin/env python3
"""
Tests for the distributed runner base classes and factory.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest

from madengine.runners.base import (
    NodeConfig,
    WorkloadSpec,
    ExecutionResult,
    DistributedResult,
    BaseDistributedRunner,
)
from madengine.runners.factory import RunnerFactory


class TestNodeConfig:
    """Test NodeConfig dataclass."""

    def test_valid_node_config(self):
        """Test valid node configuration."""
        node = NodeConfig(
            hostname="test-node",
            address="192.168.1.100",
            port=22,
            username="root",
            gpu_count=4,
            gpu_vendor="AMD",
        )

        assert node.hostname == "test-node"
        assert node.address == "192.168.1.100"
        assert node.port == 22
        assert node.username == "root"
        assert node.gpu_count == 4
        assert node.gpu_vendor == "AMD"

    def test_invalid_gpu_vendor(self):
        """Test invalid GPU vendor raises ValueError."""
        with pytest.raises(ValueError, match="Invalid gpu_vendor"):
            NodeConfig(
                hostname="test-node", address="192.168.1.100", gpu_vendor="INVALID"
            )

    def test_missing_required_fields(self):
        """Test missing required fields raises ValueError."""
        with pytest.raises(ValueError, match="hostname and address are required"):
            NodeConfig(hostname="", address="192.168.1.100")


class TestWorkloadSpec:
    """Test WorkloadSpec dataclass."""

    def test_valid_workload_spec(self):
        """Test valid workload specification."""
        # Create temporary manifest file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"built_images": {}}, f)
            manifest_file = f.name

        try:
            workload = WorkloadSpec(
                model_tags=["dummy"],
                manifest_file=manifest_file,
                timeout=3600,
                registry="localhost:5000",
            )

            assert workload.model_tags == ["dummy"]
            assert workload.manifest_file == manifest_file
            assert workload.timeout == 3600
            assert workload.registry == "localhost:5000"
        finally:
            os.unlink(manifest_file)

    def test_empty_model_tags(self):
        """Test empty model tags raises ValueError."""
        with pytest.raises(ValueError, match="model_tags cannot be empty"):
            WorkloadSpec(model_tags=[], manifest_file="nonexistent.json")

    def test_missing_manifest_file(self):
        """Test missing manifest file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            WorkloadSpec(model_tags=["dummy"], manifest_file="nonexistent.json")


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_execution_result_to_dict(self):
        """Test ExecutionResult to_dict method."""
        result = ExecutionResult(
            node_id="test-node",
            model_tag="dummy",
            status="SUCCESS",
            duration=123.45,
            performance_metrics={"fps": 30.5},
            error_message=None,
        )

        result_dict = result.to_dict()

        assert result_dict["node_id"] == "test-node"
        assert result_dict["model_tag"] == "dummy"
        assert result_dict["status"] == "SUCCESS"
        assert result_dict["duration"] == 123.45
        assert result_dict["performance_metrics"] == {"fps": 30.5}
        assert result_dict["error_message"] is None


class TestDistributedResult:
    """Test DistributedResult dataclass."""

    def test_add_successful_result(self):
        """Test adding successful result."""
        dist_result = DistributedResult(
            total_nodes=2,
            successful_executions=0,
            failed_executions=0,
            total_duration=0.0,
        )

        result = ExecutionResult(
            node_id="test-node", model_tag="dummy", status="SUCCESS", duration=100.0
        )

        dist_result.add_result(result)

        assert dist_result.successful_executions == 1
        assert dist_result.failed_executions == 0
        assert len(dist_result.node_results) == 1

    def test_add_failed_result(self):
        """Test adding failed result."""
        dist_result = DistributedResult(
            total_nodes=2,
            successful_executions=0,
            failed_executions=0,
            total_duration=0.0,
        )

        result = ExecutionResult(
            node_id="test-node",
            model_tag="dummy",
            status="FAILURE",
            duration=100.0,
            error_message="Test error",
        )

        dist_result.add_result(result)

        assert dist_result.successful_executions == 0
        assert dist_result.failed_executions == 1
        assert len(dist_result.node_results) == 1


class MockDistributedRunner(BaseDistributedRunner):
    """Mock implementation of BaseDistributedRunner for testing."""

    def setup_infrastructure(self, workload):
        return True

    def execute_workload(self, workload):
        result = DistributedResult(
            total_nodes=len(self.nodes),
            successful_executions=0,
            failed_executions=0,
            total_duration=0.0,
        )

        for node in self.nodes:
            for model_tag in workload.model_tags:
                result.add_result(
                    ExecutionResult(
                        node_id=node.hostname,
                        model_tag=model_tag,
                        status="SUCCESS",
                        duration=100.0,
                    )
                )

        return result

    def cleanup_infrastructure(self, workload):
        return True


class TestBaseDistributedRunner:
    """Test BaseDistributedRunner abstract base class."""

    def test_load_json_inventory(self):
        """Test loading JSON inventory file."""
        inventory_data = {
            "nodes": [
                {"hostname": "node1", "address": "192.168.1.101", "gpu_vendor": "AMD"},
                {
                    "hostname": "node2",
                    "address": "192.168.1.102",
                    "gpu_vendor": "NVIDIA",
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(inventory_data, f)
            inventory_file = f.name

        try:
            runner = MockDistributedRunner(inventory_file)

            assert len(runner.nodes) == 2
            assert runner.nodes[0].hostname == "node1"
            assert runner.nodes[0].gpu_vendor == "AMD"
            assert runner.nodes[1].hostname == "node2"
            assert runner.nodes[1].gpu_vendor == "NVIDIA"
        finally:
            os.unlink(inventory_file)

    def test_load_yaml_inventory(self):
        """Test loading YAML inventory file."""
        inventory_content = """
        gpu_nodes:
          - hostname: node1
            address: 192.168.1.101
            gpu_vendor: AMD
          - hostname: node2
            address: 192.168.1.102
            gpu_vendor: NVIDIA
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(inventory_content)
            inventory_file = f.name

        try:
            runner = MockDistributedRunner(inventory_file)

            assert len(runner.nodes) == 2
            assert runner.nodes[0].hostname == "node1"
            assert runner.nodes[0].gpu_vendor == "AMD"
            assert runner.nodes[1].hostname == "node2"
            assert runner.nodes[1].gpu_vendor == "NVIDIA"
        finally:
            os.unlink(inventory_file)

    def test_filter_nodes(self):
        """Test node filtering functionality."""
        inventory_data = {
            "nodes": [
                {
                    "hostname": "amd-node",
                    "address": "192.168.1.101",
                    "gpu_vendor": "AMD",
                    "labels": {"datacenter": "dc1"},
                },
                {
                    "hostname": "nvidia-node",
                    "address": "192.168.1.102",
                    "gpu_vendor": "NVIDIA",
                    "labels": {"datacenter": "dc2"},
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(inventory_data, f)
            inventory_file = f.name

        try:
            runner = MockDistributedRunner(inventory_file)

            # Test GPU vendor filtering
            amd_nodes = runner.filter_nodes({"gpu_vendor": "AMD"})
            assert len(amd_nodes) == 1
            assert amd_nodes[0].hostname == "amd-node"

            # Test label filtering
            dc1_nodes = runner.filter_nodes({"datacenter": "dc1"})
            assert len(dc1_nodes) == 1
            assert dc1_nodes[0].hostname == "amd-node"
        finally:
            os.unlink(inventory_file)

    def test_validate_workload(self):
        """Test workload validation."""
        inventory_data = {
            "nodes": [
                {"hostname": "node1", "address": "192.168.1.101", "gpu_vendor": "AMD"}
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(inventory_data, f)
            inventory_file = f.name

        # Create manifest file
        manifest_data = {"built_images": {"dummy": {}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest_data, f)
            manifest_file = f.name

        try:
            runner = MockDistributedRunner(inventory_file)

            workload = WorkloadSpec(model_tags=["dummy"], manifest_file=manifest_file)

            assert runner.validate_workload(workload) == True
        finally:
            os.unlink(inventory_file)
            os.unlink(manifest_file)

    def test_run_workflow(self):
        """Test complete run workflow."""
        inventory_data = {
            "nodes": [
                {"hostname": "node1", "address": "192.168.1.101", "gpu_vendor": "AMD"}
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(inventory_data, f)
            inventory_file = f.name

        # Create manifest file
        manifest_data = {"built_images": {"dummy": {}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest_data, f)
            manifest_file = f.name

        try:
            runner = MockDistributedRunner(inventory_file)

            workload = WorkloadSpec(model_tags=["dummy"], manifest_file=manifest_file)

            result = runner.run(workload)

            assert result.total_nodes == 1
            assert result.successful_executions == 1
            assert result.failed_executions == 0
            assert len(result.node_results) == 1
            assert result.node_results[0].status == "SUCCESS"
        finally:
            os.unlink(inventory_file)
            os.unlink(manifest_file)


class TestRunnerFactory:
    """Test RunnerFactory class."""

    def test_register_and_create_runner(self):
        """Test registering and creating a runner."""
        # Register mock runner
        RunnerFactory.register_runner("mock", MockDistributedRunner)

        # Create temporary inventory
        inventory_data = {
            "nodes": [
                {"hostname": "node1", "address": "192.168.1.101", "gpu_vendor": "AMD"}
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(inventory_data, f)
            inventory_file = f.name

        try:
            # Create runner instance
            runner = RunnerFactory.create_runner("mock", inventory_path=inventory_file)

            assert isinstance(runner, MockDistributedRunner)
            assert len(runner.nodes) == 1
            assert runner.nodes[0].hostname == "node1"
        finally:
            os.unlink(inventory_file)

    def test_unknown_runner_type(self):
        """Test creating unknown runner type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown runner type"):
            RunnerFactory.create_runner("unknown", inventory_path="test.json")

    def test_get_available_runners(self):
        """Test getting available runner types."""
        available_runners = RunnerFactory.get_available_runners()

        # Should include default runners if dependencies are available
        assert isinstance(available_runners, list)
        assert len(available_runners) > 0

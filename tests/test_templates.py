"""Tests for the template generator module.

This module tests the Jinja2-based template generation functionality
for Ansible playbooks and Kubernetes manifests.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, mock_open, MagicMock
import pytest

from madengine.runners.template_generator import (
    TemplateGenerator,
    create_ansible_playbook,
    create_kubernetes_manifests,
)


class TestTemplateGenerator(unittest.TestCase):
    """Test the template generator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = os.path.join(self.temp_dir, "templates")
        self.values_dir = os.path.join(self.temp_dir, "values")

        # Create template directories
        os.makedirs(os.path.join(self.template_dir, "ansible"))
        os.makedirs(os.path.join(self.template_dir, "k8s"))
        os.makedirs(self.values_dir)

        # Create sample templates
        self.create_sample_templates()
        self.create_sample_values()

        # Create sample manifest
        self.manifest_data = {
            "built_images": {
                "dummy_model": {
                    "docker_image": "dummy:latest",
                    "registry_image": "registry.example.com/dummy:latest",
                    "build_time": 120.5,
                }
            },
            "built_models": {
                "dummy_model": {
                    "name": "dummy",
                    "dockerfile": "docker/dummy.Dockerfile",
                    "scripts": "scripts/dummy/run.sh",
                }
            },
            "context": {
                "gpu_vendor": "nvidia",
                "docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx908"},
                "docker_env_vars": {"CUDA_VISIBLE_DEVICES": "0"},
                "docker_mounts": {"/tmp": "/tmp"},
                "docker_gpus": "all",
            },
            "registry": "registry.example.com",
            "build_timestamp": "2023-01-01T00:00:00Z",
        }

        self.manifest_file = os.path.join(self.temp_dir, "build_manifest.json")
        with open(self.manifest_file, "w") as f:
            json.dump(self.manifest_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def create_sample_templates(self):
        """Create sample template files."""
        # Ansible playbook template
        ansible_template = """---
- name: MADEngine Test Playbook
  hosts: {{ ansible.target_hosts | default('test_nodes') }}
  vars:
    registry: "{{ registry | default('') }}"
    gpu_vendor: "{{ gpu_vendor | default('') }}"
  tasks:
    - name: Test task
      debug:
        msg: "Environment: {{ environment | default('test') }}"
"""

        with open(
            os.path.join(self.template_dir, "ansible", "playbook.yml.j2"), "w"
        ) as f:
            f.write(ansible_template)

        # K8s namespace template
        k8s_namespace = """apiVersion: v1
kind: Namespace
metadata:
  name: {{ k8s.namespace | default('madengine-test') }}
  labels:
    environment: {{ environment | default('test') }}
"""

        with open(
            os.path.join(self.template_dir, "k8s", "namespace.yaml.j2"), "w"
        ) as f:
            f.write(k8s_namespace)

    def create_sample_values(self):
        """Create sample values files."""
        default_values = {
            "environment": "test",
            "ansible": {"target_hosts": "test_nodes", "become": False},
            "k8s": {"namespace": "madengine-test"},
            "execution": {"timeout": 1800, "keep_alive": False},
        }

        with open(os.path.join(self.values_dir, "default.yaml"), "w") as f:
            import yaml

            yaml.dump(default_values, f)

        dev_values = {
            "environment": "dev",
            "ansible": {"target_hosts": "dev_nodes", "become": True},
            "k8s": {"namespace": "madengine-dev"},
            "execution": {"timeout": 3600, "keep_alive": True},
        }

        with open(os.path.join(self.values_dir, "dev.yaml"), "w") as f:
            yaml.dump(dev_values, f)

    def test_template_generator_initialization(self):
        """Test template generator initialization."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)

        assert str(generator.template_dir) == self.template_dir
        assert str(generator.values_dir) == self.values_dir
        assert generator.env is not None

    def test_load_values_default(self):
        """Test loading default values."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)
        values = generator.load_values("default")

        assert values["environment"] == "test"
        assert values["ansible"]["target_hosts"] == "test_nodes"
        assert values["k8s"]["namespace"] == "madengine-test"

    def test_load_values_dev(self):
        """Test loading dev values."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)
        values = generator.load_values("dev")

        assert values["environment"] == "dev"
        assert values["ansible"]["target_hosts"] == "dev_nodes"
        assert values["k8s"]["namespace"] == "madengine-dev"

    def test_load_values_nonexistent(self):
        """Test loading non-existent values file."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)

        with pytest.raises(FileNotFoundError):
            generator.load_values("nonexistent")

    def test_merge_values(self):
        """Test merging values with manifest data."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)
        base_values = generator.load_values("default")

        merged = generator.merge_values(base_values, self.manifest_data)

        assert merged["environment"] == "test"
        assert merged["registry"] == "registry.example.com"
        assert merged["gpu_vendor"] == "nvidia"
        assert merged["images"]["dummy_model"]["docker_image"] == "dummy:latest"
        assert "generation" in merged
        assert "timestamp" in merged["generation"]

    def test_generate_ansible_playbook(self):
        """Test generating Ansible playbook."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)

        output_file = os.path.join(self.temp_dir, "test_playbook.yml")
        content = generator.generate_ansible_playbook(
            self.manifest_file, "default", output_file
        )

        assert os.path.exists(output_file)
        assert "MADEngine Test Playbook" in content
        assert "test_nodes" in content
        assert "registry.example.com" in content
        assert "nvidia" in content

    def test_generate_kubernetes_manifests(self):
        """Test generating Kubernetes manifests."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)

        output_dir = os.path.join(self.temp_dir, "k8s_output")
        generated_files = generator.generate_kubernetes_manifests(
            self.manifest_file, "default", output_dir
        )

        assert os.path.exists(output_dir)
        assert len(generated_files) > 0

        # Check namespace file
        namespace_file = os.path.join(output_dir, "namespace.yaml")
        if os.path.exists(namespace_file):
            with open(namespace_file, "r") as f:
                content = f.read()
                assert "madengine-test" in content
                assert "environment: test" in content

    def test_list_templates(self):
        """Test listing available templates."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)
        templates = generator.list_templates()

        assert "ansible" in templates
        assert "k8s" in templates
        assert "playbook.yml.j2" in templates["ansible"]
        assert "namespace.yaml.j2" in templates["k8s"]

    def test_validate_template_valid(self):
        """Test validating a valid template."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)

        # Create a simple valid template
        template_content = "Hello {{ name | default('World') }}!"
        template_file = os.path.join(self.template_dir, "test_template.j2")
        with open(template_file, "w") as f:
            f.write(template_content)

        is_valid = generator.validate_template("test_template.j2")
        assert is_valid is True

    def test_validate_template_invalid(self):
        """Test validating an invalid template."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)

        # Create an invalid template
        template_content = "Hello {{ name | invalid_filter }}!"
        template_file = os.path.join(self.template_dir, "invalid_template.j2")
        with open(template_file, "w") as f:
            f.write(template_content)

        is_valid = generator.validate_template("invalid_template.j2")
        assert is_valid is False

    def test_custom_filters(self):
        """Test custom Jinja2 filters."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)

        # Test to_yaml filter
        template = generator.env.from_string("{{ data | to_yaml }}")
        result = template.render(data={"key": "value"})
        assert "key: value" in result

        # Test to_json filter (check for JSON structure, allowing for HTML escaping)
        template = generator.env.from_string("{{ data | to_json }}")
        result = template.render(data={"key": "value"})
        assert "key" in result and "value" in result

        # Test basename filter
        template = generator.env.from_string("{{ path | basename }}")
        result = template.render(path="/path/to/file.txt")
        assert result == "file.txt"

    def test_generate_with_dev_environment(self):
        """Test generation with dev environment."""
        generator = TemplateGenerator(self.template_dir, self.values_dir)

        output_file = os.path.join(self.temp_dir, "dev_playbook.yml")
        content = generator.generate_ansible_playbook(
            self.manifest_file, "dev", output_file
        )

        assert "dev_nodes" in content
        assert "registry.example.com" in content


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manifest_file = os.path.join(self.temp_dir, "build_manifest.json")

        # Create sample manifest
        manifest_data = {
            "built_images": {"dummy": {"docker_image": "dummy:latest"}},
            "context": {"gpu_vendor": "nvidia"},
            "registry": "localhost:5000",
        }

        with open(self.manifest_file, "w") as f:
            json.dump(manifest_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("madengine.runners.template_generator.TemplateGenerator")
    def test_create_ansible_playbook_backward_compatibility(self, mock_generator_class):
        """Test backward compatibility for create_ansible_playbook."""
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

        try:
            create_ansible_playbook(
                manifest_file=self.manifest_file,
                environment="test",
                playbook_file="test.yml",
            )

            mock_generator_class.assert_called_once()
            mock_generator.generate_ansible_playbook.assert_called_once_with(
                self.manifest_file, "test", "test.yml"
            )
        finally:
            os.chdir(original_cwd)

    @patch("madengine.runners.template_generator.TemplateGenerator")
    def test_create_kubernetes_manifests_backward_compatibility(
        self, mock_generator_class
    ):
        """Test backward compatibility for create_kubernetes_manifests."""
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

        try:
            create_kubernetes_manifests(
                manifest_file=self.manifest_file,
                environment="test",
                output_dir="test-k8s",
            )

            mock_generator_class.assert_called_once()
            mock_generator.generate_kubernetes_manifests.assert_called_once_with(
                self.manifest_file, "test", "test-k8s"
            )
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

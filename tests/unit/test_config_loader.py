#!/usr/bin/env python3
"""
Unit tests for ConfigLoader.

Tests the configuration loader's ability to:
1. Apply proper defaults for minimal configs
2. Preserve full configs unchanged
3. Handle override behavior correctly
4. Auto-infer deployment types
5. Detect configuration conflicts

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import pytest
from pathlib import Path

from madengine.deployment.config_loader import ConfigLoader


# Helper function to get project root
def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


# Helper function to check if config file exists
def config_exists(relative_path):
    """Check if a config file exists."""
    full_path = get_project_root() / relative_path
    return full_path.exists()


# Helper function to load config file
def load_config_file(relative_path):
    """Load a config file if it exists."""
    full_path = get_project_root() / relative_path
    if not full_path.exists():
        pytest.skip(f"Config file not found: {relative_path}")
    
    with open(full_path) as f:
        return json.load(f)


class TestConfigLoaderBasics:
    """Test basic ConfigLoader functionality."""
    
    def test_minimal_single_gpu(self):
        """Test minimal single GPU config gets proper defaults."""
        user_config = {
            "k8s": {
                "gpu_count": 1
            }
        }
        
        result = ConfigLoader.load_k8s_config(user_config)
        
        # Validate defaults applied
        assert result["k8s"]["gpu_count"] == 1
        assert result["k8s"]["memory"] == "16Gi"
        assert result["k8s"]["cpu"] == "8"
        assert result["k8s"]["namespace"] == "default"
        assert result["gpu_vendor"] == "AMD"
        assert "OMP_NUM_THREADS" in result["env_vars"]
    
    def test_minimal_multi_gpu(self):
        """Test minimal multi-GPU config gets proper defaults."""
        user_config = {
            "k8s": {
                "gpu_count": 2
            },
            "distributed": {
                "launcher": "torchrun",
                "nnodes": 1,
                "nproc_per_node": 2
            }
        }
        
        result = ConfigLoader.load_k8s_config(user_config)
        
        # Validate multi-GPU defaults
        assert result["k8s"]["gpu_count"] == 2
        assert result["k8s"]["memory"] == "64Gi"
        assert result["k8s"]["cpu"] == "16"
        assert "NCCL_DEBUG" in result["env_vars"]
        assert result["env_vars"]["NCCL_DEBUG"] == "WARN"
        assert "MIOPEN_FIND_MODE" in result["env_vars"]
        assert result["distributed"]["backend"] == "nccl"
    
    def test_minimal_multi_node(self):
        """Test minimal multi-node config gets proper defaults."""
        user_config = {
            "k8s": {
                "gpu_count": 2
            },
            "distributed": {
                "launcher": "torchrun",
                "nnodes": 2,
                "nproc_per_node": 2
            }
        }
        
        result = ConfigLoader.load_k8s_config(user_config)
        
        # Validate multi-node defaults
        assert result["k8s"]["host_ipc"] == True
        assert "NCCL_DEBUG_SUBSYS" in result["env_vars"]
        assert "NCCL_TIMEOUT" in result["env_vars"]
    
    def test_nvidia_config(self):
        """Test NVIDIA GPU config gets proper defaults."""
        user_config = {
            "gpu_vendor": "NVIDIA",
            "k8s": {
                "gpu_count": 4
            },
            "distributed": {
                "launcher": "torchrun",
                "nnodes": 1,
                "nproc_per_node": 4
            }
        }
        
        result = ConfigLoader.load_k8s_config(user_config)
        
        # Validate NVIDIA defaults
        assert result["k8s"]["gpu_resource_name"] == "nvidia.com/gpu"
        assert "NCCL_P2P_DISABLE" in result["env_vars"]
        assert result["env_vars"]["OMP_NUM_THREADS"] == "12"
    
    def test_override_behavior(self):
        """Test that user overrides work correctly."""
        user_config = {
            "k8s": {
                "gpu_count": 1,
                "namespace": "custom-namespace",
                "memory": "32Gi"  # Override default 16Gi
            },
            "env_vars": {
                "CUSTOM_VAR": "custom_value"
            }
        }
        
        result = ConfigLoader.load_k8s_config(user_config)
        
        # Validate overrides
        assert result["k8s"]["namespace"] == "custom-namespace"
        assert result["k8s"]["memory"] == "32Gi"  # Overridden
        assert result["k8s"]["cpu"] == "8"  # Still has default
        assert "CUSTOM_VAR" in result["env_vars"]
        assert "OMP_NUM_THREADS" in result["env_vars"]  # Default still there


class TestConfigLoaderK8sConfigs:
    """Test with actual K8s config files (if they exist)."""
    
    @pytest.mark.skipif(
        not config_exists("examples/k8s-configs/basic/01-native-single-node-single-gpu.json"),
        reason="K8s config file not found"
    )
    def test_k8s_single_gpu_config(self):
        """Test K8s single GPU config file."""
        user_config = load_config_file("examples/k8s-configs/basic/01-native-single-node-single-gpu.json")
        result = ConfigLoader.load_k8s_config(user_config)
        
        # Validate key fields are preserved
        assert result["k8s"]["gpu_count"] == 1
        assert "memory" in result["k8s"]
        assert "namespace" in result["k8s"]
        assert result["gpu_vendor"] in ["AMD", "NVIDIA"]
    
    @pytest.mark.skipif(
        not config_exists("examples/k8s-configs/basic/02-torchrun-single-node-multi-gpu.json"),
        reason="K8s multi-GPU config file not found"
    )
    def test_k8s_multi_gpu_config(self):
        """Test K8s multi-GPU config file."""
        user_config = load_config_file("examples/k8s-configs/basic/02-torchrun-single-node-multi-gpu.json")
        result = ConfigLoader.load_k8s_config(user_config)
        
        # Validate multi-GPU config
        assert result["k8s"]["gpu_count"] >= 2
        assert "distributed" in result
        assert result["distributed"]["nnodes"] == 1
        assert result["distributed"]["nproc_per_node"] >= 2


class TestConfigLoaderSlurmConfigs:
    """Test with actual SLURM config files (if they exist)."""
    
    @pytest.mark.skipif(
        not config_exists("examples/slurm-configs/basic/01-single-node-single-gpu.json"),
        reason="SLURM config file not found"
    )
    def test_slurm_single_gpu_config(self):
        """Test SLURM single GPU config file."""
        user_config = load_config_file("examples/slurm-configs/basic/01-single-node-single-gpu.json")
        result = ConfigLoader.load_slurm_config(user_config)
        
        # Validate SLURM config structure
        assert "slurm" in result
        assert result["slurm"]["nodes"] == 1
        assert result["slurm"]["gpus_per_node"] >= 1
    
    @pytest.mark.skipif(
        not config_exists("examples/slurm-configs/basic/06-vllm-multi-node.json"),
        reason="SLURM vLLM multi-node config file not found"
    )
    def test_slurm_vllm_multi_node_config(self):
        """Test SLURM vLLM multi-node config file."""
        user_config = load_config_file("examples/slurm-configs/basic/06-vllm-multi-node.json")
        result = ConfigLoader.load_slurm_config(user_config)
        
        # Validate multi-node vLLM config
        assert "slurm" in result
        assert result["slurm"]["nodes"] >= 2
        assert result["slurm"]["gpus_per_node"] >= 1
        assert "distributed" in result
        
        # Check for new preflight node check parameters
        if "enable_node_check" in result["slurm"]:
            assert isinstance(result["slurm"]["enable_node_check"], bool)
        if "auto_cleanup_nodes" in result["slurm"]:
            assert isinstance(result["slurm"]["auto_cleanup_nodes"], bool)


class TestConfigLoaderDeploymentType:
    """Test deployment type inference and validation."""
    
    def test_auto_infer_k8s(self):
        """Test k8s deployment type is auto-inferred from k8s field presence."""
        user_config = {
            "k8s": {
                "gpu_count": 1
            }
        }
        
        result = ConfigLoader.load_config(user_config)
        
        # Validate k8s config was loaded and defaults applied
        assert "k8s" in result
        assert result["k8s"]["gpu_count"] == 1
        assert "memory" in result["k8s"]  # Default was applied
    
    def test_auto_infer_slurm(self):
        """Test slurm deployment type is auto-inferred from slurm field presence."""
        user_config = {
            "slurm": {
                "nodes": 1,
                "gpus_per_node": 4
            }
        }
        
        result = ConfigLoader.load_config(user_config)
        
        # Validate slurm config was loaded and defaults applied
        assert "slurm" in result
        assert result["slurm"]["nodes"] == 1
        assert result["slurm"]["gpus_per_node"] == 4
    
    def test_auto_infer_local(self):
        """Test local deployment when no k8s/slurm present."""
        user_config = {
            "env_vars": {"MY_VAR": "value"}
        }
        
        result = ConfigLoader.load_config(user_config)
        
        # Validate local config (no k8s or slurm fields)
        assert "k8s" not in result or result.get("k8s") == {}
        assert "slurm" not in result or result.get("slurm") == {}
        assert result["env_vars"]["MY_VAR"] == "value"
    
    def test_conflict_k8s_and_slurm(self):
        """Test error when both k8s and slurm fields present."""
        user_config = {
            "k8s": {"gpu_count": 1},
            "slurm": {"nodes": 2}
        }
        
        with pytest.raises(ValueError, match="Both 'k8s' and 'slurm'"):
            ConfigLoader.load_config(user_config)
    
    def test_conflict_explicit_deploy_mismatch(self):
        """Test error when explicit deploy field conflicts with config presence."""
        user_config = {
            "deploy": "slurm",
            "k8s": {"gpu_count": 1}
        }
        
        with pytest.raises(ValueError, match="Conflicting deployment"):
            ConfigLoader.load_config(user_config)
    
    def test_explicit_deploy_matching(self):
        """Test that explicit deploy field works when it matches config."""
        user_config = {
            "deploy": "k8s",
            "k8s": {"gpu_count": 1}
        }
        
        result = ConfigLoader.load_config(user_config)
        
        # Should work fine since deploy matches k8s presence
        # The deploy field may or may not be preserved in result
        assert result["k8s"]["gpu_count"] == 1
        assert "memory" in result["k8s"]  # Defaults applied


class TestConfigLoaderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_config(self):
        """Test empty config defaults to local deployment."""
        user_config = {}
        
        result = ConfigLoader.load_config(user_config)
        
        # Should default to local (no k8s or slurm fields)
        assert "k8s" not in result or result.get("k8s") == {}
        assert "slurm" not in result or result.get("slurm") == {}
        # Empty config should return as-is
        assert isinstance(result, dict)
    
    def test_deep_merge_preserves_nested(self):
        """Test that deep merge preserves nested structures."""
        user_config = {
            "k8s": {
                "gpu_count": 2,
                "labels": {
                    "app": "myapp",
                    "env": "prod"
                }
            }
        }
        
        result = ConfigLoader.load_k8s_config(user_config)
        
        # Nested structure should be preserved
        assert result["k8s"]["labels"]["app"] == "myapp"
        assert result["k8s"]["labels"]["env"] == "prod"
        # Defaults should still be applied at top level
        assert result["k8s"]["memory"] == "64Gi"


# Run pytest if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


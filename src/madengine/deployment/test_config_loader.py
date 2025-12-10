#!/usr/bin/env python3
"""
Test script to validate ConfigLoader functionality.

Tests:
1. Minimal configs get proper defaults
2. Full configs remain unchanged
3. Override behavior works correctly
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from madengine.deployment.config_loader import ConfigLoader


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_minimal_single_gpu():
    """Test minimal single GPU config."""
    print_section("TEST 1: Minimal Single GPU Config")
    
    user_config = {
        "k8s": {
            "gpu_count": 1
        }
    }
    
    result = ConfigLoader.load_k8s_config(user_config)
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    print("\nOutput (with defaults applied):")
    print(json.dumps(result, indent=2))
    
    # Validate
    assert result["k8s"]["gpu_count"] == 1
    assert result["k8s"]["memory"] == "16Gi"
    assert result["k8s"]["cpu"] == "8"
    assert result["k8s"]["namespace"] == "default"
    assert result["gpu_vendor"] == "AMD"
    assert "OMP_NUM_THREADS" in result["env_vars"]
    
    print("\n✅ Test PASSED: Single GPU defaults applied correctly")
    return True


def test_minimal_multi_gpu():
    """Test minimal multi-GPU config."""
    print_section("TEST 2: Minimal Multi-GPU Config")
    
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
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    print("\nOutput (with defaults applied):")
    print(json.dumps(result, indent=2))
    
    # Validate
    assert result["k8s"]["gpu_count"] == 2
    assert result["k8s"]["memory"] == "64Gi"
    assert result["k8s"]["cpu"] == "16"
    assert "NCCL_DEBUG" in result["env_vars"]
    assert result["env_vars"]["NCCL_DEBUG"] == "WARN"
    assert "MIOPEN_FIND_MODE" in result["env_vars"]
    assert result["distributed"]["backend"] == "nccl"
    
    print("\n✅ Test PASSED: Multi-GPU defaults applied correctly")
    return True


def test_minimal_multi_node():
    """Test minimal multi-node config."""
    print_section("TEST 3: Minimal Multi-Node Config")
    
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
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    print("\nOutput (with defaults applied):")
    print(json.dumps(result, indent=2))
    
    # Validate
    assert result["k8s"]["host_ipc"] == True
    assert "NCCL_DEBUG_SUBSYS" in result["env_vars"]
    assert "NCCL_TIMEOUT" in result["env_vars"]
    
    print("\n✅ Test PASSED: Multi-node defaults applied correctly")
    return True


def test_nvidia_config():
    """Test NVIDIA GPU config."""
    print_section("TEST 4: NVIDIA GPU Config")
    
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
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    print("\nOutput (with defaults applied):")
    print(json.dumps(result, indent=2))
    
    # Validate
    assert result["k8s"]["gpu_resource_name"] == "nvidia.com/gpu"
    assert "NCCL_P2P_DISABLE" in result["env_vars"]
    assert result["env_vars"]["OMP_NUM_THREADS"] == "12"
    
    print("\n✅ Test PASSED: NVIDIA defaults applied correctly")
    return True


def test_full_config_unchanged():
    """Test that full configs remain unchanged."""
    print_section("TEST 5: Full Config Backward Compatibility")
    
    # Load actual full config
    config_path = Path(__file__).parent.parent.parent.parent / "examples/k8s-configs/01-single-node-single-gpu.json"
    with open(config_path) as f:
        user_config = json.load(f)
    
    result = ConfigLoader.load_k8s_config(user_config)
    
    print("Input: 01-single-node-single-gpu.json")
    print(json.dumps(user_config, indent=2))
    print("\nOutput (should be mostly the same):")
    print(json.dumps(result, indent=2))
    
    # Validate key fields are preserved
    assert result["k8s"]["gpu_count"] == 1
    assert result["k8s"]["memory"] == "16Gi"
    assert result["k8s"]["namespace"] == "default"
    assert result["gpu_vendor"] == "AMD"
    
    print("\n✅ Test PASSED: Full config preserved")
    return True


def test_override_behavior():
    """Test that user overrides work correctly."""
    print_section("TEST 6: Override Behavior")
    
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
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    print("\nOutput:")
    print(json.dumps(result, indent=2))
    
    # Validate
    assert result["k8s"]["namespace"] == "custom-namespace"
    assert result["k8s"]["memory"] == "32Gi"  # Overridden
    assert result["k8s"]["cpu"] == "8"  # Still has default
    assert "CUSTOM_VAR" in result["env_vars"]
    assert "OMP_NUM_THREADS" in result["env_vars"]  # Default still there
    
    print("\n✅ Test PASSED: Overrides work correctly")
    return True


def test_full_multi_gpu_config():
    """Test full multi-GPU config backward compatibility."""
    print_section("TEST 7: Full Multi-GPU Config Backward Compatibility")
    
    config_path = Path(__file__).parent.parent.parent.parent / "examples/k8s-configs/02-single-node-multi-gpu.json"
    with open(config_path) as f:
        user_config = json.load(f)
    
    result = ConfigLoader.load_k8s_config(user_config)
    
    print("Input: 02-single-node-multi-gpu.json")
    
    # Validate key fields are preserved
    assert result["k8s"]["gpu_count"] == 2
    assert result["k8s"]["memory"] == "64Gi"
    assert result["distributed"]["nnodes"] == 1
    assert result["distributed"]["nproc_per_node"] == 2
    assert result["env_vars"]["NCCL_DEBUG"] == "WARN"
    
    print("✅ Test PASSED: Full multi-GPU config preserved")
    return True


def test_auto_infer_k8s():
    """Test k8s deployment type is auto-inferred from k8s field presence."""
    print_section("TEST 8: Auto-Infer K8s Deployment")
    
    user_config = {
        "k8s": {
            "gpu_count": 1
        }
    }
    
    result = ConfigLoader.load_config(user_config)
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    print("\nOutput:")
    print(f"  deploy field: {result.get('deploy')}")
    
    # Validate deploy field was inferred
    assert result["deploy"] == "k8s"
    
    print("\n✅ Test PASSED: Deploy type auto-inferred as 'k8s'")
    return True


def test_auto_infer_local():
    """Test local deployment when no k8s/slurm present."""
    print_section("TEST 9: Auto-Infer Local Deployment")
    
    user_config = {
        "env_vars": {"MY_VAR": "value"}
    }
    
    result = ConfigLoader.load_config(user_config)
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    print("\nOutput:")
    print(f"  deploy field: {result.get('deploy')}")
    
    # Validate deploy field was inferred as local
    assert result["deploy"] == "local"
    
    print("\n✅ Test PASSED: Deploy type auto-inferred as 'local'")
    return True


def test_conflict_k8s_and_slurm():
    """Test error when both k8s and slurm fields present."""
    print_section("TEST 10: Conflict - Both K8s and SLURM Present")
    
    user_config = {
        "k8s": {"gpu_count": 1},
        "slurm": {"nodes": 2}
    }
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    
    try:
        result = ConfigLoader.load_config(user_config)
        print("\n❌ Test FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"\nExpected error raised: {e}")
        assert "Both 'k8s' and 'slurm'" in str(e)
        print("\n✅ Test PASSED: Correctly detected conflicting configs")
        return True


def test_conflict_explicit_deploy_mismatch():
    """Test error when explicit deploy field conflicts with config presence."""
    print_section("TEST 11: Conflict - Explicit Deploy Mismatch")
    
    user_config = {
        "deploy": "slurm",
        "k8s": {"gpu_count": 1}
    }
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    
    try:
        result = ConfigLoader.load_config(user_config)
        print("\n❌ Test FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"\nExpected error raised: {e}")
        assert "Conflicting deployment" in str(e)
        print("\n✅ Test PASSED: Correctly detected conflicting deploy field")
        return True


def test_explicit_deploy_matching():
    """Test that explicit deploy field works when it matches config."""
    print_section("TEST 12: Explicit Deploy Field Matching Config")
    
    user_config = {
        "deploy": "k8s",
        "k8s": {"gpu_count": 1}
    }
    
    result = ConfigLoader.load_config(user_config)
    
    print("Input:")
    print(json.dumps(user_config, indent=2))
    print("\nOutput:")
    print(f"  deploy field: {result.get('deploy')}")
    
    # Should work fine since deploy matches k8s presence
    assert result["deploy"] == "k8s"
    assert result["k8s"]["gpu_count"] == 1
    
    print("\n✅ Test PASSED: Explicit deploy field matching config works")
    return True


def main():
    """Run all tests."""
    print("\n" + "🧪" * 40)
    print("ConfigLoader Test Suite")
    print("🧪" * 40)
    
    tests = [
        test_minimal_single_gpu,
        test_minimal_multi_gpu,
        test_minimal_multi_node,
        test_nvidia_config,
        test_full_config_unchanged,
        test_override_behavior,
        test_full_multi_gpu_config,
        test_auto_infer_k8s,
        test_auto_infer_local,
        test_conflict_k8s_and_slurm,
        test_conflict_explicit_deploy_mismatch,
        test_explicit_deploy_matching,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"\n❌ Test FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ Test ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)
    
    if failed == 0:
        print("\n✅ All tests PASSED! ConfigLoader is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())


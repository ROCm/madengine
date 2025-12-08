"""
Pytest configuration and shared fixtures for MADEngine tests.

Provides reusable fixtures for multi-platform testing (AMD GPU, NVIDIA GPU, CPU),
mock contexts, and integration test utilities.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


# ============================================================================
# Platform Configuration Fixtures
# ============================================================================

@pytest.fixture
def amd_gpu_context():
    """Mock Context for AMD GPU platform (ROCm)."""
    context = MagicMock()
    context.ctx = {
        "docker_build_arg": {
            "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a",
            "MAD_GPU_VENDOR": "AMD",
        }
    }
    context.get_gpu_vendor.return_value = "AMD"
    context.get_system_ngpus.return_value = 8
    context.get_system_gpu_architecture.return_value = "gfx90a"
    context.get_system_hip_version.return_value = "6.0"
    context.get_gpu_renderD_nodes.return_value = ["renderD128", "renderD129"]
    context.get_docker_gpus.return_value = "all"
    context.get_system_gpu_product_name.return_value = "AMD Instinct MI300X"
    return context


@pytest.fixture
def nvidia_gpu_context():
    """Mock Context for NVIDIA GPU platform (CUDA)."""
    context = MagicMock()
    context.ctx = {
        "docker_build_arg": {
            "MAD_SYSTEM_GPU_ARCHITECTURE": "sm_90",
            "MAD_GPU_VENDOR": "NVIDIA",
        }
    }
    context.get_gpu_vendor.return_value = "NVIDIA"
    context.get_system_ngpus.return_value = 8
    context.get_system_gpu_architecture.return_value = "sm_90"
    context.get_system_cuda_version.return_value = "12.1"
    context.get_docker_gpus.return_value = "all"
    context.get_system_gpu_product_name.return_value = "NVIDIA H100"
    return context


@pytest.fixture
def cpu_context():
    """Mock Context for CPU-only platform."""
    context = MagicMock()
    context.ctx = {
        "docker_build_arg": {
            "MAD_SYSTEM_GPU_ARCHITECTURE": "",
            "MAD_GPU_VENDOR": "NONE",
        }
    }
    context.get_gpu_vendor.return_value = "NONE"
    context.get_system_ngpus.return_value = 0
    context.get_system_gpu_architecture.return_value = ""
    context.get_docker_gpus.return_value = None
    return context


@pytest.fixture(params=["amd", "nvidia", "cpu"])
def multi_platform_context(request, amd_gpu_context, nvidia_gpu_context, cpu_context):
    """Parametrized fixture that tests across all platforms."""
    contexts = {
        "amd": amd_gpu_context,
        "nvidia": nvidia_gpu_context,
        "cpu": cpu_context,
    }
    return contexts[request.param]


# ============================================================================
# Mock Args Fixtures
# ============================================================================

@pytest.fixture
def mock_build_args():
    """Mock args for build command."""
    args = MagicMock()
    args.tags = []
    args.target_archs = []
    args.registry = None
    args.additional_context = None
    args.additional_context_file = None
    args.clean_docker_cache = False
    args.manifest_output = "build_manifest.json"
    args.live_output = False
    args.output = "perf.csv"
    args.ignore_deprecated_flag = False
    args.data_config_file_name = "data.json"
    args.tools_json_file_name = "tools.json"
    args.generate_sys_env_details = True
    args.force_mirror_local = False
    args.disable_skip_gpu_arch = False
    args.verbose = False
    args._separate_phases = True
    return args


@pytest.fixture
def mock_run_args():
    """Mock args for run command."""
    args = MagicMock()
    args.tags = []
    args.manifest_file = "build_manifest.json"
    args.registry = None
    args.timeout = 3600
    args.keep_alive = False
    args.keep_model_dir = False
    args.additional_context = None
    args.additional_context_file = None
    args.live_output = False
    args.output = "perf.csv"
    args.ignore_deprecated_flag = False
    args.data_config_file_name = "data.json"
    args.tools_json_file_name = "tools.json"
    args.generate_sys_env_details = True
    args.force_mirror_local = False
    args.disable_skip_gpu_arch = False
    args.verbose = False
    args._separate_phases = True
    return args


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_models():
    """Sample model data for testing."""
    return [
        {
            "name": "model1",
            "tags": ["test", "integration"],
            "dockerfile": "docker/model1.Dockerfile",
        },
        {
            "name": "model2",
            "tags": ["test"],
            "dockerfile": "docker/model2.Dockerfile",
        },
    ]


@pytest.fixture
def sample_build_summary_success():
    """Sample successful build summary."""
    return {
        "successful_builds": [
            {
                "model": "model1",
                "docker_image": "ci-model1",
                "dockerfile": "docker/model1.Dockerfile",
                "build_duration": 10.5,
                "gpu_architecture": "gfx90a",
            },
            {
                "model": "model2",
                "docker_image": "ci-model2",
                "dockerfile": "docker/model2.Dockerfile",
                "build_duration": 8.3,
                "gpu_architecture": "gfx90a",
            },
        ],
        "failed_builds": [],
        "total_build_time": 18.8,
    }


@pytest.fixture
def sample_build_summary_partial():
    """Sample partial build summary (mixed success/failure)."""
    return {
        "successful_builds": [
            {
                "model": "model1",
                "docker_image": "ci-model1",
                "dockerfile": "docker/model1.Dockerfile",
                "build_duration": 10.5,
                "gpu_architecture": "gfx90a",
            },
        ],
        "failed_builds": [
            {
                "model": "model2",
                "error": "Build failed: dependency not found",
            },
        ],
        "total_build_time": 10.5,
    }


@pytest.fixture
def sample_build_summary_all_failed():
    """Sample build summary with all failures."""
    return {
        "successful_builds": [],
        "failed_builds": [
            {
                "model": "model1",
                "error": "Build failed: base image not found",
            },
            {
                "model": "model2",
                "error": "Build failed: syntax error in Dockerfile",
            },
        ],
        "total_build_time": 0,
    }


@pytest.fixture
def sample_manifest():
    """Sample build manifest."""
    return {
        "built_images": {
            "ci-model1": {
                "docker_image": "ci-model1",
                "dockerfile": "docker/model1.Dockerfile",
                "gpu_architecture": "gfx90a",
            },
            "ci-model2": {
                "docker_image": "ci-model2",
                "dockerfile": "docker/model2.Dockerfile",
                "gpu_architecture": "gfx90a",
            },
        },
        "built_models": {
            "ci-model1": {
                "name": "model1",
                "tags": ["test"],
            },
            "ci-model2": {
                "name": "model2",
                "tags": ["test"],
            },
        },
        "summary": {
            "successful_builds": [
                {"model": "model1", "docker_image": "ci-model1"},
                {"model": "model2", "docker_image": "ci-model2"},
            ],
            "failed_builds": [],
        },
    }


# ============================================================================
# Temporary File Fixtures
# ============================================================================

@pytest.fixture
def temp_manifest_file(sample_manifest):
    """Create a temporary manifest file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(sample_manifest, f)
        manifest_path = f.name
    
    yield manifest_path
    
    # Cleanup
    if os.path.exists(manifest_path):
        os.unlink(manifest_path)


@pytest.fixture
def temp_working_dir():
    """Create a temporary working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        yield tmpdir
        
        os.chdir(original_cwd)


# ============================================================================
# Mock Builder and Runner Fixtures
# ============================================================================

@pytest.fixture
def mock_docker_builder(sample_build_summary_success):
    """Mock DockerBuilder with successful builds."""
    builder = MagicMock()
    builder.build_all_models.return_value = sample_build_summary_success
    builder.export_build_manifest.return_value = None
    builder.built_images = {
        "ci-model1": {"docker_image": "ci-model1"},
        "ci-model2": {"docker_image": "ci-model2"},
    }
    return builder


@pytest.fixture
def mock_container_runner():
    """Mock ContainerRunner with successful runs."""
    runner = MagicMock()
    runner.run_models_from_manifest.return_value = {
        "successful_runs": [
            {
                "model": "model1",
                "image": "ci-model1",
                "status": "SUCCESS",
                "performance": 1000.0,
                "duration": 30.5,
            },
            {
                "model": "model2",
                "image": "ci-model2",
                "status": "SUCCESS",
                "performance": 1200.0,
                "duration": 28.3,
            },
        ],
        "failed_runs": [],
        "total_runs": 2,
    }
    return runner


# ============================================================================
# Integration Test Helpers
# ============================================================================

@pytest.fixture
def integration_test_env():
    """Setup integration test environment variables."""
    env_vars = {
        "MODEL_DIR": "tests/fixtures/dummy",
        "MAD_SKIP_GPU_CHECK": "1",  # Skip actual GPU detection in tests
    }
    
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as fast unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU hardware"
    )
    config.addinivalue_line(
        "markers", "amd: marks tests specific to AMD GPUs"
    )
    config.addinivalue_line(
        "markers", "nvidia: marks tests specific to NVIDIA GPUs"
    )
    config.addinivalue_line(
        "markers", "cpu: marks tests for CPU-only execution"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# ============================================================================
# Utility Functions for Tests
# ============================================================================

def assert_build_manifest_valid(manifest_path):
    """Assert that a build manifest file is valid."""
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Check required keys
    assert "built_images" in manifest
    assert "built_models" in manifest
    assert "summary" in manifest
    
    # Check summary structure
    summary = manifest["summary"]
    assert "successful_builds" in summary
    assert "failed_builds" in summary
    assert isinstance(summary["successful_builds"], list)
    assert isinstance(summary["failed_builds"], list)
    
    return manifest


def assert_perf_csv_valid(csv_path):
    """Assert that a performance CSV file is valid."""
    assert os.path.exists(csv_path), f"Performance CSV not found: {csv_path}"
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_columns = ["model", "n_gpus", "gpu_architecture", "status"]
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    return df


# Export utility functions for use in tests
__all__ = [
    "assert_build_manifest_valid",
    "assert_perf_csv_valid",
]


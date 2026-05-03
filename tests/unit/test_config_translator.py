#!/usr/bin/env python3
"""Tests for ConfigTranslator."""

import pytest
from omegaconf import OmegaConf

from madengine.config.translator import ConfigTranslator


def make_cfg(overrides: dict) -> "DictConfig":
    """Build a DictConfig from a base + overrides for testing."""
    base = {
        "model": {"tags": [], "manifest_file": None, "container_image": None, "skip_run": False, "timeout": None},
        "docker": {"build_args": {}, "env_vars": {}, "mounts": {}, "gpus": None, "cpus": None, "additional_run_options": None, "keep_alive": False, "clean_cache": False},
        "build": {"registry": None, "target_archs": [], "manifest_output": "build_manifest.json"},
        "env_vars": {},
        "debug": False,
        "live_output": False,
        "log_error": {"pattern_scan": True, "benign_patterns": [], "patterns": []},
        "tools": [],
        "pre_scripts": [],
        "post_scripts": [],
        "encapsulate_script": None,
        "data_config": "data.json",
        "output": "perf.csv",
        "summary_output": None,
        "gpu_vendor": "AMD",
        "guest_os": "UBUNTU",
        "runtime": {"devices": [], "capabilities": [], "security_opts": [], "network_mode": "host", "ipc": "host", "groups": [], "use_gpu_flag": False},
        "platform": {"type": "docker"},
    }
    merged = {**base, **overrides}
    return OmegaConf.create(merged)


class TestDockerKeyMapping:
    def test_build_args_mapped(self):
        cfg = make_cfg({"docker": {"build_args": {"KEY": "val"}, "env_vars": {}, "mounts": {}, "gpus": None, "cpus": None, "additional_run_options": None, "keep_alive": False, "clean_cache": False}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["docker_build_arg"] == {"KEY": "val"}

    def test_env_vars_mapped(self):
        cfg = make_cfg({"docker": {"build_args": {}, "env_vars": {"A": "1"}, "mounts": {}, "gpus": None, "cpus": None, "additional_run_options": None, "keep_alive": False, "clean_cache": False}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["docker_env_vars"] == {"A": "1"}

    def test_null_gpus_excluded(self):
        cfg = make_cfg({})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "docker_gpus" not in ctx

    def test_non_null_gpus_included(self):
        cfg = make_cfg({"docker": {"build_args": {}, "env_vars": {}, "mounts": {}, "gpus": "0-3", "cpus": None, "additional_run_options": None, "keep_alive": False, "clean_cache": False}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["docker_gpus"] == "0-3"


class TestLogErrorMapping:
    def test_pattern_scan_mapped(self):
        cfg = make_cfg({"log_error": {"pattern_scan": False, "benign_patterns": [], "patterns": []}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["log_error_pattern_scan"] is False

    def test_patterns_mapped(self):
        cfg = make_cfg({"log_error": {"pattern_scan": True, "benign_patterns": ["OK"], "patterns": ["ERR"]}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["log_error_benign_patterns"] == ["OK"]
        assert ctx["log_error_patterns"] == ["ERR"]


class TestPassthroughKeys:
    def test_gpu_vendor_passthrough(self):
        cfg = make_cfg({"gpu_vendor": "NVIDIA"})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["gpu_vendor"] == "NVIDIA"

    def test_env_vars_passthrough(self):
        cfg = make_cfg({"env_vars": {"MY": "VAR"}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["env_vars"] == {"MY": "VAR"}

    def test_slurm_passthrough(self):
        cfg = make_cfg({"slurm": {"partition": "gpu"}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["slurm"] == {"partition": "gpu"}

    def test_distributed_passthrough(self):
        cfg = make_cfg({"distributed": {"enabled": True, "launcher": "torchrun"}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["distributed"]["launcher"] == "torchrun"

    def test_tools_passthrough(self):
        cfg = make_cfg({"tools": [{"name": "rpd"}]})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["tools"] == [{"name": "rpd"}]


class TestExtractedKeys:
    def test_model_extracted(self):
        cfg = make_cfg({"model": {"tags": ["dummy"], "manifest_file": None, "container_image": None, "skip_run": False, "timeout": 300}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "model" not in ctx
        assert meta["model"]["tags"] == ["dummy"]
        assert meta["model"]["timeout"] == 300

    def test_build_extracted(self):
        cfg = make_cfg({"build": {"registry": "myregistry.io", "target_archs": ["gfx942"], "manifest_output": "build_manifest.json"}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "build" not in ctx
        assert meta["build"]["registry"] == "myregistry.io"

    def test_platform_extracted(self):
        cfg = make_cfg({})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "platform" not in ctx
        assert meta["platform"]["type"] == "docker"

    def test_container_image_promoted(self):
        cfg = make_cfg({"model": {"tags": [], "manifest_file": None, "container_image": "myimage:latest", "skip_run": False, "timeout": None}})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert ctx["MAD_CONTAINER_IMAGE"] == "myimage:latest"

    def test_runtime_extracted(self):
        cfg = make_cfg({})
        ctx, meta = ConfigTranslator.to_additional_context(cfg)
        assert "runtime" not in ctx
        assert "runtime" in meta

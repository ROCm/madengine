#!/usr/bin/env python3
"""Tests for HydraConfigLoader."""

import os
import tempfile

import pytest
from omegaconf import DictConfig

from madengine.config.loader import HydraConfigLoader
from madengine.core.errors import ConfigurationError


class TestParseArgs:
    def test_hydra_overrides_only(self):
        user_file, overrides = HydraConfigLoader._parse_args(
            ["scheduler=slurm", "distributed.nnodes=4"]
        )
        assert user_file is None
        assert overrides == ["scheduler=slurm", "distributed.nnodes=4"]

    def test_yaml_file_only(self):
        user_file, overrides = HydraConfigLoader._parse_args(["/path/to/config.yaml"])
        assert user_file == "/path/to/config.yaml"
        assert overrides == []

    def test_yaml_file_with_overrides(self):
        user_file, overrides = HydraConfigLoader._parse_args(
            ["/path/to/config.yaml", "distributed.nnodes=8"]
        )
        assert user_file == "/path/to/config.yaml"
        assert overrides == ["distributed.nnodes=8"]

    def test_yml_extension_recognized(self):
        user_file, overrides = HydraConfigLoader._parse_args(["/path/to/config.yml"])
        assert user_file == "/path/to/config.yml"

    def test_multiple_yaml_files_raises(self):
        with pytest.raises(ConfigurationError, match="Only one YAML"):
            HydraConfigLoader._parse_args(["/path/a.yaml", "/path/b.yaml"])

    def test_append_override_not_treated_as_file(self):
        user_file, overrides = HydraConfigLoader._parse_args(["+profile=mi300x_8gpu"])
        assert user_file is None
        assert overrides == ["+profile=mi300x_8gpu"]

    def test_empty_args(self):
        user_file, overrides = HydraConfigLoader._parse_args([])
        assert user_file is None
        assert overrides == []


class TestLoad:
    def test_defaults_only(self):
        cfg = HydraConfigLoader.load([])
        assert isinstance(cfg, DictConfig)
        assert cfg.gpu_vendor == "AMD"
        assert cfg.guest_os == "UBUNTU"
        assert cfg.distributed.enabled is False

    def test_scheduler_override(self):
        cfg = HydraConfigLoader.load(["scheduler=slurm"])
        assert "slurm" in cfg
        assert cfg.slurm.partition == "amd-rccl"

    def test_launcher_override(self):
        cfg = HydraConfigLoader.load(["launcher=torchrun"])
        assert cfg.distributed.enabled is True
        assert cfg.distributed.launcher == "torchrun"

    def test_inline_value_override(self):
        cfg = HydraConfigLoader.load(["launcher=torchrun", "distributed.nnodes=4"])
        assert cfg.distributed.nnodes == 4

    def test_append_profile(self):
        cfg = HydraConfigLoader.load(["+profile=mi300x_8gpu"])
        assert cfg.gpu_type == "mi300x"
        assert cfg.distributed.nproc_per_node == 8

    def test_user_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("debug: true\nenv_vars:\n  MY_VAR: hello\n")
            f.flush()
            try:
                cfg = HydraConfigLoader.load([f.name])
                assert cfg.debug is True
                assert cfg.env_vars.MY_VAR == "hello"
            finally:
                os.unlink(f.name)

    def test_user_yaml_with_overrides(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("debug: true\n")
            f.flush()
            try:
                cfg = HydraConfigLoader.load([f.name, "scheduler=slurm"])
                assert cfg.debug is True
                assert "slurm" in cfg
            finally:
                os.unlink(f.name)

    def test_hardware_nvidia(self):
        cfg = HydraConfigLoader.load(["hardware=nvidia"])
        assert cfg.gpu_vendor == "NVIDIA"
        assert cfg.runtime.use_gpu_flag is True

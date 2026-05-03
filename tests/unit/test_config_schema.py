#!/usr/bin/env python3
"""Tests for ConfigValidator."""

import pytest
from omegaconf import OmegaConf

from madengine.config.schema import ConfigValidator


def make_cfg(data: dict) -> "DictConfig":
    return OmegaConf.create(data)


class TestConflictDetection:
    def test_slurm_and_k8s_conflict(self):
        cfg = make_cfg({"slurm": {"partition": "gpu"}, "k8s": {"namespace": "default"}})
        errors = ConfigValidator.validate(cfg)
        assert any("Cannot specify both" in e for e in errors)

    def test_slurm_only_no_conflict(self):
        cfg = make_cfg({"slurm": {"partition": "gpu"}})
        errors = ConfigValidator.validate(cfg)
        assert not any("Cannot specify both" in e for e in errors)

    def test_k8s_only_no_conflict(self):
        cfg = make_cfg({"k8s": {"namespace": "default"}})
        errors = ConfigValidator.validate(cfg)
        assert not any("Cannot specify both" in e for e in errors)


class TestDistributedValidation:
    def test_enabled_without_launcher(self):
        cfg = make_cfg({"distributed": {"enabled": True}})
        errors = ConfigValidator.validate(cfg)
        assert any("requires distributed.launcher" in e for e in errors)

    def test_enabled_with_launcher(self):
        cfg = make_cfg({"distributed": {"enabled": True, "launcher": "torchrun"}})
        errors = ConfigValidator.validate(cfg)
        assert not any("requires distributed.launcher" in e for e in errors)

    def test_invalid_nnodes(self):
        cfg = make_cfg(
            {"distributed": {"enabled": True, "launcher": "torchrun", "nnodes": -1}}
        )
        errors = ConfigValidator.validate(cfg)
        assert any("positive integer" in e for e in errors)

    def test_valid_nnodes(self):
        cfg = make_cfg(
            {"distributed": {"enabled": True, "launcher": "torchrun", "nnodes": 4}}
        )
        errors = ConfigValidator.validate(cfg)
        assert not any("positive integer" in e for e in errors)


class TestUnknownKeys:
    def test_unknown_top_level_key(self):
        cfg = make_cfg({"gpu_vendor": "AMD", "typo_key": "oops"})
        errors = ConfigValidator.validate(cfg)
        assert any("Unknown config key: 'typo_key'" in e for e in errors)

    def test_known_keys_accepted(self):
        cfg = make_cfg({"gpu_vendor": "AMD", "debug": True, "env_vars": {}})
        errors = ConfigValidator.validate(cfg)
        assert not any("Unknown config key" in e for e in errors)


class TestPlatformValidation:
    def test_unsupported_platform(self):
        cfg = make_cfg({"platform": {"type": "bare_metal"}})
        errors = ConfigValidator.validate(cfg)
        assert any("not yet supported" in e for e in errors)

    def test_docker_platform_ok(self):
        cfg = make_cfg({"platform": {"type": "docker"}})
        errors = ConfigValidator.validate(cfg)
        assert not any("not yet supported" in e for e in errors)

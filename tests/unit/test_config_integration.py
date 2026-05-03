#!/usr/bin/env python3
"""Integration tests for load_config end-to-end pipeline."""

import pytest
from pathlib import Path

from madengine.config import load_config
from madengine.core.errors import ConfigurationError


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "configs"


class TestLoadConfigEndToEnd:
    def test_defaults_produce_valid_context(self):
        ctx, meta = load_config([])
        assert ctx["gpu_vendor"] == "AMD"
        assert ctx["guest_os"] == "UBUNTU"
        assert meta["model"]["tags"] == []

    def test_scheduler_slurm(self):
        ctx, meta = load_config(["scheduler=slurm"])
        assert "slurm" in ctx
        assert ctx["slurm"]["partition"] == "amd-rccl"

    def test_launcher_torchrun(self):
        ctx, meta = load_config(["launcher=torchrun"])
        assert ctx["distributed"]["enabled"] is True
        assert ctx["distributed"]["launcher"] == "torchrun"

    def test_combined_overrides(self):
        ctx, meta = load_config([
            "scheduler=slurm",
            "launcher=torchrun",
            "distributed.nnodes=4",
            "+env=nccl_debug",
        ])
        assert ctx["distributed"]["nnodes"] == 4
        assert ctx["env_vars"]["NCCL_DEBUG"] == "INFO"
        assert "slurm" in ctx

    def test_user_yaml_file(self):
        yaml_path = str(FIXTURES_DIR / "test_slurm_job.yaml")
        ctx, meta = load_config([yaml_path])
        assert meta["model"]["tags"] == ["dummy"]
        assert ctx["slurm"]["partition"] == "test-partition"
        assert ctx["distributed"]["nnodes"] == 2
        assert ctx["env_vars"]["MY_VAR"] == "test_value"
        assert ctx["debug"] is True

    def test_user_yaml_with_override(self):
        # User YAML is merged last (highest priority). Overrides for keys present in
        # the user YAML are overwritten by it; new keys added via '+' syntax survive.
        yaml_path = str(FIXTURES_DIR / "test_slurm_job.yaml")
        ctx, meta = load_config([yaml_path, "+env_vars.EXTRA_VAR=hello"])
        # MY_VAR from user YAML is preserved
        assert ctx["env_vars"]["MY_VAR"] == "test_value"
        # EXTRA_VAR added via '+' override is also present
        assert ctx["env_vars"]["EXTRA_VAR"] == "hello"

    def test_docker_keys_translated(self):
        # Appending to an empty dict in Hydra requires the '+' prefix
        ctx, meta = load_config(["+docker.build_args.KEY=val"])
        assert ctx["docker_build_arg"]["KEY"] == "val"

    def test_slurm_and_k8s_conflict_raises(self):
        # scheduler=slurm adds 'slurm' section; +k8s.namespace appends 'k8s' section.
        # Validator detects both and raises ConfigurationError.
        with pytest.raises(ConfigurationError, match="Cannot specify both"):
            load_config(["scheduler=slurm", "+k8s.namespace=test"])

    def test_unsupported_platform_raises(self):
        with pytest.raises(ConfigurationError, match="not yet supported"):
            load_config(["platform=bare_metal"])

    def test_container_image_promoted(self):
        ctx, meta = load_config(
            ["model.container_image=myimage:latest"]
        )
        assert ctx["MAD_CONTAINER_IMAGE"] == "myimage:latest"

    def test_model_tags_in_metadata(self):
        ctx, meta = load_config(["model.tags=[dummy,bert]"])
        assert meta["model"]["tags"] == ["dummy", "bert"]
        assert "model" not in ctx

    def test_profile_append(self):
        ctx, meta = load_config(["+profile=mi300x_8gpu"])
        assert ctx["gpu_type"] == "mi300x"
        assert ctx["env_vars"]["HSA_ENABLE_SDMA"] == "0"

    def test_tools_append(self):
        ctx, meta = load_config(["+tools=rocprofv3_lightweight"])
        assert len(ctx["tools"]) == 1
        assert ctx["tools"][0]["name"] == "rocprofv3_lightweight"

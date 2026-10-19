#!/usr/bin/env python3
"""--config must emit the same additional_context keys as JSON additional-context."""

from pathlib import Path

from madengine.config import load_config

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "configs"


class TestAdditionalContextParity:
    def test_slurm_job_yaml_matches_json_shape(self):
        ctx, meta = load_config([str(FIXTURES_DIR / "test_slurm_job.yaml")])
        assert meta["model"]["tags"] == ["dummy"]
        assert ctx["slurm"]["partition"] == "test-partition"
        assert ctx["distributed"]["enabled"] is True
        assert ctx["distributed"]["launcher"] == "torchrun"
        assert ctx["distributed"]["nnodes"] == 2
        assert ctx["env_vars"]["MY_VAR"] == "test_value"
        assert ctx["debug"] is True
        assert ctx["gpu_vendor"] == "AMD"
        assert ctx["guest_os"] == "UBUNTU"
        assert "model" not in ctx
        assert "platform" not in ctx
        assert "tools" not in ctx
        assert "runtime" not in ctx
        assert "docker_keep_alive" not in ctx
        assert "docker_clean_cache" not in ctx

    def test_json_equivalent_keys_from_yaml_file(self):
        yaml_path = str(FIXTURES_DIR / "test_parity_job.yaml")
        ctx, meta = load_config([yaml_path])
        expected = {
            "slurm": {
                "partition": "gpu",
                "reservation": "myres",
                "time": "02:00:00",
            },
            "distributed": {
                "enabled": True,
                "launcher": "torchrun",
                "nnodes": 2,
                "nproc_per_node": 4,
            },
            "env_vars": {"NCCL_DEBUG": "INFO"},
            "docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"},
            "require_pinned_image": True,
            "live_output": True,
        }
        for key, value in expected.items():
            assert ctx[key] == value, f"{key}: {ctx.get(key)!r} != {value!r}"

    def test_build_ci_preset_sets_clean_cache(self):
        ctx, meta = load_config(["+build=ci"])
        assert ctx.get("docker_clean_cache") is True

    def test_build_multi_arch_preset_sets_target_archs(self):
        ctx, meta = load_config(["+build=multi_arch"])
        assert meta["build"]["target_archs"] == ["gfx90a", "gfx942"]

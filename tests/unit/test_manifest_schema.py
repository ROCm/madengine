#!/usr/bin/env python3
"""
Unit tests for build-manifest schema validation.

The manifest used to be checked for three top-level keys, so the failures below all
surfaced minutes later as a failed multi-node job. Each test pins one of them to a
startup error instead:

1. A field with the wrong type is reported with its JSON pointer.
2. `built_models` and `built_images` are joined by key, so an orphan model is fatal.
3. `slurm.nodes` and `distributed.nnodes` must agree or sbatch and the launcher
   disagree on the world size.
4. A top-level deployment block is folded into `deployment_config`, which is the one
   place the target is read from.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import copy

import pytest

from madengine.core.errors import ValidationError
from madengine.schemas import (
    load_schema,
    migrate_top_level_deployment_blocks,
    validate_build_manifest,
)


VALID_MANIFEST = {
    "built_images": {
        "img": {
            "model": "dummy",
            "docker_image": "rocm/dummy:latest",
            "dockerfile": "docker/dummy.ubuntu.amd.Dockerfile",
            "local_image": True,
        }
    },
    "built_models": {
        "img": {
            "name": "dummy",
            "scripts": "scripts/dummy/run.sh",
            "n_gpus": "-1",
            "tags": ["pyt", "training"],
            "timeout": -1,
            "args": "--model_repo dummy",
            "multiple_results": "perf_dummy.csv",
        }
    },
    "context": {
        "docker_env_vars": {"NCCL_DEBUG": "INFO"},
        "docker_mounts": {"/dev/infiniband": "/dev/infiniband"},
        "docker_build_arg": {},
        "gpu_vendor": "AMD",
        "guest_os": "UBUNTU",
        "docker_gpus": "0,1,2,3,4,5,6,7",
    },
    "deployment_config": {
        "target": "slurm",
        "slurm": {"partition": "meta64", "nodes": 2, "gpus_per_node": 8},
        "distributed": {"launcher": "torchrun", "nnodes": 2, "nproc_per_node": 8},
        "env_vars": {},
    },
}


@pytest.fixture
def manifest():
    return copy.deepcopy(VALID_MANIFEST)


class TestSchemaIsWellFormed:
    def test_schema_loads(self):
        schema = load_schema()
        assert schema["$schema"].startswith("https://json-schema.org/draft/2020-12")
        assert set(schema["required"]) == {"built_images", "built_models", "context"}

    def test_valid_manifest_passes_without_warnings(self, manifest):
        assert validate_build_manifest(manifest) == []


class TestSchemaViolations:
    def test_wrong_type_is_reported_with_json_pointer(self, manifest):
        manifest["deployment_config"]["slurm"]["nodes"] = "2"
        with pytest.raises(ValidationError) as excinfo:
            validate_build_manifest(manifest)
        assert "/deployment_config/slurm/nodes" in str(excinfo.value)

    def test_nested_env_var_value_is_rejected(self, manifest):
        """Env values are rendered into a shell command, so they must be scalars."""
        manifest["context"]["docker_env_vars"]["NCCL_IB_HCA"] = {"device": "bnxt_re0"}
        with pytest.raises(ValidationError) as excinfo:
            validate_build_manifest(manifest)
        assert "/context/docker_env_vars/NCCL_IB_HCA" in str(excinfo.value)

    def test_unknown_deployment_target_is_rejected(self, manifest):
        manifest["deployment_config"]["target"] = "sluurm"
        with pytest.raises(ValidationError) as excinfo:
            validate_build_manifest(manifest)
        assert "/deployment_config/target" in str(excinfo.value)

    def test_unknown_keys_are_allowed(self, manifest):
        """A manifest may carry consumer-specific metadata madengine does not read."""
        manifest["built_models"]["img"]["consumer_note"] = "keep me"
        manifest["rccl_ci"] = {"branch": "develop"}
        assert validate_build_manifest(manifest) == []

    def test_null_optional_fields_are_accepted(self, manifest):
        """`madengine build` copies a null from models.json straight through.

        The schema describes what madengine writes, so rejecting its own output would only
        break runs that worked: a model declaring `"timeout": null` is not a broken manifest.
        """
        manifest["built_models"]["img"].update(
            {
                "timeout": None,
                "scripts": None,
                "n_gpus": None,
                "args": None,
                "multiple_results": None,
                "tags": None,
                "env_vars": None,
                "slurm": None,
                "distributed": None,
            }
        )
        manifest["built_images"]["img"]["dockerfile"] = None
        manifest["context"]["docker_gpus"] = None

        assert validate_build_manifest(manifest) == []

    def test_error_names_the_source_file(self, manifest):
        manifest["deployment_config"]["distributed"]["port"] = 70000
        with pytest.raises(ValidationError) as excinfo:
            validate_build_manifest(manifest, source="build_manifest.json")
        assert "build_manifest.json" in str(excinfo.value)


class TestCrossFieldChecks:
    def test_model_without_matching_image_is_fatal(self, manifest):
        manifest["built_models"]["orphan"] = {"name": "orphan"}
        with pytest.raises(ValidationError) as excinfo:
            validate_build_manifest(manifest)
        assert "orphan" in str(excinfo.value)

    def test_node_count_mismatch_is_fatal(self, manifest):
        manifest["deployment_config"]["distributed"]["nnodes"] = 4
        with pytest.raises(ValidationError) as excinfo:
            validate_build_manifest(manifest)
        assert "world size" in str(excinfo.value)

    def test_image_without_model_is_a_warning(self, manifest):
        manifest["built_images"]["spare"] = {"docker_image": "rocm/spare:latest"}
        warnings = validate_build_manifest(manifest)
        assert any("spare" in w for w in warnings)

    def test_slurm_target_without_slurm_block_is_a_warning(self, manifest):
        del manifest["deployment_config"]["slurm"]
        warnings = validate_build_manifest(manifest)
        assert any("cluster defaults" in w for w in warnings)


class TestSingleSourceOfTruthForDeploymentTarget:
    """A top-level deployment block used to select the target while its values were ignored."""

    def test_top_level_block_is_moved_under_deployment_config(self, manifest):
        top_level = {"partition": "amd-rccl", "nodes": 2}
        manifest["slurm"] = top_level
        del manifest["deployment_config"]["slurm"]

        warnings = validate_build_manifest(manifest)

        assert "slurm" not in manifest
        assert manifest["deployment_config"]["slurm"] == top_level
        assert any("belongs under" in w for w in warnings)

    def test_deployment_config_wins_when_both_are_present(self, manifest):
        manifest["slurm"] = {"partition": "ignored-partition", "nodes": 2}

        warnings = validate_build_manifest(manifest)

        assert "slurm" not in manifest
        assert manifest["deployment_config"]["slurm"]["partition"] == "meta64"
        assert any("was ignored" in w for w in warnings)

    def test_migration_can_be_disabled(self, manifest):
        manifest["slurm"] = {"partition": "amd-rccl", "nodes": 2}
        validate_build_manifest(manifest, migrate=False)
        assert "slurm" in manifest

    def test_every_deployment_block_is_migrated(self):
        manifest = {
            "built_images": {},
            "built_models": {},
            "context": {},
            "distributed": {"launcher": "torchrun", "nnodes": 1},
            "kubernetes": {"namespace": "mad"},
        }
        migrate_top_level_deployment_blocks(manifest)
        assert set(manifest["deployment_config"]) == {"distributed", "kubernetes"}

#!/usr/bin/env python3
"""
Unit tests for madengine CLI validators

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import tempfile

import pytest
import typer

from madengine.core.additional_context_defaults import (
    DEFAULT_GPU_VENDOR,
    DEFAULT_GUEST_OS,
)
from madengine.cli.validators import validate_additional_context
from madengine.cli.constants import ExitCode


class TestValidateAdditionalContext:
    """Test suite for validate_additional_context function"""

    def test_validate_additional_context_with_defaults_applied(self, capsys):
        """Test that defaults are applied when context is empty"""
        result = validate_additional_context(additional_context="{}")

        assert result["gpu_vendor"] == DEFAULT_GPU_VENDOR
        assert result["guest_os"] == DEFAULT_GUEST_OS

        # Verify console output mentions defaults
        # Note: capsys won't capture Rich console output, so we just verify the result

    def test_validate_additional_context_no_defaults_when_provided(self):
        """Test that explicit values are preserved and no defaults applied"""
        explicit_context = '{"gpu_vendor": "NVIDIA", "guest_os": "CENTOS"}'
        result = validate_additional_context(additional_context=explicit_context)

        assert result["gpu_vendor"] == "NVIDIA"
        assert result["guest_os"] == "CENTOS"

    def test_validate_additional_context_partial_default_gpu_vendor(self):
        """Test that only gpu_vendor is defaulted when guest_os is provided"""
        partial_context = '{"guest_os": "CENTOS"}'
        result = validate_additional_context(additional_context=partial_context)

        assert result["gpu_vendor"] == DEFAULT_GPU_VENDOR
        assert result["guest_os"] == "CENTOS"

    def test_validate_additional_context_partial_default_guest_os(self):
        """Test that only guest_os is defaulted when gpu_vendor is provided"""
        partial_context = '{"gpu_vendor": "NVIDIA"}'
        result = validate_additional_context(additional_context=partial_context)

        assert result["gpu_vendor"] == "NVIDIA"
        assert result["guest_os"] == DEFAULT_GUEST_OS

    def test_validate_additional_context_file_with_defaults(self):
        """Test that defaults are applied after file is loaded"""
        # Create temporary config file with extra fields but missing required ones
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"extra_field": "value", "timeout": 30}, f)
            temp_file = f.name

        try:
            result = validate_additional_context(
                additional_context="{}", additional_context_file=temp_file
            )

            # Should have defaults plus the extra field
            assert result["gpu_vendor"] == DEFAULT_GPU_VENDOR
            assert result["guest_os"] == DEFAULT_GUEST_OS
            assert result["extra_field"] == "value"
            assert result["timeout"] == 30
        finally:
            import os

            os.unlink(temp_file)

    def test_validate_additional_context_invalid_values_no_defaults(self):
        """Test that invalid values cause validation error, not default application"""
        invalid_context = '{"gpu_vendor": "INVALID", "guest_os": "INVALID"}'

        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=invalid_context)

        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_empty_string_no_defaults(self):
        """Test that empty string values cause validation error, not defaults"""
        empty_string_context = '{"gpu_vendor": "", "guest_os": ""}'

        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=empty_string_context)

        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_case_insensitive(self):
        """Lowercase gpu_vendor/guest_os are accepted and normalized to canonical uppercase."""
        lowercase_context = '{"gpu_vendor": "amd", "guest_os": "ubuntu"}'
        result = validate_additional_context(additional_context=lowercase_context)

        assert result["gpu_vendor"] == "AMD"
        assert result["guest_os"] == "UBUNTU"

    def test_validate_additional_context_file_and_cli_merge_with_defaults(self):
        """Test that file + CLI merge works and defaults fill gaps"""
        # Create temporary file with one field
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"gpu_vendor": "NVIDIA"}, f)
            temp_file = f.name

        try:
            # CLI provides different field (neither provides guest_os)
            result = validate_additional_context(
                additional_context='{"timeout": 60}', additional_context_file=temp_file
            )

            # Should merge file + CLI + defaults
            assert result["gpu_vendor"] == "NVIDIA"  # From file
            assert result["guest_os"] == DEFAULT_GUEST_OS  # From defaults
            assert result["timeout"] == 60  # From CLI
        finally:
            import os

            os.unlink(temp_file)

    def test_validate_additional_context_cli_overrides_file(self):
        """Test that CLI values override file values"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"gpu_vendor": "AMD", "guest_os": "UBUNTU"}, f)
            temp_file = f.name

        try:
            # CLI overrides gpu_vendor
            result = validate_additional_context(
                additional_context='{"gpu_vendor": "NVIDIA"}',
                additional_context_file=temp_file,
            )

            assert result["gpu_vendor"] == "NVIDIA"  # From CLI (override)
            assert result["guest_os"] == "UBUNTU"  # From file
        finally:
            import os

            os.unlink(temp_file)

    def test_validate_additional_context_invalid_json(self):
        """Test that invalid JSON raises appropriate error"""
        invalid_json = '{"gpu_vendor": "AMD", invalid}'

        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=invalid_json)

        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_file_not_found(self):
        """Test that missing file raises appropriate error"""
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context="{}",
                additional_context_file="/nonexistent/file.json",
            )

        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_docker_build_arg_must_be_object(self):
        bad = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU", "docker_build_arg": "oops"}'

        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=bad)

        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_nested_docker_build_arg_ok(self):
        ctx = (
            '{"gpu_vendor": "AMD", "guest_os": "UBUNTU", '
            '"docker_build_arg": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}}'
        )
        result = validate_additional_context(additional_context=ctx)
        assert result["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"] == "gfx942"

    def test_validate_additional_context_tools_must_be_list(self):
        bad = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU", "tools": {}}'

        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=bad)

        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_tools_list_ok(self):
        ctx = '{"gpu_vendor": "AMD", "guest_os": "UBUNTU", "tools": []}'
        result = validate_additional_context(additional_context=ctx)
        assert result["tools"] == []

    def test_validate_additional_context_log_error_keys_accepted(self):
        """Valid types for log scan keys pass structure validation."""
        ctx = json.dumps(
            {
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "log_error_pattern_scan": False,
                "log_error_benign_patterns": ["noise", "(ok)"],
                "log_error_patterns": ["OOM", "Killed"],
            }
        )
        result = validate_additional_context(additional_context=ctx)
        assert result["log_error_pattern_scan"] is False
        assert result["log_error_benign_patterns"] == ["noise", "(ok)"]
        assert result["log_error_patterns"] == ["OOM", "Killed"]

    @pytest.mark.parametrize(
        "leps",
        [
            True,
            "true",
            0,
            1,
            None,
        ],
    )
    def test_validate_additional_context_log_error_pattern_scan_coercible_types(
        self, leps
    ):
        ctx = json.dumps(
            {
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "log_error_pattern_scan": leps,
            }
        )
        result = validate_additional_context(additional_context=ctx)
        assert result["log_error_pattern_scan"] == leps

    def test_validate_additional_context_log_error_pattern_scan_rejects_bad_type(self):
        bad = json.dumps(
            {
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "log_error_pattern_scan": [],
            }
        )
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=bad)
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_log_error_benign_patterns_rejects_non_list(
        self,
    ):
        bad = json.dumps(
            {
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "log_error_benign_patterns": "not-a-list",
            }
        )
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=bad)
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_log_error_benign_patterns_rejects_non_strings(
        self,
    ):
        bad = json.dumps(
            {
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "log_error_benign_patterns": ["a", 1],
            }
        )
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=bad)
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_log_error_patterns_rejects_empty_list(self):
        bad = json.dumps(
            {
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "log_error_patterns": [],
            }
        )
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=bad)
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_log_error_patterns_rejects_non_list(self):
        bad = json.dumps(
            {
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "log_error_patterns": {},
            }
        )
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=bad)
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_validate_additional_context_log_error_patterns_rejects_non_string_element(
        self,
    ):
        bad = json.dumps(
            {
                "gpu_vendor": "AMD",
                "guest_os": "UBUNTU",
                "log_error_patterns": ["ok", 2],
            }
        )
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=bad)
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS


def _cluster_context(cluster):
    """Build a minimal valid context carrying the given cluster block."""
    return json.dumps(
        {"gpu_vendor": "AMD", "guest_os": "UBUNTU", "cluster": cluster}
    )


class TestValidateClusterContext:
    """Test suite for additional_context.cluster schema validation"""

    def test_cluster_must_be_object(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(additional_context=_cluster_context("oops"))
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_full_cluster_config_accepted(self):
        cluster = {
            "rdma": {
                "enabled": True,
                "strict": False,
                "mode": "enforce",
                "apply_env": True,
                "artifact_name": "rdma_recommendation.json",
            },
            "gcm": {
                "enabled": True,
                "strict": False,
                "enabled_platforms": ["slurm"],
                "health_checks": ["check-hca", "check-ibstat"],
                "source": {
                    "repo": "https://github.com/coketaste/gcm",
                    "ref": "9fed02cd0721d3937f8749672951185f31955bd4",
                },
                "collector": {
                    "enabled": True,
                    "command": "slurm_job_monitor",
                    "once": True,
                    "sink": "file",
                    "timeout_sec": 120,
                    "max_retries": 1,
                    "best_effort": True,
                },
                "artifacts": {
                    "dir": "./slurm_results/cluster_artifacts",
                    "files": {"health_raw_log": "gcm_health_raw.log"},
                },
            },
        }
        result = validate_additional_context(
            additional_context=_cluster_context(cluster)
        )
        assert result["cluster"] == cluster

    def test_empty_cluster_accepted(self):
        result = validate_additional_context(additional_context=_cluster_context({}))
        assert result["cluster"] == {}

    # --- cluster.rdma ---------------------------------------------------

    def test_rdma_must_be_object(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"rdma": "yes"})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    @pytest.mark.parametrize("key", ["enabled", "strict", "apply_env"])
    def test_rdma_bool_keys_reject_non_bool(self, key):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"rdma": {key: "true"}})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_rdma_artifact_name_rejects_non_string(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"rdma": {"artifact_name": 5}})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    @pytest.mark.parametrize("mode", ["recommend", "enforce"])
    def test_rdma_mode_enum_accepted(self, mode):
        result = validate_additional_context(
            additional_context=_cluster_context({"rdma": {"mode": mode}})
        )
        assert result["cluster"]["rdma"]["mode"] == mode

    def test_rdma_mode_enum_rejects_unknown_value(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"rdma": {"mode": "force"}})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    # --- cluster.gcm ----------------------------------------------------

    def test_gcm_must_be_object(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"gcm": []})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    @pytest.mark.parametrize("key", ["enabled", "strict"])
    def test_gcm_bool_keys_reject_non_bool(self, key):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"gcm": {key: 1}})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_gcm_enabled_platforms_rejects_non_string_element(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context(
                    {"gcm": {"enabled_platforms": ["slurm", 2]}}
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_gcm_health_checks_allowlist_rejects_unknown_check(self):
        """Only check-hca / check-ibstat are permitted in this phase."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context(
                    {"gcm": {"health_checks": ["check-hca", "rm -rf /"]}}
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_gcm_health_checks_allowlist_accepts_known_checks(self):
        result = validate_additional_context(
            additional_context=_cluster_context(
                {"gcm": {"health_checks": ["check-ibstat"]}}
            )
        )
        assert result["cluster"]["gcm"]["health_checks"] == ["check-ibstat"]

    def test_gcm_source_rejects_non_string_ref(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"gcm": {"source": {"ref": 42}}})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_gcm_collector_command_allowlist_rejects_unknown_command(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context(
                    {"gcm": {"collector": {"command": "curl evil.example"}}}
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    @pytest.mark.parametrize("key", ["timeout_sec", "max_retries"])
    def test_gcm_collector_int_keys_reject_non_int(self, key):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"gcm": {"collector": {key: "5"}}})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    @pytest.mark.parametrize("key", ["timeout_sec", "max_retries"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_gcm_collector_int_keys_reject_below_minimum(self, key, value):
        """0/negative would make the collector loop never run."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context({"gcm": {"collector": {key: value}}})
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_gcm_artifacts_files_rejects_non_string_value(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_cluster_context(
                    {"gcm": {"artifacts": {"files": {"health_raw_log": 1}}}}
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

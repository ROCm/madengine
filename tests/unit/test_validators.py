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

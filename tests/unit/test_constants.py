"""Unit tests for core constants and _get_env_or_creds_or_default."""

import json
import os
from unittest.mock import patch

from madengine.core.constants import (
    _DEFAULT_MAD_AWS_S3,
    _DEFAULT_MAD_MINIO,
    _DEFAULT_NAS_NODES,
    _DEFAULT_PUBLIC_GITHUB_ROCM_KEY,
    MAD_AWS_S3,
    MAD_MINIO,
    NAS_NODES,
    PUBLIC_GITHUB_ROCM_KEY,
    _get_env_or_creds_or_default,
)


class TestGetEnvOrCredsOrDefault:
    """Test _get_env_or_creds_or_default helper."""

    def test_env_override_returns_parsed_json(self):
        """When env is set with valid JSON, that value is returned."""
        with patch.dict(os.environ, {"TEST_KEY": '[{"a": 1}]'}, clear=False):
            # Need to pass creds - we patch CREDS via the module
            import madengine.core.constants as constants_module

            with patch.object(constants_module, "CREDS", {}):
                result = _get_env_or_creds_or_default(
                    "TEST_KEY", "TEST_KEY", _DEFAULT_NAS_NODES
                )
        assert result == [{"a": 1}]

    def test_env_invalid_json_returns_default(self):
        """When env is set with invalid JSON, default is returned."""
        with patch.dict(os.environ, {"TEST_KEY": "not json"}, clear=False):
            import madengine.core.constants as constants_module

            with patch.object(constants_module, "CREDS", {}):
                result = _get_env_or_creds_or_default(
                    "TEST_KEY", "TEST_KEY", _DEFAULT_NAS_NODES
                )
        assert result == _DEFAULT_NAS_NODES

    def test_creds_fallback_when_env_unset(self):
        """When env is unset and creds has key, creds value is returned."""
        with patch.dict(os.environ, {}, clear=False):
            try:
                orig = os.environ.get("TEST_KEY")
                if "TEST_KEY" in os.environ:
                    del os.environ["TEST_KEY"]
            except Exception:
                pass
        import madengine.core.constants as constants_module

        with patch.object(constants_module, "CREDS", {"TEST_KEY": [{"from": "creds"}]}):
            result = _get_env_or_creds_or_default(
                "TEST_KEY", "TEST_KEY", _DEFAULT_NAS_NODES
            )
        assert result == [{"from": "creds"}]

    def test_default_when_env_and_creds_unset(self):
        """When env unset and creds missing key, default is returned."""
        import madengine.core.constants as constants_module

        with patch.dict(os.environ, {}, clear=False):
            if "TEST_KEY" in os.environ:
                del os.environ["TEST_KEY"]
        with patch.object(constants_module, "CREDS", {}):
            result = _get_env_or_creds_or_default(
                "TEST_KEY", "TEST_KEY", _DEFAULT_MAD_AWS_S3
            )
        assert result == _DEFAULT_MAD_AWS_S3


class TestConstantsPublicAPI:
    """Test that the four constants still expose correct shape (smoke)."""

    def test_nas_nodes_is_list_of_dicts(self):
        """NAS_NODES is a list of node config dicts."""
        assert isinstance(NAS_NODES, list)
        assert len(NAS_NODES) >= 1
        for node in NAS_NODES:
            assert isinstance(node, dict)
            assert "NAME" in node or "HOST" in node

    def test_mad_aws_s3_has_expected_keys(self):
        """MAD_AWS_S3 has USERNAME and PASSWORD."""
        assert isinstance(MAD_AWS_S3, dict)
        assert "USERNAME" in MAD_AWS_S3
        assert "PASSWORD" in MAD_AWS_S3

    def test_mad_minio_has_expected_keys(self):
        """MAD_MINIO has USERNAME, PASSWORD, MINIO_ENDPOINT, AWS_ENDPOINT_URL_S3."""
        assert isinstance(MAD_MINIO, dict)
        assert "USERNAME" in MAD_MINIO
        assert "PASSWORD" in MAD_MINIO
        assert "MINIO_ENDPOINT" in MAD_MINIO
        assert "AWS_ENDPOINT_URL_S3" in MAD_MINIO

    def test_public_github_rocm_key_has_expected_keys(self):
        """PUBLIC_GITHUB_ROCM_KEY has username and token (no value assert to avoid leaking secrets)."""
        assert isinstance(PUBLIC_GITHUB_ROCM_KEY, dict)
        assert set(PUBLIC_GITHUB_ROCM_KEY.keys()) >= {
            "username",
            "token",
        }, "PUBLIC_GITHUB_ROCM_KEY must have at least keys 'username' and 'token'"

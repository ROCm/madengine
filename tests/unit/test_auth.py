"""Unit tests for madengine.core.auth module."""

import os
from unittest.mock import mock_open, patch

from madengine.core.auth import load_credentials


class TestLoadCredentials:
    """Tests for load_credentials()."""

    @patch("madengine.core.auth.os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"dockerhub": {"username": "user", "password": "pass"}}',
    )
    def test_load_credentials_from_file(self, mock_file, mock_exists):
        """Valid credential.json is loaded and returned."""
        result = load_credentials()
        assert result is not None
        assert "dockerhub" in result
        assert result["dockerhub"]["username"] == "user"
        assert result["dockerhub"]["password"] == "pass"

    @patch("madengine.core.auth.os.path.exists", return_value=False)
    @patch.dict(os.environ, {}, clear=True)
    def test_load_credentials_no_file_no_env(self, mock_exists):
        """Returns None when no credential file and no env vars."""
        result = load_credentials()
        assert result is None

    @patch("madengine.core.auth.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="not valid json{{{")
    def test_load_credentials_malformed_json(self, mock_file, mock_exists):
        """Malformed credential.json is handled gracefully (returns None)."""
        # The function logs the error via handle_error but does not re-raise
        result = load_credentials()
        # credentials should be None since the file parse failed and no env vars
        assert result is None

    @patch("madengine.core.auth.os.path.exists", return_value=False)
    @patch.dict(
        os.environ,
        {"MAD_DOCKERHUB_USER": "envuser", "MAD_DOCKERHUB_PASSWORD": "envpass"},
        clear=True,
    )
    def test_load_credentials_env_vars_only(self, mock_exists):
        """Credentials from env vars when no file exists."""
        result = load_credentials()
        assert result is not None
        assert "dockerhub" in result
        assert result["dockerhub"]["username"] == "envuser"
        assert result["dockerhub"]["password"] == "envpass"
        assert "repository" not in result["dockerhub"]

    @patch("madengine.core.auth.os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"dockerhub": {"username": "fileuser", "password": "filepass"}}',
    )
    @patch.dict(
        os.environ,
        {"MAD_DOCKERHUB_USER": "envuser", "MAD_DOCKERHUB_PASSWORD": "envpass"},
        clear=True,
    )
    def test_load_credentials_env_overrides_file(self, mock_file, mock_exists):
        """Env vars override file credentials for dockerhub key."""
        result = load_credentials()
        assert result is not None
        assert result["dockerhub"]["username"] == "envuser"
        assert result["dockerhub"]["password"] == "envpass"

    @patch("madengine.core.auth.os.path.exists", return_value=False)
    @patch.dict(
        os.environ,
        {
            "MAD_DOCKERHUB_USER": "envuser",
            "MAD_DOCKERHUB_PASSWORD": "envpass",
            "MAD_DOCKERHUB_REPO": "myrepo/images",
        },
        clear=True,
    )
    def test_load_credentials_env_with_repo(self, mock_exists):
        """MAD_DOCKERHUB_REPO is included when set."""
        result = load_credentials()
        assert result is not None
        assert result["dockerhub"]["repository"] == "myrepo/images"

    @patch("madengine.core.auth.os.path.exists", return_value=False)
    @patch.dict(
        os.environ,
        {"MAD_DOCKERHUB_USER": "envuser"},
        clear=True,
    )
    def test_load_credentials_env_user_only_no_password(self, mock_exists):
        """Only MAD_DOCKERHUB_USER without PASSWORD does not create dockerhub entry."""
        result = load_credentials()
        # Without both user and password, dockerhub credentials are not created
        assert result is None

    @patch("madengine.core.auth.os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"custom_registry": {"token": "abc123"}}',
    )
    def test_load_credentials_non_dockerhub_registry(self, mock_file, mock_exists):
        """Non-dockerhub registries in credential.json are preserved."""
        result = load_credentials()
        assert result is not None
        assert "custom_registry" in result
        assert result["custom_registry"]["token"] == "abc123"

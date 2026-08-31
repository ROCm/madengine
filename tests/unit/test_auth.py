"""Unit tests for madengine.core.auth module."""

import json
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from madengine.core.auth import (
    explain_registry_denial,
    has_ambient_docker_auth,
    load_credentials,
    login_to_registry,
)


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


@patch.dict(os.environ, {"MAD_SKIP_DOCKER_LOGIN": ""}, clear=False)
@patch("madengine.core.auth.has_ambient_docker_auth", return_value=False)
class TestLoginToRegistry:
    """Tests for login_to_registry() when the machine has no existing docker login."""

    def _mocks(self):
        console = MagicMock()
        rich_console = MagicMock()
        return console, rich_console

    def test_no_credentials_returns_early(self, mock_ambient):
        """Passing None credentials logs a warning and returns without error."""
        console, rich_console = self._mocks()
        login_to_registry("docker.io", None, console, rich_console)
        console.sh.assert_not_called()

    def test_missing_registry_key_raises_when_raise_on_failure(self, mock_ambient):
        """RuntimeError raised when registry key absent and raise_on_failure=True."""
        console, rich_console = self._mocks()
        credentials = {"other_registry": {"username": "u", "password": "p"}}
        with pytest.raises(RuntimeError, match="myregistry.io"):
            login_to_registry(
                "myregistry.io",
                credentials,
                console,
                rich_console,
                raise_on_failure=True,
            )
        console.sh.assert_not_called()

    def test_missing_registry_key_returns_when_not_raise_on_failure(self, mock_ambient):
        """Returns silently when registry key absent and raise_on_failure=False."""
        console, rich_console = self._mocks()
        credentials = {"other_registry": {"username": "u", "password": "p"}}
        login_to_registry(
            "myregistry.io", credentials, console, rich_console, raise_on_failure=False
        )
        console.sh.assert_not_called()

    def test_invalid_credentials_format_raises(self, mock_ambient):
        """RuntimeError raised when username/password fields missing."""
        console, rich_console = self._mocks()
        credentials = {"dockerhub": {"token": "abc"}}
        with pytest.raises(RuntimeError, match="username|password"):
            login_to_registry(
                "docker.io", credentials, console, rich_console, raise_on_failure=True
            )
        console.sh.assert_not_called()

    def test_invalid_credentials_format_returns_when_not_raise_on_failure(
        self, mock_ambient
    ):
        """Returns silently when credentials format invalid and raise_on_failure=False."""
        console, rich_console = self._mocks()
        credentials = {"dockerhub": {"token": "abc"}}
        login_to_registry(
            "docker.io", credentials, console, rich_console, raise_on_failure=False
        )
        console.sh.assert_not_called()

    def test_blank_credentials_raise_without_ambient_auth(self, mock_ambient):
        """Placeholder credentials are treated as absent, not as credentials."""
        console, rich_console = self._mocks()
        credentials = {"dockerhub": {"repository": "r", "username": "", "password": ""}}
        with pytest.raises(RuntimeError, match="username|password"):
            login_to_registry(
                "docker.io", credentials, console, rich_console, raise_on_failure=True
            )
        console.sh.assert_not_called()

    def test_docker_io_normalised_to_dockerhub(self, mock_ambient):
        """docker.io registry is looked up under the 'dockerhub' key."""
        console, rich_console = self._mocks()
        credentials = {"dockerhub": {"username": "user", "password": "pass"}}
        login_to_registry("docker.io", credentials, console, rich_console)
        console.sh.assert_called_once()
        cmd = console.sh.call_args[0][0]
        # docker.io should not appear in the login command (uses default DockerHub endpoint)
        assert "docker.io" not in cmd

    def test_custom_registry_included_in_command(self, mock_ambient):
        """Non-DockerHub registry URL is included in the login command."""
        console, rich_console = self._mocks()
        credentials = {"myregistry.io": {"username": "user", "password": "pass"}}
        login_to_registry("myregistry.io", credentials, console, rich_console)
        console.sh.assert_called_once()
        cmd = console.sh.call_args[0][0]
        assert "myregistry.io" in cmd

    def test_login_failure_raises_when_raise_on_failure(self, mock_ambient):
        """docker login error is re-raised when raise_on_failure=True."""
        console, rich_console = self._mocks()
        console.sh.side_effect = RuntimeError("auth failed")
        credentials = {"dockerhub": {"username": "user", "password": "pass"}}
        with pytest.raises(RuntimeError, match="auth failed"):
            login_to_registry(
                None, credentials, console, rich_console, raise_on_failure=True
            )

    def test_login_failure_suppressed_when_not_raise_on_failure(self, mock_ambient):
        """docker login error is suppressed when raise_on_failure=False."""
        console, rich_console = self._mocks()
        console.sh.side_effect = RuntimeError("auth failed")
        credentials = {"dockerhub": {"username": "user", "password": "pass"}}
        login_to_registry(
            None, credentials, console, rich_console, raise_on_failure=False
        )
        # Should not propagate the exception


@patch.dict(os.environ, {"MAD_SKIP_DOCKER_LOGIN": ""}, clear=False)
class TestLoginToRegistryWithAmbientAuth:
    """Tests for login_to_registry() when the machine already has a docker login."""

    def _mocks(self):
        return MagicMock(), MagicMock()

    @patch("madengine.core.auth.has_ambient_docker_auth", return_value=True)
    def test_blank_credentials_defer_to_ambient_auth(self, mock_ambient):
        """Blank credentials never override or break an existing docker login."""
        console, rich_console = self._mocks()
        credentials = {"dockerhub": {"repository": "r", "username": "", "password": ""}}
        # No raise even with raise_on_failure=True: the machine is authenticated.
        login_to_registry(
            "docker.io", credentials, console, rich_console, raise_on_failure=True
        )
        console.sh.assert_not_called()

    @patch("madengine.core.auth.has_ambient_docker_auth", return_value=True)
    def test_missing_registry_key_defers_to_ambient_auth(self, mock_ambient):
        """A registry with no credential.json entry falls back to the existing login."""
        console, rich_console = self._mocks()
        credentials = {"other_registry": {"username": "u", "password": "p"}}
        login_to_registry(
            "myregistry.io", credentials, console, rich_console, raise_on_failure=True
        )
        console.sh.assert_not_called()

    @patch("madengine.core.auth.has_ambient_docker_auth", return_value=True)
    def test_explicit_credentials_win_over_ambient_auth(self, mock_ambient):
        """Usable explicit credentials still trigger a login (explicit wins)."""
        console, rich_console = self._mocks()
        credentials = {"dockerhub": {"username": "user", "password": "pass"}}
        login_to_registry("docker.io", credentials, console, rich_console)
        console.sh.assert_called_once()
        assert "--username user" in console.sh.call_args[0][0]

    @patch("madengine.core.auth.has_ambient_docker_auth", return_value=False)
    def test_whitespace_only_credentials_are_not_credentials(self, mock_ambient):
        """Whitespace-only values are treated as blank."""
        console, rich_console = self._mocks()
        credentials = {"dockerhub": {"username": "  ", "password": "\t"}}
        with pytest.raises(RuntimeError, match="username|password"):
            login_to_registry(
                "docker.io", credentials, console, rich_console, raise_on_failure=True
            )
        console.sh.assert_not_called()


class TestSkipDockerLogin:
    """Tests for the MAD_SKIP_DOCKER_LOGIN escape hatch."""

    @patch.dict(os.environ, {"MAD_SKIP_DOCKER_LOGIN": "1"}, clear=False)
    def test_skip_env_var_bypasses_login(self):
        """MAD_SKIP_DOCKER_LOGIN=1 defers to ambient credentials unconditionally."""
        console, rich_console = MagicMock(), MagicMock()
        credentials = {"dockerhub": {"username": "user", "password": "pass"}}
        login_to_registry(
            "docker.io", credentials, console, rich_console, raise_on_failure=True
        )
        console.sh.assert_not_called()


class TestHasAmbientDockerAuth:
    """Tests for has_ambient_docker_auth()."""

    def _write_config(self, tmp_path, config):
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return {"DOCKER_CONFIG": str(tmp_path)}

    def test_dockerhub_auth_entry_detected(self, tmp_path):
        """A Docker Hub entry with an auth blob counts as authenticated."""
        env = self._write_config(
            tmp_path, {"auths": {"https://index.docker.io/v1/": {"auth": "abc123"}}}
        )
        with patch.dict(os.environ, env, clear=False):
            assert has_ambient_docker_auth(None) is True
            assert has_ambient_docker_auth("docker.io") is True
            assert has_ambient_docker_auth("docker.io/rocm/mad-private") is True
            assert has_ambient_docker_auth("myregistry.io") is False

    def test_identity_token_entry_detected(self, tmp_path):
        """An identitytoken-only entry counts as authenticated."""
        env = self._write_config(
            tmp_path, {"auths": {"index.docker.io": {"identitytoken": "tok"}}}
        )
        with patch.dict(os.environ, env, clear=False):
            assert has_ambient_docker_auth("docker.io") is True

    def test_cred_helper_detected(self, tmp_path):
        """A credHelpers entry counts as authenticated."""
        env = self._write_config(
            tmp_path, {"credHelpers": {"myregistry.io": "ecr-login"}}
        )
        with patch.dict(os.environ, env, clear=False):
            assert has_ambient_docker_auth("myregistry.io/team/img") is True
            assert has_ambient_docker_auth("docker.io") is False

    def test_creds_store_with_empty_auth_entry(self, tmp_path):
        """credsStore-managed entries are stored empty but are still credentials."""
        env = self._write_config(
            tmp_path,
            {"credsStore": "desktop", "auths": {"https://index.docker.io/v1/": {}}},
        )
        with patch.dict(os.environ, env, clear=False):
            assert has_ambient_docker_auth("docker.io") is True
            assert has_ambient_docker_auth("other.io") is False

    def test_empty_auth_entry_without_creds_store(self, tmp_path):
        """An empty entry with no credential store is not usable."""
        env = self._write_config(
            tmp_path, {"auths": {"https://index.docker.io/v1/": {}}}
        )
        with patch.dict(os.environ, env, clear=False):
            assert has_ambient_docker_auth("docker.io") is False

    def test_registry_with_port_matches_host(self, tmp_path):
        """host:port registries are matched on the host segment."""
        env = self._write_config(tmp_path, {"auths": {"localhost:5000": {"auth": "x"}}})
        with patch.dict(os.environ, env, clear=False):
            assert has_ambient_docker_auth("localhost:5000/team/img") is True

    def test_missing_config_returns_false(self, tmp_path):
        """A missing config.json yields False rather than raising."""
        with patch.dict(os.environ, {"DOCKER_CONFIG": str(tmp_path)}, clear=False):
            assert has_ambient_docker_auth("docker.io") is False

    def test_corrupt_config_returns_false(self, tmp_path):
        """A corrupt config.json yields False rather than raising."""
        (tmp_path / "config.json").write_text("not json{{{", encoding="utf-8")
        with patch.dict(os.environ, {"DOCKER_CONFIG": str(tmp_path)}, clear=False):
            assert has_ambient_docker_auth("docker.io") is False


class TestExplainRegistryDenial:
    """Tests for explain_registry_denial()."""

    def test_insufficient_scope_is_an_authorization_problem(self):
        """insufficient_scope is reported as authorization, not authentication."""
        log = (
            "#6 ERROR: pull access denied, repository does not exist or may require "
            "authorization: server message: insufficient_scope: authorization failed"
        )
        hint = explain_registry_denial(log, "rocm/triton-inference-server-dev:x")
        assert hint is not None
        assert "authorization problem" in hint
        assert "rocm/triton-inference-server-dev:x" in hint
        assert "will not fix it" in hint

    def test_plain_denial_points_at_login(self):
        """A denial without insufficient_scope points at supplying credentials."""
        log = "Error response from daemon: pull access denied for rocm/private"
        hint = explain_registry_denial(log, "rocm/private:latest")
        assert hint is not None
        assert "docker login" in hint
        assert "authorization problem" not in hint

    def test_non_dockerhub_denial_names_that_registry(self):
        """A denial for another registry does not suggest Docker Hub credentials."""
        log = "Error response from daemon: pull access denied for ghcr.io/org/app"
        hint = explain_registry_denial(log, "ghcr.io/org/app:latest")
        assert hint is not None
        assert "docker login ghcr.io" in hint
        assert '"ghcr.io": {"username"' in hint
        assert "dockerhub" not in hint
        assert "MAD_DOCKERHUB" not in hint

    def test_dockerhub_namespace_still_suggests_dockerhub(self):
        """A bare namespace/repo reference is Docker Hub, not a registry host."""
        log = "Error response from daemon: pull access denied for rocm/private"
        hint = explain_registry_denial(log, "rocm/private:latest")
        assert hint is not None
        assert '"dockerhub"' in hint
        assert "MAD_DOCKERHUB_USER" in hint

    def test_unrelated_failure_returns_none(self):
        """Non-registry build failures produce no hint."""
        assert (
            explain_registry_denial("RUN apt-get install failed: exit code 100") is None
        )

    def test_empty_log_returns_none(self):
        """Empty output produces no hint."""
        assert explain_registry_denial("") is None

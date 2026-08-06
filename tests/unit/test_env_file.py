#!/usr/bin/env python3
"""
Tests for `deployment_config.env_file` loading.

The contract these lock down is the one an operator already relies on when they run
`source mad.env` by hand: the file is real shell, the values it sets win over whatever
was inherited, and a path that does not exist stops the run at submit time instead of
surfacing as an empty `MODEL_DIR` twenty minutes into an allocation.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os

import pytest

from madengine.core.env_file import apply_env_file, load_env_file
from madengine.core.errors import ValidationError


@pytest.fixture
def env_dir(tmp_path):
    """Directory holding env files, standing in for a run directory."""
    return tmp_path


@pytest.fixture(autouse=True)
def restore_environ():
    """Keep `apply_env_file` from leaking into the rest of the suite."""
    saved = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved)


class TestLoadEnvFile:
    """Reading values out of an env file."""

    def test_plain_assignments_are_returned(self, env_dir):
        """`KEY=value` lines come back without needing an explicit export."""
        env_file = env_dir / "mad.env"
        env_file.write_text("MODEL_DIR=/shared/models\nMAD_DOCKER_BUILDS=/shared/builds\n")

        loaded = load_env_file(str(env_file))

        assert loaded["MODEL_DIR"] == "/shared/models"
        assert loaded["MAD_DOCKER_BUILDS"] == "/shared/builds"

    def test_shell_constructs_are_evaluated(self, env_dir):
        """The file is sourced, so expansion and defaults behave as in a shell."""
        env_file = env_dir / "mad.env"
        env_file.write_text(
            'MAD_STORAGE=/shared/mad\n'
            'MAD_DOCKER_BUILDS="$MAD_STORAGE/docker_builds"\n'
            'MAD_PARTITION="${MAD_PARTITION:-gpu}"\n'
        )

        loaded = load_env_file(str(env_file))

        assert loaded["MAD_DOCKER_BUILDS"] == "/shared/mad/docker_builds"
        assert loaded["MAD_PARTITION"] == "gpu"

    def test_inherited_value_wins_over_default(self, env_dir, monkeypatch):
        """A `${VAR:-default}` respects what the caller already exported."""
        monkeypatch.setenv("MAD_PARTITION", "debug")
        env_file = env_dir / "mad.env"
        env_file.write_text('MAD_PARTITION="${MAD_PARTITION:-gpu}"\n')

        loaded = load_env_file(str(env_file))

        # Unchanged relative to the current environment, so nothing to report.
        assert "MAD_PARTITION" not in loaded

    def test_unchanged_variables_are_not_reported(self, env_dir, monkeypatch):
        """Only what the file actually changes is returned."""
        monkeypatch.setenv("MODEL_DIR", "/shared/models")
        env_file = env_dir / "mad.env"
        env_file.write_text("MODEL_DIR=/shared/models\nEXTRA=1\n")

        loaded = load_env_file(str(env_file))

        assert loaded == {"EXTRA": "1"}

    def test_bash_bookkeeping_is_dropped(self, env_dir):
        """`_`, `SHLVL` and friends are the subshell's, not the file's."""
        env_file = env_dir / "mad.env"
        env_file.write_text("MODEL_DIR=/shared/models\n")

        loaded = load_env_file(str(env_file))

        assert set(loaded) == {"MODEL_DIR"}

    def test_values_with_newlines_survive(self, env_dir):
        """NUL-delimited output keeps multi-line values intact."""
        env_file = env_dir / "mad.env"
        env_file.write_text('MAD_BANNER="line one\nline two"\n')

        loaded = load_env_file(str(env_file))

        assert loaded["MAD_BANNER"] == "line one\nline two"

    def test_relative_path_resolves_against_base_dir(self, env_dir):
        """A manifest names its env file relative to itself, not to the CWD."""
        (env_dir / "mad.env").write_text("MODEL_DIR=/shared/models\n")

        loaded = load_env_file("mad.env", base_dir=str(env_dir))

        assert loaded["MODEL_DIR"] == "/shared/models"

    def test_absolute_path_ignores_base_dir(self, env_dir, tmp_path):
        """An absolute path is taken as given."""
        env_file = env_dir / "mad.env"
        env_file.write_text("MODEL_DIR=/shared/models\n")

        loaded = load_env_file(str(env_file), base_dir=str(tmp_path / "elsewhere"))

        assert loaded["MODEL_DIR"] == "/shared/models"

    def test_path_with_spaces_is_quoted(self, env_dir):
        """The path reaches bash as one word."""
        directory = env_dir / "run dir"
        directory.mkdir()
        env_file = directory / "mad.env"
        env_file.write_text("MODEL_DIR=/shared/models\n")

        loaded = load_env_file(str(env_file))

        assert loaded["MODEL_DIR"] == "/shared/models"


class TestEnvFileFailures:
    """A bad env file has to stop the run where it can still be explained."""

    def test_missing_file_names_the_resolved_path(self, env_dir):
        """The error shows where madengine looked, not just what the manifest said."""
        with pytest.raises(ValidationError) as exc_info:
            load_env_file("absent.env", base_dir=str(env_dir))

        assert str(env_dir / "absent.env") in str(exc_info.value)

    def test_directory_is_not_a_file(self, env_dir):
        """Pointing at a directory fails the same way a missing file does."""
        with pytest.raises(ValidationError):
            load_env_file(str(env_dir))

    def test_shell_error_is_reported(self, env_dir):
        """A non-zero exit from bash carries stderr into the message."""
        env_file = env_dir / "mad.env"
        env_file.write_text("exit 3\n")

        with pytest.raises(ValidationError) as exc_info:
            load_env_file(str(env_file))

        assert "Failed to source" in str(exc_info.value)

    def test_syntax_error_is_reported(self, env_dir):
        """Malformed shell is a fatal, named error."""
        env_file = env_dir / "mad.env"
        env_file.write_text('MODEL_DIR="/unterminated\n')

        with pytest.raises(ValidationError):
            load_env_file(str(env_file))


class TestApplyEnvFile:
    """Applying the file to the running process."""

    def test_variables_land_in_environ(self, env_dir):
        """What the file sets is visible to everything downstream."""
        env_file = env_dir / "mad.env"
        env_file.write_text("MAD_DOCKER_BUILDS=/shared/builds\n")

        apply_env_file(str(env_file))

        assert os.environ["MAD_DOCKER_BUILDS"] == "/shared/builds"

    def test_file_wins_over_inherited_value(self, env_dir, monkeypatch):
        """Sourcing overwrites, and so does this."""
        monkeypatch.setenv("MAD_DOCKER_BUILDS", "/tmp/stale")
        env_file = env_dir / "mad.env"
        env_file.write_text("MAD_DOCKER_BUILDS=/shared/builds\n")

        apply_env_file(str(env_file))

        assert os.environ["MAD_DOCKER_BUILDS"] == "/shared/builds"

    def test_unrelated_variables_are_left_alone(self, env_dir, monkeypatch):
        """Loading an env file is additive, not a replacement of the environment."""
        monkeypatch.setenv("MAD_KEEP_ME", "yes")
        env_file = env_dir / "mad.env"
        env_file.write_text("MODEL_DIR=/shared/models\n")

        apply_env_file(str(env_file))

        assert os.environ["MAD_KEEP_ME"] == "yes"

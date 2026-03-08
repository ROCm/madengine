"""Unit tests for deployment base (create_jinja_env)."""

import tempfile
from pathlib import Path

import pytest

from madengine.deployment.base import create_jinja_env


class TestCreateJinjaEnv:
    """Test create_jinja_env helper."""

    def test_returns_environment_with_dirname_basename_filters(self):
        """create_jinja_env returns Environment with dirname and basename filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "test.j2").write_text("dir={{ path | dirname }} name={{ path | basename }}")
            env = create_jinja_env(p)
            template = env.get_template("test.j2")
            out = template.render(path="/foo/bar/baz.txt")
            assert "dir=/foo/bar" in out or "dir=foo/bar" in out
            assert "name=baz.txt" in out

    def test_template_dir_must_exist(self):
        """create_jinja_env works when template_dir exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_jinja_env(Path(tmpdir))
            assert env is not None
            assert env.filters.get("dirname") is not None
            assert env.filters.get("basename") is not None

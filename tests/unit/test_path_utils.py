"""Unit tests for path_utils (scripts_base_dir_from, get_madengine_root)."""

import os
from pathlib import Path

import pytest

from madengine.utils.path_utils import scripts_base_dir_from, get_madengine_root


class TestScriptsBaseDirFrom:
    """Test scripts_base_dir_from helper."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert scripts_base_dir_from(None) is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert scripts_base_dir_from("") is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only string returns None (stripped empty)."""
        assert scripts_base_dir_from("   ") is None

    def test_file_path_returns_parent_dir(self):
        """Path to a file returns its parent directory."""
        assert scripts_base_dir_from("/foo/bar/script.sh") == "/foo/bar"
        assert scripts_base_dir_from("scripts/run.sh") == "scripts"

    def test_dir_path_returns_parent_of_dir(self):
        """Path to a directory returns the parent of that directory."""
        assert scripts_base_dir_from("/foo/bar/baz") == "/foo/bar"

    def test_single_segment_returns_empty_string(self):
        """Single segment (no slash) returns empty string from dirname."""
        assert scripts_base_dir_from("script.sh") == ""


class TestGetMadengineRoot:
    """Test get_madengine_root helper."""

    def test_returns_path(self):
        """Returns a Path instance."""
        root = get_madengine_root()
        assert isinstance(root, Path)

    def test_resolves_to_madengine_package(self):
        """Resolved path ends with 'madengine' (package name)."""
        root = get_madengine_root()
        assert root.name == "madengine"

    def test_path_exists(self):
        """Returned path exists and is a directory."""
        root = get_madengine_root()
        assert root.exists()
        assert root.is_dir()

"""Unit tests for utils: path_utils and run_details."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from madengine.utils.path_utils import scripts_base_dir_from, get_madengine_root
from madengine.utils.run_details import (
    get_pipeline,
    get_build_number,
    flatten_tags_in_place,
)


# ---- path_utils ----

class TestScriptsBaseDirFrom:
    """Test scripts_base_dir_from helper."""

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_falsy_input_returns_none(self, value):
        """None, empty string, or whitespace return None."""
        assert scripts_base_dir_from(value) is None

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

    def test_returns_madengine_package_path(self):
        """Returns a Path that exists, is a directory, and ends with 'madengine'."""
        root = get_madengine_root()
        assert isinstance(root, Path)
        assert root.name == "madengine"
        assert root.exists() and root.is_dir()


# ---- run_details ----

class TestGetPipeline:
    @pytest.mark.parametrize("env_val,expected", [({}, ""), ({"pipeline": "ci-mad"}, "ci-mad")])
    def test_pipeline_from_env_or_default(self, env_val, expected):
        with patch.dict(os.environ, env_val, clear=False):
            if not env_val and "pipeline" in os.environ:
                del os.environ["pipeline"]
            assert get_pipeline() == expected


class TestGetBuildNumber:
    @pytest.mark.parametrize("env_val,expected", [({}, "0"), ({"BUILD_NUMBER": "42"}, "42")])
    def test_build_number_from_env_or_default(self, env_val, expected):
        with patch.dict(os.environ, env_val, clear=False):
            if not env_val and "BUILD_NUMBER" in os.environ:
                del os.environ["BUILD_NUMBER"]
            assert get_build_number() == expected


class TestFlattenTagsInPlace:
    def test_list_converted_to_comma_string(self):
        record = {"tags": ["a", "b", "c"], "other": 1}
        flatten_tags_in_place(record)
        assert record["tags"] == "a,b,c"
        assert record["other"] == 1

    def test_non_list_unchanged(self):
        record = {"tags": "already,a,string"}
        flatten_tags_in_place(record)
        assert record["tags"] == "already,a,string"

    def test_missing_tags_unchanged(self):
        record = {"model": "m1"}
        flatten_tags_in_place(record)
        assert "tags" not in record
        assert record["model"] == "m1"

    def test_empty_list_converted_to_empty_string(self):
        record = {"tags": []}
        flatten_tags_in_place(record)
        assert record["tags"] == ""

"""Unit tests for run_details (get_pipeline, get_build_number, flatten_tags_in_place)."""

import os
from unittest.mock import patch

import pytest

from madengine.utils.run_details import (
    get_pipeline,
    get_build_number,
    flatten_tags_in_place,
)


class TestGetPipeline:
    def test_default_empty(self):
        with patch.dict(os.environ, {}, clear=False):
            if "pipeline" in os.environ:
                del os.environ["pipeline"]
            assert get_pipeline() == ""

    def test_from_env(self):
        with patch.dict(os.environ, {"pipeline": "ci-mad"}, clear=False):
            assert get_pipeline() == "ci-mad"


class TestGetBuildNumber:
    def test_default_zero(self):
        with patch.dict(os.environ, {}, clear=False):
            if "BUILD_NUMBER" in os.environ:
                del os.environ["BUILD_NUMBER"]
            assert get_build_number() == "0"

    def test_from_env(self):
        with patch.dict(os.environ, {"BUILD_NUMBER": "42"}, clear=False):
            assert get_build_number() == "42"


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

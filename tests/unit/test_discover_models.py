"""Unit tests for DiscoverModels tag selection (including scope/tag)."""

import argparse

import pytest

from madengine.utils.discover_models import DiscoverModels


class TestScopedTags:
    """--tags scope/filter limits to scripts/<scope>/ models (name prefix scope/)."""

    def test_scoped_inference_tag_only_models_in_scope(self):
        dm = DiscoverModels(args=argparse.Namespace(tags=["MAD-private/inference"]))
        dm.models = [
            {"name": "other", "tags": ["inference"], "args": ""},
            {"name": "MAD-private/a", "tags": ["inference"], "args": ""},
            {"name": "MAD-private/b", "tags": ["training"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert [m["name"] for m in dm.selected_models] == ["MAD-private/a"]

    def test_scoped_all_selects_every_model_in_scope(self):
        dm = DiscoverModels(args=argparse.Namespace(tags=["sub/all"]))
        dm.models = [
            {"name": "sub/x", "tags": [], "args": ""},
            {"name": "sub/y", "tags": [], "args": ""},
            {"name": "other/z", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == ["sub/x", "sub/y"]

    def test_scoped_select_by_short_model_name(self):
        dm = DiscoverModels(args=argparse.Namespace(tags=["sub/myname"]))
        dm.models = [
            {"name": "sub/myname", "tags": [], "args": ""},
            {"name": "sub/other", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert [m["name"] for m in dm.selected_models] == ["sub/myname"]

    def test_unscoped_inference_still_matches_all_repos(self):
        dm = DiscoverModels(args=argparse.Namespace(tags=["inference"]))
        dm.models = [
            {"name": "root", "tags": ["inference"], "args": ""},
            {"name": "MAD-private/x", "tags": ["inference"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == [
            "MAD-private/x",
            "root",
        ]

    def test_colon_in_tag_not_treated_as_scoped(self):
        """model:arg keeps legacy behavior (no scope/tag split on /)."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["MAD-private/foo:batch-size=32"]))
        dm.models = [
            {"name": "MAD-private/foo", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert len(dm.selected_models) == 1
        assert dm.selected_models[0]["name"] == "MAD-private/foo"
        assert "batch-size 32" in dm.selected_models[0]["args"]

    def test_scoped_no_match_raises(self):
        dm = DiscoverModels(args=argparse.Namespace(tags=["sub/unknown"]))
        dm.models = [{"name": "sub/x", "tags": ["a"], "args": ""}]
        dm.custom_models = []
        with pytest.raises(ValueError, match="unknown"):
            dm.select_models()

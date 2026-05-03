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
        dm = DiscoverModels(
            args=argparse.Namespace(tags=["MAD-private/foo:batch-size=32"])
        )
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


class TestUnscopedTagSelection:
    """Unscoped --tags: name-based matching is root-only (no scope prefix crossing),
    but tag-field matching is scope-agnostic and can select models in any scope."""

    def test_unscoped_tag_matches_root_model_by_name(self):
        """--tags pyt_foo matches a root-level model named exactly pyt_foo."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["pyt_foo"]))
        dm.models = [
            {"name": "pyt_foo", "tags": [], "args": ""},
            {"name": "pyt_bar", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert [m["name"] for m in dm.selected_models] == ["pyt_foo"]

    def test_unscoped_tag_matches_by_tag_field(self):
        """--tags inference selects all root-level models with inference in their tags field."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["inference"]))
        dm.models = [
            {"name": "pyt_foo", "tags": ["inference"], "args": ""},
            {"name": "pyt_bar", "tags": ["training"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert [m["name"] for m in dm.selected_models] == ["pyt_foo"]

    def test_unscoped_tag_does_not_cross_scope_boundary(self):
        """--tags pyt_foo must NOT match MAD/pyt_foo — scopes are not crossed."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["pyt_foo"]))
        dm.models = [
            {"name": "MAD/pyt_foo", "tags": ["inference"], "args": ""},
        ]
        dm.custom_models = []
        with pytest.raises(ValueError):
            dm.select_models()

    def test_unscoped_tag_matches_scoped_models_by_tag_field(self):
        """--tags inference matches any model carrying that tag, regardless of scope prefix.
        Tag-list matching is always scope-agnostic; only name-based matching is scope-strict.
        """
        dm = DiscoverModels(args=argparse.Namespace(tags=["inference"]))
        dm.models = [
            {"name": "MAD/pyt_foo", "tags": ["inference"], "args": ""},
            {"name": "MAD/pyt_bar", "tags": ["inference"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == [
            "MAD/pyt_bar",
            "MAD/pyt_foo",
        ]

    def test_unscoped_all_selects_every_model(self):
        """--tags all selects every model regardless of scope."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["all"]))
        dm.models = [
            {"name": "pyt_foo", "tags": [], "args": ""},
            {"name": "MAD/pyt_bar", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == [
            "MAD/pyt_bar",
            "pyt_foo",
        ]

    def test_unscoped_tag_matches_root_and_scoped_by_tag_field(self):
        """--tags inference selects root AND scoped models that carry that tag."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["inference"]))
        dm.models = [
            {"name": "root_model", "tags": ["inference"], "args": ""},
            {"name": "MAD/pyt_foo", "tags": ["inference"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == [
            "MAD/pyt_foo",
            "root_model",
        ]

    def test_unscoped_tag_with_extra_args_matches_by_tag_field(self):
        """--tags inference:batch-size=32 selects by tag 'inference', not 'inference:batch-size=32'."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["inference:batch-size=32"]))
        dm.models = [
            {"name": "pyt_foo", "tags": ["inference"], "args": ""},
            {"name": "pyt_bar", "tags": ["training"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert len(dm.selected_models) == 1
        assert dm.selected_models[0]["name"] == "pyt_foo"
        assert "--batch-size 32" in dm.selected_models[0]["args"]

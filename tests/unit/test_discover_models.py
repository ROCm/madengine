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


class TestShortNameBackwardCompat:
    """Short-name (unscoped) matching resolves dir-prefixed model names for backward compat.

    After migrating from root models.json to per-directory models.json, model names gain
    a directory prefix (e.g., ``pyt_foo`` becomes ``dir/pyt_foo``). Users should still
    be able to reference models by their original flat name via ``--tags pyt_foo``.
    """

    def test_short_name_matches_dir_prefixed_model(self):
        """--tags pyt_foo resolves dir/pyt_foo even without 'pyt_foo' in the tags list."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["pyt_foo"]))
        dm.models = [
            {"name": "dir/pyt_foo", "tags": ["inference"], "args": ""},
            {"name": "dir/pyt_bar", "tags": ["training"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert [m["name"] for m in dm.selected_models] == ["dir/pyt_foo"]

    def test_flat_name_unaffected(self):
        """--tags foo still resolves a root (non-prefixed) model named 'foo'."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["foo"]))
        dm.models = [
            {"name": "foo", "tags": [], "args": ""},
            {"name": "bar", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert [m["name"] for m in dm.selected_models] == ["foo"]

    def test_short_name_matches_custom_model(self):
        """Short-name matching also works for custom models from get_models_json.py."""
        from madengine.utils.discover_models import CustomModel

        dm = DiscoverModels(args=argparse.Namespace(tags=["my_model"]))
        dm.models = []
        cm = CustomModel(
            name="mydir/my_model",
            dockerfile="../../docker/mydir",
            scripts="run.sh",
            tags=["perf"],
        )
        dm.custom_models = [cm]
        dm.select_models()
        assert len(dm.selected_models) == 1
        assert dm.selected_models[0]["name"] == "mydir/my_model"

    def test_scoped_tag_unaffected_by_short_name_matching(self):
        """A scoped tag dir/model_name selects only that model, not other dirs' models
        with the same short name."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["dirA/pyt_foo"]))
        dm.models = [
            {"name": "dirA/pyt_foo", "tags": [], "args": ""},
            {"name": "dirB/pyt_foo", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert [m["name"] for m in dm.selected_models] == ["dirA/pyt_foo"]

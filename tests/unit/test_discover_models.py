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

    def test_unscoped_tag_matches_scoped_model_by_short_name(self):
        """--tags pyt_foo matches MAD/pyt_foo via short-name backward-compat matching.

        Name-based matching now also falls back to the short name (the part after the
        last '/'), so a scoped model can still be reached by its unscoped short name.
        See TestShortNameBackwardCompat for dedicated coverage of this behavior.
        """
        dm = DiscoverModels(args=argparse.Namespace(tags=["pyt_foo"]))
        dm.models = [
            {"name": "MAD/pyt_foo", "tags": ["inference"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert [m["name"] for m in dm.selected_models] == ["MAD/pyt_foo"]

    def test_unscoped_tag_matches_scoped_models_by_tag_field(self):
        """--tags inference matches any model carrying that tag, regardless of scope prefix.
        Tag-list matching is always scope-agnostic; only name-based matching is scope-strict."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["inference"]))
        dm.models = [
            {"name": "MAD/pyt_foo", "tags": ["inference"], "args": ""},
            {"name": "MAD/pyt_bar", "tags": ["inference"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == ["MAD/pyt_bar", "MAD/pyt_foo"]

    def test_unscoped_all_selects_every_model(self):
        """--tags all selects every model regardless of scope."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["all"]))
        dm.models = [
            {"name": "pyt_foo", "tags": [], "args": ""},
            {"name": "MAD/pyt_bar", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == ["MAD/pyt_bar", "pyt_foo"]

    def test_unscoped_tag_matches_root_and_scoped_by_tag_field(self):
        """--tags inference selects root AND scoped models that carry that tag."""
        dm = DiscoverModels(args=argparse.Namespace(tags=["inference"]))
        dm.models = [
            {"name": "root_model", "tags": ["inference"], "args": ""},
            {"name": "MAD/pyt_foo", "tags": ["inference"], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == ["MAD/pyt_foo", "root_model"]

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

    def test_ambiguous_short_name_matches_both_flat_and_prefixed_model(self):
        """When both foo and dir/foo exist, --tags foo currently matches both.

        Short-name matching is purely additive alongside exact-name matching, so there
        is no precedence between an exact match and a short-name match; this pins the
        current (both-selected) behavior for the ambiguous case.
        """
        dm = DiscoverModels(args=argparse.Namespace(tags=["foo"]))
        dm.models = [
            {"name": "foo", "tags": [], "args": ""},
            {"name": "dir/foo", "tags": [], "args": ""},
            {"name": "bar", "tags": [], "args": ""},
        ]
        dm.custom_models = []
        dm.select_models()
        assert sorted(m["name"] for m in dm.selected_models) == ["dir/foo", "foo"]

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


class TestNestedSubmoduleDiscovery:
    """Test discovery of models from nested submodule structures (e.g., scripts/Model-Repo1/category/)."""

    def test_nested_submodule_discovery(self, tmp_path, monkeypatch):
        """Discover models from nested submodule directory structure."""
        import os
        import json

        # Create nested directory structure simulating a git submodule
        scripts_dir = tmp_path / "scripts" / "Model-Repo1" / "category1"
        scripts_dir.mkdir(parents=True)

        # Create models.json in nested directory
        models_json = scripts_dir / "models.json"
        models_json.write_text(json.dumps([
            {
                "name": "model1",
                "dockerfile": "../../docker/dummy",
                "scripts": "run.sh",
                "n_gpus": "1",
                "tags": ["category1", "test"],
                "args": ""
            }
        ]))

        # Create root models.json
        root_models = tmp_path / "models.json"
        root_models.write_text("[]")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Discover models
        dm = DiscoverModels(args=argparse.Namespace(tags=None))
        dm.discover_models()

        # Verify model was discovered with correct path
        assert len(dm.models) == 1
        model = dm.models[0]
        assert model["name"] == "Model-Repo1/category1/model1"
        assert model["dockerfile"] == os.path.normpath("scripts/Model-Repo1/category1/../../docker/dummy")
        assert model["scripts"] == os.path.normpath("scripts/Model-Repo1/category1/run.sh")
        assert "category1" in model["tags"]

    def test_scoped_tag_selects_nested_submodule_models(self, tmp_path, monkeypatch):
        """Scoped tag Model-Repo1/category1 selects models from scripts/Model-Repo1/category1/ by tag."""
        import os
        import json

        # Create nested directory structure
        scripts_dir = tmp_path / "scripts" / "Model-Repo1" / "category1"
        scripts_dir.mkdir(parents=True)

        models_json = scripts_dir / "models.json"
        models_json.write_text(json.dumps([
            {
                "name": "model1",
                "dockerfile": "../../docker/dummy",
                "scripts": "run.sh",
                "n_gpus": "1",
                "tags": ["category1"],
                "args": ""
            },
            {
                "name": "model2",
                "dockerfile": "../../docker/dummy",
                "scripts": "run.sh",
                "n_gpus": "2",
                "tags": ["category1"],
                "args": ""
            }
        ]))

        root_models = tmp_path / "models.json"
        root_models.write_text("[]")

        monkeypatch.chdir(tmp_path)

        # Discover and select with scoped tag
        dm = DiscoverModels(args=argparse.Namespace(tags=["Model-Repo1/category1"]))
        dm.discover_models()
        dm.select_models()

        # Should select both models from the nested directory
        assert len(dm.selected_models) == 2
        names = sorted(m["name"] for m in dm.selected_models)
        assert names == ["Model-Repo1/category1/model1", "Model-Repo1/category1/model2"]

    def test_multiple_nested_submodules(self, tmp_path, monkeypatch):
        """Discover models from multiple nested submodule directories."""
        import json

        # Create Model-Repo1/category1
        repo1_dir = tmp_path / "scripts" / "Model-Repo1" / "category1"
        repo1_dir.mkdir(parents=True)
        (repo1_dir / "models.json").write_text(json.dumps([
            {"name": "m1", "dockerfile": "../../docker/dummy", "scripts": "run.sh", "tags": ["category1"], "args": ""}
        ]))

        # Create Model-Repo2/inference
        repo2_dir = tmp_path / "scripts" / "Model-Repo2" / "inference"
        repo2_dir.mkdir(parents=True)
        (repo2_dir / "models.json").write_text(json.dumps([
            {"name": "m2", "dockerfile": "../../docker/dummy", "scripts": "run.sh", "tags": ["inference"], "args": ""}
        ]))

        (tmp_path / "models.json").write_text("[]")
        monkeypatch.chdir(tmp_path)

        dm = DiscoverModels(args=argparse.Namespace(tags=None))
        dm.discover_models()

        # Should discover both models
        assert len(dm.models) == 2
        names = sorted(m["name"] for m in dm.models)
        assert names == ["Model-Repo1/category1/m1", "Model-Repo2/inference/m2"]

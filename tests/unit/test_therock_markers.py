"""
Unit tests for TheRock on-disk marker helpers.
"""

from pathlib import Path

import pytest

from madengine.utils.therock_markers import (
    is_therock_tree,
    therock_dist_info_path,
    therock_manifest_path,
)


@pytest.mark.unit
def test_is_therock_tree_manifest_only(tmp_path: Path) -> None:
    p = tmp_path / "share" / "therock" / "therock_manifest.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}", encoding="utf-8")
    assert is_therock_tree(tmp_path)


@pytest.mark.unit
def test_is_therock_tree_false_without_markers(tmp_path: Path) -> None:
    assert not is_therock_tree(tmp_path)


@pytest.mark.unit
def test_relpath_helpers_match_expected_layout() -> None:
    root = Path("/opt/rocm-7.0.0")
    assert (
        therock_manifest_path(root)
        == root / "share" / "therock" / "therock_manifest.json"
    )
    assert therock_dist_info_path(root) == root / "share" / "therock" / "dist_info.json"

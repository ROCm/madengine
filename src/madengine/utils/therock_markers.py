# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
r"""TheRock on-disk markers under a ROCm (or TheRock) root.

These paths are the **minimal** "file marker" test used for ROCm path resolution
(:mod:`madengine.utils.rocm_path_resolver`) and for richer detection in
``scripts/common/tools/therock_detector.py``. TheRock may also be inferable
from symlinks, binaries, or Python packages; those are handled elsewhere.

**Canonical tree** (at least one file present):

* ``<root>/share/therock/therock_manifest.json``
* ``<root>/share/therock/dist_info.json``
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

# Relative to a putative ROCm / TheRock install root.
THEROCK_SHARE_DIR: Final = Path("share") / "therock"
THEROCK_MANIFEST_RELPATH: Final = THEROCK_SHARE_DIR / "therock_manifest.json"
THEROCK_DIST_INFO_RELPATH: Final = THEROCK_SHARE_DIR / "dist_info.json"


def therock_manifest_path(root: Path) -> Path:
    return root / THEROCK_MANIFEST_RELPATH


def therock_dist_info_path(root: Path) -> Path:
    return root / THEROCK_DIST_INFO_RELPATH


def is_therock_tree(root: Path) -> bool:
    """True if the tree has standard TheRock ``share/therock`` file markers."""
    m = therock_manifest_path(root)
    d = therock_dist_info_path(root)
    return m.is_file() or d.is_file()

"""Path resolution helpers for madengine.

Provides scripts_base_dir_from() and get_madengine_root() for consistent
path handling across execution and deployment.
"""

import os
from pathlib import Path
from typing import Optional


def scripts_base_dir_from(scripts_path: Optional[str]) -> Optional[str]:
    """Return the directory containing the given scripts path, or None.

    Args:
        scripts_path: Path to a script file (or directory). Can be None or empty.

    Returns:
        Parent directory as string, or None if scripts_path is None/empty.
    """
    if not scripts_path or not str(scripts_path).strip():
        return None
    return os.path.dirname(scripts_path)


def get_madengine_root() -> Path:
    """Return the madengine package root directory (where madengine/__init__.py lives).

    Returns:
        Path to the madengine package root.
    """
    import madengine
    return Path(madengine.__file__).resolve().parent

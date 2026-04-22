# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
r"""ROCm root resolution: automatic discovery (TheRock + traditional) and overrides.

This module exposes a :class:`RocmPathResolver` (single place for detection order and
injectable dependencies for tests) plus small module-level functions for backward
compatibility. The resolution pipeline is **not** the historical ``rocEnvTool``/v1
``RocmPathResolver`` from madenginev1; it reuses the same *ideas* in a self-contained
helper for madengine (versioned ``/opt/rocm-*``, TheRock markers, ``PATH`` tools).
See **ADR 0001** in ``docs/adr/0001-rocm-path-resolution.md`` for the full design.

Versioned installs (e.g. ``/opt/rocm-7.13.0`` with ``bin/amd-smi`` and ``bin/rocm-smi``)
are detected after ``/opt/rocm``. ``which(amd-smi)`` / ``which(rocm-smi)`` infer
the root when that tree validates (skipping a Python-only ``/opt/python`` stub that
fails the root check).

**Host** overrides (precedence):

1. Top-level ``MAD_ROCM_PATH`` in additional context.
2. ``--rocm-path`` (CLI) — host-only; same meaning as (1).
3. Auto-detection (when ``MAD_AUTO_ROCM_PATH`` is not ``"0"``).
4. ``ROCM_PATH`` environment variable.
5. Default ``/opt/rocm``.

**Container** (``docker_env_vars``):

* ``MAD_ROCM_PATH`` — container ROCm root; sets ``ROCM_PATH`` for Docker; key is
  consumed and not passed to the workload.
* If only ``ROCM_PATH`` is set in ``docker_env_vars`` by the user, it is kept.
* Otherwise the host-resolved path is mirrored into ``ROCM_PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .therock_markers import is_therock_tree

#: Key in top-level context and ``docker_env_vars`` for ROCm root overrides.
MAD_ROCM_PATH = "MAD_ROCM_PATH"

OptionalPathStr = Optional[str]
WhichFn = Callable[..., Optional[str]]


def normalize_rocm_path(path: str) -> str:
    """Return an absolute path with no trailing separator."""
    p = os.path.abspath(os.path.expanduser((path or "").strip()))
    return p.rstrip(os.sep) or os.sep


def _environ_get(environ: Mapping[str, str], key: str, default: str = "") -> str:
    return (environ.get(key) or default).strip()


class RocmPathResolver:
    """Resolves a ROCm installation root using a fixed, documented probe order.

    * ``environ`` defaults to :data:`os.environ` for ``ROCM_*`` / ``HIP_*`` checks.
    * ``which`` defaults to :func:`shutil.which` and is used to locate tools on ``PATH``;
      tests may pass a custom callable to simulate ``which`` without touching the
      process environment.
    """

    __slots__ = ("_environ", "_which")

    def __init__(
        self,
        environ: Optional[Mapping[str, str]] = None,
        which: Optional[WhichFn] = None,
    ) -> None:
        self._environ: Mapping[str, str] = (
            os.environ if environ is None else environ
        )
        self._which: WhichFn = which if which is not None else shutil.which

    @staticmethod
    def rocm_root_from_bin_tool(tool_path: str) -> Optional[Path]:
        """Map .../<root>/bin/{rocminfo,amd-smi,rocm-smi} to <root>."""
        p = Path(tool_path).resolve()
        if p.name not in ("rocminfo", "amd-smi", "rocm-smi"):
            return None
        if p.parent.name != "bin":
            return None
        return p.parent.parent

    def looks_like_rocm_root(self, root: Path) -> bool:
        if not root.is_dir():
            return False
        if is_therock_tree(root):
            return True
        if (root / "bin" / "rocminfo").is_file():
            return True
        # Versioned apt/tar layouts (e.g. /opt/rocm-7.13.0) and many TheRock images
        if (root / "bin" / "amd-smi").is_file() and (root / "bin" / "rocm-smi").is_file():
            return True
        if (root / "bin" / "rocm-smi").is_file() and (root / ".info" / "version").is_file():
            return True
        if (root / "bin" / "amd-smi").is_file() and (root / ".info" / "version").is_file():
            return True
        if (root / ".info" / "version").is_file():
            return True
        return False

    def versioned_opt_rocm_dirs(self) -> List[Path]:
        """Return ``/opt/rocm-*`` versioned directories (e.g. ``/opt/rocm-7.13.0``)."""
        try:
            opt = Path("/opt")
            if not opt.is_dir():
                return []
            return sorted(p for p in opt.glob("rocm-*") if p.is_dir())
        except OSError:
            return []

    def infer_from_path_tools(self) -> OptionalPathStr:
        """Use ``which`` on rocminfo, amd-smi, rocm-smi; return first plausible root."""
        from madengine.utils import rocm_path_resolver as m  # same module; for patch hooks

        for name in ("rocminfo", "amd-smi", "rocm-smi"):
            w = self._which(name)  # type: ignore[operator]
            if not w:
                continue
            inferred = self.rocm_root_from_bin_tool(w)
            if inferred is not None and m._looks_like_rocm_root(inferred):
                return normalize_rocm_path(str(inferred))
        return None

    def auto_detect(self) -> OptionalPathStr:
        """Heuristic search for a usable ROCm installation (see class doc + module doc)."""
        from madengine.utils import rocm_path_resolver as m  # same module; for patch hooks

        opt = Path("/opt/rocm")
        if m._looks_like_rocm_root(opt):
            return normalize_rocm_path(str(opt))

        for cand in m._versioned_opt_rocm_dirs():
            if m._looks_like_rocm_root(cand):
                return normalize_rocm_path(str(cand))

        if self._which("rocm-sdk"):
            try:
                r = subprocess.run(
                    ["rocm-sdk", "path", "--root"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if r.returncode == 0 and r.stdout.strip():
                    root = Path(r.stdout.strip())
                    if m._looks_like_rocm_root(root):
                        return normalize_rocm_path(str(root))
            except (OSError, subprocess.SubprocessError):
                pass

        for var in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
            raw = _environ_get(self._environ, var, "")
            if not raw:
                continue
            root = Path(os.path.expanduser(raw))
            if not root.is_dir():
                continue
            if m._looks_like_rocm_root(root):
                return normalize_rocm_path(str(root))
            for _ in range(4):
                if m._looks_like_rocm_root(root):
                    return normalize_rocm_path(str(root))
                if root == root.parent:
                    break
                root = root.parent

        candidates: List[Path] = [
            Path("/usr/local/rocm"),
            Path.home() / "rocm",
            Path.home() / "therock",
        ]
        for c in candidates:
            if m._looks_like_rocm_root(c):
                return normalize_rocm_path(str(c))

        found = m._infer_root_from_path_tools()
        if found:
            return found

        return None


# --- Module-level hooks: tests may monkeypatch ``_looks_like_rocm_root``,
# ``_versioned_opt_rocm_dirs``, ``_infer_root_from_path_tools``, and related names.
# Implementation delegates to :class:`RocmPathResolver` (default instance) for the logic.


def _looks_like_rocm_root(root: Path) -> bool:
    return RocmPathResolver().looks_like_rocm_root(root)


def _rocm_root_from_bin_tool(tool_path: str) -> Optional[Path]:
    return RocmPathResolver.rocm_root_from_bin_tool(tool_path)


def _versioned_opt_rocm_dirs() -> List[Path]:
    return RocmPathResolver().versioned_opt_rocm_dirs()


def _infer_root_from_path_tools() -> OptionalPathStr:
    return RocmPathResolver().infer_from_path_tools()


def auto_detect_rocm_path() -> OptionalPathStr:
    return RocmPathResolver().auto_detect()


def get_rocm_path_legacy(override: Optional[str] = None) -> str:
    """No auto-detect: CLI override, then ``ROCM_PATH`` env, then default."""
    if override and str(override).strip():
        return normalize_rocm_path(str(override).strip())
    e = os.environ.get("ROCM_PATH", "").strip()
    if e:
        return normalize_rocm_path(e)
    return normalize_rocm_path("/opt/rocm")


def resolve_host_rocm_path(
    ctx: Optional[Dict[str, Any]] = None,
    cli_rocm_path: Optional[str] = None,
) -> str:
    """
    Resolve the host ROCm installation root for madengine (GPU tools, validation).

    See module docstring for precedence.
    """
    ctx = ctx or {}
    mad = ctx.get(MAD_ROCM_PATH)
    if isinstance(mad, str) and mad.strip():
        return normalize_rocm_path(mad)
    if cli_rocm_path and str(cli_rocm_path).strip():
        return normalize_rocm_path(str(cli_rocm_path).strip())

    if os.environ.get("MAD_AUTO_ROCM_PATH", "1").strip() == "0":
        return get_rocm_path_legacy(None)

    found = auto_detect_rocm_path()
    if found:
        return found

    return get_rocm_path_legacy(None)


def resolve_container_rocm_path(docker_env: Dict[str, Any], host_path: str) -> str:
    """
    Set ``docker_env['ROCM_PATH']`` and return the container ROCm root.

    * If ``MAD_ROCM_PATH`` is in ``docker_env``, use it, remove the key, set ``ROCM_PATH``.
    * Elif ``ROCM_PATH`` already set, normalize and return (no host mirror).
    * Else mirror ``host_path``.
    """
    d = docker_env
    if MAD_ROCM_PATH in d and d[MAD_ROCM_PATH] is not None:
        s = d.pop(MAD_ROCM_PATH)
        if isinstance(s, (int, float)):
            s = str(s)
        if not isinstance(s, str) or not str(s).strip():
            s = host_path
        croot = normalize_rocm_path(str(s).strip())
    elif d.get("ROCM_PATH") not in (None, ""):
        croot = normalize_rocm_path(str(d["ROCM_PATH"]).strip())
    else:
        croot = host_path
    d["ROCM_PATH"] = croot
    return croot

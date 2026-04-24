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
2. Auto-detection (when ``MAD_AUTO_ROCM_PATH`` is not ``"0"``).
3. ``ROCM_PATH`` environment variable.
4. Default ``/opt/rocm``.

**Container** (``docker_env_vars``) — set when the run uses Docker (``run_container``):

* ``MAD_ROCM_PATH`` — in-image ROCm root; the key is consumed and mapped to ``ROCM_PATH``.
* If the user set ``ROCM_PATH`` in ``docker_env_vars`` in additional context, it is kept
  (normalized) until the run; it wins over OCI / probe in :func:`finalize_container_rocm_path`
  if already present after :func:`apply_container_rocm_path_overrides`.
* Full in-container resolution order is implemented in
  :func:`finalize_container_rocm_path` (not host mirroring): (1) existing ``ROCM_PATH``
  in ``docker_env`` after overrides, (2) ``ROCM_PATH`` / ``ROCM_HOME`` from
  ``docker image inspect`` OCI ``Config.Env``, (3) ``docker run --rm`` with an
  in-image shell probe aligned with :class:`RocmPathResolver` heuristics, (4) default
  ``/opt/rocm`` with a warning.
"""

from __future__ import annotations

import json
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
) -> str:
    """
    Resolve the host ROCm installation root for madengine (GPU tools, validation).

    See module docstring for precedence.
    """
    ctx = ctx or {}
    mad = ctx.get(MAD_ROCM_PATH)
    if isinstance(mad, str) and mad.strip():
        return normalize_rocm_path(mad)

    if os.environ.get("MAD_AUTO_ROCM_PATH", "1").strip() == "0":
        return get_rocm_path_legacy(None)

    found = auto_detect_rocm_path()
    if found:
        return found

    return get_rocm_path_legacy(None)


# POSIX ``/bin/sh`` script run *inside* the target image to infer ROCm root (TheRock +
# traditional trees). Kept in sync with :class:`RocmPathResolver` intent.
_CONTAINER_PROBE_SH = r"""try_root() {
  d="$1"
  [ -d "$d" ] || return 1
  { [ -f "$d/bin/rocminfo" ] || [ -f "$d/bin/amd-smi" ] || [ -f "$d/share/therock/therock_manifest.json" ] || [ -f "$d/share/therock/dist_info.json" ]; } && return 0
  return 1
}
if command -v rocm-sdk >/dev/null 2>&1; then
  r=$(rocm-sdk path --root 2>/dev/null)
  if [ -n "$r" ] && try_root "$r"; then
    printf %s "$r"
    exit 0
  fi
fi
if try_root /opt/rocm; then
  printf %s /opt/rocm
  exit 0
fi
for d in /opt/rocm-*/; do
  d=${d%/}
  [ -d "$d" ] && try_root "$d" && { printf %s "$d"; exit 0; }
done
for d in /opt/*/; do
  d=${d%/}
  case $d in /opt/rocm) continue ;; esac
  [ -d "$d" ] && try_root "$d" && { printf %s "$d"; exit 0; }
done
if [ -d /usr/local/rocm ] && try_root /usr/local/rocm; then
  printf %s /usr/local/rocm
  exit 0
fi
exit 1
"""


def rocm_path_from_docker_image_config(docker_image: str) -> Optional[str]:
    """Read ``ROCM_PATH`` (else ``ROCM_HOME``) from the image OCI config via ``docker image inspect``."""
    if not (docker_image or "").strip():
        return None
    try:
        proc = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                docker_image.strip(),
                "--format",
                "{{json .Config.Env}}",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.SubprocessError):
        return None
    try:
        env_list: Any = json.loads((proc.stdout or "").strip() or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(env_list, list):
        return None
    for key in ("ROCM_PATH", "ROCM_HOME"):
        for entry in env_list:
            if not isinstance(entry, str):
                continue
            if entry.startswith(f"{key}="):
                val = entry.split("=", 1)[1].strip()
                if val:
                    return val
    return None


def infer_rocm_path_via_docker_run(
    docker_image: str,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    One-shot ``docker run --rm`` in-image probe using :data:`_CONTAINER_PROBE_SH`.
    """
    _print = log or print
    if not (docker_image or "").strip():
        return None
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "/bin/sh",
        docker_image.strip(),
        "-c",
        _CONTAINER_PROBE_SH,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as e:
        _print(
            f"Warning: in-container ROCm path probe (docker run) failed for "
            f"'{docker_image}': {e}"
        )
        return None
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        if err:
            _print(
                f"Warning: in-container ROCm path probe failed (exit {proc.returncode}): "
                f"{err[:500]}"
            )
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    line = out.splitlines()[0].strip()
    if not line:
        return None
    return line


def apply_container_rocm_path_overrides(
    docker_env: Dict[str, Any], host_path: str
) -> Optional[str]:
    """
    Map ``MAD_ROCM_PATH`` to ``ROCM_PATH`` and normalize a user ``ROCM_PATH`` in
    ``docker_env``. Does **not** set ``ROCM_PATH`` from the host; see
    :func:`finalize_container_rocm_path` for run-time resolution.

    Returns the resolved path if one was set, else ``None``.
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
        return None
    d["ROCM_PATH"] = croot
    return croot


def finalize_container_rocm_path(
    docker_env: Dict[str, Any],
    docker_image: str,
    host_path: str,
    *,
    log: Callable[[str], None] = print,
) -> str:
    """
    Set ``docker_env['ROCM_PATH']`` for container workloads (AMD Docker runs).

    Precedence:

    1. If ``ROCM_PATH`` is already in ``docker_env`` (e.g. from
       :func:`apply_container_rocm_path_overrides`), use it.
    2. Else if OCI :envvar:`ROCM_PATH` / :envvar:`ROCM_HOME` is set on the image, use
       that (``docker image inspect``).
    3. Else log a short warning, run the in-image shell probe, use its output if
       successful.
    4. Else default to ``/opt/rocm`` and log a warning.

    ``host_path`` is reserved for callers that need a fallback; it is **not** used
    for mirroring unless future policy adds an explicit opt-in.
    """
    d = docker_env
    # Re-run overrides so ``docker_env_vars`` merged in ``run_container`` (late) still
    # maps ``MAD_ROCM_PATH`` to ``ROCM_PATH`` and consumes the ``MAD_`` key.
    apply_container_rocm_path_overrides(d, host_path)

    existing = d.get("ROCM_PATH")
    if existing not in (None, ""):
        croot = normalize_rocm_path(str(existing).strip())
        d["ROCM_PATH"] = croot
        return croot

    oci = rocm_path_from_docker_image_config(docker_image)
    if oci:
        croot = normalize_rocm_path(oci)
        d["ROCM_PATH"] = croot
        log(
            f"ROCm container ROCM_PATH from image OCI config ({docker_image}): {croot}"
        )
        return croot

    log(
        f"Warning: image {docker_image!r} has no ROCM_PATH/ROCM_HOME in OCI Config.Env; "
        f"probing in-container for ROCm root (docker run --rm). "
        f"Set docker_env_vars.MAD_ROCM_PATH to skip."
    )
    probed = infer_rocm_path_via_docker_run(docker_image, log=log)
    if probed:
        croot = normalize_rocm_path(probed)
        d["ROCM_PATH"] = croot
        log(f"ROCm container ROCM_PATH from in-image probe: {croot}")
        return croot

    croot = normalize_rocm_path("/opt/rocm")
    d["ROCM_PATH"] = croot
    log(
        "Warning: could not infer ROCm in container; defaulting ROCM_PATH to "
        f"{croot} (set docker_env_vars.MAD_ROCM_PATH if wrong)."
    )
    return croot


def resolve_container_rocm_path(docker_env: Dict[str, Any], host_path: str) -> str:
    """
    Apply :func:`apply_container_rocm_path_overrides` only (used from
    :class:`~madengine.core.context.Context` init). Does not mirror the host
    or run Docker inspect/probe. Run-time finalization is
    :func:`finalize_container_rocm_path`.

    For backward compatibility, returns the in-``docker_env`` ``ROCM_PATH`` if set,
    else an empty string.
    """
    out = apply_container_rocm_path_overrides(docker_env, host_path)
    if out is not None:
        return out
    return ""

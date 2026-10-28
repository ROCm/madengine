"""Configure Podman behind the existing Docker command path.

The local runner already shells out to the Docker CLI. When
``additional_context["platform"]`` is ``"podman"``, this module puts a
short-lived ``docker`` shim ahead of the real Docker binary on ``PATH``.
Docker and bare-metal calls are otherwise unchanged.
"""

import atexit
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from madengine.core.errors import ConfigurationError

_shim_dir: Optional[Path] = None
_path_before_shim: Optional[str] = None


def configure_container_cli(platform: str) -> None:
    """Configure the local Docker command path for an execution platform.

    Args:
        platform: A resolved ``platform`` value.

    Raises:
        ConfigurationError: Podman is requested but unavailable.
    """
    global _path_before_shim, _shim_dir
    if platform != "podman":
        reset_container_cli()
        return
    if _shim_dir is not None:
        return

    podman = shutil.which("podman")
    if not podman:
        raise ConfigurationError(
            "podman is not on PATH. Install the podman OS package; it is not a "
            "Python dependency.",
            suggestions=["Confirm with: podman version"],
        )

    _path_before_shim = os.environ.get("PATH", "")
    _shim_dir = Path(tempfile.mkdtemp(prefix="madengine-podman-"))
    shim = _shim_dir / "docker"
    podman_command = shlex.quote(podman)
    shim.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${{CONTAINER_HOST:-}}" && -S /run/podman/podman.sock ]]; then
    export CONTAINER_HOST=unix:///run/podman/podman.sock
fi
# Podman manifest inspect of a normal image omits the digest field. The
# builder reads the first "digest" value, so print the image digest in that
# shape before the remaining arguments are forwarded.
if [[ "${{1:-}}" == "manifest" && "${{2:-}}" == "inspect" ]]; then
    digest="$({podman_command} image inspect --format '{{{{.Digest}}}}' "${{3:-}}" 2>/dev/null || true)"
    if [[ -z "$digest" || "$digest" == "<no value>" ]]; then
        image_id="$({podman_command} image inspect --format '{{{{.Id}}}}' "${{3:-}}")"
        digest="sha256:${{image_id#sha256:}}"
    fi
    printf '"digest": "%s"\\n' "$digest"
    exit 0
fi
args=()
while (($#)); do
    if [[ "$1" == "--security-opt" && "${{2:-}}" == "apparmor=unconfined" ]]; then
        shift 2
    elif [[ "$1" == name=^/*\\$ ]]; then
        args+=("name=^/?${{1#name=^/}}")
        shift
    else
        args+=("$1")
        shift
    fi
done
exec {podman_command} "${{args[@]}}"
""",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    os.environ["PATH"] = f"{_shim_dir}{os.pathsep}{_path_before_shim}"


def reset_container_cli() -> None:
    """Restore the process ``PATH`` after a Podman configuration."""
    global _path_before_shim, _shim_dir
    if _shim_dir is None:
        return
    if _path_before_shim is not None:
        os.environ["PATH"] = _path_before_shim
    shutil.rmtree(_shim_dir, ignore_errors=True)
    _path_before_shim = None
    _shim_dir = None


atexit.register(reset_container_cli)

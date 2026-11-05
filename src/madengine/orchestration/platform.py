"""Resolve the execution platform from an additional_context dict.

``platform`` is a string. Omitted means Docker, which is the historical
default. ``"podman"`` uses the same local container path with the Podman
CLI. ``"bare_metal"`` selects the conda backend. Conda settings live in
the ``conda`` object, not in a ``bare_metal`` key.
"""

from typing import Any, Dict, Optional

from madengine.core.container_runtime import configure_container_cli
from madengine.core.errors import ConfigurationError

SUPPORTED_CONTEXT_PLATFORMS = ("docker", "podman", "bare_metal")
_SCHEDULER_KEYS = ("slurm", "k8s", "kubernetes", "llm_d")


def resolve_context_platform(config: Optional[Dict[str, Any]]) -> str:
    """Return ``docker``, ``podman``, or ``bare_metal`` for this context.

    Raises:
        ConfigurationError: The dict uses the retired ``bare_metal`` key,
            names an unknown platform, puts a ``conda`` block on a container
            run, or combines ``bare_metal`` or ``podman`` with a scheduler
            block.
    """
    config = config or {}
    if "bare_metal" in config:
        raise ConfigurationError(
            "The 'bare_metal' key no longer selects the conda backend. "
            'Set "platform": "bare_metal" and put settings under "conda".',
            suggestions=[
                'Set "platform": "bare_metal".',
                "Put conda settings in a 'conda' object "
                "(conda_bin, conda_env, env_root, reuse_env, cleanup_env).",
            ],
        )

    raw = config.get("platform", "docker")
    if raw is None:
        raw = "docker"
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(
            'platform must be a string such as "docker", "podman", or "bare_metal".',
            suggestions=[
                'Example: {"platform": "podman"}',
                'Example: {"platform": "bare_metal", "conda": {...}}',
            ],
        )
    platform = raw.strip().lower()
    if platform not in SUPPORTED_CONTEXT_PLATFORMS:
        raise ConfigurationError(
            f"Unknown platform {platform!r}.",
            suggestions=[
                "Supported platforms: " + ", ".join(SUPPORTED_CONTEXT_PLATFORMS) + ".",
            ],
        )
    if platform == "podman" and str(config.get("gpu_vendor", "")).upper() == "NVIDIA":
        raise ConfigurationError(
            "platform 'podman' does not yet support NVIDIA GPU runs.",
            suggestions=["Use platform 'docker' for NVIDIA GPUs."],
        )

    conda = config.get("conda")
    if conda is not None and not isinstance(conda, dict):
        raise ConfigurationError(
            "conda must be an object of conda settings.",
            suggestions=[
                'Example: {"platform": "bare_metal", "conda": {"conda_env": "myenv"}}'
            ],
        )
    if conda is not None and platform != "bare_metal":
        raise ConfigurationError(
            'A \'conda\' block requires "platform": "bare_metal".',
            suggestions=[
                'Add "platform": "bare_metal", or remove the conda block.',
            ],
        )

    if platform in ("bare_metal", "podman"):
        present = [key for key in _SCHEDULER_KEYS if key in config]
        if present:
            if platform == "bare_metal":
                suggestion = "Remove the scheduler block to run conda on this machine."
            else:
                suggestion = "Remove the scheduler block to run Podman on this machine."
            raise ConfigurationError(
                f"platform {platform!r} cannot be combined with "
                + ", ".join(present)
                + ".",
                suggestions=[suggestion],
            )
    return platform


def apply_context_platform(config: Optional[Dict[str, Any]]) -> str:
    """Resolve ``platform`` and configure the local Docker command path.

    ``podman`` installs a process-local Docker-to-Podman shim. ``docker`` and
    ``bare_metal`` leave the existing Docker command path unchanged.

    Args:
        config: The additional_context dict, or None.

    Returns:
        str: The resolved platform.
    """
    platform = resolve_context_platform(config)
    configure_container_cli(platform)
    return platform

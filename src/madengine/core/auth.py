#!/usr/bin/env python3
"""
Shared authentication utilities for madengine.

Centralises credential loading logic used by both BuildOrchestrator and
RunOrchestrator so that fixes and improvements only need to be made once.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import shlex
from pathlib import Path
from typing import Dict, Optional, Tuple

from madengine.core.errors import (
    ConfigurationError,
    create_error_context,
    handle_error,
)

# Keys under which the Docker CLI stores Docker Hub credentials in config.json.
# Docker has used several spellings over the years and any of them means
# "this machine is authenticated to Docker Hub".
_DOCKERHUB_CONFIG_KEYS: Tuple[str, ...] = (
    "https://index.docker.io/v1/",
    "index.docker.io",
    "registry-1.docker.io",
    "docker.io",
)

# Registry values that madengine treats as "Docker Hub" rather than a host.
_DOCKERHUB_ALIASES = ("docker.io", "dockerhub")


def load_credentials() -> Optional[Dict]:
    """Load credentials from credential.json and environment variables.

    Precedence (highest wins):
      1. ``MAD_DOCKERHUB_USER`` / ``MAD_DOCKERHUB_PASSWORD`` environment vars
         (merged into the ``dockerhub`` key of the returned dict)
      2. ``credential.json`` in the current working directory

    Returns:
        Credentials dict (keyed by registry name), or ``None`` if no
        credentials are found.
    """
    credentials: Optional[Dict] = None

    credential_file = "credential.json"
    if os.path.exists(credential_file):
        try:
            with open(credential_file) as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("credential.json must contain a JSON object, not " + type(loaded).__name__)
            credentials = loaded
            print(
                f"Loaded credentials from {credential_file}: "
                f"{list(credentials.keys())}"
            )
        except Exception as e:
            context = create_error_context(
                operation="load_credentials",
                component="auth",
                file_path=credential_file,
            )
            handle_error(
                ConfigurationError(
                    f"Could not load credentials: {e}",
                    context=context,
                    suggestions=[
                        "Check if credential.json exists and has valid JSON format"
                    ],
                )
            )

    # Environment variables override / supplement file credentials
    docker_hub_user = os.environ.get("MAD_DOCKERHUB_USER")
    docker_hub_password = os.environ.get("MAD_DOCKERHUB_PASSWORD")
    docker_hub_repo = os.environ.get("MAD_DOCKERHUB_REPO")

    if docker_hub_user and docker_hub_password:
        print("Found Docker Hub credentials in environment variables")
        if credentials is None:
            credentials = {}
        credentials["dockerhub"] = {
            "username": docker_hub_user,
            "password": docker_hub_password,
        }
        if docker_hub_repo:
            credentials["dockerhub"]["repository"] = docker_hub_repo

    return credentials


def _registry_config_keys(registry: Optional[str]) -> Tuple[str, ...]:
    """Map a madengine registry value to the keys Docker uses in config.json.

    Args:
        registry: Registry URL (e.g. ``"localhost:5000"``, ``"docker.io/rocm"``),
            or ``None``/empty string for Docker Hub.

    Returns:
        The config.json ``auths`` keys that would hold credentials for it.
    """
    if not registry or registry.lower() in _DOCKERHUB_ALIASES:
        return _DOCKERHUB_CONFIG_KEYS
    # Downstream code derives the registry host the same way (docker login <host>).
    host = registry.split("/")[0]
    if host.lower() in _DOCKERHUB_ALIASES:
        return _DOCKERHUB_CONFIG_KEYS
    return (host,)


def has_ambient_docker_auth(registry: Optional[str]) -> bool:
    """Report whether the local Docker CLI is already authenticated to ``registry``.

    Reads ``${DOCKER_CONFIG:-~/.docker}/config.json`` the same way the Docker CLI
    does, so an existing ``docker login`` (e.g. an organisation access token) is
    honoured instead of being overridden or reported as "no credentials".

    Args:
        registry: Registry URL, or ``None``/empty string for Docker Hub.

    Returns:
        ``True`` if a usable credential entry exists for the registry. Any read
        or parse problem yields ``False``; this function never raises and never
        logs credential material.
    """
    config_dir = os.environ.get("DOCKER_CONFIG") or os.path.join(
        os.path.expanduser("~"), ".docker"
    )
    try:
        config = json.loads(Path(config_dir, "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False

    keys = _registry_config_keys(registry)
    auths = config.get("auths") or {}
    cred_helpers = config.get("credHelpers") or {}
    creds_store = config.get("credsStore")

    if isinstance(cred_helpers, dict) and any(key in cred_helpers for key in keys):
        return True
    if not isinstance(auths, dict):
        return False
    for key in keys:
        entry = auths.get(key)
        if not isinstance(entry, dict):
            continue
        if entry.get("auth") or entry.get("identitytoken") or entry.get("username"):
            return True
        # Credential-store-managed entries are persisted as an empty object;
        # the secret itself lives in the external store.
        if creds_store:
            return True
    return False


def _usable_credentials(creds: object) -> bool:
    """Report whether a credential entry carries a non-blank username and password.

    Placeholder entries such as ``{"username": "", "password": ""}`` are treated
    as "not configured" rather than as credentials, matching
    :func:`madengine.deployment.k8s_secrets.build_registry_secret_data`.
    """
    if not isinstance(creds, dict):
        return False
    return bool(str(creds.get("username") or "").strip()) and bool(
        str(creds.get("password") or "").strip()
    )


def _registry_from_image(image: Optional[str]) -> Optional[str]:
    """Extract the registry host from an image reference.

    Args:
        image: Image reference (e.g. ``"ghcr.io/org/app:tag"``), or ``None``.

    Returns:
        The registry host, or ``None`` when the reference targets Docker Hub,
        is malformed, or was not supplied.
    """
    if not image or not image.strip():
        return None
    ref = image.strip()
    if "/" not in ref:
        return None
    # Docker treats the first component as a registry only when it looks like a
    # host; otherwise it is a Docker Hub namespace (e.g. "rocm/private").
    host = ref.split("/")[0]
    if not ("." in host or ":" in host or host == "localhost"):
        return None
    if host.lower() in _DOCKERHUB_ALIASES:
        return None
    return host


def explain_registry_denial(
    log_text: str, image: Optional[str] = None
) -> Optional[str]:
    """Turn a registry denial in Docker output into an actionable explanation.

    Distinguishes "authenticated but not authorized for this repository" from
    "not authenticated at all", because the two need completely different fixes.

    Args:
        log_text: Docker build/pull output to inspect.
        image: Optional image reference the denial refers to, for the message.

    Returns:
        A multi-line hint, or ``None`` if the output shows no registry denial.
    """
    lowered = (log_text or "").lower()
    subject = image.strip() if image and image.strip() else "the base image"

    if "insufficient_scope" in lowered or "authorization failed" in lowered:
        return (
            f"Base image pull was denied: {subject}\n"
            "   The registry ACCEPTED the credentials but granted no pull scope "
            "for this repository.\n"
            "   This is an authorization problem, not a login problem:\n"
            "     - the access token is not scoped to this repository, or\n"
            "     - the repository/tag does not exist under that namespace.\n"
            "   Re-running `docker login` will not fix it; widen the token's "
            "repository scope instead."
        )

    denied = (
        "pull access denied" in lowered
        or "requested access to the resource is denied" in lowered
        or "authentication required" in lowered
        or "unauthorized" in lowered
    )
    if not denied:
        return None

    registry = _registry_from_image(image)
    if registry is None:
        fixes = (
            "     - `docker login` on this machine (madengine reuses an existing "
            "login)\n"
            '     - add {"dockerhub": {"username": "...", "password": "..."}} to '
            "credential.json\n"
            "     - export MAD_DOCKERHUB_USER and MAD_DOCKERHUB_PASSWORD"
        )
    else:
        fixes = (
            f"     - `docker login {registry}` on this machine (madengine reuses "
            "an existing login)\n"
            f'     - add {{"{registry}": {{"username": "...", "password": "..."}}}} '
            "to credential.json"
        )

    return (
        f"Base image pull was denied: {subject}\n"
        "   No usable credentials were presented to the registry.\n"
        "   Fix with any one of:\n" + fixes
    )


def login_to_registry(
    registry: Optional[str],
    credentials: Optional[Dict],
    console,
    rich_console,
    raise_on_failure: bool = True,
) -> None:
    """Login to a Docker registry.

    This is the single shared implementation used by both DockerBuilder
    and ContainerRunner.

    Args:
        registry: Registry URL (e.g., "localhost:5000", "docker.io"), or
            ``None``/empty string to target DockerHub.
        credentials: Credentials dictionary keyed by registry name.
        console: A ``Console`` instance for shell execution.
        rich_console: A Rich ``Console`` instance for formatted output.
        raise_on_failure: If ``True`` (default), raise ``RuntimeError`` on any
            failure (missing key, invalid format, or docker login error).
            Set to ``False`` to log and return instead, allowing the caller
            to fall back to pulling public images.

    Precedence: explicit credentials (``credential.json`` / ``MAD_DOCKERHUB_*``)
    win when they carry a non-blank username and password. Otherwise an existing
    ``docker login`` on this machine is reused and no login is attempted, so a
    placeholder credential entry never overrides or breaks working ambient auth.
    Set ``MAD_SKIP_DOCKER_LOGIN=1`` to always defer to ambient credentials.
    """
    if os.environ.get("MAD_SKIP_DOCKER_LOGIN") == "1":
        rich_console.print(
            "[yellow]MAD_SKIP_DOCKER_LOGIN=1 - using existing docker login for "
            f"{registry or 'DockerHub'}[/yellow]"
        )
        return

    registry_key = registry if registry else "dockerhub"

    # Normalise docker.io → dockerhub
    if registry and registry.lower() == "docker.io":
        registry_key = "dockerhub"

    entry = (credentials or {}).get(registry_key)
    creds: Dict = entry if isinstance(entry, dict) else {}

    if not _usable_credentials(creds):
        # No explicit credentials configured for this registry. If the machine
        # is already logged in (e.g. an organisation access token), reuse that
        # instead of failing or clobbering it.
        if has_ambient_docker_auth(registry):
            rich_console.print(
                f"[green]Using existing docker login for "
                f"{registry or 'DockerHub'} (no explicit credentials "
                f"configured)[/green]"
            )
            return

        if not credentials:
            rich_console.print(
                "[yellow]No credentials provided for registry login[/yellow]"
            )
            return

        if registry_key not in credentials:
            error_msg = f"No credentials found for registry: {registry_key}"
        else:
            error_msg = (
                f"Invalid credentials format for registry: {registry_key}"
                f"\nCredentials must contain non-empty 'username' and "
                f"'password' fields"
            )
        if registry_key == "dockerhub":
            error_msg += (
                f"\nPlease add dockerhub credentials to credential.json:\n"
                "{\n"
                '  "dockerhub": {\n'
                '    "repository": "your-repository",\n'
                '    "username": "your-dockerhub-username",\n'
                '    "password": "your-dockerhub-password-or-token"\n'
                "  }\n"
                "}"
            )
        else:
            error_msg += (
                f"\nPlease add {registry_key} credentials to credential.json:\n"
                "{\n"
                f'  "{registry_key}": {{\n'
                f'    "repository": "your-repository",\n'
                f'    "username": "your-{registry_key}-username",\n'
                f'    "password": "your-{registry_key}-password"\n'
                "  }\n"
                "}"
            )
        error_msg += "\nAlternatively, run `docker login` on this machine."
        rich_console.print(f"[red]{error_msg}[/red]")
        if raise_on_failure:
            raise RuntimeError(error_msg)
        return

    username = str(creds["username"])
    password = str(creds["password"])

    # Pass the password via an environment variable so it never appears in
    # the process argument list (visible via /proc or ps to other users).
    quoted_username = shlex.quote(username)
    login_command = "printf %s \"$MAD_REGISTRY_PASSWORD\" | docker login"
    if registry and registry.lower() not in ["docker.io", "dockerhub"]:
        login_command += f" {shlex.quote(str(registry))}"
    login_command += f" --username {quoted_username} --password-stdin"

    login_env = {**os.environ, "MAD_REGISTRY_PASSWORD": password}

    try:
        console.sh(login_command, secret=True, env=login_env)
        rich_console.print(
            f"[green]Successfully logged in to registry: "
            f"{registry or 'DockerHub'}[/green]"
        )
    except Exception as e:
        rich_console.print(
            f"[red]Failed to login to registry {registry}: {e}[/red]"
        )
        if raise_on_failure:
            raise

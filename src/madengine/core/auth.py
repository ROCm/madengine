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
from typing import Dict, Optional

from madengine.core.errors import (
    ConfigurationError,
    create_error_context,
    handle_error,
)


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
                credentials = json.load(f)
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
    """
    if not credentials:
        rich_console.print(
            "[yellow]No credentials provided for registry login[/yellow]"
        )
        return

    registry_key = registry if registry else "dockerhub"

    # Normalise docker.io → dockerhub
    if registry and registry.lower() == "docker.io":
        registry_key = "dockerhub"

    if registry_key not in credentials:
        error_msg = f"No credentials found for registry: {registry_key}"
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
        rich_console.print(f"[red]{error_msg}[/red]")
        if raise_on_failure:
            raise RuntimeError(error_msg)
        return

    creds = credentials[registry_key]

    if "username" not in creds or "password" not in creds:
        error_msg = (
            f"Invalid credentials format for registry: {registry_key}"
            f"\nCredentials must contain 'username' and 'password' fields"
        )
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

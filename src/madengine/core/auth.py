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
    """Load credentials from credential.json and environment variables."""
    credentials: Optional[Dict] = None
    credential_file = "credential.json"
    if os.path.exists(credential_file):
        try:
            with open(credential_file) as f:
                credentials = json.load(f)
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
                    suggestions=["Check if credential.json exists and has valid JSON format"],
                )
            )
    docker_hub_user = os.environ.get("MAD_DOCKERHUB_USER")
    docker_hub_password = os.environ.get("MAD_DOCKERHUB_PASSWORD")
    docker_hub_repo = os.environ.get("MAD_DOCKERHUB_REPO")
    if docker_hub_user and docker_hub_password:
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
    """Login to a Docker registry (shared implementation for all orchestrators)."""
    if not credentials:
        rich_console.print("[yellow]No credentials provided for registry login[/yellow]")
        return
    registry_key = registry if registry else "dockerhub"
    if registry and registry.lower() == "docker.io":
        registry_key = "dockerhub"
    if registry_key not in credentials:
        error_msg = f"No credentials found for registry: {registry_key}"
        rich_console.print(f"[red]{error_msg}[/red]")
        if raise_on_failure:
            raise RuntimeError(error_msg)
        return
    creds = credentials[registry_key]
    if "username" not in creds or "password" not in creds:
        error_msg = f"Invalid credentials format for registry: {registry_key}"
        rich_console.print(f"[red]{error_msg}[/red]")
        if raise_on_failure:
            raise RuntimeError(error_msg)
        return
    username = str(creds["username"])
    password = str(creds["password"])
    quoted_username = shlex.quote(username)
    login_command = "printf %s \"$MAD_REGISTRY_PASSWORD\" | docker login"
    if registry and registry.lower() not in ["docker.io", "dockerhub"]:
        login_command += f" {shlex.quote(str(registry))}"
    login_command += f" --username {quoted_username} --password-stdin"
    login_env = {**os.environ, "MAD_REGISTRY_PASSWORD": password}
    try:
        console.sh(login_command, secret=True, env=login_env)
        rich_console.print(f"[green]Successfully logged in to registry: {registry or 'DockerHub'}[/green]")
    except Exception as e:
        rich_console.print(f"[red]Failed to login to registry {registry}: {e}[/red]")
        if raise_on_failure:
            raise

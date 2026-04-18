#!/usr/bin/env python3
"""
Shared authentication utilities for madengine.

Centralises credential loading logic used by both BuildOrchestrator and
RunOrchestrator so that fixes and improvements only need to be made once.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
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

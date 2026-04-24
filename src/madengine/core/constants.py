#!/usr/bin/env python
"""Module to define constants.

This module provides the constants used in the MAD Engine.

Environment Variables:
    - MAD_VERBOSE_CONFIG: Set to "true" to enable verbose configuration logging
    - MAD_SETUP_MODEL_DIR: Set to "true" to enable automatic MODEL_DIR setup during import
    - MODEL_DIR: Path to model directory to copy to current working directory
    - MAD_MINIO: JSON string with MinIO configuration
    - MAD_AWS_S3: JSON string with AWS S3 configuration
    - NAS_NODES: JSON string with NAS nodes configuration
    - PUBLIC_GITHUB_ROCM_KEY: JSON string with GitHub token configuration

Configuration Loading:
    All configuration constants follow a priority order:
    1. Environment variables (as JSON strings)
    2. credential.json file
    3. Built-in defaults

    Invalid JSON in environment variables will fall back to defaults with error logging.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import json
import logging


# Utility function for optional verbose logging of configuration
def _log_config_info(message: str, force_print: bool = False):
    """Log configuration information either to logger or print if specified."""
    if force_print or os.environ.get("MAD_VERBOSE_CONFIG", "").lower() == "true":
        print(message)
    else:
        logging.debug(message)


# third-party modules
from madengine.core.console import Console

# Get the model directory, if it is not set, default to "." (current directory)
MODEL_DIR = os.environ.get("MODEL_DIR", ".")


def _setup_model_dir():
    """Setup model directory if MODEL_DIR environment variable is set.
    
    MODEL_DIR defaults to "." (current directory) if not set.
    Only copies if MODEL_DIR points to a different directory than current working directory.
    """
    # Get absolute paths to compare
    model_dir_abs = os.path.abspath(MODEL_DIR)
    cwd_abs = os.path.abspath(".")
    
    # Only copy if MODEL_DIR points to a different directory (not current dir)
    if model_dir_abs != cwd_abs:
        # Copy MODEL_DIR to the current working directory.
        _log_config_info(f"Current working directory: {cwd_abs}")
        _log_config_info(f"MODEL_DIR: {MODEL_DIR} (different from current dir)")
        console = Console(live_output=True)
        # copy the MODEL_DIR to the current working directory
        console.sh(f"cp -vLR --preserve=all {MODEL_DIR}/* {cwd_abs}")
        _log_config_info(f"Model dir: {MODEL_DIR} copied to current dir: {cwd_abs}")


# Only setup model directory if explicitly requested (when not just importing for constants)
if os.environ.get("MAD_SETUP_MODEL_DIR", "").lower() == "true":
    _setup_model_dir()

# madengine credentials configuration
CRED_FILE = "credential.json"


def _load_credentials():
    """Load credentials from file with proper error handling."""
    try:
        # read credentials
        with open(CRED_FILE) as f:
            creds = json.load(f)
        _log_config_info(f"Credentials loaded from {CRED_FILE}")
        return creds
    except FileNotFoundError:
        _log_config_info(f"Credentials file {CRED_FILE} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        _log_config_info(f"Error parsing {CRED_FILE}: {e}, using defaults")
        return {}
    except Exception as e:
        _log_config_info(f"Unexpected error loading {CRED_FILE}: {e}, using defaults")
        return {}


CREDS = _load_credentials()

# Default value used for NAS_NODES when neither env nor creds provide it.
_DEFAULT_NAS_NODES = [
    {
        "NAME": "DEFAULT",
        "HOST": "localhost",
        "PORT": 22,
        "USERNAME": "username",
        "PASSWORD": "password",
    }
]

# Default value used for MAD_AWS_S3 when neither env nor creds provide it.
_DEFAULT_MAD_AWS_S3 = {"USERNAME": None, "PASSWORD": None}

# Default value used for MAD_MINIO when neither env nor creds provide it.
_DEFAULT_MAD_MINIO = {
    "USERNAME": None,
    "PASSWORD": None,
    "MINIO_ENDPOINT": "http://localhost:9000",
    "AWS_ENDPOINT_URL_S3": "http://localhost:9000",
}

# Default value used for PUBLIC_GITHUB_ROCM_KEY when neither env nor creds provide it.
_DEFAULT_PUBLIC_GITHUB_ROCM_KEY = {"username": None, "token": None}


def _get_env_or_creds_or_default(env_key: str, creds_key: str, default):
    """Load config from env (JSON), creds file, or default.

    Priority: 1) environment variable (parsed as JSON), 2) credential file, 3) default.
    Invalid JSON in env falls back to default with logging.

    Args:
        env_key: Environment variable name (e.g. "NAS_NODES").
        creds_key: Key in CREDS dict (e.g. "NAS_NODES").
        default: Default value if env unset and creds missing or env JSON invalid.

    Returns:
        Loaded value (same type as default, or from env/creds).
    """
    if env_key not in os.environ:
        _log_config_info(f"{env_key} environment variable is not set.")
        if creds_key in CREDS:
            _log_config_info(f"{creds_key} loaded from credentials file.")
            return CREDS[creds_key]
        _log_config_info(f"{creds_key} is using default values.")
        return default
    _log_config_info(f"{env_key} is loaded from env variables.")
    try:
        return json.loads(os.environ[env_key])
    except json.JSONDecodeError as e:
        _log_config_info(
            f"Error parsing {env_key} environment variable: {e}, using defaults"
        )
        return default


def _get_nas_nodes():
    """Initialize NAS_NODES configuration."""
    return _get_env_or_creds_or_default("NAS_NODES", "NAS_NODES", _DEFAULT_NAS_NODES)


NAS_NODES = _get_nas_nodes()


def _get_mad_aws_s3():
    """Initialize MAD_AWS_S3 configuration."""
    return _get_env_or_creds_or_default("MAD_AWS_S3", "MAD_AWS_S3", _DEFAULT_MAD_AWS_S3)


MAD_AWS_S3 = _get_mad_aws_s3()


def _get_mad_minio():
    """Initialize MAD_MINIO configuration (dict with USERNAME, PASSWORD, MINIO_ENDPOINT, etc.)."""
    return _get_env_or_creds_or_default("MAD_MINIO", "MAD_MINIO", _DEFAULT_MAD_MINIO)


MAD_MINIO = _get_mad_minio()


def _get_public_github_rocm_key():
    """Initialize PUBLIC_GITHUB_ROCM_KEY configuration.

    Returned dict always has keys 'username' and 'token' (public API).
    Credential files may use 'password' for the token; that is normalized to 'token'.
    """
    raw = _get_env_or_creds_or_default(
        "PUBLIC_GITHUB_ROCM_KEY",
        "PUBLIC_GITHUB_ROCM_KEY",
        _DEFAULT_PUBLIC_GITHUB_ROCM_KEY,
    )
    if not isinstance(raw, dict):
        return dict(_DEFAULT_PUBLIC_GITHUB_ROCM_KEY)
    # Normalize so public API always has username + token (accept creds that use "password")
    token = raw.get("token") or raw.get("password")
    return {"username": raw.get("username"), "token": token}


PUBLIC_GITHUB_ROCM_KEY = _get_public_github_rocm_key()


def get_rocm_path(override=None):
    """Return ROCm installation root (legacy, no automatic filesystem scan).

    For full resolution (MAD_ROCM_PATH, auto-detect) use
    :func:`madengine.utils.rocm_path_resolver.resolve_host_rocm_path` in
    :class:`~madengine.core.context.Context`.

    Resolution: ``override`` -> :envvar:`ROCM_PATH` -> ``/opt/rocm``.

    Args:
        override: Optional path overriding env and default.

    Returns:
        str: Absolute ROCm root path.
    """
    from madengine.utils.rocm_path_resolver import get_rocm_path_legacy

    return get_rocm_path_legacy(override)

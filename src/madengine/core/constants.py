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


def _get_nas_nodes():
    """Initialize NAS_NODES configuration."""
    if "NAS_NODES" not in os.environ:
        _log_config_info("NAS_NODES environment variable is not set.")
        if "NAS_NODES" in CREDS:
            _log_config_info("NAS_NODES loaded from credentials file.")
            return CREDS["NAS_NODES"]
        else:
            _log_config_info("NAS_NODES is using default values.")
            return [
                {
                    "NAME": "DEFAULT",
                    "HOST": "localhost",
                    "PORT": 22,
                    "USERNAME": "username",
                    "PASSWORD": "password",
                }
            ]
    else:
        _log_config_info("NAS_NODES is loaded from env variables.")
        try:
            return json.loads(os.environ["NAS_NODES"])
        except json.JSONDecodeError as e:
            _log_config_info(
                f"Error parsing NAS_NODES environment variable: {e}, using defaults"
            )
            return [
                {
                    "NAME": "DEFAULT",
                    "HOST": "localhost",
                    "PORT": 22,
                    "USERNAME": "username",
                    "PASSWORD": "password",
                }
            ]


NAS_NODES = _get_nas_nodes()


def _get_mad_aws_s3():
    """Initialize MAD_AWS_S3 configuration."""
    if "MAD_AWS_S3" not in os.environ:
        _log_config_info("MAD_AWS_S3 environment variable is not set.")
        if "MAD_AWS_S3" in CREDS:
            _log_config_info("MAD_AWS_S3 loaded from credentials file.")
            return CREDS["MAD_AWS_S3"]
        else:
            _log_config_info("MAD_AWS_S3 is using default values.")
            return {
                "USERNAME": None,
                "PASSWORD": None,
            }
    else:
        _log_config_info("MAD_AWS_S3 is loaded from env variables.")
        try:
            return json.loads(os.environ["MAD_AWS_S3"])
        except json.JSONDecodeError as e:
            _log_config_info(
                f"Error parsing MAD_AWS_S3 environment variable: {e}, using defaults"
            )
            return {
                "USERNAME": None,
                "PASSWORD": None,
            }


MAD_AWS_S3 = _get_mad_aws_s3()


# Check the MAD_MINIO environment variable which is a dict.
def _get_mad_minio():
    """Initialize MAD_MINIO configuration."""
    if "MAD_MINIO" not in os.environ:
        _log_config_info("MAD_MINIO environment variable is not set.")
        if "MAD_MINIO" in CREDS:
            _log_config_info("MAD_MINIO loaded from credentials file.")
            return CREDS["MAD_MINIO"]
        else:
            _log_config_info("MAD_MINIO is using default values.")
            return {
                "USERNAME": None,
                "PASSWORD": None,
                "MINIO_ENDPOINT": "http://localhost:9000",
                "AWS_ENDPOINT_URL_S3": "http://localhost:9000",
            }
    else:
        _log_config_info("MAD_MINIO is loaded from env variables.")
        try:
            return json.loads(os.environ["MAD_MINIO"])
        except json.JSONDecodeError as e:
            _log_config_info(
                f"Error parsing MAD_MINIO environment variable: {e}, using defaults"
            )
            return {
                "USERNAME": None,
                "PASSWORD": None,
                "MINIO_ENDPOINT": "http://localhost:9000",
                "AWS_ENDPOINT_URL_S3": "http://localhost:9000",
            }


MAD_MINIO = _get_mad_minio()


def _get_public_github_rocm_key():
    """Initialize PUBLIC_GITHUB_ROCM_KEY configuration."""
    if "PUBLIC_GITHUB_ROCM_KEY" not in os.environ:
        _log_config_info("PUBLIC_GITHUB_ROCM_KEY environment variable is not set.")
        if "PUBLIC_GITHUB_ROCM_KEY" in CREDS:
            _log_config_info("PUBLIC_GITHUB_ROCM_KEY loaded from credentials file.")
            return CREDS["PUBLIC_GITHUB_ROCM_KEY"]
        else:
            _log_config_info("PUBLIC_GITHUB_ROCM_KEY is using default values.")
            return {
                "username": None,
                "token": None,
            }
    else:
        _log_config_info("PUBLIC_GITHUB_ROCM_KEY is loaded from env variables.")
        try:
            return json.loads(os.environ["PUBLIC_GITHUB_ROCM_KEY"])
        except json.JSONDecodeError as e:
            _log_config_info(
                f"Error parsing PUBLIC_GITHUB_ROCM_KEY environment variable: {e}, using defaults"
            )
            return {
                "username": None,
                "token": None,
            }


PUBLIC_GITHUB_ROCM_KEY = _get_public_github_rocm_key()

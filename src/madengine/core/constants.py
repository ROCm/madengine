#!/usr/bin/env python
"""Module to define constants.

This module provides the constants used in the MAD Engine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import json
# third-party modules
from madengine.core.console import Console

# Get the model directory, if it is not set, set it to None.
MODEL_DIR = os.environ.get("MODEL_DIR")
    
# MADEngine update
if MODEL_DIR:
    # Copy MODEL_DIR to the current working directory.
    cwd_path = os.getcwd()
    print(f"Current working directory: {cwd_path}")
    console = Console(live_output=True)
    # copy the MODEL_DIR to the current working directory
    console.sh(f"sudo cp -vLR --preserve=all {MODEL_DIR}/* {cwd_path}")
    print(f"Model dir: {MODEL_DIR} copied to current dir: {cwd_path}")

# MADEngine update
CRED_FILE = "credential.json"

try:
    # read credentials
    with open(CRED_FILE) as f:
        CREDS = json.load(f)
except FileNotFoundError:
    CREDS = {}

if "NAS_NODES" not in os.environ:
    if "NAS_NODES" in CREDS:
        NAS_NODES = CREDS["NAS_NODES"]
    else:
        NAS_NODES = [{
            "NAME": "DEFAULT",
            "HOST": "localhost",
            "PORT": 22,
            "USERNAME": "username",
            "PASSWORD": "password",
        }]
else:
    NAS_NODES = json.loads(os.environ["NAS_NODES"])

# Check the MAD_AWS_S3 environment variable which is a dict, if it is not set, set its element to default values.
if "MAD_AWS_S3" not in os.environ:
    # Check if the MAD_AWS_S3 is in the credentials.json file.
    if "MAD_AWS_S3" in CREDS:
        MAD_AWS_S3 = CREDS["MAD_AWS_S3"]
    else:
        MAD_AWS_S3 = {
            "USERNAME": None,
            "PASSWORD": None,
        }
else:
    MAD_AWS_S3 = json.loads(os.environ["MAD_AWS_S3"])

# Check the MAD_MINIO environment variable which is a dict.
if "MAD_MINIO" not in os.environ:
    print("MAD_MINIO environment variable is not set.")
    if "MAD_MINIO" in CREDS:
        MAD_MINIO = CREDS["MAD_MINIO"]
    else:
        print("MAD_MINIO is using default values.")
        MAD_MINIO = {
            "USERNAME": None,
            "PASSWORD": None,
            "MINIO_ENDPOINT": "http://localhost:9000",  
            "AWS_ENDPOINT_URL_S3": "http://localhost:9000",
        }
else:
    print("MAD_MINIO is loaded from env variables.")
    MAD_MINIO = json.loads(os.environ["MAD_MINIO"])

# Check the auth GitHub token environment variable which is a dict, if it is not set, set it to None.
if "PUBLIC_GITHUB_ROCM_KEY" not in os.environ:
    if "PUBLIC_GITHUB_ROCM_KEY" in CREDS:
        PUBLIC_GITHUB_ROCM_KEY = CREDS["PUBLIC_GITHUB_ROCM_KEY"]
    else:
        PUBLIC_GITHUB_ROCM_KEY = {
            "username": None,
            "token": None,
        }
else:
    PUBLIC_GITHUB_ROCM_KEY = json.loads(os.environ["PUBLIC_GITHUB_ROCM_KEY"])

#!/usr/bin/env python3
"""Utility module for helper functions

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
from statistics import mean
import typing

# third-party modules
import pandas as pd
import numpy as np


def get_env_vars() -> dict:
    """Utility function to get MAD/DLM specific env_vars

    env_vars:
    - TUNA_DB_USER_NAME
    - TUNA_DB_USER_PASSWORD
    - TUNA_DB_HOSTNAME
    - TUNA_DB_PORT
    - TUNA_DB_NAME
    - TUNA_SSH_USER
    - TUNA_SSH_PASSWORD
    - TUNA_SSH_HOSTNAME
    - TUNA_SSH_PORT
    - SLURM_CPUS_ON_NODE
    - LOG_LEVEL
    - MODEL_DIR

    Returns:
        dict: Dictionary of DLM specific env_vars
    """
    # init env vars
    env_vars = {}

    if "TUNA_DB_USER_NAME" in os.environ:
        env_vars["user_name"] = os.environ["TUNA_DB_USER_NAME"]
    else:
        env_vars["user_name"] = ""
    if "TUNA_DB_USER_PASSWORD" in os.environ:
        env_vars["user_password"] = os.environ["TUNA_DB_USER_PASSWORD"]
    else:
        env_vars["user_password"] = ""
    if "TUNA_DB_HOSTNAME" in os.environ:
        env_vars["db_hostname"] = os.environ["TUNA_DB_HOSTNAME"]
    else:
        env_vars["db_hostname"] = "localhost"
    if "TUNA_DB_PORT" in os.environ:
        env_vars["db_port"] = str(os.environ["TUNA_DB_PORT"])
    else:
        env_vars["db_port"] = "3306"
    if "TUNA_DB_NAME" in os.environ:
        env_vars["db_name"] = os.environ["TUNA_DB_NAME"]
    else:
        env_vars["db_name"] = "dlm_db"
    if "SLURM_CPUS_ON_NODE" in os.environ:
        env_vars["slurm_cpus"] = str(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        env_vars["slurm_cpus"] = "0"
    if "TUNA_SSH_USER" in os.environ:
        env_vars["ssh_user"] = os.environ["TUNA_SSH_USER"]
    else:
        env_vars["ssh_user"] = ""
    if "TUNA_SSH_PASSWORD" in os.environ:
        env_vars["ssh_password"] = os.environ["TUNA_SSH_PASSWORD"]
    else:
        env_vars["ssh_password"] = ""
    if "TUNA_SSH_HOSTNAME" in os.environ:
        env_vars["ssh_hostname"] = os.environ["TUNA_SSH_HOSTNAME"]
    else:
        env_vars["ssh_hostname"] = "localhost"
    if "TUNA_SSH_PORT" in os.environ:
        env_vars["ssh_port"] = str(os.environ["TUNA_SSH_PORT"])
    else:
        env_vars["ssh_port"] = "22"

    return env_vars


def get_avg_perf(
    entry_list: typing.List[dict], n: int = 5
) -> typing.Tuple[float, typing.List[float]]:
    """Get average performance from the last n entries

    Args:
        entry_list (list): List of entries
        n (int): Number of entries to consider

    Returns:
        tuple: Tuple of average performance and list of performances
    """
    perfs = []
    for m in entry_list:
        if m["performance"]:
            perfs.append(float(m["performance"]))
    perfs = perfs[-n:]

    if perfs:
        avg = mean(perfs)
        print("{} avg from the last {} entries".format(avg, len(perfs)))
        return avg, perfs
    else:
        return None, None


def replace_nans_with_None(data: pd.DataFrame) -> pd.DataFrame:
    """Replace NaNs with None in the dataframe

    Args:
        data (pd.DataFrame): Dataframe to replace NaNs with None

    Returns:
        pd.DataFrame: Dataframe with NaNs replaced with None
    """
    # change nans to None to avoid errors
    # data = data.where((pd.notnull(data)), None)
    data = data.replace({np.nan: None})
    return data


def load_perf_csv(csv: str) -> pd.DataFrame:
    """Load performance csv file

    Args:
        csv (str): Path to the performance csv file

    Returns:
        pd.DataFrame: Dataframe of the performance csv file
    """
    df = pd.read_csv(csv)
    df = df.drop(
        columns=[
            "dataname",
            "data_provider_type",
            "data_size",
            "data_download_duration",
            "build_number",
        ],
        errors="ignore",
    )
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df = df.rename(columns=lambda x: x.strip())
    df = df.where((pd.notnull(df)), None)

    def trim_strings(x):
        return x.strip() if isinstance(x, str) else x

    df = df.applymap(trim_strings)
    df = replace_nans_with_None(df)
    return df


def dataFrame_to_list(df: pd.DataFrame) -> typing.List[dict]:
    """Convert dataframe to list of dictionaries

    Args:
        df (pd.DataFrame): Dataframe to convert

    Returns:
        list: List of dictionaries
    """
    return df.to_dict(orient="records")

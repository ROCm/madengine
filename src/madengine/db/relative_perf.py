"""Module to get the relative performance of the models.

This module contains functions to get the relative performance of the models.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import argparse
import ast
from statistics import mean
import typing

# third-party modules
import pandas as pd

# MAD Engine modules
from database import ENGINE, create_tables, LOGGER
from utils import get_avg_perf, load_perf_csv, dataFrame_to_list
from database_functions import get_all_gpu_archs, get_matching_db_entries


def get_baseline_configs(
    recent_entry: typing.Dict[str, typing.Any],
    baseline_params: typing.Dict[str, typing.Any],
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Get the baseline configurations.

    This function gets the baseline configurations from the database.

    Args:
        recent_entry (typing.Dict[str, typing.Any]): The recent entry.
        baseline_params (typing.Dict[str, typing.Any]): The baseline parameters.

    Returns:
        typing.List[typing.Dict[str, typing.Any]]: The baseline configurations.
    """
    # create sample_config
    sample_baseline_config = recent_entry
    for k, v in baseline_params.items():
        sample_baseline_config[k] = v

    # search database for similar configs
    last_successful_matching_entries = get_matching_db_entries(
        recent_entry,
        filters={"status": "SUCCESS", "base_docker": recent_entry["base_docker"]},
    )

    return last_successful_matching_entries


def relative_perf(
    data: pd.DataFrame, base_line_params: typing.Dict[str, typing.Any]
) -> pd.DataFrame:
    """Get the relative performance.

    This function gets the relative performance of the models.

    Args:
        data (pd.DataFrame): The data.
        base_line_params (typing.Dict[str, typing.Any]): The baseline parameters.

    Returns:
        pd.DataFrame: The data.
    """
    LOGGER.info("Checking relative performance against {}".format(base_line_params))
    print(data)
    # get the most recent entries
    most_recent_entries = dataFrame_to_list(data)

    # compare new data with avg of last succesfull runs in database
    for i, recent_entry in enumerate(most_recent_entries):

        # find matching entries to current entry
        baseline_configs = get_baseline_configs(recent_entry, base_line_params)
        baseline_avg, baseline_perfs = get_avg_perf(baseline_configs, 5)
        if recent_entry["performance"] and baseline_avg:
            print(
                "Current Performance is {} {}".format(
                    recent_entry["performance"], recent_entry["metric"]
                )
            )
            relative_perf = (float(recent_entry["performance"]) / baseline_avg) * 100
            print(
                "Relative perf {:.2f}% against {}".format(
                    relative_perf, base_line_params
                )
            )
        else:
            relative_perf = None

        entry_relative_change = {
            "pct_change": relative_perf,
            "baseline_avg": baseline_avg,
            "sample_count": len(baseline_perfs) if baseline_perfs else None,
        }

        # add pct_change info
        if data.loc[i, "relative_change"]:
            relative_change = ast.literal_eval(data.loc[i, "relative_change"])
            relative_change[base_line_params["gpu_architecture"]] = (
                entry_relative_change
            )
        else:
            relative_change = {
                base_line_params["gpu_architecture"]: entry_relative_change
            }
        data.loc[i, "relative_change"] = str(relative_change)

    print(data)
    return data


def relative_perf_all_configs(data: pd.DataFrame) -> pd.DataFrame:
    """Get the relative performance of all configurations.

    This function gets the relative performance of all configurations.

    Args:
        data (pd.DataFrame): The data.

    Returns:
        pd.DataFrame: The data.
    """
    archs = get_all_gpu_archs()
    print(archs)
    for a in archs:
        data = relative_perf(data, {"gpu_architecture": a})
    return data

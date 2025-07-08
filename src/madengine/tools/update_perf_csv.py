"""Module to update the performance csv file with the latest performance data.

This module is used to update the performance csv file with the latest performance data.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# build-in imports
import os
import json
import argparse
import typing
# third-party imports
import pandas as pd


def df_strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the column names of a DataFrame.
    
    Args:
        df: The DataFrame to strip the column names of.
    
    Returns:
        The DataFrame with stripped column names.
    """
    df.columns = df.columns.str.strip()
    return df


def read_json(js: str) -> dict:
    """Read a JSON file.
    
    Args:
        js: The path to the JSON file.
    
    Returns:
        The JSON dictionary.
    """
    # Input to this function should always be a path
    js_dict = json.load(open(js))
    return js_dict


def flatten_tags(perf_entry: dict):
    """Flatten the tags of a performance entry.
    
    Args:
        perf_entry: The performance entry.

    Returns:
        The performance entry with flattened tags.
    """
    # flatten tags to a string, if tags is a list.
    if type(perf_entry["tags"]) == list:
        perf_entry["tags"] = ",".join(str(item) for item in perf_entry["tags"])


def perf_entry_df_to_csv(perf_entry: pd.DataFrame) -> None:
    """Write the performance entry DataFrame to a CSV file.
    
    Args:
        perf_entry: The performance entry DataFrame.

    Returns:
        The performance entry DataFrame written to a CSV file.
    """
    perf_entry.to_csv("perf_entry.csv", index=False)


def perf_entry_dict_to_csv(perf_entry: typing.Dict) -> None:
    """Write the performance entry dictionary to a CSV file.
    
    Args:
        perf_entry: The performance entry dictionary.
    """
    flatten_tags(perf_entry)
    js_df = pd.DataFrame(perf_entry, index=[0])
    perf_entry_df_to_csv(js_df)


def handle_multiple_results(
        perf_csv_df: pd.DataFrame, 
        multiple_results: str, 
        common_info: str, 
        model_name: str
    ) -> pd.DataFrame:
    """Handle multiple results.
    
    Args:
        perf_csv_df: The performance csv DataFrame.
        multiple_results: The path to the multiple results CSV file.
        common_info: The path to the common info JSON file.
        model_name: The model name.
        
    Returns:
        The updated performance csv DataFrame.
    
    Raises:
        AssertionError: If the number of columns in the performance csv DataFrame is not equal to the length of the row.
    """
    # Check that the multiple results CSV has three columns and has the following format:
    # model, performance, metric
    multiple_results_df = df_strip_columns(pd.read_csv(multiple_results))
    multiple_results_header = multiple_results_df.columns.tolist()
    # if (len(multiple_results_header) != 3):
    #     raise RuntimeError("Multiple Results CSV file must have three columns: model, performance, metric")
    headings = ['model', 'performance', 'metric']
    for heading in headings:
        if not(heading in multiple_results_header):
            raise RuntimeError("Multiple Results CSV file is missing the " + heading + " column")

    common_info_json = read_json(common_info)
    flatten_tags(common_info_json)

    final_multiple_results_df = pd.DataFrame()
    # add results to perf.csv
    for r in multiple_results_df.to_dict(orient="records"):
        row = common_info_json.copy()
        row["model"] = model_name + "_" + str(r["model"])
        row["performance"] = r["performance"]
        row["metric"] = r["metric"]

        if r["performance"] is not None and pd.notna(r["performance"]):
            row["status"] = "SUCCESS"
        else:
            row["status"] = "FAILURE"

        assert perf_csv_df.columns.size == len(row)
        final_multiple_results_df = pd.concat(
            [final_multiple_results_df, pd.DataFrame(row, index=[0])], ignore_index=True
        )

    final_multiple_results_df = final_multiple_results_df[perf_csv_df.columns]
    perf_entry_df_to_csv(final_multiple_results_df)
    perf_csv_df = pd.concat([perf_csv_df, final_multiple_results_df])
    return perf_csv_df


def handle_single_result(
        perf_csv_df: pd.DataFrame, 
        single_result: str
    ) -> pd.DataFrame:
    """Handle a single result.
    
    Args:
        perf_csv_df: The performance csv DataFrame.
        single_result: The path to the single result JSON file.
    
    Returns:
        The updated performance csv DataFrame.

    Raises:
        AssertionError: If the number of columns in the performance csv DataFrame is not equal
    """
    single_result_json = read_json(single_result)
    perf_entry_dict_to_csv(single_result_json)
    perf_csv_df = pd.concat(
        [perf_csv_df, pd.DataFrame(single_result_json, index=[0])], ignore_index=True
    )

    return perf_csv_df


def handle_exception_result(
        perf_csv_df: pd.DataFrame, 
        exception_result: str
    ) -> pd.DataFrame:
    """Handle an exception result.
    
    Args:
        perf_csv_df: The performance csv DataFrame.
        exception_result: The path to the exception result JSON file.
    
    Returns:
        The updated performance csv DataFrame.

    Raises:
        AssertionError: If there is already an entry for the model in the performance csv DataFrame.
    """
    exception_result_json = read_json(exception_result)
    perf_entry_dict_to_csv(exception_result_json)
    perf_csv_df = pd.concat(
        [perf_csv_df, pd.DataFrame(exception_result_json, index=[0])], ignore_index=True
    )

    return perf_csv_df


def update_perf_csv(
        perf_csv: str,
        multiple_results: typing.Optional[str] = None,
        single_result: typing.Optional[str] = None,
        exception_result: typing.Optional[str] = None,
        common_info: typing.Optional[str] = None,
        model_name: typing.Optional[str] = None,
    ):
    """Update the performance csv file with the latest performance data."""
    print(f"Attaching performance metrics of models to perf.csv")
    # read perf.csv
    perf_csv_df = df_strip_columns(pd.read_csv(perf_csv))

    # handle multiple_results, single_result, and exception_result
    if multiple_results:
        perf_csv_df = handle_multiple_results(
            perf_csv_df,
            multiple_results,
            common_info,
            model_name,
        )
    elif single_result:
        perf_csv_df = handle_single_result(perf_csv_df, single_result)
    elif exception_result:
        perf_csv_df = handle_exception_result(
            perf_csv_df, exception_result
        )
    else:
        print("No results to update in perf.csv")

    # write new perf.csv
    # Note that this file will also generate a perf_entry.csv regardless of the output file args.
    perf_csv_df.to_csv(perf_csv, index=False)


class UpdatePerfCsv:
    """Class to update the performance csv file with the latest performance data.

    This class is used to update the performance csv file with the latest performance data.
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize the UpdatePerfCsv class.

        Args:
            args: The command-line arguments.
        """
        self.args = args
        self.return_status = False

    def run(self):
        """Update the performance csv file with the latest performance data."""
        print(f"Updating performance metrics of models perf.csv to database")
        # read perf.csv
        perf_csv_df = df_strip_columns(pd.read_csv(self.args.perf_csv))

        # handle multiple_results, single_result, and exception_result
        if self.args.multiple_results:
            perf_csv_df = handle_multiple_results(
                perf_csv_df,
                self.args.multiple_results,
                self.args.common_info,
                self.args.model_name,
            )
        elif self.args.single_result:
            perf_csv_df = handle_single_result(perf_csv_df, self.args.single_result)
        elif self.args.exception_result:
            perf_csv_df = handle_exception_result(
                perf_csv_df, self.args.exception_result
            )
        else:
            print("No results to update in perf.csv")

        # write new perf.csv
        # Note that this file will also generate a perf_entry.csv regardless of the output file args.
        perf_csv_df.to_csv(self.args.perf_csv, index=False)

        self.return_status = True
        return self.return_status

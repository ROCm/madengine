"""Module for converting a CSV file to an HTML file.

This module is responsible for converting a CSV file to an HTML file.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in imports
import os
import argparse

# third-party imports
import pandas as pd


def convert_csv_to_html(file_path: str):
    """Convert the CSV file to an HTML file.

    Args:
        file_path: The path to the CSV file.
    """
    # get file names
    base_path = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_name)[0]

    output_name = ""
    if base_path:
        output_name = base_path + "/"
    output_name += file_name + ".html"
    # read csv
    df = pd.read_csv(file_path)

    # Use beautiful formatting for dataframe display
    try:
        from madengine.utils.log_formatting import print_dataframe_beautiful

        print_dataframe_beautiful(df, f"Converting CSV: {file_name}")
    except ImportError:
        # Fallback to basic formatting if utils not available
        print(f"\n📊 Converting CSV: {file_name}")
        print("=" * 80)
        print(df.to_string(max_rows=20, max_cols=10))
        print("=" * 80)

    # Use the .to_html() to get your table in html
    df_html = df.to_html(index=False)
    perf_html = open(output_name, "w")
    n = perf_html.write(df_html)
    perf_html.close()


class ConvertCsvToHtml:
    def __init__(self, args: argparse.Namespace):
        """Initialize the ConvertCsvToHtml object.

        Args:
            args: The command-line arguments.
        """
        self.args = args
        self.return_status = False

    def run(self):
        """Convert the CSV file to an HTML file."""
        file_path = self.args.csv_file_path
        print(f"Converting CSV file to HTML file: {file_path}")

        # get file names
        base_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]

        output_name = ""
        if base_path:
            output_name = base_path + "/"

        output_name += file_name + ".html"

        # read csv
        df = pd.read_csv(file_path)

        # Use beautiful formatting for dataframe display
        try:
            from madengine.utils.log_formatting import print_dataframe_beautiful

            print_dataframe_beautiful(df, f"CSV Data from {file_name}")
        except ImportError:
            # Fallback to basic formatting if utils not available
            print(f"\n📊 CSV Data from {file_name}")
            print("=" * 80)
            print(df.to_string(max_rows=20, max_cols=10))
            print("=" * 80)

        # Use the .to_html() to get your table in html
        df_html = df.to_html(index=False)
        perf_html = open(output_name, "w")
        n = perf_html.write(df_html)
        perf_html.close()

        self.return_status = True
        return self.return_status

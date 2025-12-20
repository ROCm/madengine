"""Module for converting CSV files to HTML reports.

This module provides functionality to convert CSV files to HTML format
for generating performance reports and visualizations.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import argparse
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def convert_csv_to_html(
    file_path: str,
    output_path: Optional[str] = None,
    include_index: bool = False
) -> str:
    """Convert a CSV file to an HTML file.

    Args:
        file_path: The path to the CSV file.
        output_path: Optional custom output path. If None, creates HTML in same directory.
        include_index: Whether to include DataFrame index in HTML output.

    Returns:
        The path to the generated HTML file.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is not a CSV file.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    # Validate input
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    if not file_path.endswith('.csv'):
        raise ValueError(f"File must be a CSV file: {file_path}")

    # Determine output path
    if output_path is None:
        base_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]
        
        output_path = os.path.join(base_path, f"{file_name}.html") if base_path else f"{file_name}.html"

    # Read CSV file
    logger.info(f"Reading CSV file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise

    # Display DataFrame (with beautiful formatting if available)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        from madengine.utils.log_formatting import print_dataframe_beautiful
        print_dataframe_beautiful(df, f"Converting CSV: {file_name}")
    except ImportError:
        # Fallback to basic formatting if utils not available
        print(f"\n📊 Converting CSV: {file_name}")
        print("=" * 80)
        print(df.to_string(max_rows=20, max_cols=10))
        print("=" * 80)

    # Convert DataFrame to HTML
    logger.info(f"Converting to HTML: {output_path}")
    df_html = df.to_html(index=include_index)
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as html_file:
        html_file.write(df_html)

    logger.info(f"✅ Successfully converted {file_path} to {output_path}")
    return output_path


class ConvertCsvToHtml:
    """Handler class for CSV to HTML conversion command.
    
    This class provides a command-line interface wrapper for converting
    CSV files to HTML format.
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize the ConvertCsvToHtml handler.

        Args:
            args: Command-line arguments containing csv_file_path.
        """
        self.args = args
        self.return_status = False

    def run(self) -> bool:
        """Execute the CSV to HTML conversion.
        
        Returns:
            True if conversion was successful, False otherwise.
        """
        file_path = self.args.csv_file_path
        
        print("\n" + "=" * 80)
        print("🔄 CONVERTING CSV TO HTML REPORT")
        print("=" * 80)
        print(f"📂 Input file: {file_path}")

        try:
            output_path = convert_csv_to_html(file_path)
            print(f"📄 Output file: {output_path}")
            print("✅ Conversion completed successfully")
            print("=" * 80 + "\n")
            self.return_status = True
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("=" * 80 + "\n")
            self.return_status = False
        except ValueError as e:
            print(f"❌ Error: {e}")
            print("=" * 80 + "\n")
            self.return_status = False
        except Exception as e:
            print(f"❌ Unexpected error during conversion: {e}")
            logger.exception("Conversion failed")
            print("=" * 80 + "\n")
            self.return_status = False

        return self.return_status


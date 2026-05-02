"""Module for converting CSV files to email-ready HTML reports.

This module provides functionality to convert multiple CSV files in a directory
to a consolidated HTML report suitable for email distribution.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import argparse
import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def find_csv_files(directory: str) -> List[str]:
    """Find all CSV files in the specified directory.

    Args:
        directory: Path to the directory to search.

    Returns:
        List of CSV file paths found in the directory.
    """
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(directory, filename))
    return sorted(csv_files)


def csv_to_html_section(file_path: str) -> Tuple[str, str]:
    """Convert a CSV file to an HTML section with header.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Tuple of (section_name, html_content).
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Get section name from file path
    base_name = os.path.basename(file_path)
    section_name = os.path.splitext(base_name)[0]
    
    # Convert DataFrame to HTML
    html_table = df.to_html(index=False)
    
    # Create HTML section with header
    html_section = f"<h2>{section_name}</h2>\n{html_table}\n"
    
    return section_name, html_section


def convert_directory_csvs_to_html(
    directory_path: str,
    output_file: str = "run_results.html"
) -> str:
    """Convert all CSV files in a directory to a single HTML file.

    Args:
        directory_path: Path to the directory containing CSV files.
        output_file: Name of the output HTML file.

    Returns:
        Path to the generated HTML file.

    Raises:
        NotADirectoryError: If the path is not a directory.
        FileNotFoundError: If the directory does not exist.
    """
    # Validate input
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")

    # Find all CSV files
    csv_files = find_csv_files(directory_path)
    
    if not csv_files:
        logger.warning(f"No CSV files found in directory: {directory_path}")
        print(f"⚠️  No CSV files found in {directory_path}")
        return None

    print(f"📊 Found {len(csv_files)} CSV file(s) to process")
    
    # Process each CSV file and combine HTML
    full_html_content = ""
    for csv_file in csv_files:
        try:
            section_name, html_section = csv_to_html_section(csv_file)
            full_html_content += html_section
            logger.info(f"Processed: {section_name}")
            print(f"  ✓ Converted {os.path.basename(csv_file)}")
        except Exception as e:
            logger.error(f"Failed to process {csv_file}: {e}")
            print(f"  ✗ Failed to convert {os.path.basename(csv_file)}: {e}")

    # Write combined HTML to output file
    output_path = os.path.join(directory_path, output_file) if directory_path != "." else output_file
    
    with open(output_path, 'w', encoding='utf-8') as html_file:
        html_file.write(full_html_content)
    
    logger.info(f"Generated HTML report: {output_path}")
    return output_path


class ConvertCsvToEmail:
    """Handler class for CSV to email-ready HTML conversion command.
    
    This class provides a command-line interface wrapper for converting
    multiple CSV files in a directory to a consolidated HTML report.
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize the ConvertCsvToEmail handler.

        Args:
            args: Command-line arguments containing path to CSV directory.
        """
        self.args = args
        self.return_status = False

    def run(self) -> bool:
        """Execute the CSV to email HTML conversion.
        
        Returns:
            True if conversion was successful, False otherwise.
        """
        directory_path = getattr(self.args, 'csv_file_path', '.') or '.'
        output_file = getattr(self.args, 'output_file', 'run_results.html')
        
        print("\n" + "=" * 80)
        print("📧 CONVERTING CSV FILES TO EMAIL REPORT")
        print("=" * 80)
        print(f"📂 Input directory: {directory_path}")

        try:
            output_path = convert_directory_csvs_to_html(directory_path, output_file)
            
            if output_path:
                print(f"📄 Output file: {output_path}")
                print("✅ Email report generated successfully")
            else:
                print("ℹ️  No files to process")
                
            print("=" * 80 + "\n")
            self.return_status = True
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"❌ Error: {e}")
            print("=" * 80 + "\n")
            self.return_status = False
        except Exception as e:
            print(f"❌ Unexpected error during conversion: {e}")
            logger.exception("Email report generation failed")
            print("=" * 80 + "\n")
            self.return_status = False

        return self.return_status


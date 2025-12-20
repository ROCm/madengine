#!/usr/bin/env python3
"""
Report command for madengine CLI

This module provides report generation commands including CSV to HTML
and CSV to email conversions.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
from pathlib import Path

import typer
from rich.panel import Panel

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.reporting.csv_to_html import ConvertCsvToHtml
from madengine.reporting.csv_to_email import ConvertCsvToEmail

from ..constants import ExitCode
from ..utils import console, setup_logging, create_args_namespace


# Create a sub-app for report commands
report_app = typer.Typer(
    name="report",
    help="📊 Generate reports from CSV files",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@report_app.command("to-html")
def to_html(
    csv_file: Annotated[
        str,
        typer.Option(
            "--csv-file",
            help="Path to the CSV file to convert to HTML"
        ),
    ],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    📄 Convert a single CSV file to HTML report.
    
    This command converts a CSV file to an HTML table format,
    useful for viewing performance metrics in a web browser.
    
    Examples:
        madengine-cli report to-html --csv-file perf_amd.csv
        madengine-cli report to-html --csv-file results/perf_mi300.csv
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"📄 [bold cyan]Converting CSV to HTML[/bold cyan]\n"
            f"Input file: [yellow]{csv_file}[/yellow]",
            title="CSV to HTML Report",
            border_style="blue",
        )
    )

    # Validate input
    if not os.path.exists(csv_file):
        console.print(f"❌ [bold red]Error: CSV file not found: {csv_file}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    
    if not os.path.isfile(csv_file):
        console.print(f"❌ [bold red]Error: Path is not a file: {csv_file}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    
    if not csv_file.endswith('.csv'):
        console.print(f"❌ [bold red]Error: File must be a CSV file: {csv_file}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)

    try:
        # Create args namespace for compatibility with existing code
        args = create_args_namespace(csv_file_path=csv_file)
        
        # Use ConvertCsvToHtml class
        converter = ConvertCsvToHtml(args=args)
        result = converter.run()
        
        if result:
            # Determine output file name
            output_file = str(Path(csv_file).with_suffix('.html'))
            console.print(f"✅ [bold green]Successfully converted to: {output_file}[/bold green]")
        else:
            console.print("❌ [bold red]Conversion failed[/bold red]")
            raise typer.Exit(ExitCode.FAILURE)

    except Exception as e:
        console.print(f"💥 [bold red]Conversion failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


@report_app.command("to-email")
def to_email(
    directory: Annotated[
        str,
        typer.Option(
            "--directory",
            "--dir",
            help="Path to directory containing CSV files to consolidate"
        ),
    ] = ".",
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output HTML filename"
        ),
    ] = "run_results.html",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    📧 Convert all CSV files in a directory to consolidated email-ready HTML report.
    
    This command scans a directory for CSV files and combines them into a single
    HTML report with sections for each CSV file, suitable for email distribution.
    
    Examples:
        madengine-cli report to-email
        madengine-cli report to-email --directory ./results
        madengine-cli report to-email --dir ./results --output summary.html
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"📧 [bold cyan]Converting CSV Files to Email Report[/bold cyan]\n"
            f"Input directory: [yellow]{directory}[/yellow]\n"
            f"Output file: [yellow]{output}[/yellow]",
            title="CSV to Email Report",
            border_style="blue",
        )
    )

    # Validate input
    if not os.path.exists(directory):
        console.print(f"❌ [bold red]Error: Directory not found: {directory}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    
    if not os.path.isdir(directory):
        console.print(f"❌ [bold red]Error: Path is not a directory: {directory}[/bold red]")
        console.print(f"💡 [cyan]Tip: Use 'to-html' command for single CSV files[/cyan]")
        raise typer.Exit(ExitCode.FAILURE)

    try:
        # Create args namespace for compatibility with existing code
        # The old code expects 'csv_file_path' to be the directory
        args = create_args_namespace(csv_file_path=directory, output_file=output)
        
        # Use ConvertCsvToEmail class
        converter = ConvertCsvToEmail(args=args)
        result = converter.run()
        
        if result:
            output_path = os.path.join(directory, output) if directory != "." else output
            console.print(f"✅ [bold green]Successfully generated email report: {output_path}[/bold green]")
        else:
            console.print("⚠️  [yellow]No CSV files found to process[/yellow]")

    except Exception as e:
        console.print(f"💥 [bold red]Report generation failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


# Export the report app
def report() -> typer.Typer:
    """Return the report sub-app."""
    return report_app


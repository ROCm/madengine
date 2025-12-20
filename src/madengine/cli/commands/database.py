#!/usr/bin/env python3
"""
Database command for madengine CLI

This module provides MongoDB upload functionality.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os

import typer
from rich.panel import Panel

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.database.mongodb import MongoDBHandler

from ..constants import ExitCode
from ..utils import console, setup_logging, create_args_namespace


def database(
    csv_file: Annotated[
        str,
        typer.Option(
            "--csv-file",
            help="Path to the CSV file to upload to MongoDB"
        ),
    ] = "perf_entry.csv",
    database_name: Annotated[
        str,
        typer.Option(
            "--database-name",
            "--db",
            help="Name of the MongoDB database"
        ),
    ] = None,
    collection_name: Annotated[
        str,
        typer.Option(
            "--collection-name",
            "--collection",
            help="Name of the MongoDB collection"
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    💾 Upload CSV data to MongoDB database.
    
    This command uploads CSV file data to a specified MongoDB database and collection.
    MongoDB connection details are read from environment variables:
    - MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASSWORD
    
    Examples:
        madengine-cli database --csv-file perf.csv --db mydb --collection results
        madengine-cli database --csv-file perf_entry.csv --database-name test --collection-name perf
    """
    setup_logging(verbose)

    # Validate required parameters
    if not database_name:
        console.print("❌ [bold red]Error: --database-name is required[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    
    if not collection_name:
        console.print("❌ [bold red]Error: --collection-name is required[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)

    console.print(
        Panel(
            f"💾 [bold cyan]Uploading to MongoDB[/bold cyan]\n"
            f"CSV file: [yellow]{csv_file}[/yellow]\n"
            f"Database: [yellow]{database_name}[/yellow]\n"
            f"Collection: [yellow]{collection_name}[/yellow]",
            title="MongoDB Upload",
            border_style="blue",
        )
    )

    # Validate CSV file exists
    if not os.path.exists(csv_file):
        console.print(f"❌ [bold red]Error: CSV file not found: {csv_file}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)

    try:
        # Create args namespace for compatibility
        args = create_args_namespace(
            csv_file_path=csv_file,
            database_name=database_name,
            collection_name=collection_name
        )
        
        # Use MongoDBHandler class
        handler = MongoDBHandler(args=args)
        result = handler.run()
        
        if result:
            console.print(f"✅ [bold green]Successfully uploaded to MongoDB[/bold green]")
        else:
            console.print("❌ [bold red]Upload failed[/bold red]")
            raise typer.Exit(ExitCode.FAILURE)

    except Exception as e:
        console.print(f"💥 [bold red]Upload failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


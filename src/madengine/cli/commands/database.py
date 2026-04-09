#!/usr/bin/env python3
"""
Database command for madengine CLI - MongoDB upload.

Modern implementation with auto-detection and intelligent defaults.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
from pathlib import Path

import typer
from rich.panel import Panel
from rich.console import Console

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from madengine.database.mongodb import (
    upload_file_to_mongodb,
    MongoDBConfig,
    UploadOptions
)
from ..constants import ExitCode
from ..utils import setup_logging

console = Console()


def database(
    file: Annotated[
        str,
        typer.Option(
            "--file", "-f",
            help="Path to file (CSV or JSON, auto-detected)"
        ),
    ],
    database: Annotated[
        str,
        typer.Option(
            "--database", "--db",
            help="MongoDB database name"
        ),
    ],
    collection: Annotated[
        str,
        typer.Option(
            "--collection", "-c",
            help="MongoDB collection name"
        ),
    ],
    unique_key: Annotated[
        str,
        typer.Option(
            "--unique-key", "-k",
            help="Unique field(s) for deduplication (comma-separated, auto-detected if not specified)"
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for bulk operations"
        ),
    ] = 1000,
    no_upsert: Annotated[
        bool,
        typer.Option(
            "--no-upsert",
            help="Insert only (don't update existing documents)"
        ),
    ] = False,
    no_index: Annotated[
        bool,
        typer.Option(
            "--no-index",
            help="Skip automatic index creation"
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate without uploading"
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help="Verbose output"
        ),
    ] = False,
) -> None:
    """
    💾 Upload CSV or JSON files to MongoDB.
    
    Supports intelligent type preservation, automatic deduplication,
    and bulk operations for optimal performance.
    
    \b
    Examples:
        # Upload JSON with auto-detection
        madengine database -f perf_entry_super.json --db mydb -c perf_super
        
        # Upload CSV with custom unique key
        madengine database -f perf.csv --db test -c results -k model,timestamp
        
        # Dry run to validate
        madengine database -f data.json --db test -c data --dry-run
        
    \b
    Environment Variables:
        MONGO_HOST        MongoDB host (default: localhost)
        MONGO_PORT        MongoDB port (default: 27017)
        MONGO_USER        MongoDB username
        MONGO_PASSWORD    MongoDB password
    """
    
    setup_logging(verbose)
    
    # Display configuration
    file_path = Path(file)
    
    console.print(
        Panel(
            f"💾 [bold cyan]MongoDB Upload[/bold cyan]\n\n"
            f"File: [yellow]{file_path.name}[/yellow]\n"
            f"Database: [yellow]{database}[/yellow]\n"
            f"Collection: [yellow]{collection}[/yellow]\n"
            f"Unique Key: [yellow]{unique_key or 'auto-detect'}[/yellow]\n"
            f"Mode: [yellow]{'Dry Run' if dry_run else 'Upload'}[/yellow]",
            border_style="cyan",
        )
    )
    
    # Validate file exists
    if not file_path.exists():
        console.print(f"❌ [bold red]File not found: {file}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    
    # Prepare configuration
    config = MongoDBConfig.from_env()
    
    # Parse unique fields
    unique_fields = None
    if unique_key:
        unique_fields = [k.strip() for k in unique_key.split(',')]
    
    # Prepare options
    options = UploadOptions(
        unique_fields=unique_fields,
        upsert=not no_upsert,
        batch_size=batch_size,
        create_indexes=not no_index,
        dry_run=dry_run
    )
    
    try:
        # Perform upload
        result = upload_file_to_mongodb(
            file_path=str(file_path),
            database_name=database,
            collection_name=collection,
            config=config,
            options=options
        )
        
        # Display results
        console.print()
        result.print_summary()
        
        # Show errors if any
        if result.errors and verbose:
            console.print("\n⚠️  [yellow]Errors:[/yellow]")
            for i, error in enumerate(result.errors[:10], 1):
                console.print(f"   {i}. {error}")
            if len(result.errors) > 10:
                console.print(f"   ... and {len(result.errors) - 10} more errors")
        
        # Exit with appropriate code
        if result.status == "success":
            raise typer.Exit(ExitCode.SUCCESS)
        elif result.status == "partial":
            raise typer.Exit(ExitCode.SUCCESS if result.documents_inserted + result.documents_updated > 0 else ExitCode.FAILURE)
        else:
            raise typer.Exit(ExitCode.FAILURE)
            
    except typer.Exit:
        # Re-raise typer.Exit without catching it
        raise
    except Exception as e:
        console.print(f"\n💥 [bold red]Upload failed:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)

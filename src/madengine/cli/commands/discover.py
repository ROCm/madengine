#!/usr/bin/env python3
"""
Discover command for madengine CLI

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from typing import List

import typer
from rich.panel import Panel

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.utils.discover_models import DiscoverModels

from ..constants import ExitCode
from ..utils import console, setup_logging, split_comma_separated_tags, create_args_namespace


def discover(
    tags: Annotated[
        List[str],
        typer.Option("--tags", "-t", help="Model tags to discover (can specify multiple)"),
    ] = [],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🔍 Discover all models in the project.

    This command discovers all available models in the project based on the
    specified tags. If no tags are provided, all models will be discovered.
    """
    setup_logging(verbose)

    # Process tags to handle comma-separated values
    processed_tags = split_comma_separated_tags(tags)

    console.print(
        Panel(
            f"🔍 [bold cyan]Discovering Models[/bold cyan]\n"
            f"Tags: [yellow]{processed_tags if processed_tags else 'All models'}[/yellow]",
            title="Model Discovery",
            border_style="blue",
        )
    )

    try:
        # Create args namespace similar to mad.py
        args = create_args_namespace(tags=processed_tags)
        
        # Use DiscoverModels class
        # Note: DiscoverModels prints output directly and returns None
        discover_models_instance = DiscoverModels(args=args)
        result = discover_models_instance.run()
        
        console.print("✅ [bold green]Model discovery completed successfully[/bold green]")

    except Exception as e:
        console.print(f"💥 [bold red]Model discovery failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


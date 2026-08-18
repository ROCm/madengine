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
    full: Annotated[
        bool,
        typer.Option("--full", "-f", help="Output full JSON with all discovered models and their tags"),
    ] = False,
    json: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output plain JSON only (no formatting, no status messages)"),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🔍 Discover all models in the project.

    This command discovers all available models in the project based on the
    specified tags. If no tags are provided, all models will be discovered.

    **Scoped tags** (``scope/tag``): exactly one ``/`` and no ``:`` in the tag
    limits selection to models under ``scripts/<scope>/`` (e.g.
    ``MAD-private/inference`` → models named ``MAD-private/...`` with tag
    ``inference``). Use ``scope/all`` for every model in that scope.

    **Full JSON output** (``--full``): outputs complete model cards with all tags
    and metadata in JSON format, similar to ``--tags`` output but for all models.

    **Plain JSON output** (``--json``): outputs only pure JSON without any formatting
    or status messages. Useful for piping to other tools or CI/CD pipelines.
    """
    # Skip logging setup if json mode is enabled
    if not json:
        setup_logging(verbose)

    # Process tags to handle comma-separated values
    processed_tags = split_comma_separated_tags(tags)

    # Skip console output if json mode is enabled
    if not json:
        console.print(
            Panel(
                f"🔍 [bold cyan]Discovering Models[/bold cyan]\n"
                f"Tags: [yellow]{processed_tags if processed_tags else 'All models'}[/yellow]\n"
                f"Full JSON: [yellow]{full}[/yellow]",
                title="Model Discovery",
                border_style="blue",
            )
        )

    try:
        # Create args namespace similar to mad.py
        args = create_args_namespace(tags=processed_tags, full=full, json=json)

        # Use DiscoverModels class
        # Note: DiscoverModels prints output directly and returns None
        discover_models_instance = DiscoverModels(args=args)
        result = discover_models_instance.run()

        # Skip success message if json mode is enabled
        if not json:
            console.print("✅ [bold green]Model discovery completed successfully[/bold green]")

    except Exception as e:
        # In json mode, write errors to stderr to keep stdout clean
        if json:
            import sys
            print(f"Error: {e}", file=sys.stderr)
        else:
            console.print(f"💥 [bold red]Model discovery failed: {e}[/bold red]")
            if verbose:
                console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)


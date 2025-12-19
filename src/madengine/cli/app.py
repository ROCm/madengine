#!/usr/bin/env python3
"""
Main CLI Application for madengine

This module contains the main Typer app and entry point for the madengine CLI.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import sys

import typer
from rich.traceback import install

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from .commands import build, run, discover
from .constants import ExitCode
from .utils import console

# Install rich traceback handler for better error displays
install(show_locals=True)

# Initialize the main Typer app
app = typer.Typer(
    name="madengine-cli",
    help="🚀 madengine Distributed Orchestrator - Build and run AI models in distributed scenarios",
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
)

# Register commands
app.command()(build)
app.command()(run)
app.command()(discover)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", help="Show version and exit")
    ] = False,
) -> None:
    """
    🚀 madengine Distributed Orchestrator

    Modern CLI for building and running AI models in distributed scenarios.
    Built with Typer and Rich for a beautiful, production-ready experience.
    """
    if version:
        # You might want to get the actual version from your package
        console.print(
            "🚀 [bold cyan]madengine-cli[/bold cyan] version [green]2.0.0[/green]"
        )
        raise typer.Exit()

    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        ctx.exit()


def cli_main() -> None:
    """Entry point for the CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Operation cancelled by user[/yellow]")
        sys.exit(ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]Unexpected error: {e}[/bold red]")
        console.print_exception()
        sys.exit(ExitCode.FAILURE)


if __name__ == "__main__":
    cli_main()


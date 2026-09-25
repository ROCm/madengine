#!/usr/bin/env python3
"""
Setup-page command for madengine CLI.

Generates a self-contained, PyTorch-style setup picker HTML page from the
current repository's ``models.json`` and the madengine context schema.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from typing import Optional

import typer
from rich.panel import Panel

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.setup_page.generator import DEFAULT_INSTALL_CMD, generate_setup_page

from ..constants import ExitCode
from ..utils import console, setup_logging


def setup_page(
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output HTML file path"),
    ] = "index.html",
    title: Annotated[
        str,
        typer.Option("--title", help="Page title / heading"),
    ] = "madengine Setup Picker",
    repo_url: Annotated[
        Optional[str],
        typer.Option("--repo-url", help="Model repo URL (used for clone instructions)"),
    ] = None,
    install_cmd: Annotated[
        str,
        typer.Option("--install-cmd", help="madengine install command shown on the page"),
    ] = DEFAULT_INSTALL_CMD,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🧩 Generate a setup picker page from this repo's models.json.

    Produces a single self-contained HTML file where users select every
    relevant madengine dimension (model/tags plus the full context schema) and
    copy the exact ``madengine run`` command. The page has no external assets,
    so it can be published directly to GitHub Pages.
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"🧩 [bold cyan]Generating Setup Page[/bold cyan]\n"
            f"Output: [yellow]{output}[/yellow]",
            title="Setup Page",
            border_style="blue",
        )
    )

    try:
        written = generate_setup_page(
            output=output,
            title=title,
            repo_url=repo_url or "",
            install_cmd=install_cmd,
        )
        console.print(
            f"✅ [bold green]Setup page written to:[/bold green] [cyan]{written}[/cyan]"
        )
    except FileNotFoundError:
        console.print(
            "💥 [bold red]models.json not found in the current directory.[/bold red]"
        )
        raise typer.Exit(ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]Setup page generation failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)

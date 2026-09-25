#!/usr/bin/env python3
"""
Report command for madengine CLI

This module provides report generation commands including CSV to HTML
and CSV to email conversions.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.reporting.csv_to_html import ConvertCsvToHtml
from madengine.reporting.csv_to_email import ConvertCsvToEmail
from madengine.reporting.tracelens_report import (
    TraceLensNotInstalledError,
    compare_tracelens_reports,
    discover_traces,
    generate_tracelens_reports,
)

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
    csv_file_path: Annotated[
        str,
        typer.Option(
            "--csv-file-path",
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
        madengine report to-html --csv-file-path perf_amd.csv
        madengine report to-html --csv-file-path results/perf_mi300.csv
    """
    setup_logging(verbose)

    console.print(
        Panel(
            f"📄 [bold cyan]Converting CSV to HTML[/bold cyan]\n"
            f"Input file: [yellow]{csv_file_path}[/yellow]",
            title="CSV to HTML Report",
            border_style="blue",
        )
    )

    # Validate input
    if not os.path.exists(csv_file_path):
        console.print(f"❌ [bold red]Error: CSV file not found: {csv_file_path}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    
    if not os.path.isfile(csv_file_path):
        console.print(f"❌ [bold red]Error: Path is not a file: {csv_file_path}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    
    if not csv_file_path.endswith('.csv'):
        console.print(f"❌ [bold red]Error: File must be a CSV file: {csv_file_path}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)

    try:
        # Create args namespace for compatibility with existing code
        args = create_args_namespace(csv_file_path=csv_file_path)
        
        # Use ConvertCsvToHtml class
        converter = ConvertCsvToHtml(args=args)
        result = converter.run()
        
        if result:
            # Determine output file name
            output_file = str(Path(csv_file_path).with_suffix('.html'))
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
        madengine report to-email
        madengine report to-email --directory ./results
        madengine report to-email --dir ./results --output summary.html
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


def _print_tracelens_results(summary: dict) -> None:
    """Render the analyzer's per-trace results as a table."""
    results = summary.get("results") or []
    if not results:
        return

    table = Table(title="TraceLens reports", show_lines=False)
    table.add_column("Status")
    table.add_column("Trace", overflow="fold")
    table.add_column("Kind")
    table.add_column("Report")
    table.add_column("Detail", overflow="fold")

    # Trace paths and TraceLens error text routinely contain square brackets
    # (e.g. "[rocprofv3]"), which rich would otherwise parse as markup.
    styles = {"SUCCESS": "green", "FAILURE": "red", "SKIPPED": "yellow"}
    for result in results:
        status = str(result.get("status", ""))
        table.add_row(
            f"[{styles.get(status, 'white')}]{status}[/]",
            escape(str(result.get("trace_file", ""))),
            escape(str(result.get("kind", ""))),
            escape(
                str(result.get("tracelens_tool", "")).replace(
                    "TraceLens_generate_perf_report_", ""
                )
            ),
            escape(str(result.get("detail", ""))),
        )
    console.print(table)


@report_app.command("tracelens")
def tracelens(
    root: Annotated[
        str,
        typer.Option(
            "--root",
            "-r",
            help="Directory to search recursively for trace artifacts",
        ),
    ] = ".",
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory for generated reports"),
    ] = "tracelens_output",
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Restrict analysis to one trace kind: auto, pytorch, rocprof, pftrace, collective",
        ),
    ] = "auto",
    python: Annotated[
        Optional[str],
        typer.Option("--python", help="Interpreter that has TraceLens installed"),
    ] = None,
    gpu_arch: Annotated[
        Optional[str],
        typer.Option(
            "--gpu-arch",
            help="TraceLens GPU arch platform (e.g. MI300X) for roofline bound classification",
        ),
    ] = None,
    world_size: Annotated[
        int,
        typer.Option(
            "--world-size",
            help="Rank count for the collective report (default: number of traces found)",
        ),
    ] = 0,
    max_traces: Annotated[
        int,
        typer.Option("--max-traces", help="Cap traces analyzed per kind (0 = no cap)"),
    ] = 0,
    discover_only: Annotated[
        bool,
        typer.Option(
            "--discover-only",
            help="List discovered traces without running TraceLens",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    🔬 Generate TraceLens performance reports from collected GPU traces.

    Discovers trace artifacts a run left behind (rocprof_output/,
    torch_profiler_output/, slurm_results/, k8s_results/) and generates the
    matching TraceLens report for each: operator and roofline analysis for
    torch.profiler traces, kernel summaries for rocprofv3 JSON, and
    activity/API/memory-copy reports for pftrace.

    Requires TraceLens: pip install 'madengine[tracelens]'

    Examples:
        madengine report tracelens
        madengine report tracelens --discover-only
        madengine report tracelens --gpu-arch MI300X
        madengine report tracelens --root slurm_results --mode collective --world-size 8
    """
    setup_logging(verbose)

    valid_modes = ("auto", "pytorch", "rocprof", "pftrace", "collective")
    if mode not in valid_modes:
        console.print(
            f"❌ [bold red]Error: invalid --mode '{mode}'. "
            f"Choose one of: {', '.join(valid_modes)}[/bold red]"
        )
        raise typer.Exit(ExitCode.INVALID_ARGS)

    if not os.path.isdir(root):
        console.print(f"❌ [bold red]Error: directory not found: {root}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)

    console.print(
        Panel(
            f"🔬 [bold cyan]TraceLens Analysis[/bold cyan]\n"
            f"Search root: [yellow]{root}[/yellow]\n"
            f"Output directory: [yellow]{output_dir}[/yellow]\n"
            f"Mode: [yellow]{mode}[/yellow]",
            title="TraceLens Report",
            border_style="blue",
        )
    )

    try:
        if discover_only:
            summary = discover_traces(root=root, output_dir=output_dir)
            discovered = summary.get("discovered") or {}
            if not discovered and not summary.get("unsupported"):
                console.print(
                    f"⚠️  [yellow]No trace artifacts found under {root}[/yellow]"
                )
            for kind, count in discovered.items():
                console.print(f"  [cyan]{kind}[/cyan]: {count} trace(s)")
            for item in summary.get("unsupported") or []:
                console.print(
                    f"  [yellow]unsupported[/yellow]: {escape(item['path'])} — "
                    f"{escape(item['reason'])}"
                )
            return

        summary = generate_tracelens_reports(
            root=root,
            output_dir=output_dir,
            mode=mode,
            python=python,
            gpu_arch=gpu_arch,
            world_size=world_size,
            max_traces=max_traces,
        )
    except TraceLensNotInstalledError as e:
        console.print(f"❌ [bold red]{escape(str(e))}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]TraceLens analysis failed: {escape(str(e))}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)

    _print_tracelens_results(summary)

    succeeded = int(summary.get("succeeded", 0))
    failed = int(summary.get("failed", 0))
    skipped = int(summary.get("skipped", 0))

    if not succeeded and not failed:
        console.print(
            f"⚠️  [yellow]No supported trace artifacts found under {root}. "
            "Stack a profiling tool (torch_profiler_dynolog, rocprofv3_lightweight, "
            "rocprofv3_perfetto) on the run first.[/yellow]"
        )
        return

    console.print(
        f"📄 [bold]Reports written to:[/bold] [yellow]{output_dir}[/yellow] "
        f"(summary: {os.path.join(output_dir, 'tracelens_summary.csv')})"
    )
    if failed:
        console.print(
            f"⚠️  [yellow]{succeeded} report(s) generated, {failed} failed, "
            f"{skipped} skipped[/yellow]"
        )
        raise typer.Exit(ExitCode.FAILURE)

    console.print(
        f"✅ [bold green]{succeeded} report(s) generated"
        + (f", {skipped} skipped" if skipped else "")
        + "[/bold green]"
    )


@report_app.command("tracelens-compare")
def tracelens_compare(
    reports: Annotated[
        List[str],
        typer.Argument(
            help="Two or more TraceLens reports (.xlsx files or per-sheet CSV directories)"
        ),
    ],
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output comparison workbook"),
    ] = "tracelens_comparison.xlsx",
    names: Annotated[
        Optional[List[str]],
        typer.Option("--names", help="Display tag per report (repeat the flag)"),
    ] = None,
    python: Annotated[
        Optional[str],
        typer.Option("--python", help="Interpreter that has TraceLens installed"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """
    ⚖️  Compare two or more TraceLens reports into a single diff workbook.

    The first report is the baseline; every metric gains ``_diff`` and ``_pct``
    columns relative to it. Use this to quantify the effect of a change across
    two madengine runs.

    Examples:
        madengine report tracelens-compare baseline.xlsx candidate.xlsx
        madengine report tracelens-compare a.xlsx b.xlsx --names before --names after -o diff.xlsx
    """
    setup_logging(verbose)

    missing = [r for r in reports if not os.path.exists(r)]
    if missing:
        console.print(
            f"❌ [bold red]Error: report(s) not found: {', '.join(missing)}[/bold red]"
        )
        raise typer.Exit(ExitCode.FAILURE)

    console.print(
        Panel(
            f"⚖️  [bold cyan]Comparing TraceLens Reports[/bold cyan]\n"
            f"Reports: [yellow]{', '.join(reports)}[/yellow]\n"
            f"Output: [yellow]{output}[/yellow]",
            title="TraceLens Comparison",
            border_style="blue",
        )
    )

    try:
        summary = compare_tracelens_reports(
            reports=reports, output=output, names=names or (), python=python
        )
    except (TraceLensNotInstalledError, ValueError) as e:
        console.print(f"❌ [bold red]{escape(str(e))}[/bold red]")
        raise typer.Exit(ExitCode.FAILURE)
    except Exception as e:
        console.print(f"💥 [bold red]Comparison failed: {escape(str(e))}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(ExitCode.FAILURE)

    if summary.get("status") == "SUCCESS":
        console.print(f"✅ [bold green]Comparison written to: {output}[/bold green]")
    else:
        console.print(
            f"💥 [bold red]Comparison failed: "
            f"{escape(str(summary.get('detail', '')))}[/bold red]"
        )
        raise typer.Exit(ExitCode.FAILURE)


# Export the report app
def report() -> typer.Typer:
    """Return the report sub-app."""
    return report_app


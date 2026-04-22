#!/usr/bin/env python3
"""
Utility functions for formatting and displaying data in logs.

This module provides enhanced formatting utilities for better log readability,
including dataframe formatting and other display utilities.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pandas as pd
import typing
from rich.table import Table
from rich.console import Console as RichConsole
from rich.text import Text


def format_dataframe_for_log(
    df: pd.DataFrame, title: str = "DataFrame", max_rows: int = 20, max_cols: int = 10
) -> str:
    """
    Format a pandas DataFrame for beautiful log output.

    Args:
        df: The pandas DataFrame to format
        title: Title for the dataframe display
        max_rows: Maximum number of rows to display (if None, use all rows)
        max_cols: Maximum number of columns to display

    Returns:
        str: Beautifully formatted string representation of the DataFrame
    """
    if df.empty:
        return f"\n📊 {title}\n{'='*60}\n❌ DataFrame is empty\n{'='*60}\n"

    # Define key columns to display for performance results
    key_columns = [
        "model",
        "n_gpus",
        "docker_file",
        "machine_name",
        "gpu_architecture",
        "performance",
        "metric",
        "status",
        "dataname",
    ]

    # Filter DataFrame to show only key columns that exist
    available_columns = [col for col in key_columns if col in df.columns]
    if available_columns:
        display_df = df[available_columns].copy()
        total_columns_note = (
            f"(showing {len(available_columns)} of {len(df.columns)} columns)"
        )
    else:
        # If no key columns found, show all columns as fallback with truncation
        display_df = df.copy()
        total_columns_note = f"(showing all {len(df.columns)} columns)"
        if len(df.columns) > max_cols:
            display_df = display_df.iloc[:, :max_cols]
            total_columns_note = (
                f"(showing first {max_cols} of {len(df.columns)} columns)"
            )

    # Use all rows if max_rows is None
    if max_rows is None:
        max_rows = len(display_df)

    # Truncate rows if necessary (show latest rows)
    truncated_rows = False
    if len(display_df) > max_rows:
        display_df = display_df.tail(max_rows)
        truncated_rows = True

    # Create header
    header = f"\n📊 {title} {total_columns_note}\n"
    header += f"{'='*80}\n"
    if available_columns:
        header += f"📏 Shape: {df.shape[0]} rows × {len(available_columns)} key columns (total: {df.shape[1]} columns)\n"
    else:
        header += f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"

    if truncated_rows:
        header += f"⚠️  Display truncated: showing first {max_rows} rows\n"

    header += f"{'='*80}\n"

    # Format the DataFrame with nice styling
    formatted_df = display_df.to_string(
        index=True, max_rows=max_rows, width=None, float_format="{:.4f}".format
    )

    # Add some visual separators
    footer = f"\n{'='*80}\n"

    return header + formatted_df + footer


def format_dataframe_rich(
    df: pd.DataFrame, title: str = "DataFrame", max_rows: int = 20
) -> None:
    """
    Display a pandas DataFrame using Rich formatting for enhanced readability.

    Args:
        df: The pandas DataFrame to display
        title: Title for the table
        max_rows: Maximum number of rows to display
    """
    console = RichConsole()

    if df.empty:
        console.print(
            f"📊 [bold cyan]{title}[/bold cyan]: [red]DataFrame is empty[/red]"
        )
        return

    # Define key columns to display for performance results
    key_columns = [
        "model",
        "n_gpus",
        "machine_name",
        "gpu_architecture",
        "performance",
        "metric",
        "status",
        "dataname",
    ]

    # Filter DataFrame to show only key columns that exist
    available_columns = [col for col in key_columns if col in df.columns]
    if available_columns:
        display_df = df[available_columns]
        total_columns_note = (
            f"(showing {len(available_columns)} of {len(df.columns)} columns)"
        )
    else:
        # If no key columns found, show all columns as fallback
        display_df = df
        total_columns_note = f"(showing all {len(df.columns)} columns)"

    # Create Rich table
    table = Table(
        title=f"📊 {title} {total_columns_note}",
        show_header=True,
        header_style="bold magenta",
    )

    # Add index column
    table.add_column("Index", style="dim", width=8)

    # Add data columns
    for col in display_df.columns:
        table.add_column(str(col), style="cyan")

    # Add rows (truncate if necessary, show latest rows)
    if len(display_df) > max_rows:
        truncated_df = display_df.tail(max_rows)
        truncated_indices = truncated_df.index
        display_rows = max_rows
    else:
        truncated_df = display_df
        truncated_indices = truncated_df.index
        display_rows = len(truncated_df)

    for i in range(display_rows):
        row_data = [str(truncated_indices[i])]
        for col in truncated_df.columns:
            value = truncated_df.iloc[i][col]
            if pd.isna(value):
                row_data.append("[dim]NaN[/dim]")
            elif isinstance(value, float):
                row_data.append(f"{value:.4f}")
            else:
                row_data.append(str(value))
        table.add_row(*row_data)

    # Show truncation info
    if len(display_df) > max_rows:
        table.add_row(*["..." for _ in range(len(truncated_df.columns) + 1)])
        console.print(
            f"[yellow]⚠️  Showing latest {max_rows} of {len(display_df)} rows[/yellow]"
        )

    console.print(table)
    console.print(
        f"[green]✨ DataFrame shape: {df.shape[0]} rows × {len(available_columns)} key columns (total: {df.shape[1]} columns)[/green]"
    )


def print_dataframe_beautiful(
    df: pd.DataFrame, title: str = "Data", use_rich: bool = True
) -> None:
    """
    Print a pandas DataFrame with beautiful formatting.

    Args:
        df: The pandas DataFrame to print
        title: Title for the display
        use_rich: Whether to use Rich formatting (if available) or fall back to simple formatting
    """
    try:
        if use_rich:
            format_dataframe_rich(df, title)
        else:
            raise ImportError("Fallback to simple formatting")
    except (ImportError, Exception):
        # Fallback to simple but nice formatting
        formatted_output = format_dataframe_for_log(df, title)
        print(formatted_output)


def highlight_log_section(title: str, content: str, style: str = "info") -> str:
    """
    Create a highlighted log section with borders and styling.

    Args:
        title: Section title
        content: Section content
        style: Style type ('info', 'success', 'warning', 'error')

    Returns:
        str: Formatted log section
    """
    styles = {
        "info": {"emoji": "ℹ️", "border": "-"},
        "success": {"emoji": "✅", "border": "="},
        "warning": {"emoji": "⚠️", "border": "!"},
        "error": {"emoji": "❌", "border": "#"},
    }

    style_config = styles.get(style, styles["info"])
    emoji = style_config["emoji"]
    border_char = style_config["border"]

    border = border_char * 80
    header = f"\n{border}\n{emoji} {title.upper()}\n{border}"
    footer = f"{border}\n"

    return f"{header}\n{content}\n{footer}"

#!/usr/bin/env python3
"""
NaN debugging commands for the madengine CLI

`madengine nan <command>` is a thin front for the deterministic tools in
madengine.nan_doctor (standard library only, no LLM): run a command in a model's image on
the GPUs, group and replay NaN capsules, package a standalone reproducer, keep the
investigation ledger and report. Each command parses its own options; see
`madengine nan <command> --help`.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import typer

from madengine.nan_doctor.cli import COMMANDS
from madengine.nan_doctor.cli import main as nan_main

nan_app = typer.Typer(
    name="nan",
    help="🔎 Capture, replay and package NaNs of PyTorch workloads (nan-doctor)",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _register(name: str, help_text: str) -> None:
    @nan_app.command(
        name,
        help=help_text,
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "help_option_names": [],  # `--help` goes to the command's own parser
        },
    )
    def _command(ctx: typer.Context) -> None:
        raise typer.Exit(nan_main([name] + list(ctx.args)))


for _name, (_module, _convention, _help) in COMMANDS.items():
    _register(_name, _help)

#!/usr/bin/env python3
"""
Cleanup command for madengine CLI

Removes benchmark containers left behind by runs that died without getting to
clean up after themselves.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import typing

import typer

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.core import lifecycle

from ..constants import ExitCode
from ..utils import console


def _pid_alive(pid: int) -> bool:
    """Report whether a process id is still running.

    Args:
        pid (int): The process id to check.

    Returns:
        bool: True if a process with that id exists.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user, but alive.
        return True
    except (OSError, ValueError):
        return False
    return True


def _list_madengine_containers() -> typing.List[typing.Dict]:
    """List every container madengine has labelled.

    Returns:
        list: Dicts with ``id``, ``name``, ``owner_pid`` and ``state``.
    """
    code, out = lifecycle.docker(
        [
            "ps",
            "-a",
            "--no-trunc",
            "--filter",
            f"label={lifecycle.LABEL_SESSION}",
            "--format",
            '{{.ID}}\t{{.Names}}\t{{.Label "' + lifecycle.LABEL_OWNER_PID + '"}}\t{{.State}}',
        ],
        lifecycle.INSPECT_TIMEOUT,
    )
    if code != 0 or not out:
        return []

    containers = []
    for line in out.splitlines():
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        try:
            owner_pid = int(fields[2])
        except ValueError:
            owner_pid = 0
        containers.append(
            {
                "id": fields[0],
                "name": fields[1],
                "owner_pid": owner_pid,
                "state": fields[3],
            }
        )
    return containers


def cleanup(
    all_containers: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Also remove containers whose owning madengine process is still running",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List what would be removed, remove nothing"),
    ] = False,
) -> None:
    """🧹 Remove benchmark containers orphaned by interrupted runs.

    By default only containers whose owning madengine process is gone are
    removed, so this is safe to run from an ``if: always()`` CI step or from
    cron while other benchmarks are in flight.
    """
    containers = _list_madengine_containers()
    if not containers:
        console.print("🧹 [green]No madengine containers found.[/green]")
        raise typer.Exit(ExitCode.SUCCESS)

    stale = [
        c
        for c in containers
        if all_containers or not (c["owner_pid"] and _pid_alive(c["owner_pid"]))
    ]
    skipped = len(containers) - len(stale)
    if skipped:
        console.print(
            f"[dim]Skipping {skipped} container(s) still owned by a running "
            f"madengine process (use --all to override).[/dim]"
        )

    if not stale:
        raise typer.Exit(ExitCode.SUCCESS)

    if dry_run:
        for container in stale:
            console.print(
                f"  would remove [cyan]{container['name']}[/cyan] "
                f"({container['id'][:12]}, state={container['state']}, "
                f"owner pid {container['owner_pid'] or 'unknown'})"
            )
        raise typer.Exit(ExitCode.SUCCESS)

    wedged = []
    for container in stale:
        console.print(
            f"🧹 Removing [cyan]{container['name']}[/cyan] "
            f"({container['id'][:12]}, state={container['state']})"
        )
        result = lifecycle.reap(container["id"], container["id"])
        if result["wedged"]:
            wedged.append(container["name"])

    removed = len(stale) - len(wedged)
    console.print(f"🧹 [green]Removed {removed} container(s).[/green]")

    if wedged:
        console.print(
            f"💥 [bold red]{len(wedged)} container(s) could not be removed: "
            f"{', '.join(wedged)}[/bold red]"
        )
        console.print(
            "[red]The GPU is wedged; this host needs a reboot before it can "
            "run another benchmark.[/red]"
        )
        raise typer.Exit(ExitCode.GPU_WEDGED)

    raise typer.Exit(ExitCode.SUCCESS)

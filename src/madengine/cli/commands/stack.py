"""
`madengine stack`: inspect, check, compare and harvest stack locks.

Locks live in the MAD checkout under compat/stacks/; run these from the MAD root.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
from collections import Counter
from typing import List, Optional

import typer
from rich.markup import escape
from rich.table import Table

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.stack import dockerfile, lock, wheels

from ..constants import ExitCode
from ..utils import console

stack_app = typer.Typer(
    name="stack",
    help="🧱 Inspect, check, compare and harvest stack locks (compat/stacks/)",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

GpuArch = Annotated[
    Optional[str], typer.Option("--gpu-arch", "-a", help="Target, e.g. gfx942")
]


def _load(ref: str, gpu_arch: Optional[str]) -> dict:
    try:
        return lock.resolve(lock.load(ref), gpu_arch)
    except lock.LockError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(ExitCode.INVALID_ARGS)


@stack_app.command("show")
def show(
    ref: Annotated[str, typer.Argument(help="Lock name or path")],
    gpu_arch: GpuArch = None,
    dockerfile_out: Annotated[
        bool, typer.Option("--dockerfile", help="Print the generated Dockerfile")
    ] = False,
) -> None:
    """Show the artifacts of a lock for one target."""
    resolved = _load(ref, gpu_arch)
    if dockerfile_out:
        completion = (
            lock.completion_requirements(resolved) if resolved.get("exclude") else None
        )
        console.print(
            dockerfile.generate(resolved, completion=completion),
            markup=False,
            highlight=False,
        )
        return
    platform = resolved.get("platform") or {}
    table = Table(title=f"{resolved['name']}: {resolved.get('description', '')}")
    for col in ("component", "pin", "from"):
        table.add_column(col)
    for a in resolved.get("artifacts") or []:
        table.add_row(a["name"], lock.artifact_version(a),
                      a.get("built_by") or a.get("index") or "url")
    for s in resolved.get("sources") or []:
        table.add_row(s["name"], lock.components({"sources": [s]})[
            wheels.normalize(s["name"])], "source build")
    console.print(table)
    console.print(f"platform: {json.dumps(platform)}")
    for e in resolved.get("exclude") or []:
        console.print(f"[yellow]excluded {e['name']}: {e['reason']}[/yellow]")
    if resolved.get("validated"):
        console.print(f"validated: {json.dumps(resolved['validated'])}")


@stack_app.command("check")
def check(
    refs: Annotated[List[str], typer.Argument(help="Lock names or paths")],
    gpu_arch: GpuArch = None,
) -> None:
    """Verify declared pins between a lock's artifacts (reads wheel metadata)."""
    failed = False
    for ref in refs:
        resolved = _load(ref, gpu_arch)
        problems = lock.check(resolved)
        if problems:
            failed = True
            console.print(f"[red]✗ {resolved['name']}[/red]")
            for p in problems:
                console.print(f"   {p}", markup=False)
        else:
            console.print(f"[green]✓ {resolved['name']}[/green]: artifact pins consistent")
    raise typer.Exit(ExitCode.FAILURE if failed else ExitCode.SUCCESS)


@stack_app.command("diff")
def diff(
    a: Annotated[str, typer.Argument(help="First lock")],
    b: Annotated[str, typer.Argument(help="Second lock")],
    gpu_arch: GpuArch = None,
) -> None:
    """Components on which two locks differ (what a customer running both sees)."""
    la, lb = _load(a, gpu_arch), _load(b, gpu_arch)
    rows = lock.diff(la, lb)
    if not rows:
        console.print("identical components")
        return
    table = Table()
    table.add_column("component")
    table.add_column(la["name"])
    table.add_column(lb["name"])
    for row in rows:
        table.add_row(*(escape(c) for c in row))
    console.print(table)


@stack_app.command("harvest")
def harvest(
    index: Annotated[str, typer.Argument(help="Wheel index URL (PEP 503)")],
    output: Annotated[str, typer.Option("--output", "-o", help="Catalog JSON")] = "",
    project: Annotated[
        Optional[List[str]], typer.Option("--project", "-p", help="Only these projects")
    ] = None,
    python: Annotated[
        Optional[str], typer.Option("--python", help="Only this python tag, e.g. cp312")
    ] = None,
    metadata: Annotated[
        bool, typer.Option("--metadata", help="Also read each wheel's dependencies")
    ] = False,
) -> None:
    """List every wheel in an index, with provenance from its local version.

    Nested per-stack indexes are followed and recorded (index=vllm-cdna), which
    is how two builds of the same component by different teams stay apart.
    """
    found = wheels.walk_index(index, project)
    if python:
        found = [w for w in found if python in w.python.split(".") or w.python.startswith("py3")]
    if metadata:
        for w in found:
            try:
                wheels.fetch_metadata(w)
            except Exception as e:
                console.print(f"[yellow]{w.url}: {e}[/yellow]")
    by = Counter((w.index or "-", w.name) for w in found)
    table = Table(title=f"{len(found)} wheels in {index}")
    for col in ("index", "project", "wheels", "latest"):
        table.add_column(col)
    for (idx, name), n in sorted(by.items()):
        latest = sorted((w.version for w in found if (w.index or "-", w.name) == (idx, name)),
                        key=_version_key)[-1]
        table.add_row(idx, name, str(n), latest)
    console.print(table)
    if output:
        with open(output, "w") as f:
            json.dump({"index": index, "wheels": [w.to_dict() for w in found]}, f, indent=1)
        console.print(f"wrote {output}")


def _version_key(version: str):
    from packaging.version import InvalidVersion, Version

    try:
        return (0, Version(version))
    except InvalidVersion:
        return (-1, version)

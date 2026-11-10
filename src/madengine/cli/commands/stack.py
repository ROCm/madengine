"""
`madengine stack`: inspect workload image specs against the compatibility matrix.

Run from the MAD root. A SPEC argument is a workload folder (scripts/vllm) or an
image_spec.yaml path; it is composed against compat/rocm.yaml.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
from collections import Counter
from typing import List, Optional

import typer
from rich.markup import escape
from rich.table import Table

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

from madengine.execution import image_spec
from madengine.stack import compose, dockerfile, wheels

from ..constants import ExitCode
from ..utils import console

stack_app = typer.Typer(
    name="stack",
    help="🧱 Inspect workload image specs against the compatibility matrix",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

GpuArch = Annotated[
    Optional[str], typer.Option("--gpu-arch", "-a", help="Target, e.g. gfx942")
]
SpecArg = Annotated[str, typer.Argument(help="Workload folder or image_spec.yaml")]


def _compose(ref: str, gpu_arch: Optional[str]) -> dict:
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    path = os.path.join(ref, image_spec.SPEC_FILE) if os.path.isdir(ref) else ref
    try:
        both = OmegaConf.to_container(
            OmegaConf.create({"spec": OmegaConf.load(path),
                              "compat": image_spec.matrix_node()}),
            resolve=True,
        )
        plan = compose.compose(both["spec"], both["compat"], gpu_arch)
    except (OSError, OmegaConfBaseException, image_spec.ImageSpecError,
            compose.ComposeError) as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(ExitCode.INVALID_ARGS)
    plan["name"] = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return plan


@stack_app.command("show")
def show(
    ref: SpecArg,
    gpu_arch: GpuArch = None,
    dockerfile_out: Annotated[
        bool, typer.Option("--dockerfile", help="Print the generated Dockerfile")
    ] = False,
) -> None:
    """Components of a spec in install order, and why each is there."""
    plan = _compose(ref, gpu_arch)
    if dockerfile_out:
        completion = compose.completion_requirements(plan) if plan["exclude"] else None
        console.print(dockerfile.generate(plan, plan["name"], completion=completion),
                      markup=False, highlight=False)
        return
    table = Table(title=f"{plan['name']} (install order, {plan['gpu_arch']})")
    for col in ("#", "component", "build", "from", "why"):
        table.add_column(col)
    for n, item in enumerate(plan["items"], 1):
        entry = item["entry"]
        origin = entry.get("built_by") or entry.get("index") or (
            "source" if entry.get("source") else "url")
        why = item["why"] if item["authorized"] else f"{item['why']}, NOT AUTHORIZED"
        table.add_row(str(n), item["name"], escape(item["key"]), origin, why)
    console.print(table)
    console.print(f"base: {plan['base']}   python: {plan['python'] or 'base image'}")
    for e in plan["exclude"]:
        console.print(f"[yellow]excluded {e['name']}: {escape(e['reason'])}[/yellow]")


@stack_app.command("check")
def check(refs: Annotated[List[str], typer.Argument(help="Workload folders or specs")],
          gpu_arch: GpuArch = None) -> None:
    """Compare what each wheel declares (METADATA) with the composed plan."""
    failed = False
    for ref in refs:
        plan = _compose(ref, gpu_arch)
        problems = compose.check(plan)
        if problems:
            failed = True
            console.print(f"[red]✗ {plan['name']}[/red]")
            for p in problems:
                console.print(f"   {p}", markup=False)
        else:
            console.print(f"[green]✓ {plan['name']}[/green]: wheel metadata agrees")
    raise typer.Exit(ExitCode.FAILURE if failed else ExitCode.SUCCESS)


@stack_app.command("diff")
def diff(a: SpecArg, b: SpecArg, gpu_arch: GpuArch = None) -> None:
    """Components on which two workloads' images differ."""
    pa, pb = _compose(a, gpu_arch), _compose(b, gpu_arch)
    rows = compose.diff(pa, pb)
    if not rows:
        console.print("identical components")
        return
    table = Table()
    for col in ("component", pa["name"], pb["name"]):
        table.add_column(col)
    for row in rows:
        table.add_row(*(escape(c) for c in row))
    console.print(table)


@stack_app.command("infer")
def infer_cmd(
    image: Annotated[Optional[str], typer.Argument(help="Built image to read")] = None,
    base: Annotated[
        Optional[str], typer.Option("--base", "-b", help="Base image it was built on")
    ] = None,
    image_info: Annotated[
        Optional[str], typer.Option("--image-info", help="image_info JSON instead of IMAGE")
    ] = None,
    base_info: Annotated[
        Optional[str], typer.Option("--base-info", help="image_info JSON of the base")
    ] = None,
    output: Annotated[
        str, typer.Option("--output", "-o", help="Write the draft image_spec.yaml here")
    ] = "",
) -> None:
    """Draft an image_spec for an image built from a hand-written Dockerfile.

    Reads what the image added on top of its base (probe, or saved image_info
    JSON), matches the added packages to authorized builds in compat, and lists
    what compat has no build for yet.
    """
    import yaml

    from madengine.core.console import Console
    from madengine.stack import infer

    def info(ref: Optional[str], path: Optional[str]) -> Optional[dict]:
        if path:
            with open(path) as f:
                return json.load(f)
        if ref:
            return image_spec.read_image_info(Console(live_output=False), ref, probe=True)
        return None

    if not (image or image_info) or not (base or base_info):
        console.print("[red]need IMAGE or --image-info, and --base or --base-info[/red]")
        raise typer.Exit(ExitCode.INVALID_ARGS)
    built, before = info(image, image_info), info(base, base_info)
    if not built:
        console.print(f"[red]could not read the software stack of {image}[/red]")
        raise typer.Exit(ExitCode.FAILURE)
    try:
        matrix = image_spec.load_matrix()
    except image_spec.ImageSpecError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(ExitCode.INVALID_ARGS)
    spec, findings = infer.infer(built, before, base or "<base image>", matrix)
    text = yaml.safe_dump(spec, sort_keys=False)
    if output:
        with open(output, "w") as f:
            f.write(f"# Drafted by madengine stack infer from {image or image_info}\n"
                    + text)
        console.print(f"wrote {output}")
    else:
        console.print(text, markup=False, highlight=False)
    for finding in findings:
        status = finding["status"]
        if status == "dependencies":
            console.print(f"[dim]{finding['package']} from package indexes: pip installs "
                          f"them as dependencies[/dim]")
        elif status == "ambiguous":
            console.print(f"[yellow]{finding['package']} {finding['version']}: several "
                          f"compat builds, chose {escape(finding['chosen'])}[/yellow]")
        else:
            known = finding.get("authorized")
            extra = f" (authorized: {', '.join(known)})" if known else ""
            console.print(f"[red]{finding['package']} {escape(finding['version'])}: "
                          f"{status}{escape(extra)}[/red]")


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
    The output is the raw material for new compat entries.
    """
    found = wheels.walk_index(index, project)
    if python:
        found = [w for w in found
                 if python in w.python.split(".") or w.python.startswith("py3")]
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
        versions = [w.version for w in found if (w.index or "-", w.name) == (idx, name)]
        table.add_row(idx, name, str(n), sorted(versions, key=_version_key)[-1])
    console.print(table)
    if output:
        with open(output, "w") as f:
            json.dump({"index": index, "wheels": [w.to_dict() for w in found]}, f,
                      indent=1)
        console.print(f"wrote {output}")


def _version_key(version: str):
    from packaging.version import InvalidVersion, Version

    try:
        return (0, Version(version))
    except InvalidVersion:
        return (-1, version)

"""Compose a workload's image from its image_spec and the authorized matrix.

The compatibility matrix (``compat/rocm.yaml`` in the MAD checkout) lists what
MAD supports: ROCm releases, wheel indexes, and every component build the team
has authorized, keyed by name and build. A build key is ``<version>@<origin>``
when the same version exists as different builds (one per framework stack)::

    components:
      amd-aiter:
        "0.1.22.post2.dev0+gb4d9154d1.d20261002@frameworks-vllm":
          url: https://.../amd_aiter-...whl
          requires: ["torch==2.13.0+rocm10.1.0"]
      nixl:
        "778edd1d@kvcache":
          source: {repo: ..., commit: ..., cherry_pick: [...], steps: [...]}
          requires: ["ucx"]

A workload's image_spec (``scripts/<dir>/image_spec.yaml``) selects builds by
key. Whatever a spec selects is what that workload is locked to::

    rocm: "10.1"
    python: "3.14"
    components:
      vllm: "0.29.1.dev0+rocm10.1.0.g98dff2a81.d20261002@frameworks-vllm"

``compose`` adds every authorized build that the selection requires, checks
each requirement against what is selected, and orders the result so that every
component installs after what it requires. A spec can also force a relative
install order with ``order: [a, b, c]`` (a before b before c); that adds to the
requirement order and cannot contradict it. A requirement that no authorized
build satisfies, or a cycle, fails before any build. A spec may also give a
full entry instead of a key; that build is not authorized and is marked so.
"""

import typing

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from madengine.stack import wheels

ROCM = "rocm"
_ENTRY_KEYS = {"url", "requirement", "index", "source", "requires", "exclude",
               "env", "system_packages", "gpu_arch", "python", "built_by",
               "sha256", "note", "version"}
_SOURCE_KEYS = {"repo", "commit", "cherry_pick", "steps", "path"}


class ComposeError(ValueError):
    """A spec that the authorized matrix cannot satisfy."""


def build_version(key: str, entry: dict) -> str:
    return str(entry.get("version") or key.split("@", 1)[0])


def _version(text: str) -> typing.Optional[Version]:
    try:
        return Version(text)
    except InvalidVersion:
        return None


def _satisfies(req: Requirement, version: str) -> bool:
    if not req.specifier:
        return True
    v = _version(version)
    return v is not None and req.specifier.contains(v, prereleases=True)


def _check_entry(name: str, key: str, entry: dict) -> None:
    where = f"{name} {key}"
    unknown = set(entry) - _ENTRY_KEYS
    if unknown:
        raise ComposeError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
    kinds = [k for k in ("url", "requirement", "source") if entry.get(k)]
    if len(kinds) != 1:
        raise ComposeError(f"{where}: needs exactly one of url, requirement, source")
    if entry.get("source"):
        source = entry["source"]
        if not (source.get("repo") and source.get("commit")):
            raise ComposeError(f"{where}: source needs repo and commit")
        unknown = set(source) - _SOURCE_KEYS
        if unknown:
            raise ComposeError(f"{where}: unknown source key(s) {', '.join(sorted(unknown))}")
    for item in entry.get("exclude") or []:
        if not (item.get("name") and item.get("reason")):
            raise ComposeError(f"{where}: exclude needs a name and the reason")


def _rocm_entry(matrix: dict, spec_rocm, gpu_arch: str) -> typing.Tuple[str, dict]:
    """The implicit 'rocm' component for the spec's ROCm release."""
    rocm = spec_rocm if isinstance(spec_rocm, dict) else {"version": spec_rocm}
    release_key = str(rocm.get("version") or matrix.get("default_release") or "")
    releases = matrix.get("releases") or {}
    if release_key not in releases:
        raise ComposeError(
            f"ROCm {release_key or '(unset)'} is not authorized in compat; "
            f"known: {', '.join(releases) or 'none'}"
        )
    release = releases[release_key]
    if (rocm.get("install") or release.get("default_install")) != "therock":
        raise ComposeError(
            f"ROCm {release_key}: composed images install TheRock wheels; "
            f"use a base image for legacy ROCm"
        )
    pin = release.get("pip_spec", "")
    return release_key, {
        "requirement": f"rocm[libraries,devel,device-{gpu_arch}]{pin}",
        "index": release["index"],
        "version": release_key,
    }


def compose(
    spec: dict, matrix: dict, gpu_arch: typing.Optional[str] = None
) -> dict:
    """Ordered install plan for *spec*: base, python, components, env, excludes.

    Returns a dict with ``items`` in install order; each item has ``name``,
    ``key``, ``version``, ``entry``, ``authorized`` and ``why`` (``selected``,
    ``rocm`` or ``required by <name>``).
    """
    arch = spec.get("gpu_arch") or gpu_arch or "all"
    catalog = matrix.get("components") or {}
    python = spec.get("python")
    chosen: typing.Dict[str, dict] = {}
    order_hint: typing.List[str] = []

    def add(name: str, key: str, entry: dict, authorized: bool, why: str) -> None:
        name = wheels.normalize(name)
        _check_entry(name, key, entry)
        chosen[name] = {"name": name, "key": key, "version": build_version(key, entry),
                        "entry": entry, "authorized": authorized, "why": why}
        order_hint.append(name)

    if spec.get("rocm"):
        key, entry = _rocm_entry(matrix, spec["rocm"], arch)
        add(ROCM, key, entry, True, "rocm")

    for name, selection in (spec.get("components") or {}).items():
        norm = wheels.normalize(name)
        if isinstance(selection, dict):
            add(norm, selection.get("version") or "custom", dict(selection), False,
                "selected")
            continue
        builds = catalog.get(norm) or catalog.get(name) or {}
        if str(selection) not in builds:
            raise ComposeError(
                f"{name} {selection} is not authorized in compat; authorized builds: "
                f"{', '.join(builds) or 'none'}"
            )
        add(norm, str(selection), dict(builds[str(selection)]), True, "selected")

    # Close over requirements: check selected builds, pull in authorized ones.
    pending = list(chosen)
    while pending:
        item = chosen[pending.pop(0)]
        for line in item["entry"].get("requires") or []:
            try:
                req = Requirement(line)
            except InvalidRequirement:
                raise ComposeError(f"{item['name']}: bad requirement '{line}'") from None
            dep = wheels.normalize(req.name)
            if dep in chosen:
                if not _satisfies(req, chosen[dep]["version"]):
                    raise ComposeError(
                        f"{item['name']} {item['key']} requires {line}, "
                        f"but the spec has {dep} {chosen[dep]['key']}"
                    )
                continue
            candidates = [
                (k, e) for k, e in (catalog.get(dep) or {}).items()
                if _satisfies(req, build_version(k, e))
            ]
            if not candidates:
                raise ComposeError(
                    f"{item['name']} {item['key']} requires {line}; no authorized "
                    f"build in compat satisfies it"
                )
            k, e = max(candidates, key=lambda c: (_version(build_version(*c)) or Version("0"), c[0]))
            add(dep, k, dict(e), True, f"required by {item['name']}")
            pending.append(dep)

    for item in chosen.values():
        allowed = item["entry"].get("python")
        if python and allowed and str(python) not in [str(p) for p in allowed]:
            raise ComposeError(
                f"{item['name']} {item['key']} is built for python "
                f"{', '.join(map(str, allowed))}, spec uses {python}"
            )
        limit = item["entry"].get("gpu_arch")
        if limit and arch != "all":
            limit = [limit] if isinstance(limit, str) else limit
            if arch not in limit:
                raise ComposeError(
                    f"{item['name']} {item['key']} is built for {', '.join(limit)}, not {arch}"
                )

    order = [wheels.normalize(n) for n in spec.get("order") or []]
    unknown = [n for n in order if n not in chosen]
    if unknown:
        raise ComposeError(
            f"order names {', '.join(unknown)}, not in the plan "
            f"({', '.join(chosen)})"
        )
    items = _topological(chosen, order_hint, order)
    exclude = [x for i in items for x in i["entry"].get("exclude") or []]
    env: typing.Dict[str, str] = {}
    for i in items:
        env.update(i["entry"].get("env") or {})
    env.update(spec.get("env") or {})
    return {
        "base": spec.get("base") or matrix.get("bare_os_base", "ubuntu:24.04"),
        "python": python,
        "gpu_arch": arch,
        "indexes": dict(matrix.get("indexes") or {}),
        "setup": list(spec.get("setup") or []),
        "system_packages": list(spec.get("system_packages") or []),
        "env": env,
        "exclude": exclude,
        "items": items,
    }


def _topological(
    chosen: dict, hint: typing.List[str], order: typing.Sequence[str] = ()
) -> typing.List[dict]:
    """Dependencies first, then the spec's ``order``; otherwise the spec's order.

    ``order`` adds an edge between each consecutive pair, so it can move
    components relative to each other but not ahead of what they require: a
    contradiction shows up as a cycle.
    """
    deps = {}
    for name, item in chosen.items():
        deps[name] = {
            wheels.normalize(Requirement(r).name)
            for r in item["entry"].get("requires") or []
        } & set(chosen)
    for before, after in zip(order, order[1:]):
        deps[after].add(before)
    rank = {n: i for i, n in enumerate(dict.fromkeys(hint))}
    done: typing.List[str] = []
    ready = sorted((n for n in chosen if not deps[n]), key=rank.get)
    while ready:
        n = ready.pop(0)
        done.append(n)
        for m in chosen:
            if m not in done and m not in ready and deps[m] <= set(done):
                ready.append(m)
        ready.sort(key=rank.get)
    if len(done) != len(chosen):
        stuck = sorted(set(chosen) - set(done))
        raise ComposeError(
            f"requirement cycle among: {', '.join(stuck)}"
            + (" (check the spec's order against what these components require)"
               if order else "")
        )
    return [chosen[n] for n in done]


def fill(text: str, gpu_arch: str) -> str:
    """``{gpu_arch}`` in requirements and build steps; TheRock calls 'all' device-all."""
    return text.replace("{gpu_arch}", gpu_arch or "all")


def components(plan: dict) -> typing.Dict[str, str]:
    """name -> build key, for display and diff."""
    return {i["name"]: i["key"] for i in plan["items"]}


def diff(a: dict, b: dict) -> typing.List[typing.Tuple[str, str, str]]:
    """Rows (component, build in a, build in b) where two plans differ."""
    ca, cb = components(a), components(b)
    rows = [(n, ca.get(n, "-"), cb.get(n, "-"))
            for n in sorted(set(ca) | set(cb)) if ca.get(n) != cb.get(n)]
    for key in ("python", "base"):
        if a.get(key) != b.get(key):
            rows.insert(0, (f"[{key}]", str(a.get(key) or "-"), str(b.get(key) or "-")))
    return rows


def check(plan: dict, fetch=None) -> typing.List[str]:
    """Problems in the wheels' own declared dependencies against the plan.

    The matrix's ``requires`` is what the team states; a wheel's METADATA is
    what its builder declared. This reads the latter for every wheel in the
    plan and reports pins on other plan components that do not match, and
    python ranges that exclude the plan's python.
    """
    fetch = fetch or wheels.fetch_metadata
    have = {i["name"]: i["version"] for i in plan["items"]}
    py = plan.get("python")
    problems = []
    for item in plan["items"]:
        url = item["entry"].get("url")
        if not url:
            continue
        wheel = wheels.parse_wheel_filename(url)
        if wheel is None:
            problems.append(f"{item['name']}: url is not a wheel")
            continue
        try:
            fetch(wheel)
        except Exception as e:  # network or archive errors are reported, not fatal
            problems.append(f"{item['name']}: could not read metadata ({e})")
            continue
        if py and wheel.requires_python and not SpecifierSet(
            wheel.requires_python
        ).contains(f"{py}.0"):
            problems.append(
                f"{item['name']}: requires python {wheel.requires_python}, spec uses {py}"
            )
        for line in wheel.requires or []:
            try:
                req = Requirement(line)
            except InvalidRequirement:
                continue
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            dep = wheels.normalize(req.name)
            if dep in have and not _satisfies(req, have[dep]):
                problems.append(
                    f"{item['name']} {wheel.version} declares {line}, plan has {have[dep]}"
                )
    return problems


def completion_requirements(plan: dict, fetch=None) -> typing.List[str]:
    """Declared dependencies of the plan's wheels, minus plan members and excludes.

    Used when the plan excludes something: pip cannot skip one dependency of a
    package it resolves, so the dependencies are listed explicitly and the
    wheels themselves stay out of that resolution. Markers are evaluated for
    the plan's python on Linux x86_64.
    """
    fetch = fetch or wheels.fetch_metadata
    py = str(plan.get("python") or "3")
    env = {"python_version": py, "python_full_version": f"{py}.0", "extra": "",
           "sys_platform": "linux", "platform_system": "Linux",
           "platform_machine": "x86_64", "os_name": "posix"}
    skip = {i["name"] for i in plan["items"]}
    skip |= {wheels.normalize(e["name"]) for e in plan.get("exclude") or []}
    out: typing.Dict[str, str] = {}
    for item in plan["items"]:
        url = item["entry"].get("url")
        if not url:
            continue
        wheel = fetch(wheels.parse_wheel_filename(url))
        for line in wheel.requires or []:
            try:
                req = Requirement(line)
            except InvalidRequirement:
                continue
            if req.marker and not req.marker.evaluate(env):
                continue
            if wheels.normalize(req.name) in skip:
                continue
            req.marker = None
            out.setdefault(str(req), str(req))
    return sorted(out)

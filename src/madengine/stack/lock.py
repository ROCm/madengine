"""Stack locks: the exact artifacts of one best-known stack on one platform.

A lock is a small JSON file in the MAD checkout (``compat/stacks/<name>.json``)::

    {
      "name": "vllm",
      "description": "...",
      "platform": {"vendor": "AMD", "rocm": "10.1", "python": "3.14",
                   "gpu_arch": ["gfx942", "gfx950"], "nic": []},
      "base": "ubuntu:24.04",
      "system_packages": ["libgomp1"],
      "indexes": {"therock": "https://...", "vllm": "https://..."},
      "artifacts": [
        {"name": "rocm", "index": "therock",
         "requirement": "rocm[libraries,devel,device-{gpu_arch}]==10.1.0"},
        {"name": "vllm", "url": "https://.../vllm-0.29.1...whl", "built_by": "frameworks/vllm"}
      ],
      "setup": ["<shell run before system packages>"],
      "sources": [
        {"name": "mori", "repo": "https://github.com/ROCm/mori.git", "commit": "<sha>",
         "path": "/opt/mad/src/mori", "cherry_pick": [], "system_packages": [],
         "env": {}, "steps": ["pip install ."]}
      ],
      "env": {"VLLM_ROCM_USE_AITER": "1"},
      "exclude": [{"name": "amd-quark", "reason": "pinned by vllm, no build for this python"}],
      "validated": {"date": "...", "by": "...", "run": "..."}
    }

Artifacts install in list order, one image layer each, then sources build in
list order (``{gpu_arch}`` is also replaced in their steps). ``env`` is set
before the source builds, so later builds can use earlier ones. An artifact is either an
exact wheel ``url`` or a pip ``requirement`` resolved against a named index;
``{gpu_arch}`` in either is replaced with the target, and ``gpu_arch`` on an
artifact or source limits it to those targets.
"""

import copy
import json
import os
import re
import typing

from packaging.requirements import InvalidRequirement, Requirement

from madengine.stack import wheels

STACKS_DIR = os.path.join("compat", "stacks")
_ARTIFACT_KEYS = {"name", "url", "requirement", "index", "gpu_arch", "built_by",
                  "sha256", "size", "etag", "note", "overridden"}
_SOURCE_KEYS = {"name", "repo", "commit", "cherry_pick", "steps", "env", "path",
                "system_packages", "gpu_arch", "note", "overridden"}


class LockError(ValueError):
    """A lock that is malformed, unknown, or inconsistent."""


def lock_path(ref: str, root: str = ".") -> str:
    if ref.endswith(".json"):
        return ref if os.path.isabs(ref) else os.path.join(root, ref)
    return os.path.join(root, STACKS_DIR, f"{ref}.json")


def load(ref: str, root: str = ".") -> dict:
    path = lock_path(ref, root)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        known = sorted(
            n[:-5] for n in os.listdir(os.path.join(root, STACKS_DIR))
            if n.endswith(".json")
        ) if os.path.isdir(os.path.join(root, STACKS_DIR)) else []
        raise LockError(
            f"stack lock '{ref}' not found at {path}; known: {', '.join(known) or 'none'}"
        ) from None
    validate(data)
    return data


def validate(data: dict) -> None:
    for key in ("name", "platform"):
        if key not in data:
            raise LockError(f"lock is missing '{key}'")
    for i, item in enumerate(data.get("exclude") or []):
        if not (isinstance(item, dict) and item.get("name") and item.get("reason")):
            raise LockError(
                f"{data['name']}: exclude[{i}] needs a 'name' and the 'reason'"
                " it is left out"
            )
    names = set()
    for kind, items, allowed in (
        ("artifacts", data.get("artifacts") or [], _ARTIFACT_KEYS),
        ("sources", data.get("sources") or [], _SOURCE_KEYS),
    ):
        for i, item in enumerate(items):
            where = f"{data['name']}: {kind}[{i}]"
            if not item.get("name"):
                raise LockError(f"{where} needs a name")
            unknown = set(item) - allowed
            if unknown:
                raise LockError(f"{where} has unknown key(s): {', '.join(sorted(unknown))}")
            if kind == "artifacts" and bool(item.get("url")) == bool(item.get("requirement")):
                raise LockError(f"{where} needs exactly one of 'url' or 'requirement'")
            if kind == "artifacts" and item.get("requirement"):
                index = item.get("index")
                if not index or index not in (data.get("indexes") or {}):
                    raise LockError(f"{where} names index '{index}', not in 'indexes'")
            if kind == "sources" and not (item.get("repo") and item.get("commit")):
                raise LockError(f"{where} needs 'repo' and 'commit'")
            key = (kind, wheels.normalize(item["name"]))
            if key in names:
                raise LockError(f"{where} duplicates '{item['name']}'")
            names.add(key)


def _for_arch(item: dict, gpu_arch: typing.Optional[str]) -> bool:
    limit = item.get("gpu_arch")
    if not limit:
        return True
    limit = [limit] if isinstance(limit, str) else limit
    return gpu_arch in limit


def _fill(text: str, gpu_arch: typing.Optional[str]) -> str:
    # TheRock device extras: device-gfx942, or device-all when the target is unknown.
    return text.replace("{gpu_arch}", gpu_arch or "all")


def resolve(
    data: dict,
    gpu_arch: typing.Optional[str] = None,
    override: typing.Optional[dict] = None,
) -> dict:
    """The lock for one target: arch-filtered, overrides applied, placeholders filled.

    *override* maps an artifact or source name to the fields to replace, e.g.
    ``{"vllm": {"url": "<my wheel>"}}``; an override for a name the lock does
    not have is appended as a new artifact (url/requirement) or source (repo).
    """
    platform = data.get("platform") or {}
    archs = platform.get("gpu_arch") or []
    if archs and gpu_arch and gpu_arch not in archs:
        raise LockError(
            f"stack '{data['name']}' is locked for {', '.join(archs)}, not {gpu_arch}"
        )
    out = copy.deepcopy(data)
    for name, fields in (override or {}).items():
        target = None
        for kind in ("artifacts", "sources"):
            for item in out.get(kind) or []:
                if wheels.normalize(item["name"]) == wheels.normalize(name):
                    target = item
        if target is None:
            kind = "sources" if "repo" in fields else "artifacts"
            out.setdefault(kind, []).append({"name": name})
            target = out[kind][-1]
        if "url" in fields:
            target.pop("requirement", None)
            target.pop("index", None)
        if "requirement" in fields:
            target.pop("url", None)
        for drop in ("sha256", "size", "etag", "built_by"):
            target.pop(drop, None)
        target.update(fields)
        target["overridden"] = True
    validate(out)
    out["artifacts"] = [
        {**a, **({"url": _fill(a["url"], gpu_arch)} if a.get("url") else {}),
         **({"requirement": _fill(a["requirement"], gpu_arch)} if a.get("requirement") else {})}
        for a in out.get("artifacts") or [] if _for_arch(a, gpu_arch)
    ]
    out["sources"] = [
        {**s, "steps": [_fill(step, gpu_arch) for step in s.get("steps") or []]}
        for s in out.get("sources") or [] if _for_arch(s, gpu_arch)
    ]
    out["gpu_arch"] = gpu_arch or ""
    return out


def artifact_version(artifact: dict) -> str:
    """Version an artifact pins: from the wheel filename or an ``==`` requirement."""
    if artifact.get("url"):
        wheel = wheels.parse_wheel_filename(artifact["url"])
        return wheel.version if wheel else ""
    try:
        req = Requirement(artifact["requirement"])
    except InvalidRequirement:
        return ""
    pins = [s.version for s in req.specifier if s.operator in ("==", "===")]
    return pins[0] if len(pins) == 1 else str(req.specifier)


def check(resolved: dict, fetch=None) -> typing.List[str]:
    """Cross-artifact problems in a resolved lock.

    Reads each wheel artifact's declared dependencies and reports every
    ``Requires-Dist`` on another artifact of the lock that the locked version
    does not satisfy (for example vllm pinning a different amd-aiter build).
    Dependencies on packages outside the lock are left to pip.
    """
    fetch = fetch or wheels.fetch_metadata
    versions = {wheels.normalize(a["name"]): artifact_version(a)
                for a in resolved.get("artifacts") or []}
    problems = []
    for artifact in resolved.get("artifacts") or []:
        if not artifact.get("url"):
            continue
        wheel = wheels.parse_wheel_filename(artifact["url"])
        if wheel is None:
            problems.append(f"{artifact['name']}: url is not a wheel")
            continue
        try:
            fetch(wheel)
        except Exception as e:  # network or archive errors are reported, not fatal
            problems.append(f"{artifact['name']}: could not read metadata ({e})")
            continue
        py = (resolved.get("platform") or {}).get("python")
        if py and wheel.requires_python:
            from packaging.specifiers import SpecifierSet

            if not SpecifierSet(wheel.requires_python).contains(f"{py}.0"):
                problems.append(
                    f"{artifact['name']}: requires python {wheel.requires_python}, "
                    f"lock uses {py}"
                )
        for line in wheel.requires or []:
            try:
                req = Requirement(line)
            except InvalidRequirement:
                continue
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            have = versions.get(wheels.normalize(req.name))
            if have is None or not req.specifier:
                continue
            if not req.specifier.contains(have, prereleases=True):
                problems.append(
                    f"{artifact['name']} {wheel.version} requires "
                    f"{req.name}{req.specifier}, lock has {have}"
                )
    return problems


def completion_requirements(resolved: dict, fetch=None) -> typing.List[str]:
    """Declared dependencies of the lock's wheel artifacts, minus excludes.

    Used only when a lock excludes something: pip cannot be told to skip one
    dependency of a package it resolves, so the dependencies are listed
    explicitly instead and the wheel artifacts themselves stay out of the
    resolution. Lock members and excluded names are dropped; markers are
    evaluated for the lock's python on Linux x86_64.
    """
    fetch = fetch or wheels.fetch_metadata
    py = str((resolved.get("platform") or {}).get("python") or "3")
    env = {"python_version": py, "python_full_version": f"{py}.0", "extra": "",
           "sys_platform": "linux", "platform_system": "Linux",
           "platform_machine": "x86_64", "os_name": "posix"}
    skip = {wheels.normalize(a["name"]) for a in resolved.get("artifacts") or []}
    skip |= {wheels.normalize(e["name"]) for e in resolved.get("exclude") or []}
    out: typing.Dict[str, str] = {}
    for artifact in resolved.get("artifacts") or []:
        if not artifact.get("url"):
            continue
        wheel = fetch(wheels.parse_wheel_filename(artifact["url"]))
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


def components(resolved: dict) -> typing.Dict[str, str]:
    """name -> pinned version or source commit, for display and diff."""
    out = {wheels.normalize(a["name"]): artifact_version(a)
           for a in resolved.get("artifacts") or []}
    for s in resolved.get("sources") or []:
        picks = s.get("cherry_pick") or []
        out[wheels.normalize(s["name"])] = s["commit"][:12] + (
            f" +{len(picks)} cherry-pick" if picks else ""
        )
    return out


def diff(a: dict, b: dict) -> typing.List[typing.Tuple[str, str, str]]:
    """Rows (component, version in a, version in b) where the two locks differ."""
    ca, cb = components(a), components(b)
    rows = []
    for name in sorted(set(ca) | set(cb)):
        va, vb = ca.get(name, "-"), cb.get(name, "-")
        if va != vb:
            rows.append((name, va, vb))
    for key in ("rocm", "python"):
        pa, pb = (a.get("platform") or {}).get(key), (b.get("platform") or {}).get(key)
        if pa != pb:
            rows.insert(0, (f"[platform] {key}", str(pa or "-"), str(pb or "-")))
    return rows

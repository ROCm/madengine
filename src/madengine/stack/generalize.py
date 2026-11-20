"""Generalize working stacks from build manifests into the compatibility matrix.

compat/rocm.yaml is the generalized dependency state: which component versions work together and
how each installs. A build manifest records, per card, the image's stack (built_images.<image>:
image_info for any image when image_probe is on, build_plan for an image_spec image); a run's
results say whether the card worked. A stack that worked is evidence; this turns it into the
general facts compat lacks, and nothing else:

* a component compat knows, at a version it does not list, or a custom build of one: a new build
  entry when the version's wheel is in one of the matrix's indexes (its url, and its requirements
  on components compat knows, from the wheel's metadata);
* a version no index serves (a source build, a local wheel): listed, not invented -- someone writes
  its build entry;
* a requirement of an authorized build that a working stack contradicts: listed for review.

Plain index packages (dependencies pip resolves) are not listed; compat does not hold them. A stack
compat already covers changes nothing, so the result is the same however often it runs.
"""

import csv
import typing

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from madengine.stack import wheels
from madengine.stack.compose import build_version
from madengine.stack.infer import infer

NOTE = "generalized from a working build manifest"


def passed_models(result_files: typing.Iterable[str]) -> typing.Set[str]:
    """Model names whose rows in madengine's perf CSVs all have status SUCCESS."""
    status: typing.Dict[str, bool] = {}
    for path in result_files:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("model") or "").strip()
                if name:
                    ok = (row.get("status") or "").strip().upper() == "SUCCESS"
                    status[name] = status.get(name, True) and ok
    return {n for n, ok in status.items() if ok}


def proven_stacks(manifest: dict, passed: typing.Set[str]) -> typing.List[typing.Tuple[str, dict]]:
    """(card, image_info) for each built card that passed and whose image's stack was recorded."""
    out = []
    images = manifest.get("built_images") or {}
    for key, model in (manifest.get("built_models") or {}).items():
        name = model.get("name") or key
        if not ({name, name.rsplit("/", 1)[-1]} & passed):
            continue
        image = model.get("docker_image") or model.get("image") or key
        info = (images.get(image) or {}).get("image_info")
        if info:
            out.append((name, info))
    return out


def _version(text: str) -> typing.Optional[Version]:
    try:
        return Version(text)
    except InvalidVersion:
        return None


def index_finder(matrix: dict) -> typing.Callable[[str, str], typing.Optional[typing.Tuple[str, "wheels.Wheel"]]]:
    """find(name, version) -> (matrix index name, wheel with metadata) from the matrix's indexes."""
    cache: typing.Dict[str, typing.List[wheels.Wheel]] = {}

    def find(name, version):
        for index, url in (matrix.get("indexes") or {}).items():
            key = f"{index}:{name}"
            if key not in cache:
                try:
                    cache[key] = wheels.walk_index(url, [name])
                except Exception:
                    cache[key] = []
            for w in cache[key]:
                if w.version == version:
                    return index, wheels.fetch_metadata(w)
        return None

    return find


def _requires(wheel: "wheels.Wheel", matrix: dict) -> typing.List[str]:
    """The wheel's requirements on components compat knows (unconditional ones)."""
    known = {wheels.normalize(n) for n in (matrix.get("components") or {})} | {"rocm"}
    out = []
    for line in wheel.requires or []:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        if req.marker is not None or wheels.normalize(req.name) not in known:
            continue
        out.append(str(req))
    return out


def generalize(stacks: typing.Iterable[typing.Tuple[str, dict]], matrix: dict,
               find=None) -> dict:
    """The compat delta that working *stacks* establish: {"add": {component: {key: entry}},
    "unresolved": [...], "contradicts": [...]}. *find* looks a version up in the indexes."""
    find = find or index_finder(matrix)
    catalog = matrix.get("components") or {}
    add: typing.Dict[str, dict] = {}
    unresolved: typing.Dict[typing.Tuple[str, str], dict] = {}
    contradicts: typing.Dict[typing.Tuple[str, str, str], dict] = {}
    for card, info in stacks:
        spec, findings = infer(info, None, "", matrix)
        packages = {wheels.normalize(n): v for n, v in (info.get("packages") or {}).items()}
        for f in findings:
            if f["status"] not in ("unauthorized", "new"):
                continue
            name, version = f["package"], f["version"]
            if name == "rocm":
                unresolved.setdefault((name, version), {
                    "component": name, "version": version, "seen_in": [],
                    "why": "a ROCm release compat does not list (releases:)"})["seen_in"].append(card)
                continue
            if any(build_version(k, e) == version for k, e in (add.get(name) or {}).items()):
                continue
            got = find(name, version)
            if got is None:
                unresolved.setdefault((name, version), {
                    "component": name, "version": version, "seen_in": [],
                    "why": "no wheel in the matrix's indexes: a source build or a local wheel needs "
                           "its build entry"})["seen_in"].append(card)
                continue
            index, wheel = got
            entry = {"url": wheel.url}
            reqs = _requires(wheel, matrix)
            if reqs:
                entry["requires"] = reqs
            if wheel.python and wheel.python.startswith("cp3"):
                entry["python"] = [f"3.{wheel.python[3:]}"]
            entry["note"] = NOTE
            add.setdefault(name, {})[f"{version}@{index}"] = entry
        # Requirements of the authorized builds this stack uses, against what it ran with.
        for name, key in (spec.get("components") or {}).items():
            for line in (catalog.get(name) or {}).get(key, {}).get("requires") or []:
                try:
                    req = Requirement(line)
                except InvalidRequirement:
                    continue
                dep = wheels.normalize(req.name)
                have = packages.get(dep)
                v = _version(have) if have else None
                if v is not None and req.specifier and not req.specifier.contains(v, prereleases=True):
                    contradicts.setdefault((name, key, line), {
                        "component": name, "build": key, "requires": line, "ran_with": f"{dep}=={have}",
                        "seen_in": []})["seen_in"].append(card)
    return {"add": add, "unresolved": list(unresolved.values()), "contradicts": list(contradicts.values())}


def insert_builds(text: str, builds: typing.Dict[str, dict]) -> str:
    """compat/rocm.yaml text with *builds* added under `components:`, comments kept: under an
    existing component's block, or as a new component before the next top-level key. Builds already
    present are left as they are."""
    import yaml

    have = (yaml.safe_load(text) or {}).get("components") or {}
    lines = text.rstrip("\n").split("\n")
    start = next(i for i, l in enumerate(lines) if l.rstrip() == "components:")

    def section_end():
        nxt = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace()
                    and not lines[i].startswith("#")), len(lines))
        while nxt > start + 1 and (not lines[nxt - 1].strip() or lines[nxt - 1].startswith("#")):
            nxt -= 1
        return nxt

    for comp, entries in builds.items():
        new = {k: e for k, e in entries.items() if k not in (have.get(comp) or {})}
        if not new:
            continue
        block = ["    " + l for l in yaml.safe_dump(new, sort_keys=False, width=1 << 16).rstrip("\n").split("\n")]
        head = next((i for i in range(start + 1, section_end()) if lines[i].rstrip() == f"  {comp}:"), None)
        if head is None:
            end = section_end()
            lines[end:end] = ["", f"  {comp}:"] + block
            continue
        end = head + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith("    ")):
            end += 1
        while end > head + 1 and not lines[end - 1].strip():
            end -= 1
        lines[end:end] = block
    out = "\n".join(lines) + "\n"
    got = (yaml.safe_load(out) or {}).get("components") or {}
    for comp, entries in builds.items():
        for k in entries:
            assert k in (got.get(comp) or {}), f"{comp} {k} was not added"
    return out

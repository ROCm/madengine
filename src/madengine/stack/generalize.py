"""The generalized form of a built image: an image spec over the compatibility matrix.

compat/rocm.yaml is the generalized dependency state: which component versions work together and
how each installs. A build manifest is the specific instance: per image, its stack (image_info, or
build_plan for an image_spec image) and its base. When a card's configuration is generated
(`madengine build --emit-config`, the pipeline's STANDALONE record), its image is generalized here:

* the image as a spec: the base it was built on and the compat builds of what it added;
* what compat lacks for that spec, as general facts: a build for a component version an
  authorized index serves (its url, and its requirements on components compat knows, from the
  wheel's metadata) -- merged into a copy of the matrix beside the generated configs, for the PR
  of whoever made the run;
* what cannot be generalized (a version no index serves: a source build or a local wheel; a
  requirement the stack contradicts): listed with the card's image, not invented. A source build at
  a commit the matrix lacks comes with a draft entry: the matrix's build of that repository at the
  new commit, for whoever made the run to confirm.

A hand-written Dockerfile's image is generalized as what it added to its base (the build probes
both: the Python packages each holds); an image_spec image is its own spec. Either is then placed
under the named stack (compat/images) it is closest to: `extends: ${images.<stack>}` and only what
differs -- so it inherits what that stack holds and the probe cannot read (native libraries), and
a stack the image is exactly is just its reference.

A stack the matrix covers adds nothing, so the copy changes only when a run brings something new.
"""

import copy
import typing

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from madengine.stack import wheels
from madengine.stack.compose import build_version
from madengine.stack.infer import build_commit, infer

NOTE = "generalized from a working build manifest"


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


def _draft(name: str, commit: str, catalog: dict) -> typing.Optional[typing.Dict[str, dict]]:
    """{key: entry}: the matrix's source build of *name* at *commit* (its last build of that
    repository, an unpatched one first; patches left out: the image's are unknown), or None without
    one to start from."""
    sources = [(k, e) for k, e in (catalog.get(name) or {}).items()
               if isinstance(e, dict) and build_commit(k, e) and (e.get("source") or {}).get("repo")]
    if not sources or len({e["source"]["repo"] for _, e in sources}) != 1:
        return None
    key, entry = ([s for s in sources if not s[1]["source"].get("patches")] or sources)[-1]
    entry = copy.deepcopy(entry)
    entry["source"]["commit"] = commit
    entry["source"].pop("patches", None)
    entry.pop("version", None)
    entry["note"] = f"drafted from {name} {key} at the commit a working build used; confirm the steps"
    return {f"{commit[:8]}@{key.split('@', 1)[-1]}": entry}


def generalize(stacks: typing.Iterable[tuple], matrix: dict, find=None) -> dict:
    """The compat delta that working *stacks* establish: {"add": {component: {key: entry}},
    "unresolved": [...], "contradicts": [...]}. Each stack is (card, image_info[, base_info]);
    *find* looks a version up in the indexes."""
    find = find or index_finder(matrix)
    catalog = matrix.get("components") or {}
    add: typing.Dict[str, dict] = {}
    unresolved: typing.Dict[typing.Tuple[str, str], dict] = {}
    contradicts: typing.Dict[typing.Tuple[str, str, str], dict] = {}
    for card, info, *base in stacks:
        spec, findings = infer(info, base[0] if base else None, "", matrix)
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
                item = {"component": name, "version": version, "seen_in": [],
                        "why": "no wheel in the matrix's indexes: a source build or a local wheel needs "
                               "its build entry"}
                draft = _draft(name, f["commit"], catalog) if f.get("commit") else None
                if draft:
                    item.update(commit=f["commit"], draft=draft,
                                why="built from source at a commit the matrix has no build for")
                unresolved.setdefault((name, version), item)["seen_in"].append(card)
                continue
            index, wheel = got
            entry = {"url": wheel.url}
            reqs = []
            for line in _requires(wheel, matrix):
                req = Requirement(line)
                have = packages.get(wheels.normalize(req.name))
                v = _version(have) if have else None
                if v is not None and req.specifier and not req.specifier.contains(v, prereleases=True):
                    # The working stack shows the wheel runs without it: reported, not recorded.
                    contradicts.setdefault((name, f"{version}@{index}", line), {
                        "component": name, "build": f"{version}@{index}", "requires": line,
                        "ran_with": f"{wheels.normalize(req.name)}=={have}", "seen_in": []})["seen_in"].append(card)
                    continue
                reqs.append(line)
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


def _selections(spec: dict):
    """(arch, {component: key}) of a named spec: its default selection, then each GPU's variant."""
    from madengine.stack.compose import _for_arch

    yield None, dict(spec.get("components") or {})
    for arch in sorted(spec.get("variants") or {}):
        yield arch, dict(_for_arch(spec, arch).get("components") or {})


def _strip_digest(image: str) -> str:
    return str(image or "").split("@", 1)[0]


def place(spec: dict, images: typing.Dict[str, dict]) -> dict:
    """*spec* under the named stack it is closest to (the most builds in common):
    {"extends": "${images.<stack>}", and only what differs}; *spec* itself when no stack shares a
    build with it. A spec equal to a named stack is that stack."""
    for name, named in images.items():
        if spec == named:
            return {"extends": "${images.%s}" % name}
    have = spec.get("components") or {}
    best = None
    for name, named in images.items():
        for arch, sel in _selections(named):
            common = sum(1 for k, v in have.items() if sel.get(k) == v)
            if common and (best is None or common > best[0]):
                best = (common, name, arch, sel)
    if best is None:
        return spec
    _, name, arch, sel = best
    out: dict = {"extends": "${images.%s}" % name}
    base, named_base = spec.get("base"), images[name].get("base")
    if isinstance(named_base, dict):
        named_base = named_base.get(arch) if arch else next(iter(named_base.values()), None)
    if base and _strip_digest(base) != _strip_digest(named_base):
        out["base"] = {arch: base} if arch and isinstance(images[name].get("base"), dict) else base
    for key in ("python", "rocm"):
        if spec.get(key) and spec[key] != images[name].get(key):
            out[key] = spec[key]
    differs = {k: v for k, v in have.items() if sel.get(k) != v}
    if differs:
        out["components"] = differs
    return out


def image_of(image_build: dict, matrix: dict, card: str = "", find=None,
             images: typing.Optional[typing.Dict[str, dict]] = None) -> typing.Tuple[dict, dict]:
    """(the image as a spec, the compat delta it establishes) from its manifest entry
    (built_images.<image>). An image_spec image is its own spec; an image with a recorded stack
    (image_info) is matched against the matrix; otherwise ({}, empty delta). With *images* (the
    named stacks), the spec is placed under the closest one (place)."""
    empty = {"add": {}, "unresolved": [], "contradicts": []}
    plan = (image_build or {}).get("build_plan") or {}
    if plan.get("spec"):
        spec = dict(plan["spec"])
        return (place(spec, images) if images else spec), empty
    info = (image_build or {}).get("image_info")
    if not info:
        return {}, empty
    base = (image_build or {}).get("base_info")
    spec, _ = infer(info, base, (image_build or {}).get("base_docker") or "", matrix)
    if not spec.get("base"):
        spec.pop("base", None)
    delta = generalize([(card, info, base)], matrix, find)
    # The builds this stack establishes are its own: the spec selects them, against the matrix
    # they are merged into; a drafted build too, which it selects once someone confirms it.
    for name, entries in delta["add"].items():
        spec.setdefault("components", {})[name] = next(iter(entries))
    for item in delta["unresolved"]:
        if item.get("draft"):
            spec.setdefault("components", {})[item["component"]] = next(iter(item["draft"]))
    return (place(spec, images) if images else spec), delta


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

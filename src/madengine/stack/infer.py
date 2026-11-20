"""Infer an image_spec from an image that was built another way.

The loop that moves an existing workload onto a spec:

    hand-written Dockerfile -> image -> image_info (probe)
    -> what the Dockerfile added on top of its base (image_info minus the base's)
    -> match each added package against the authorized builds in compat
    -> draft image_spec.yaml + the packages compat has no build for yet

The draft keeps the original base image, so building it should reproduce the
image; the unmatched packages are what the matrix needs before the draft can
be built (add them to compat/rocm.yaml, e.g. from ``madengine stack harvest``).
"""

import typing

from madengine.stack import wheels
from madengine.stack.compose import build_version


def added_packages(image: dict, base: typing.Optional[dict]) -> typing.Dict[str, str]:
    """Packages the image has that its base does not, or has at another version."""
    have = image.get("packages") or {}
    before = (base or {}).get("packages") or {}
    return {
        wheels.normalize(n): v for n, v in sorted(have.items())
        if before.get(n) != v and before.get(wheels.normalize(n)) != v
    }


def _rocm_release(matrix: dict, version: str) -> typing.Optional[str]:
    """Matrix release whose pin contains *version* (or whose key prefixes it)."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        v = Version(version)
    except InvalidVersion:
        v = None
    for key, release in (matrix.get("releases") or {}).items():
        pin = release.get("pip_spec")
        if pin and v is not None and SpecifierSet(pin).contains(v, prereleases=True):
            return key
        if version == key or version.startswith(key + "."):
            return key
    return None


def infer(
    image: dict,
    base: typing.Optional[dict],
    base_image: str,
    matrix: dict,
) -> typing.Tuple[dict, typing.List[dict]]:
    """Draft spec and findings for an image built on *base_image*.

    *image* and *base* are image_info dicts (python, rocm_version, packages).
    Returns (spec, findings); each finding has ``package``, ``version`` and
    ``status``:

    * ``unauthorized``: compat has the component, but not at this version;
    * ``new``: a custom build (local version such as ``+rocm10.1.0`` or
      ``+g<sha>``) that compat does not list; a candidate for a new entry;
    * ``ambiguous``: several compat builds share the version; the first was
      chosen and the alternatives are listed;
    * ``dependencies``: one summary of the plain package-index versions added,
      which pip installs as dependencies and compat does not need to list.
    """
    spec: dict = {"base": base_image}
    findings: typing.List[dict] = []
    py, base_py = image.get("python") or "", (base or {}).get("python") or ""
    if py and py.rsplit(".", 1)[0] != base_py.rsplit(".", 1)[0]:
        spec["python"] = py.rsplit(".", 1)[0]

    rocm, base_rocm = image.get("rocm_version"), (base or {}).get("rocm_version")
    if rocm and rocm != base_rocm:
        release = _rocm_release(matrix, rocm)
        if release:
            spec["rocm"] = release
        else:
            findings.append({"package": "rocm", "version": rocm, "status": "unauthorized",
                             "authorized": list(matrix.get("releases") or {})})

    catalog = matrix.get("components") or {}
    components: typing.Dict[str, str] = {}
    plain: typing.List[str] = []
    for name, version in added_packages(image, base).items():
        if name == "rocm" or name.startswith("rocm-sdk"):
            continue  # part of the rocm release above
        builds = catalog.get(name) or {}
        matches = [k for k, e in builds.items() if build_version(k, e) == version]
        if not matches:
            if builds:
                findings.append({
                    "package": name, "version": version, "status": "unauthorized",
                    "authorized": [build_version(k, e) for k, e in builds.items()]})
            elif "+" in version:
                findings.append({"package": name, "version": version, "status": "new"})
            else:
                plain.append(f"{name}=={version}")
            continue
        components[name] = matches[0]
        if len(matches) > 1:
            findings.append({"package": name, "version": version, "status": "ambiguous",
                             "chosen": matches[0], "alternatives": matches[1:]})
    if components:
        spec["components"] = components
    if plain:
        findings.append({"package": f"{len(plain)} packages", "version": "",
                         "status": "dependencies", "packages": plain})
    return spec, findings

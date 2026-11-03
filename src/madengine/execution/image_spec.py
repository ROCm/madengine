"""Declarative image specs for model cards.

A workload folder may keep its image spec next to its cards, in
``scripts/<dir>/image_spec.yaml``. A card opts in by referencing it with an
OmegaConf interpolation, and may override any part of it::

    "image_spec": "${image_spec}"
    "image_spec": {"extends": "${image_spec}", "components": {"vllm": "<build>"}}

Interpolations can also reach the compatibility matrix, e.g.
``${compat.indexes.therock_stable}``. Cards without ``image_spec`` build their
hand-written ``dockerfile`` exactly as before, and ``--additional-context``
``image_spec`` is deep-merged over the card's for one run.

The compatibility matrix ``compat/rocm.yaml`` (MAD checkout) is the set of
authorized capabilities: ROCm releases, wheel indexes, and component builds. A
spec selects builds from it; whatever the spec selects is what the workload is
locked to. madengine adds what those builds require, orders everything by
dependency (and the spec's ``order``), and generates the Dockerfile
(madengine.stack)::

    rocm: "10.1"
    python: "3.14"
    components: {vllm: "0.29.1.dev0+rocm10.1.0.g98dff2a81.d20261002@frameworks-vllm"}
    order: [amd-aiter, flash-attn]     # optional relative install order
"""

import hashlib
import json
import os
import shlex
import typing
from dataclasses import asdict, dataclass, field

SPEC_KEY = "image_spec"
SPEC_FILE = "image_spec.yaml"
MATRIX_FILE = os.path.join("compat", "rocm.yaml")
BUILD_PLAN_ARG = "MAD_BUILD_PLAN"
BUILD_INFO_PATH = "/opt/mad/build_info.json"

_SPEC_KEYS = {
    "rocm", "python", "components", "order", "base", "gpu_arch", "env", "setup",
    "system_packages",
}

# Reports the software stack inside an image as JSON. The backend Dockerfiles
# run the same logic at build time and write BUILD_INFO_PATH; this copy is the
# opt-in probe for images built from hand-written Dockerfiles.
PROBE_SCRIPT = r"""
import json, os, sys
from importlib import metadata
info = {"python": "%d.%d.%d" % sys.version_info[:3]}
try:
    info["rocm_version"] = open("/opt/rocm/.info/version").read().strip()
except OSError:
    pass
pkgs = {}
for dist in metadata.distributions():
    name = dist.metadata.get("Name")
    if name:
        pkgs[name.lower()] = dist.version
info["packages"] = dict(sorted(pkgs.items()))
if "rocm_version" not in info and "rocm" in pkgs:
    info["rocm_version"] = pkgs["rocm"]
print(json.dumps(info))
"""


class ImageSpecError(ValueError):
    """An image_spec that cannot be resolved against the compat matrix."""


@dataclass
class BuildPlan:
    """A resolved image_spec."""

    vendor: str
    dockerfile_prefix: str
    build_args: typing.Dict[str, str]
    spec: typing.Dict[str, typing.Any]
    resolved: typing.Dict[str, typing.Any] = field(default_factory=dict)

    def digest(self) -> str:
        """Stable short hash of what the image is built from."""
        args = {k: v for k, v in self.build_args.items() if k != BUILD_PLAN_ARG}
        payload = json.dumps(
            {"vendor": self.vendor, "prefix": self.dockerfile_prefix, "args": args,
             "resolved": self.resolved},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        out = asdict(self)
        out["digest"] = self.digest()
        return out


def _deep_merge(base: dict, over: dict) -> dict:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def matrix_node(root: str = "."):
    """The compatibility matrix as an unresolved OmegaConf node.

    Unresolved, so it can sit next to a spec in one config and be resolved once:
    the matrix backslash-escapes shell ${VAR} references, which a second
    resolution would read as interpolations.
    """
    from omegaconf import OmegaConf

    path = os.path.join(root, MATRIX_FILE)
    if not os.path.exists(path):
        raise ImageSpecError(
            f"{path} not found; image_spec needs the MAD compatibility matrix "
            f"(run from the MAD root or set MODEL_DIR)"
        )
    return OmegaConf.load(path)


def load_matrix(root: str = ".") -> typing.Dict:
    """The compatibility matrix (``compat/rocm.yaml``) of the MAD checkout."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(matrix_node(root), resolve=True)


def _card_spec(model_info: typing.Dict, root: str) -> typing.Optional[dict]:
    """The card's image_spec with OmegaConf interpolations resolved.

    ``${image_spec}`` is the folder's image_spec.yaml and ``${compat}`` the
    matrix; an ``extends`` key is merged under the card's own keys.
    """
    value = model_info.get(SPEC_KEY)
    if value in (None, {}, ""):
        return None
    if not isinstance(value, (str, dict)):
        raise ImageSpecError(
            f"{SPEC_KEY} in model card must be an object or '${{image_spec}}', "
            f"got {type(value).__name__}"
        )
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    folder = model_info.get("_fs_rel_path")
    spec_path = os.path.join(root, "scripts", folder, SPEC_FILE) if folder else None
    nodes: typing.Dict[str, typing.Any] = {"card": value}
    if spec_path and os.path.exists(spec_path):
        nodes[SPEC_KEY] = OmegaConf.to_container(OmegaConf.load(spec_path))
    if os.path.exists(os.path.join(root, MATRIX_FILE)):
        nodes["compat"] = matrix_node(root)
    try:
        resolved = OmegaConf.to_container(OmegaConf.create(nodes), resolve=True)["card"]
    except OmegaConfBaseException as e:
        where = (spec_path if spec_path and os.path.exists(spec_path)
                 else f"no {SPEC_FILE} next to the card's models.json")
        raise ImageSpecError(f"{model_info.get('name')}: {SPEC_KEY} reference: {e} "
                             f"[{where}]") from None
    if isinstance(resolved, dict) and "extends" in resolved:
        base = resolved.pop("extends")
        if not isinstance(base, dict):
            raise ImageSpecError(f"{model_info.get('name')}: 'extends' must name a spec")
        resolved = _deep_merge(base, resolved)
    if not isinstance(resolved, dict):
        raise ImageSpecError(f"{model_info.get('name')}: {SPEC_KEY} must resolve to an object")
    return resolved


def effective_image_spec(
    model_info: typing.Dict, ctx: typing.Dict, root: str = "."
) -> typing.Optional[typing.Dict]:
    """Card ``image_spec`` with context ``image_spec`` deep-merged over it.

    Returns None when neither declares one, which keeps the card on its
    hand-written Dockerfile.
    """
    card = _card_spec(model_info, root) or {}
    override = (ctx.get(SPEC_KEY) if isinstance(ctx, dict) else None) or {}
    if not isinstance(override, dict):
        raise ImageSpecError(
            f"{SPEC_KEY} in additional context must be an object, "
            f"got {type(override).__name__}"
        )
    if not card and not override:
        return None
    spec = _deep_merge(card, override)
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ImageSpecError(
            f"unknown {SPEC_KEY} key(s): {', '.join(sorted(unknown))}; "
            f"expected a subset of {', '.join(sorted(_SPEC_KEYS))}"
        )
    return spec


def dockerfile_prefix(
    spec: typing.Dict,
    gpu_arch: typing.Optional[str] = None,
    root: str = ".",
    name: str = SPEC_KEY,
) -> str:
    """Prefix of the Dockerfile generated for a spec (under .madengine/build/)."""
    return _resolve_composed(spec, gpu_arch, root, name).dockerfile_prefix


def _resolve_composed(
    spec: dict, gpu_arch: typing.Optional[str], root: str, name: str
) -> BuildPlan:
    from madengine.stack import compose, dockerfile

    try:
        plan = compose.compose(spec, load_matrix(root), gpu_arch)
    except compose.ComposeError as e:
        raise ImageSpecError(f"{name}: {e}") from None
    if any(not i["authorized"] for i in plan["items"]):
        # A build outside the matrix: verify what its wheel declares against the
        # rest of the plan before spending a build on it.
        problems = compose.check(plan)
        if problems:
            raise ImageSpecError(
                f"{name}: unauthorized component is inconsistent:\n  "
                + "\n  ".join(problems)
            )
    resolved = {
        "components": {i["name"]: i["key"] for i in plan["items"]},
        "order": [i["name"] for i in plan["items"]],
        "pulled_in": {i["name"]: i["why"] for i in plan["items"]
                      if i["why"].startswith("required by")},
        "unauthorized": [i["name"] for i in plan["items"] if not i["authorized"]],
        "excluded": plan["exclude"],
        "base": plan["base"],
        "python": plan["python"],
        "gpu_arch": plan["gpu_arch"],
    }
    digest = hashlib.sha256(json.dumps(
        {"plan": [(i["name"], i["key"], i["entry"]) for i in plan["items"]],
         "spec": spec}, sort_keys=True, default=str).encode()).hexdigest()[:12]
    completion = compose.completion_requirements(plan) if plan["exclude"] else None
    prefix = dockerfile.write(plan, name, digest, "AMD", root, completion)
    return BuildPlan("AMD", prefix, {}, spec, resolved)


def resolve(
    spec: typing.Dict,
    vendor: str,
    gpu_arch: typing.Optional[str] = None,
    root: str = ".",
    name: str = SPEC_KEY,
) -> BuildPlan:
    """Resolve *spec* into a BuildPlan. image_spec builds AMD images."""
    if (vendor or "").upper() != "AMD":
        raise ImageSpecError(f"image_spec does not support GPU vendor '{vendor}'")
    plan = _resolve_composed(spec, gpu_arch, root, name)
    plan.build_args[BUILD_PLAN_ARG] = json.dumps(
        {"digest": plan.digest(), "spec": spec, "resolved": plan.resolved},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return plan


def read_image_info(console, image: str, probe: bool) -> typing.Optional[typing.Dict]:
    """Software stack inside *image*.

    Reads BUILD_INFO_PATH, which the backend Dockerfiles write at build time.
    With *probe*, falls back to running PROBE_SCRIPT in the image, which works
    for any image that has python3. Returns None when neither is available.
    """
    quoted = shlex.quote(image)
    try:
        out = console.sh(
            f"docker run --rm --entrypoint cat {quoted} {BUILD_INFO_PATH} 2>/dev/null",
            canFail=True,
        )
        if out and out.strip().startswith("{"):
            return json.loads(out)
    except Exception:
        pass
    if not probe:
        return None
    try:
        out = console.sh(
            f"docker run --rm --entrypoint python3 {quoted} "
            f"-c {shlex.quote(PROBE_SCRIPT)}",
            canFail=True,
        )
        return json.loads(out.strip().splitlines()[-1]) if out else None
    except Exception:
        return None

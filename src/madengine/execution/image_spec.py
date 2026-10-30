"""Declarative image specs for model cards.

A model card (or ``--additional-context``) may carry an ``image_spec`` instead
of, or on top of, a hand-written ``dockerfile``. The spec states intent (ROCm or
CUDA version, torch, extra components); the resolver turns it into the two
things the build already understands: a Dockerfile prefix and build args.

Nothing here replaces the existing path. A card without ``image_spec`` builds
exactly as before. A card with one builds the shared backend Dockerfile
(``docker/global_univ2`` by default) with resolved build args, and the resolved
plan is recorded in ``build_manifest.json`` so it can be read back, diffed, or
passed again as ``image_spec`` to rebuild the same stack.

Version knowledge lives in the MAD checkout, not in madengine:

    compat/rocm.json   ROCm releases, install types, index URLs, version pins
    compat/cuda.json   torch/CUDA/cuDNN compatibility, synced from replicate/cog

Spec shape (all keys optional unless noted)::

    image_spec:
      backend: docker/global_univ2     # Dockerfile prefix the spec compiles to
      base: auto | <image>             # auto = matrix default for the vendor
      python: "3.12"                   # NVIDIA backend only
      torch: "2.10"
      rocm: {version: "7.2", install: therock | legacy, index_url: <url>}
      cuda: {version: "12.8"}
      gpu_arch: gfx942                 # default MAD_SYSTEM_GPU_ARCHITECTURE, else all
      components:                      # installed in order
        - {name: migraphx, url: <wheel or deb>, sha256: <hex>}
        - {name: x, amd: {url: ...}, nvidia: {url: ...}}   # per-vendor URLs
      sources:
        - {repo: https://github.com/org/repo, commit: <sha>}
      pip: [numpy==1.26.4]
"""

import hashlib
import json
import os
import re
import shlex
import typing
from dataclasses import asdict, dataclass, field

SPEC_KEY = "image_spec"
DEFAULT_BACKEND = "docker/global_univ2"
COMPAT_DIR = "compat"
BUILD_PLAN_ARG = "MAD_BUILD_PLAN"
BUILD_INFO_PATH = "/opt/mad/build_info.json"

_VENDOR_MATRIX = {"AMD": "rocm.json", "NVIDIA": "cuda.json"}
_SPEC_KEYS = {
    "backend",
    "base",
    "python",
    "torch",
    "rocm",
    "cuda",
    "gpu_arch",
    "components",
    "sources",
    "pip",
}

# Reports the software stack inside an image as JSON. The backend Dockerfiles
# run the same logic at build time and write BUILD_INFO_PATH; this copy is the
# opt-in probe for images built from hand-written Dockerfiles.
PROBE_SCRIPT = r"""
import json, os, sys
from importlib import metadata
info = {"python": "%d.%d.%d" % sys.version_info[:3]}
for key, path in (("rocm_version", "/opt/rocm/.info/version"),):
    try:
        info[key] = open(path).read().strip()
    except OSError:
        pass
try:
    info["cuda_version"] = json.load(open("/usr/local/cuda/version.json"))["cuda"]["version"]
except Exception:
    if os.environ.get("CUDA_VERSION"):
        info["cuda_version"] = os.environ["CUDA_VERSION"]
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
    """A resolved image_spec for one GPU vendor."""

    vendor: str
    dockerfile_prefix: str
    build_args: typing.Dict[str, str]
    spec: typing.Dict[str, typing.Any]
    resolved: typing.Dict[str, typing.Any] = field(default_factory=dict)

    def digest(self) -> str:
        """Stable short hash of what the image is built from."""
        args = {k: v for k, v in self.build_args.items() if k != BUILD_PLAN_ARG}
        payload = json.dumps(
            {"vendor": self.vendor, "prefix": self.dockerfile_prefix, "args": args},
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


def effective_image_spec(
    model_info: typing.Dict, ctx: typing.Dict
) -> typing.Optional[typing.Dict]:
    """Card ``image_spec`` with context ``image_spec`` deep-merged over it.

    Returns None when neither declares one, which keeps the card on its
    hand-written Dockerfile.
    """
    card = model_info.get(SPEC_KEY) or {}
    override = (ctx.get(SPEC_KEY) if isinstance(ctx, dict) else None) or {}
    for name, value in (("model card", card), ("additional context", override)):
        if not isinstance(value, dict):
            raise ImageSpecError(
                f"{SPEC_KEY} in {name} must be an object, got {type(value).__name__}"
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


def backend_prefix(spec: typing.Dict) -> str:
    return spec.get("backend") or DEFAULT_BACKEND


def load_matrix(vendor: str, root: str = ".") -> typing.Dict:
    """Load ``compat/<rocm|cuda>.json`` from the MAD checkout at *root*."""
    filename = _VENDOR_MATRIX.get(vendor.upper())
    if not filename:
        raise ImageSpecError(f"no compat matrix for GPU vendor '{vendor}'")
    path = os.path.join(root, COMPAT_DIR, filename)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ImageSpecError(
            f"{path} not found; image_spec needs the MAD compat matrices "
            f"(run from the MAD root or set MODEL_DIR)"
        ) from None


def _version_key(version: str) -> typing.Tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", version.split("+")[0]))


def _version_matches(want: str, have: str) -> bool:
    """``2.6`` matches ``2.6.0+cu124``; ``2.6.1`` does not match ``2.6.0``."""
    want_key = _version_key(want)
    return _version_key(have)[: len(want_key)] == want_key


def _component_url(component: typing.Dict, vendor: str) -> typing.Optional[str]:
    per_vendor = component.get(vendor.lower())
    if isinstance(per_vendor, dict) and per_vendor.get("url"):
        return per_vendor["url"]
    if "amd" in component or "nvidia" in component:
        return None  # vendor-specific component not meant for this vendor
    return component.get("url")


def _common_args(spec: typing.Dict, vendor: str) -> typing.Tuple[dict, dict]:
    args: typing.Dict[str, str] = {}
    resolved: typing.Dict[str, typing.Any] = {}

    components = spec.get("components") or []
    urls, used = [], []
    for i, component in enumerate(components):
        if not isinstance(component, dict):
            raise ImageSpecError(f"components[{i}] must be an object")
        url = _component_url(component, vendor)
        if not url:
            continue
        if "," in url:
            raise ImageSpecError(
                f"components[{i}] url must be a single URL (',' separates components)"
            )
        urls.append(url)
        entry = {k: v for k, v in component.items() if k not in ("amd", "nvidia")}
        entry["url"] = url
        used.append(entry)
    resolved["components"] = used

    sources = []
    for i, source in enumerate(spec.get("sources") or []):
        repo, commit = (source or {}).get("repo"), (source or {}).get("commit")
        if not repo or not commit:
            raise ImageSpecError(f"sources[{i}] needs both 'repo' and 'commit'")
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
            raise ImageSpecError(
                f"sources[{i}].commit must be a commit SHA, got '{commit}'"
            )
        sources.append(f"{repo.rstrip('/')}/commit/{commit}")
    if sources:
        args["COMPONENT_REPOS"] = ",".join(sources)

    pip = spec.get("pip") or []
    if isinstance(pip, str):
        pip = pip.split()
    if pip:
        args["PIP_PREREQS_EXTRA"] = " ".join(pip)

    shas = [c.get("sha256", "") for c in used]
    if any(shas):
        args["COMPONENT_SHA256"] = ",".join(shas)

    return args, {"component_urls": urls, **resolved}


def _resolve_amd(spec: dict, matrix: dict, gpu_arch: typing.Optional[str]) -> BuildPlan:
    rocm_spec = spec.get("rocm") or {}
    version = str(rocm_spec.get("version") or matrix.get("default_version") or "")
    releases = matrix.get("releases") or {}
    if version not in releases:
        raise ImageSpecError(
            f"ROCm {version or '(unset)'} is not in compat/rocm.json; "
            f"known: {', '.join(sorted(releases, key=_version_key)) or 'none'}"
        )
    release = releases[version]
    install = rocm_spec.get("install") or release.get("default_install", "therock")
    if install not in ("therock", "legacy"):
        raise ImageSpecError(f"rocm.install must be therock or legacy, got '{install}'")

    args, resolved = _common_args(spec, "AMD")
    urls = resolved.pop("component_urls")
    args["INSTALL_TYPE"] = install
    arch = spec.get("gpu_arch") or gpu_arch
    # TheRock wheels carry GPU kernels as per-target extras (device-gfx942);
    # device-all covers every target when the arch is unknown at build time.
    device = f"device-{arch}" if arch else "device-all"
    if install == "therock":
        index = rocm_spec.get("index_url") or release.get("therock_index_url")
        if not index:
            raise ImageSpecError(
                f"ROCm {version} has no TheRock index in compat/rocm.json"
            )
        args["ROCM_RELEASE_URL_THEROCK"] = index
        pin = release.get("rocm_pip_spec", "")
        args["ROCM_EXTRAS"] = f"rocm[libraries,devel,{device}]{pin}"
        if urls:
            args["COMPONENT_RELEASE_URL_THEROCK"] = ",".join(urls)
    else:
        installer = rocm_spec.get("index_url") or release.get("legacy_installer_url")
        if not installer:
            raise ImageSpecError(
                f"ROCm {version} has no legacy installer in compat/rocm.json"
            )
        args["ROCM_RELEASE_URL_LEGACY"] = installer
        if urls:
            args["COMPONENT_RELEASE_URL_LEGACY"] = ",".join(urls)

    base = spec.get("base") or "auto"
    if base == "auto":
        args["BASE_DOCKER"] = matrix.get("bare_os_base", "ubuntu:24.04")
        args["DOCKER_TYPE"] = "BARE_OS"
    else:
        args["BASE_DOCKER"] = base
        args["DOCKER_TYPE"] = "auto"

    if arch:
        args["GPU_ARCH"] = arch

    torch = str(spec.get("torch") or "")
    if torch:
        index = (release.get("torch_index_url") or {}).get(install)
        if not index:
            raise ImageSpecError(
                f"compat/rocm.json has no torch index for ROCm {version} ({install})"
            )
        # "2.10" means any 2.10.x; a full version pins exactly.
        pin = f"{torch}.*" if torch.count(".") == 1 else torch
        extras = f"[{device}]" if install == "therock" else ""
        args["TORCH_PACKAGES"] = f"torch{extras}=={pin}"
        # One TheRock index serves torch builds for several ROCm releases, and
        # each pins its own rocm; resolving both together picks the matching one.
        if install == "therock" and release.get("rocm_pip_spec"):
            args["TORCH_PACKAGES"] += f" rocm{release['rocm_pip_spec']}"
        args["TORCH_INDEX_URL"] = index

    resolved.update(
        {"rocm_version": version, "install": install, "base": args["BASE_DOCKER"],
         "gpu_arch": arch or ""}
    )
    return BuildPlan("AMD", backend_prefix(spec), args, spec, resolved)


def _pick_torch(matrix: dict, torch: str, cuda: typing.Optional[str]) -> dict:
    candidates = [
        e for e in matrix.get("torch") or []
        if e.get("CUDA") and _version_matches(torch, e["Torch"])
        and (not cuda or _version_key(e["CUDA"]) <= _version_key(cuda))
    ]
    if not candidates:
        raise ImageSpecError(
            f"no torch {torch} build for CUDA {cuda or 'any'} in compat/cuda.json"
        )
    return max(
        candidates, key=lambda e: (_version_key(e["CUDA"]), _version_key(e["Torch"]))
    )


def _pick_cuda_base(matrix: dict, cuda: str) -> dict:
    candidates = [
        e for e in matrix.get("base_images") or []
        if _version_matches(cuda, e["CUDA"])
    ]
    if not candidates:
        raise ImageSpecError(
            f"no nvidia/cuda base image for CUDA {cuda} in compat/cuda.json"
        )
    # Devel images carry nvcc, which source components need.
    return max(
        candidates,
        key=lambda e: (
            bool(e.get("IsDevel")),
            _version_key(e["CUDA"]),
            _version_key(e.get("CuDNN", "0")),
            _version_key(e.get("Ubuntu", "0")),
        ),
    )


def _resolve_nvidia(spec: dict, matrix: dict) -> BuildPlan:
    cuda_spec = spec.get("cuda") or {}
    cuda = str(cuda_spec.get("version") or "") or None
    torch = spec.get("torch")

    args, resolved = _common_args(spec, "NVIDIA")
    urls = resolved.pop("component_urls")
    if urls:
        args["COMPONENT_WHEEL_URLS"] = ",".join(urls)

    if torch:
        entry = _pick_torch(matrix, str(torch), cuda)
        # Build on the CUDA torch was compiled for; a requested version is a ceiling.
        if cuda:
            resolved["cuda_requested"] = cuda
        cuda = entry["CUDA"]
        packages = [f"torch=={entry['Torch'].split('+')[0]}"]
        if entry.get("Torchvision"):
            packages.append(f"torchvision=={entry['Torchvision'].split('+')[0]}")
        if entry.get("Torchaudio"):
            packages.append(f"torchaudio=={entry['Torchaudio'].split('+')[0]}")
        args["TORCH_PACKAGES"] = " ".join(packages)
        if entry.get("ExtraIndexURL"):
            args["TORCH_INDEX_URL"] = entry["ExtraIndexURL"]
        resolved["torch"] = entry["Torch"]
    cuda = cuda or matrix.get("default_cuda")
    if not cuda:
        raise ImageSpecError("image_spec needs cuda.version or torch for NVIDIA")

    base = spec.get("base") or "auto"
    if base == "auto":
        image = _pick_cuda_base(matrix, cuda)
        args["BASE_DOCKER"] = f"nvidia/cuda:{image['Tag']}"
    else:
        args["BASE_DOCKER"] = base
    if spec.get("python"):
        args["PYTHON_VERSION"] = str(spec["python"])

    resolved.update({"cuda_version": cuda, "base": args["BASE_DOCKER"]})
    return BuildPlan("NVIDIA", backend_prefix(spec), args, spec, resolved)


def resolve(
    spec: typing.Dict,
    vendor: str,
    gpu_arch: typing.Optional[str] = None,
    root: str = ".",
    matrix: typing.Optional[typing.Dict] = None,
) -> BuildPlan:
    """Resolve *spec* for one vendor into a BuildPlan."""
    vendor = (vendor or "").upper()
    if matrix is None:
        matrix = load_matrix(vendor, root)
    if vendor == "AMD":
        plan = _resolve_amd(spec, matrix, gpu_arch)
    elif vendor == "NVIDIA":
        plan = _resolve_nvidia(spec, matrix)
    else:
        raise ImageSpecError(f"image_spec does not support GPU vendor '{vendor}'")
    plan.build_args[BUILD_PLAN_ARG] = json.dumps(
        {"digest": plan.digest(), "spec": spec, "resolved": plan.resolved},
        sort_keys=True,
        separators=(",", ":"),
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

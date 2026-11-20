"""The submit-time environment of a slurm_multi card, with where each value came from.

A slurm_multi card runs its own launcher script (MAD's run_*.slurm), which owns its
environment: it fills site defaults, applies the model's recipe and records every
value with the layer that set it (MAD scripts/common/config/layers.sh). madengine's
part is the top layer, "submit time", which it builds from, weakest first:

    mad-config    the card's layered config (mad-config.yaml model: / benchmark:)
    docker-env-vars   the manifest's / additional context's docker_env_vars
    card          the card's env_vars
    additional-context   the run's env_vars, without keys only a preset set
    distributed   xP / yD from the card's sglang_disagg block, when not set above

and then the values madengine itself decides (the image the nodes pull, MAD_TOOLS,
the Kubernetes LOG_PATH, MAD_IMAGE_COMPONENTS -- the builds an image_spec image holds), labelled
"madengine".

Every path that runs a slurm_multi card -- SLURM, Spur (a SlurmDeployment), and
Kubernetes per pod -- builds this one environment here, so a card gets the same
submit layer wherever it runs. The source of each name travels with it as
MAD_ENV_SOURCES ("NAME=label,..."), which the launcher folds into its record.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import PRESET_ENV_KEYS
from .layered_config import resolve_for_model

SOURCES_VAR = "MAD_ENV_SOURCES"
COMPONENTS_VAR = "MAD_IMAGE_COMPONENTS"


class SubmitEnv:
    """Ordered env plus the source of each name; later set() wins, like dict.update."""

    def __init__(self):
        self.env: Dict[str, str] = {}
        self.sources: Dict[str, str] = {}

    def update(self, values: Optional[Dict[str, Any]], source: str) -> None:
        for k, v in (values or {}).items():
            self.set(k, v, source)

    def set(self, key: str, value: Any, source: str) -> None:
        self.env[key] = str(value)
        self.sources[key] = source

    def setdefault(self, key: str, value: Any, source: str) -> str:
        if key not in self.env:
            self.set(key, value, source)
        return self.env[key]

    def encoded_sources(self) -> str:
        return ",".join(f"{k}={v}" for k, v in sorted(self.sources.items()) if k != SOURCES_VAR)

    def final(self) -> Dict[str, str]:
        """The env to export, with MAD_ENV_SOURCES."""
        out = dict(self.env)
        out[SOURCES_VAR] = self.encoded_sources()
        return out


def user_env_vars(additional_context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """additional_context.env_vars without keys only a preset set, and the keys dropped.

    For the self-managed path. load_slurm_config merges the slurm preset
    profiles' env_vars into additional_context.env_vars, where they become
    indistinguishable from what the user asked for. The templated launchers
    rely on them. A slurm_multi card does not: it runs its own launcher
    script, which owns its environment, and the same card submitted without
    madengine (plain `sbatch --export=ALL` of the card script)
    never receives them. Exporting them anyway makes the madengine run a
    different experiment from the standalone one.

    It is not hypothetical. MAD's run_xPyD_models.slurm sources a connector
    env that sets HSA_ENABLE_SDMA=1 but forwards `-e K=${K:-default}`, so the
    multi-node profile's HSA_ENABLE_SDMA=0 won: SDMA was off under madengine
    and on under plain sbatch. The same profile's NCCL_SOCKET_IFNAME=eth0
    preempts the card's own fabric detection (eno0 on AINIC, fenic0 on
    Thor2), and NCCL_IB_DISABLE=1 would push NCCL onto TCP in any launcher
    that forwards it.

    Only preset-ONLY keys are dropped: a user who sets NCCL_IB_DISABLE
    themselves still gets exactly their value, and the card's env_vars,
    docker_env_vars and layered config are untouched.
    """
    env = dict(additional_context.get("env_vars") or {})
    preset_only = additional_context.get(PRESET_ENV_KEYS) or []
    dropped = sorted(k for k in preset_only if k in env)
    for key in dropped:
        del env[key]
    return env, dropped


def submit_env(
    model_info: Dict[str, Any],
    additional_context: Dict[str, Any],
    scripts_dir: Optional[Path],
    distributed_config: Optional[Dict[str, Any]] = None,
    image_build: Optional[Dict[str, Any]] = None,
) -> Tuple[SubmitEnv, List[str], List[str]]:
    """The card's submit layer: (env with sources, layered-config warnings, preset keys dropped).

    Raises LayeredConfigError when the card's layered config is refused.
    """
    user_env, dropped = user_env_vars(additional_context)
    out = SubmitEnv()
    layered_env, warnings = resolve_for_model(model_info, scripts_dir, runtime_env=user_env)
    out.update(layered_env, "mad-config")
    out.update(additional_context.get("docker_env_vars"), "docker-env-vars")
    out.update(model_info.get("env_vars"), "card")
    out.update(user_env, "additional-context")
    sglang = ((model_info.get("distributed") or {}).get("sglang_disagg")
              or (distributed_config or {}).get("sglang_disagg") or {})
    if sglang:
        out.setdefault("xP", sglang.get("prefill_nodes", 1), "distributed")
        out.setdefault("yD", sglang.get("decode_nodes", 1), "distributed")
    components = image_components(image_build)
    if components:
        out.setdefault(COMPONENTS_VAR, components, "madengine")
    return out, warnings, dropped


def image_components(image_build: Optional[Dict[str, Any]]) -> Optional[str]:
    """The builds an image_spec image holds, as MAD's recipe check reads them
    (MAD_IMAGE_COMPONENTS: name=key,...), from its manifest entry (built_images.<image>.build_plan);
    None for an image from a hand-written Dockerfile."""
    resolved = ((image_build or {}).get("build_plan") or {}).get("resolved") or {}
    comps = resolved.get("components") or {}
    return ",".join(f"{n}={k}" for n, k in sorted(comps.items())) or None   # by name: one form everywhere


def recipe_precheck(env: Dict[str, str], script_path: Path, gpu_arch: Optional[str]) -> Optional[str]:
    """MAD's recipe gate before submitting, when the target GPU is known (slurm.gpu_arch): the
    refusal text, or None. Runs the card checkout's scripts/common/recipe/precheck.py with the
    card's submit env; a checkout without it, or no known GPU, checks nothing. The launcher still
    checks on the allocated nodes -- this only refuses before the job waits in a queue."""
    import os
    import subprocess
    import sys

    if not gpu_arch:
        return None
    tool = Path(script_path).resolve().parent.parent / "common" / "recipe" / "precheck.py"
    if not tool.is_file():
        return None
    p = subprocess.run([sys.executable, str(tool), "--script", str(script_path), "--arch", str(gpu_arch)],
                       env={**os.environ, **env}, capture_output=True, text=True, timeout=120)
    if p.returncode == 0:
        return None
    return (p.stderr or p.stdout).strip() or f"recipe check refused (exit {p.returncode})"


MULTINODE_CATALOGS = ("vllm_dissag", "vllm_multinode", "sglang_disagg")
EMIT_CARD_KEYS = ("name", "scripts", "tags", "dockerfile", "docker_build_arg", "image_spec")


def emit_config(model_info: Dict[str, Any], out_dir: Path, image_build: Optional[Dict[str, Any]] = None,
                find=None) -> Optional[Dict[str, str]]:
    """The card's configuration in its generalized form, written by the card checkout's
    scripts/common/config/emit.py (the card on MAD's shared layers) -- with its image generalized
    against the checkout's compat matrix from its manifest entry (*image_build*:
    built_images.<image>): `image.spec` the image as a spec over compat builds, `image.unresolved`
    what compat cannot express yet; the general facts compat lacks are merged into
    <out_dir>/compat/rocm.yaml, a copy of the checkout's matrix for the PR. Returns {"path",
    "sha256"[, "compat"]}, or None for a card that is not a multinode launcher's, or a checkout
    without the tool. Shared by `madengine build --emit-config` and the pipeline's STANDALONE mode,
    so both write the same files."""
    import hashlib
    import json as _json
    import os
    import subprocess
    import sys
    import tempfile

    script = Path(model_info.get("scripts") or "")
    catalog = script.parent.name
    if catalog not in MULTINODE_CATALOGS:
        return None
    tool = script.resolve().parent.parent / "common" / "config" / "emit.py"
    if not tool.is_file():
        return None
    # madengine names a directory's cards <dir>/<card>: the file is named for the card.
    out = Path(out_dir) / catalog / f"{str(model_info['name']).rsplit('/', 1)[-1]}.yaml"
    # The card's own fields and the run's image, however this build path recorded them, so
    # every path (built, prebuilt, STANDALONE) writes the same file for the same card and image.
    card = {k: model_info[k] for k in EMIT_CARD_KEYS if model_info.get(k) not in (None, "", {}, [])}
    env = dict(model_info.get("env_vars") or {})
    image = model_info.get("docker_image") or env.pop("DOCKER_IMAGE_NAME", None)
    env.pop("DOCKER_IMAGE_NAME", None)
    if image:
        env["DOCKER_IMAGE_NAME"] = image          # last, wherever a path put it
    card["env_vars"] = env
    root = script.resolve().parent.parent.parent
    delta = None
    if image_build and (root / "compat" / "rocm.yaml").is_file():
        from madengine.execution import image_spec as _image_spec
        from madengine.stack import generalize as _gen

        spec, delta = _gen.image_of(image_build, _image_spec.load_matrix(str(root)), card["name"], find,
                                    _image_spec.named_specs(str(root)))
        if spec:
            card["_image_spec"] = spec
        unresolved = [{k: v for k, v in u.items() if k != "seen_in"}
                      for u in delta["unresolved"] + delta["contradicts"]]
        if unresolved:
            card["_image_unresolved"] = unresolved
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        _json.dump(card, f, default=str)
    try:
        p = subprocess.run([sys.executable, str(tool), "--card-json", f.name, "--catalog", catalog,
                            "-o", str(out)], capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(f.name)
    if p.returncode != 0:
        raise RuntimeError(f"{model_info['name']}: emit.py failed: {(p.stderr or p.stdout).strip()}")
    got = {"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    if delta and delta["add"]:
        from madengine.stack import generalize as _gen

        matrix = Path(out_dir) / "compat" / "rocm.yaml"
        text = matrix.read_text() if matrix.is_file() else (root / "compat" / "rocm.yaml").read_text()
        matrix.parent.mkdir(parents=True, exist_ok=True)
        matrix.write_text(_gen.insert_builds(text, delta["add"]))
        got["compat"] = str(matrix)
    return got


def placement_directives(env: Dict[str, str], script_path: Path) -> List[str]:
    """sbatch options for the run's topology placement: when MAD_TOPOLOGY names a spec whose
    placement needs the scheduler (e.g. a role held inside one multi-node scale-up domain), the card
    checkout's scripts/common/topology/topology.py scheduler-hints says what to ask for
    (["--segment=18"] on SLURM's block topology). Nothing without a spec or the tool. Shared by
    madengine's SLURM path and the pipeline's STANDALONE mode, so both request the same."""
    import json as _json
    import os
    import subprocess
    import sys

    if not env.get("MAD_TOPOLOGY"):
        return []
    script_path = Path(script_path).resolve()
    tool = script_path.parent.parent / "common" / "topology" / "topology.py"
    if not tool.is_file():
        return []
    p = subprocess.run([sys.executable, str(tool), "scheduler-hints", "--scheduler", "slurm",
                        "--launcher", script_path.parent.name],
                       env={**os.environ, **env}, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise ValueError(f"MAD_TOPOLOGY: {(p.stderr or p.stdout).strip()}")
    return [o for o in _json.loads(p.stdout).get("options") or [] if isinstance(o, str)]


def platform_gpus_per_node(env: Dict[str, str], script_path: Path) -> Optional[int]:
    """GPUs per node of the run's platform (MAD_PLATFORM, else the catalog's default_platform),
    from the card checkout's capability catalog -- compat/hardware.yaml (compat/rocm.yaml in a
    checkout from before MAD split it) -- what MAD's launchers use when nothing passes
    GPUS_PER_NODE. None without the catalog."""
    import yaml as _yaml

    compat = Path(script_path).resolve().parent.parent.parent / "compat"
    catalog = next((compat / f for f in ("hardware.yaml", "rocm.yaml") if (compat / f).is_file()), None)
    if catalog is None:
        return None
    with open(catalog) as f:
        c = _yaml.safe_load(f) or {}
    name = env.get("MAD_PLATFORM") or c.get("default_platform")
    p = (c.get("platforms") or {}).get(name) or {}
    return int(p["gpus_per_node"]) if p.get("gpus_per_node") else None


def k8s_domain_affinity(env: Dict[str, str], script_path: Path, domain_label: Optional[str]) -> Optional[str]:
    """The node label to hold the whole Job inside one scale-up domain on, when the run's topology
    (MAD_TOPOLOGY) places it in one and the site names its domain label; else None. A placement one
    Job cannot express raises ValueError. From the card checkout's topology.py scheduler-hints, as
    MAD's own Kubernetes submit uses it."""
    import json as _json
    import os
    import subprocess
    import sys

    if not env.get("MAD_TOPOLOGY") or not domain_label:
        return None
    script_path = Path(script_path).resolve()
    tool = script_path.parent.parent / "common" / "topology" / "topology.py"
    if not tool.is_file():
        return None
    p = subprocess.run([sys.executable, str(tool), "scheduler-hints", "--scheduler", "kubernetes",
                        "--launcher", script_path.parent.name, "--domain-label", domain_label],
                       env={**os.environ, **env}, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise ValueError(f"MAD_TOPOLOGY: {(p.stderr or p.stdout).strip()}")
    for o in _json.loads(p.stdout).get("options") or []:
        if isinstance(o, dict) and o.get("pod_affinity_topology_key"):
            return o["pod_affinity_topology_key"]
    return None

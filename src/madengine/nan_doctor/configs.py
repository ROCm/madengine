#!/usr/bin/env python3
"""The switch catalogue of differential mode: environment settings that make a stack compute the same function
through a different implementation (mad-debug-numeric, `step2-diff.md`).

    madengine nan arms detect [--root DIR] <command ...>          -> FRAMEWORK=<vllm|sglang|torch>
    madengine nan arms arms --framework F [--only a,b] [--extra fp32] [--drop a,b] [--set VAR=VALUE]
        -> one arm per line, `<name>|<VAR=VALUE ...>`: A, A2, REF, then flip_<switch> per switch
    madengine nan arms arms ... --format context --out-dir DIR
        -> one arm per line, `<name><TAB><JSON>`: a madengine --additional-context whose
           docker_env_vars set the arm's switches and turn the tap on (ND_DIFF_OUT=/myworkspace/DIR/<name>)
    madengine nan arms list [--framework F]                       -> the catalogue, one switch per line

Every switch is environment-only (no model edit): a stack's own variable, or one that nan_doctor_rt/diff_tap's
sitecustomize turns into an engine argument / a torch setting (ND_VLLM_*, ND_SGLANG_*, ND_DIFF_TF32,
ND_DIFF_SDPA). The REFERENCE configuration is the user's with every switch on its conservative side; an arm
flip_<s> is the user's with switch s alone flipped -- the arm that AGREEs with the reference while the user's
configuration DIVERGEs names the switch (`madengine nan diff`: ND_DIFF_SWITCH). `extra` switches (fp32) are
never in the default reference: they change too much at once and cost memory; --extra adds them.

Checked on vLLM 0.33.0, SGLang 0.5.23 and torch 2.10 / transformers 5.20.2 (the model still runs, and the
switch changes the outputs). INERT there, by design: vLLM `aiter` (VLLM_ROCM_USE_AITER defaults to 0),
`custom_ops` (compiled vLLM already runs custom_ops=none; it matters with compilation off, as in the
reference), `graphs` (replay runs the same kernels; its log shows the capture skipped); SGLang `graphs`; HF
fp32 `tf32`/`blas`/`sdpa` on a model already in fp32 math. Re-check after an upgrade.
"""

# flake8: noqa: E741,E731,E402  (terse variable names, kept from the original scripts)

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

# name, framework, env (conservative side), what it replaces, extra (not in the default reference)
SWITCHES = [
    # vLLM (EngineArgs fields through diff_tap; VLLM_* read by vLLM itself)
    ("graphs", "vllm", {"ND_VLLM_CUDAGRAPH_MODE": "NONE"}, "HIP-graph replay", False),
    (
        "compile",
        "vllm",
        {"ND_VLLM_COMPILE": "0"},
        "torch.compile / Inductor fusion (compilation mode NONE)",
        False,
    ),
    (
        "custom_ops",
        "vllm",
        {"ND_VLLM_CUSTOM_OPS": "none"},
        "vLLM's fused custom ops (_C.*, aiter) -> their forward_native PyTorch code",
        False,
    ),
    (
        "aiter",
        "vllm",
        {"VLLM_ROCM_USE_AITER": "0"},
        "every aiter kernel (MLA, RMSNorm, MoE, linear, ...)",
        False,
    ),
    (
        "attention",
        "vllm",
        {"ND_VLLM_ATTENTION_BACKEND": "TRITON_ATTN"},
        "the attention backend -> TRITON_ATTN (an MLA model: set TRITON_MLA with --set)",
        False,
    ),
    (
        "skinny_gemm",
        "vllm",
        {"VLLM_ROCM_USE_SKINNY_GEMM": "0"},
        "ROCm skinny GEMM kernels -> torch GEMM",
        False,
    ),
    ("fp32", "vllm", {"ND_VLLM_DTYPE": "float32"}, "the model dtype -> float32", True),
    # SGLang (Engine kwargs through diff_tap; SGLANG_* read by SGLang itself)
    (
        "graphs",
        "sglang",
        {"ND_SGLANG_DISABLE_CUDA_GRAPH": "1"},
        "HIP-graph replay",
        False,
    ),
    (
        "fused_ops",
        "sglang",
        {"SGLANG_FORCE_FUSED_OP_BACKEND": "torch"},
        "fused ops -> their torch code",
        False,
    ),
    (
        "attention",
        "sglang",
        {"ND_SGLANG_ATTENTION_BACKEND": "triton"},
        "the attention backend -> triton",
        False,
    ),
    (
        "fp32",
        "sglang",
        {"ND_SGLANG_DTYPE": "float32"},
        "the model dtype -> float32",
        True,
    ),
    # plain PyTorch / Hugging Face / a training script
    ("compile", "torch", {"TORCHDYNAMO_DISABLE": "1"}, "torch.compile -> eager", False),
    (
        "tf32",
        "torch",
        {"ND_DIFF_TF32": "0", "HIPBLASLT_ALLOW_TF32": "0"},
        "TF32 GEMMs / convolutions -> fp32",
        False,
    ),
    (
        "blas",
        "torch",
        {"TORCH_BLAS_PREFER_HIPBLASLT": "0"},
        "hipBLASLt -> rocBLAS",
        False,
    ),
    (
        "sdpa",
        "torch",
        {"ND_DIFF_SDPA": "math"},
        "flash / mem-efficient attention -> the math SDPA",
        False,
    ),
]
FRAMEWORKS = ("vllm", "sglang", "torch")
# where a madengine run sees its MAD checkout, and the tap inside it (scripts/common is
# populated from the madengine package for the run)
WORKSPACE = "/myworkspace"
TAP_DIR = WORKSPACE + "/scripts/common/tools/nan_doctor/nan_doctor_rt/diff_tap"


def switches(framework, extra=(), drop=(), only=()):
    out = []
    for name, fw, env, what, is_extra in SWITCHES:
        if fw != framework or name in drop or (only and name not in only):
            continue
        if is_extra and name not in extra:
            continue
        out.append((name, env, what))
    return out


def _env_words(env):
    return " ".join("%s=%s" % (k, shlex.quote(v)) for k, v in env.items())


def arms(framework, extra=(), drop=(), only=(), overrides=None):
    """[(arm name, env dict)]: A, A2, REF, flip_<switch> per switch."""
    sw = switches(framework, extra, drop, only)
    if overrides:
        sw = [
            (n, {k: overrides.get(k, v) for k, v in env.items()}, w) for n, env, w in sw
        ]
    ref = {}
    for _, env, _ in sw:
        ref.update(env)
    out = [("A", {}), ("A2", {}), ("REF", ref)]
    if len(sw) > 1:  # with one switch, its flip IS the reference
        out += [("flip_" + n, dict(env)) for n, env, _ in sw]
    return out


def context(name, env, out_dir):
    """A madengine --additional-context for one arm: its switches and the tap, as
    docker_env_vars (ND_DIFF_OUT under the run's /myworkspace)."""
    dv = dict(env)
    dv["ND_DIFF_OUT"] = "%s/%s/%s" % (WORKSPACE, out_dir.strip("/"), name)
    dv["PYTHONPATH"] = TAP_DIR
    return {"docker_env_vars": dv}


_PY = re.compile(r"[\w./$@{}-]+\.py\b")


def detect(cmd, root=None):
    """The framework a model command runs: the first .py file in it that imports vllm / sglang; else torch.
    A path with $ND_REPO / @REPO@ is resolved against root (the repo)."""
    text = " ".join(cmd)
    for m in _PY.finditer(text):
        p = m.group(0)
        if root:
            p = (
                p.replace("$ND_REPO", root)
                .replace("${ND_REPO}", root)
                .replace("@REPO@", root)
            )
        if not os.path.isfile(p):
            continue
        try:
            src = Path(p).read_text(errors="replace")
        except OSError:
            continue
        v = re.search(r"^\s*(from|import)\s+vllm\b", src, re.M)
        g = re.search(r"^\s*(from|import)\s+sglang\b", src, re.M)
        if v and g:
            print(
                "madengine nan arms detect: %s imports both vllm and sglang -- set FRAMEWORK"
                % p,
                file=sys.stderr,
            )
            return "ambiguous"
        if v:
            return "vllm"
        if g:
            return "sglang"
    if re.search(r"\bvllm\b", text):
        return "vllm"
    if re.search(r"\bsglang\b", text):
        return "sglang"
    return "torch"


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="madengine nan arms", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("command", nargs=argparse.REMAINDER)
    d.add_argument("--root", default=os.environ.get("NAN_DOCTOR_ROOT"))
    a = sub.add_parser("arms")
    a.add_argument("--framework", required=True, choices=FRAMEWORKS)
    a.add_argument(
        "--extra", default="", help="comma list of extra switches to add (fp32)"
    )
    a.add_argument("--drop", default="", help="comma list of switches to leave out")
    a.add_argument("--only", default="", help="comma list: only these switches")
    a.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="override a switch's conservative value (ND_VLLM_ATTENTION_BACKEND=TRITON_MLA)",
    )
    a.add_argument(
        "--format",
        choices=("env", "context"),
        default="env",
        help="env: `<name>|<VAR=VALUE ...>`; context: `<name><TAB><additional-context JSON>`",
    )
    a.add_argument(
        "--out-dir",
        help="--format context: the tap's output dir, relative to the MAD checkout "
        "(each arm writes <out-dir>/<arm>/outputs.jsonl)",
    )
    l_ = sub.add_parser("list")
    l_.add_argument("--framework", choices=FRAMEWORKS)
    args = p.parse_args(argv)
    if args.cmd == "detect":
        print("FRAMEWORK=%s" % detect(args.command, args.root))
    elif args.cmd == "arms":
        if args.format == "context" and not args.out_dir:
            p.error("--format context needs --out-dir")
        split = lambda s: tuple(x for x in s.split(",") if x)  # noqa: E731
        ov = dict(s.split("=", 1) for s in args.set)
        for name, env in arms(
            args.framework, split(args.extra), split(args.drop), split(args.only), ov
        ):
            if args.format == "context":
                print("%s\t%s" % (name, json.dumps(context(name, env, args.out_dir))))
            else:
                print("%s|%s" % (name, _env_words(env)))
    else:
        for name, fw, env, what, is_extra in SWITCHES:
            if args.framework and fw != args.framework:
                continue
            print(
                "%-7s %-11s %-60s %s%s"
                % (fw, name, _env_words(env), what, "  [extra]" if is_extra else "")
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

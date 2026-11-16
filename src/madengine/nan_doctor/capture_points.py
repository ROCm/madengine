#!/usr/bin/env python3
"""Point at the lines of a model (and its launcher) that mad-debug-nan steps 4 and
5a-load need, so the agent reads windows instead of whole files.

    madengine nan capture-points <model.py> [<launcher.sh> ...] [--context N]

Static and model-independent (no import, no GPU).  For .py files it uses the AST:

  IMPORTS   last top-level import       -> where `from nan_doctor_rt.op_nan_capture import ...` goes
  HOOK      framework-owned loops (Lightning training_step / Trainer.fit, HF
            Trainer.train): no loop in user code -> wire the capture as a callback
  LOOP      loops whose body does training/repro work (backward, optimizer/scheduler
            step, pipeline progress, train/training_step, ...), ranked; the top ones
            get a code window -> where cap.step() / cap.close() go
  LAZY      torch.compile, TrainPipeline*, CUDA graphs: built on the first call, so
            arm after step 1 (NAN_CAPTURE_ARM_AFTER_STEPS=1)
  SWALLOW   `except Exception` / bare `except` around the work: capture exceptions
            get swallowed -> judge by the capsule dir, and the model may print its
            own NaN line on the capture's message
  NANCHECK  the model's own NaN/inf checks (and any step interval) -> the control
            arm's floor (run length must reach the first check)
  DIST      torch.distributed / torchrun -> world size > 1 needs sync_stop (step()
            does it)
  FANOUT    multiprocessing / spawn / DataLoader workers / extra streams / env
            amplifiers / thread counts -> 5a-load's <CONTENTION>
For shell launchers: concurrent trial loops (`&`), device pinning, container
launches, torchrun --nproc-per-node, env amplifiers -> Declare-contention.

It is a heuristic map, not an analysis: `no LOOP candidate` means read the entry
point (main / __main__) yourself -- never guess a placement from silence.
"""

import argparse
import ast
import re

WORK = [  # (regex on a call's dotted name, weight, label)
    (r"(^|\.)backward$", 5, "backward"),
    (r"(optim|optimizer|opt)\w*\.step$", 5, "optimizer.step"),
    (r"(^|\.)(engine|model_engine|ds_engine)\.step$", 5, "engine.step"),
    (r"(^|\.)(scaler)\.step$", 4, "scaler.step"),
    (r"(sched|scheduler|lr_sched)\w*\.step$", 2, "scheduler.step"),
    (r"(^|\.)progress$", 4, "pipeline.progress"),
    (
        r"(^|\.)(train_step|training_step|train_one_step|train_batch|step_fn)$",
        4,
        "train step",
    ),
    (r"(^|\.)(zero_grad)$", 2, "zero_grad"),
    (
        r"scaled_dot_product_attention$|(^|\.)(matmul|mm|bmm|linear|addmm)$",
        2,
        "compute op",
    ),
    (r"(^|\.)(forward|model|net|module)$", 1, "forward"),
]
LAZY = [
    (r"torch\.compile|(^|\.)compile$", "torch.compile (built on first call)"),
    (r"TrainPipeline\w*", "torchrec TrainPipeline (built in first progress())"),
    (r"CUDAGraph|cuda\.graph|make_graphed_callables", "CUDA graph capture"),
]
FAN_PY = [
    (
        r"multiprocessing|mp\.Process|mp\.Pool|\.spawn\(|set_start_method|ProcessPoolExecutor",
        "process fan-out",
    ),
    (r"DataLoader\(.*num_workers", "DataLoader workers"),
    (r"cuda\.Stream\(|torch\.cuda\.stream\(", "extra CUDA streams"),
    (
        r"HSA_DISABLE_CACHE|TORCH_ROCM_FA_PREFER_CK|GPU_MAX_HW_QUEUES|TORCH_BLAS_PREFER_HIPBLASLT|AITER_\w+",
        "env amplifier / backend switch",
    ),
    (r"OMP_NUM_THREADS|set_num_threads", "host threads"),
]
HOOK_DEFS = {"training_step", "train_step", "on_train_batch_start", "compute_loss"}
HOOK_CALLS = r"(^|\.)(fit|train)$|(^|\.)Trainer$"
DIST = r"torch\.distributed|init_process_group|dist\.\w+\(|LOCAL_RANK|WORLD_SIZE"
NAN = r"isnan|isfinite|isinf|nan_to_num|math\.isnan"
FAN_SH = [
    (r"^\s*for\b.*\bin\b", "trial loop"),
    (r"&\s*(#.*)?$", "backgrounded launch (concurrent)"),
    (r"\bwait\b", "wait"),
    (
        r"HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|--gpus\b|ROCR_VISIBLE_DEVICES",
        "device pinning",
    ),
    (
        r"docker run|apptainer exec|singularity exec|lux_exec|srun\b",
        "container/process launch",
    ),
    (r"torchrun|--nproc[-_]per[-_]node|--ntasks-per-node|mpirun", "distributed launch"),
    (r"%\s*\$?\{?\w*(gpu|GPU)", "trial -> GPU arithmetic"),
    (
        r"HSA_DISABLE_CACHE|TORCH_ROCM_FA_PREFER_CK|GPU_MAX_HW_QUEUES|OMP_NUM_THREADS|AITER_\w+",
        "env amplifier / threads",
    ),
    (r"NAN_CAPTURE_\w+", "capture knob"),
    (r"grep .*-q|SUCCESS|REPRODUCED|FAILURE", "NaN-signal check"),
]


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return (base + "." if base else "") + node.attr
    if isinstance(node, ast.Call):
        return dotted(node.func)
    return ""


def enclosing(tree):
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                if hasattr(n, "lineno"):
                    owner.setdefault(n.lineno, fn.name)
    return owner


def scan_py(path, src, ctx, show):
    lines = src.splitlines()
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print("%s: cannot parse (%s) -- regex hits only" % (path, e))
        tree = None
    if tree is not None:
        imps = [
            n.lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        if imps:
            out.append((max(imps), "IMPORTS", "last top-level import"))
        owner = enclosing(tree)
        loops = []
        for n in ast.walk(tree):
            if not isinstance(n, (ast.For, ast.While, ast.AsyncFor)):
                continue
            score, why = 0, set()
            for c in ast.walk(n):
                if isinstance(c, ast.Call):
                    name = dotted(c.func)
                    for rx, w, lab in WORK:
                        if re.search(rx, name):
                            score += w
                            why.add(lab)
                            break
            if score:
                loops.append(
                    (score, n.lineno, getattr(n, "end_lineno", n.lineno), sorted(why))
                )
        # an outer loop scores its inner loops too; prefer the innermost loop that
        # still carries the step-level work (backward / optimizer / progress)
        loops.sort(key=lambda t: (-t[0], t[2] - t[1]))  # equal work: innermost first
        for k, (sc, a, b, why) in enumerate(loops[:6], 1):
            out.append(
                (
                    a,
                    "LOOP",
                    "#%d score %d, lines %d-%d, in %s(): %s"
                    % (k, sc, a, b, owner.get(a, "<module>"), ", ".join(why)),
                )
            )
        for n in ast.walk(tree):
            if isinstance(n, ast.ExceptHandler) and (
                n.type is None or dotted(n.type) in ("Exception", "BaseException")
            ):
                out.append(
                    (
                        n.lineno,
                        "SWALLOW",
                        "except %s in %s()"
                        % (dotted(n.type) or "(bare)", owner.get(n.lineno, "<module>")),
                    )
                )
        hooks = []
        for n in ast.walk(tree):
            if (
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in HOOK_DEFS
            ):
                hooks.append(
                    (n.lineno, "HOOK", "def %s: the framework owns the loop" % n.name)
                )
            if isinstance(n, ast.Call) and re.search(HOOK_CALLS, dotted(n.func)):
                hooks.append(
                    (
                        n.lineno,
                        "HOOK",
                        "%s(): framework-owned loop -> callback" % dotted(n.func),
                    )
                )
        out += hooks
        if not loops:
            out.append(
                (
                    0,
                    "LOOP",
                    "no LOOP candidate%s"
                    % (
                        ": the framework owns the loop -- wire the capture as a callback (step4-capture.md)"
                        if hooks
                        else ": read the entry point (main / __main__) yourself"
                    ),
                )
            )
    doc = set()  # docstring lines: prose, not code
    if tree is not None:
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Expr)
                and isinstance(getattr(n, "value", None), ast.Constant)
                and isinstance(n.value.value, str)
            ):
                doc.update(range(n.lineno, getattr(n, "end_lineno", n.lineno) + 1))
    for i, ln in enumerate(lines, 1):
        if i in doc:
            continue
        code = ln.split("#", 1)[0]
        if re.match(
            r"\s*(print|logger\.\w+|logging\.\w+|log\.\w+)\(", code
        ) or re.match(r'\s*(f?["\']|\+ *f?["\'])', code):
            continue  # log/print lines and string continuations
        for rx, lab in LAZY:
            if re.search(rx, code):
                out.append((i, "LAZY", lab))
        if re.search(NAN, code) and not code.strip().startswith(("import", "from")):
            itv = None  # the guard's step interval, if one is near
            for j in sorted(
                range(max(1, i - 12), min(len(lines), i + 12) + 1),
                key=lambda j: abs(j - i),
            ):
                m = re.search(
                    r"%\s*\(?\s*([\w.]+)\s*\)?\s*[=!<>]=?|==\s*([\w.]+\s*-\s*[\w.]+)",
                    lines[j - 1],
                )
                if m:
                    itv = "guarded at L%d: %s" % (j, lines[j - 1].strip()[:40])
                    break
            out.append(
                (
                    i,
                    "NANCHECK",
                    itv or "every iteration? (no step guard within +-12 lines)",
                )
            )
        if re.search(DIST, code):
            out.append((i, "DIST", "distributed"))
        for rx, lab in FAN_PY:
            if re.search(rx, code):
                out.append((i, "FANOUT", lab))
    return lines, out


def scan_sh(path, src):
    lines = src.splitlines()
    out = []
    for i, ln in enumerate(lines, 1):
        if ln.lstrip().startswith("#"):
            continue
        for rx, lab in FAN_SH:
            if re.search(rx, ln):
                out.append((i, "LAUNCH", lab))
    return lines, out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="+")
    p.add_argument(
        "--context",
        type=int,
        default=4,
        help="window half-height for the top LOOP (default 4)",
    )
    p.add_argument(
        "--windows", type=int, default=2, help="how many top loops get a code window"
    )
    p.add_argument("--max-hits", type=int, default=6, help="per category, per file")
    a = p.parse_args()
    for path in a.files:
        src = open(path, errors="replace").read()
        if path.endswith(".py"):
            lines, hits = scan_py(path, src, a.context, a.windows)
        else:
            lines, hits = scan_sh(path, src)
        print("== %s (%d lines)" % (path, len(lines)))
        seen, shown = {}, set()
        for ln, cat, what in sorted(
            hits,
            key=lambda h: (h[1] != "LOOP", h[1], h[2] if h[1] == "LOOP" else "", h[0]),
        ):
            key = (cat, what if cat in ("FANOUT", "LAUNCH", "LAZY") else "")
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > a.max_hits or (ln, cat, what) in shown:
                continue
            shown.add((ln, cat, what))
            code = lines[ln - 1].strip()[:100] if ln else ""
            if not ln:  # an instruction, not a hit: never truncate it
                print("  %-8s L-     %s" % (cat, what))
                continue
            print("  %-8s L%-5s %-44s %s" % (cat, ln, what[:44], code))
        loops = [h for h in hits if h[1] == "LOOP" and h[0]]
        for ln, _, what in loops[: a.windows]:
            lo, hi = max(1, ln - a.context), min(len(lines), ln + a.context)
            print("  -- window L%d-%d (%s)" % (lo, hi, what.split(",")[0]))
            for i in range(lo, hi + 1):
                print("  %5d| %s" % (i, lines[i - 1][:110]))
        over = {k: v for k, v in seen.items() if v > a.max_hits}
        if over:
            print(
                "  (capped: %s)"
                % ", ".join("%s %s x%d" % (k[0], k[1], v) for k, v in over.items())
            )


if __name__ == "__main__":
    main()

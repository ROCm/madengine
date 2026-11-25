#!/usr/bin/env python3
"""Can this run script see a NaN, and can a capture be wired into its model?
One report instead of a dozen greps (mad-debug-numeric steps 1 and 4.1).

    madengine nan preflight <run script> [--source <model.py>] [--signal REGEX]
                         [--root DIR] [--venv DIR]

Checks, each one line plus where it looked:

  IMAGE    every docker image the run script names (IMAGE=${IMAGE:-tag}) is on this host --
           MISSING prints the pull command.
  SIGNAL   every alternative of the run script's NaN signal (from --signal, else
           the patterns its grep/zgrep -E calls use) is searched for in the model
           repo and its venv.  NOT RAISED ANYWHERE = the script can never report
           that NaN: it will score NaN runs as clean.  A regex alternative is
           searched as a regex (not found = unverifiable, not an alarm); a run
           script whose greps mention no NaN gets the source's own not-finite
           messages as --signal candidates instead.  `grep -o` field extractions
           (`grep -o 'nan_request=[^ ]*'`) are not tests and are skipped; a
           pattern that also matches the field's clean value (`nan_request=-`)
           is flagged MATCHES CLEAN OUTPUT.  A run script with arms (`./run.sh
           <arm>`: an ARMS="..." table, want="..." groups) is read per arm: a
           grep of an arm that does not run the --source is skipped, and the
           default run's candidates come first.  A verdict field the run script
           compares to a bad value (`wrong=yes`) is a wrong-value signal.
  MASK     nan_to_num / masked loss in the repo: a NaN that never reaches a check.
  HOOK     is a nan-doctor capture already wired (capture_from_env /
           OpNanCapture / cap.step) -- and where.
  ENV      how the run script's env reaches the trainer (set -a + source, docker
           -e / --env-file, env -i), and subprocess env dicts in the repo that do
           not start from os.environ (they would drop NAN_* settings).
  LAZY     torch.compile / compile: true in specs -> NAN_CAPTURE_ARM_AFTER_STEPS=1.
  VLLM     a vLLM engine in the source: is it in-process (the capture sees the
           model only then), tensor-parallel, a server; its compiled default.
  SGLANG   an sglang.Engine in the source: the model runs in its scheduler
           subprocess -- the capture goes there (nan_doctor_rt/sglang_capture).
  LOOP     `madengine nan capture-points`' top loop for --source (where cap.step() goes).

Static: reads files, imports nothing, runs no GPU work.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=300).stdout
    except Exception:  # noqa: BLE001
        return ""


def grep(
    pattern,
    paths,
    fixed=False,
    include=("*.py", "*.sh", "*.yaml", "*.yml", "*.env"),
    venv_ok=False,
):
    """[(file, line, text)] for `pattern` under `paths` (grep -rn, skipping run dirs)."""
    # -H: with a single file argument grep omits the file name, and every hit was dropped
    args = [
        "grep",
        "-rnIH",
        "-F" if fixed else "-E",
        "--exclude-dir=.git",
        "--exclude-dir=__pycache__",
        "--exclude-dir=qualification-runs*",
        "--exclude-dir=artifacts",
        # a framework's unit tests raise and compile on purpose
        "--exclude-dir=tests",
        "--exclude-dir=test",
        "--exclude-dir=unit_tests",
    ]
    if not venv_ok:
        args.append("--exclude-dir=.venv*")  # repo-scoped: libraries are noise here
    args += ["--include=%s" % i for i in include]
    out = []
    for line in sh(args + ["--", pattern] + [p for p in paths if p]).splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            out.append((parts[0], parts[1], parts[2].strip()))
    return out


def short(hits, n=3, root=""):
    return "; ".join(
        "%s:%s" % (os.path.relpath(f, root) if root else f, ln) for f, ln, _ in hits[:n]
    ) + (" (+%d)" % (len(hits) - n) if len(hits) > n else "")


# What a signal pattern mentions when it tests for a bad value: a NaN / Inf, or a wrong
# finite value (`RESULT ... wrong=yes` is a valid signal, step4-explode.md).
_SIGNAL_WORDS = r"nan|finite|\binf\b|wrong|mismatch|differ"
# a verdict field's bad value, when the run script compares the field to literals
_BAD_VALUES = ("yes", "true", "1", "nan", "wrong", "bad")


def _shell_mask(src):
    """`src` with quoted text and comments blanked (newlines kept): shell keywords (`if`,
    `fi`) found in it are code, not the words of an echo or a comment."""
    out, i, n, q = list(src), 0, len(src), None
    while i < n:
        c = src[i]
        if q:
            if c == "\\" and q == '"' and i + 1 < n:
                out[i] = " "
                if src[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
                continue
            if c == q:
                q = None
            elif c != "\n":
                out[i] = " "
        elif c in "'\"":
            q = c
        elif c == "\\":
            i += 1
        elif c == "#" and (i == 0 or src[i - 1] in " \t\n;"):
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


class Arms:
    """The run script's arms (`./run.sh <arm>`: an `ARMS="..."` table, `want="..."`
    groups), which arms each line of the script belongs to (`if [ "$name" = op_issue ];
    then ... else ... fi`, `$OUT_DIR/op_issue.log`), which arms run the model source, and
    which `./run.sh` runs bare.  Without it an issue-repro arm's `grep -q 'nan_count=0/'`
    was taken as the signal of the serve script, whose arms print `Logits are not
    finite`.  No arm table -> every line belongs to every run (names empty)."""

    def __init__(self, src, sources):
        self.src = src
        groups = {}
        for var, body in re.findall(r"""^\s*(\w+)=["']([^"'$]*)["']""", src, re.M):
            groups.setdefault(var, body)
        names = []
        m = re.search(r"""^\s*ARMS=["']([^"']*)["']""", src, re.M)
        if m:
            names += [l.split()[0] for l in m.group(1).splitlines() if l.split()]
        for body in re.findall(r"""\bwant=["']([^"'$]*)["']""", src):
            names += body.split()
        self.names = [
            x for x in dict.fromkeys(names) if re.match(r"[A-Za-z_][\w-]*$", x)
        ]
        self.default = None
        m = re.search(r"""case\s+"?\$\{1:-([\w-]+)\}"?\s+in""", src)
        if m and self.names:
            d = re.search(
                r"^\s*%s\)\s*want=(?:[\"']([^\"'$]*)[\"']|\$\{?(\w+)\}?)"
                % re.escape(m.group(1)),
                src,
                re.M,
            )
            if d:
                self.default = set(
                    (
                        d.group(1)
                        if d.group(1) is not None
                        else groups.get(d.group(2), "")
                    ).split()
                )
        self.lines = src.splitlines()
        self._ctx = self._contexts() if self.names else None
        self.source_arms = set(self.names)
        runs = []
        for s in sources:
            b = re.escape(os.path.basename(s))
            runs += [
                i
                for i, l in enumerate(self.lines)
                if re.search(r"""(python[\d.]*|torchrun)\b[^|;&]*?["'/\s]%s\b""" % b, l)
            ]
        if self.names and runs:
            self.source_arms = set().union(*(self.of_line(i) for i in runs))

    def _contexts(self):
        """Per line: [(arm names in an enclosing if/elif condition, negated: in its else)]."""
        masked = _shell_mask(self.src).splitlines()
        stack, ctx = [], []
        for i, ml in enumerate(masked):
            line_ctx = [tuple(x) for x in stack]
            for m in re.finditer(r"(?<![\w$-])(if|elif|else|fi)(?![\w-])", ml):
                kw = m.group(1)
                if kw == "if":
                    cond = re.split(r"\bthen\b", self.lines[i][m.end() :])[0]
                    stack.append([self._named(cond), False, set()])
                elif kw == "elif" and stack:
                    seen = stack[-1][2] | (stack[-1][0] if not stack[-1][1] else set())
                    cond = re.split(r"\bthen\b", self.lines[i][m.end() :])[0]
                    stack[-1] = [self._named(cond), False, seen]
                elif kw == "else" and stack:
                    seen = stack[-1][2] | (stack[-1][0] if not stack[-1][1] else set())
                    stack[-1] = [seen, True, seen]
                elif kw == "fi" and stack:
                    stack.pop()
                # the code after a keyword on the same line runs in the new context
                line_ctx = [tuple(x) for x in stack] if kw != "fi" else line_ctx
            ctx.append(line_ctx)
        return ctx

    def _named(self, text):
        return {
            a
            for a in self.names
            if re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(a), text)
        }

    def of_line(self, i):
        """The arms whose run executes line i (0-based)."""
        if not self.names:
            return set()
        cur = set(self.names)
        for arms, neg, _ in self._ctx[i]:
            if arms:
                cur = cur - arms if neg else cur & arms
        on_line = self._named(self.lines[i])  # `"$OUT_DIR/op_issue.log"`
        if on_line:
            cur &= on_line
        return cur

    def label(self, arms):
        if not self.names or arms == set(self.names):
            return "all arms"
        txt = "arm%s %s" % (
            "s" if len(arms) > 1 else "",
            ",".join(a for a in self.names if a in arms),
        )
        if self.default is not None:
            txt += (
                " (default run)" if arms & self.default else " (not in the default run)"
            )
        return txt


def signal_alternatives(run_script, given, sources=()):
    """The run script's signal candidates: [(alternative, is_regex, arms label, kind)],
    kind 'grep' (a literal / regex alternative of a grep) or 'field' (a verdict field of a
    result line, `res=$(grep '^RESULT' ...); wrong=$(grep -o 'wrong=[^ ]*' <<< "$res");
    [ "$wrong" = yes ]` -> `RESULT .*wrong=yes`), and [notes] on candidates dropped
    because they test another arm.

    `grep -E 'a|b'` is split on `|`; any other grep (`grep -q "..."`, `grep -F`, plain)
    is a single literal -- splitting it would invent alternatives that were never there.
    """
    if given:
        return [(given, True, "", "grep")], []
    src = Path(run_script).read_text()
    arms = Arms(src, sources)
    line_of = lambda pos: src.count("\n", 0, pos)  # noqa: E731
    pats, notes, fields = [], [], []
    for m in re.finditer(
        r"""(?<![\w-])z?grep\b((?:\s+-[a-zA-Z0-9]+)*)\s+['"]([^'"]+)['"]""", src
    ):
        flags, pat = m.group(1), m.group(2)
        # (?<![\w-]): `pgrep -af 'worker|torchrun'` (an idle-GPU check) is not a log test.
        # `grep -o 'nan_request=[^ ]*'` extracts a field, NaN or not: taken as the signal
        # it matches every clean RESULT line too.
        if "o" in re.sub(r"[^a-zA-Z]", "", flags):
            continue
        ln = line_of(m.start())
        on = arms.of_line(ln)
        if arms.names and not on & arms.source_arms:
            if re.search(_SIGNAL_WORDS, pat, re.I):
                notes.append(
                    "'%s' (run.sh:%d) tests %s, which does not run the --source -- skipped"
                    % (pat[:32], ln + 1, arms.label(on))
                )
            continue
        pats.append(
            (
                pat,
                "E" in flags and "F" not in flags,
                arms.label(on) if arms.names else "",
                on,
            )
        )
    # verdict fields of a result line: VAR=$(grep -o 'KEY=[^ ]*' <<< "$RES") compared to a literal
    for m in re.finditer(
        r"""(\w+)=\$\(\s*grep\s+-o\s+['"]\s*(\w+)=[^'"]*['"]\s*<<<\s*"?\$\{?(\w+)""",
        src,
    ):
        var, key, res = m.groups()
        if not re.search(_SIGNAL_WORDS, key, re.I):
            continue
        ln = line_of(m.start())
        on = arms.of_line(ln)
        if arms.names and not on & arms.source_arms:
            continue
        vals = re.findall(
            r"""["']?\$\{?%s(?::-[^}]*)?\}?["']?\s*!?==?\s*["']?(\w+)""" % var, src
        )
        bad = next((v for v in _BAD_VALUES if v in vals), None)
        if not bad:
            continue
        r = re.search(
            r"""\b%s=\$\(\s*grep\b[^'"\n]*['"]\^?(\w+)[^'"]*['"]""" % res, src
        )
        prefix = (r.group(1) + " .*") if r else ""
        fields.append(
            (
                "%s%s=%s" % (prefix, key, bad),
                True,
                arms.label(on) if arms.names else "",
                on,
            )
        )
    # A run script greps for more than its NaN line (exit codes, step numbers): keep the
    # patterns that mention a NaN or a wrong value, unless none does.
    pats = [x for x in pats if re.search(_SIGNAL_WORDS, x[0], re.I)] or (
        [] if fields else pats
    )
    # the default run's candidates first (`./run.sh` bare), then the other arms'
    rank = lambda x: (  # noqa: E731
        0 if arms.default is None or not arms.names or x[3] & arms.default else 1
    )
    alts = []
    for p, regex, lab, on in sorted(pats, key=rank):
        for a in p.split("|") if regex else [p]:
            a = a.strip("()^$ ") if regex else a
            if len(a) >= 6 and a not in [x[0] for x in alts]:
                alts.append((a, regex, lab, "grep"))
    for a, regex, lab, on in sorted(fields, key=rank):
        if a not in [x[0] for x in alts]:
            alts.append((a, regex, lab, "field"))
    return alts, notes


def raised_messages(scan):
    """[(file, line, message, code line)] for the source's own not-finite messages."""
    out = []
    for f, ln, text in grep(
        r"""["'][^"']*(not finite|non-finite|[Ii]s ?[Nn]a[Nn]|NaN detected)""",
        scan,
        include=("*.py",),
    ):
        m = re.search(
            r"""["']([^"']*(?:not finite|non-finite|[Ii]s ?[Nn]a[Nn]|NaN detected)[^"'{]*)""",
            text,
        )
        out.append((f, ln, (m.group(1) if m else text).strip(), text))
    return out


# Values a key=value field takes when nothing went wrong.
# Not "0": a step or request index can be 0 (`nan_step=[0-9]` is a valid signal).
_CLEAN_VALUES = ("-", "None", "none", "False", "false")


def matches_clean(alt):
    """A regex alternative that also matches its own field's clean value
    (`nan_request=[^ ]*` matches `nan_request=-`): the literal prefix up to the first
    regex metacharacter, followed by each clean value.  Only for key=value / key: value
    shaped patterns."""
    m = re.match(r"([^\\\[\](){}*+?.|^$]*[=:]\s*)", alt)
    if not m:
        return None
    for v in _CLEAN_VALUES:
        try:
            if re.search(alt, m.group(1) + v):
                return m.group(1) + v
        except re.error:
            return None
    return None


def _dedup(hits):
    seen, out = set(), []
    for h in hits:
        k = (os.path.realpath(h[0]), h[1])
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def vllm_check(files, rs, root):
    """VLLM: a vLLM engine in the model source, and what the capture needs from it.
    vLLM V1 runs the engine core (and with it the model) in a subprocess unless
    VLLM_ENABLE_V1_MULTIPROCESSING=0: a capture wired into the serving loop then sees no
    model op.  `files`: the model source(s) -- never the whole repo (a vendored torchtitan
    has RL actors that import vLLM without being on the run's path)."""
    serve = re.search(
        r"\bvllm\s+serve\b|vllm\.entrypoints", Path(rs).read_text(errors="replace")
    )
    uses = (
        _dedup(grep(r"^\s*(from vllm\b|import vllm\b)", files, include=("*.py",)))
        if files
        else []
    )
    if not uses and not serve:
        return
    if serve and not uses:
        print(
            "VLLM     the run script starts a vLLM server (`vllm serve` / api_server): the model "
            "runs in the\n           server's engine processes, not in a loop of yours -- not "
            "wireable as is; drive the same\n           model through offline vllm.LLM in one "
            "process (step1-ask.md *vLLM*)"
        )
        return
    both = files + [rs]
    inproc = [
        h
        for h in _dedup(
            grep(r"VLLM_ENABLE_V1_MULTIPROCESSING", both, include=("*.py", "*.sh"))
        )
        if re.search(r"""^\W*["']?0\b""", h[2].split("MULTIPROCESSING", 1)[1])
    ]
    if inproc:
        print(
            "VLLM     %s: in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0 at %s) -- the "
            "capture sees the model"
            % (short(uses, 1, root=root), short(inproc, 2, root=root))
        )
    else:
        print(
            "VLLM     %s: vLLM V1 runs the model in an engine-core SUBPROCESS by default -- a "
            "capture in\n           this process sees no model op: set "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 in the run's env\n           (madengine: "
            "docker_env_vars) (step1-ask.md *vLLM*)" % short(uses, 1, root=root)
        )
    tp = [
        h
        for h in _dedup(
            grep(
                r"tensor_parallel_size\s*=|pipeline_parallel_size\s*=|--tensor-parallel-size",
                both,
                include=("*.py", "*.sh"),
            )
        )
        if not re.search(r"=\s*1\b", h[2])
    ]
    if tp:
        print(
            "VLLM     parallel size set at %s: > 1 puts the model in one worker process per "
            "rank -- capture lines\n           get nan_doctor_rt/vllm_capture on PYTHONPATH "
            "(the capture is built inside every TP worker);\n           pipeline parallel > 1 "
            "or several nodes: not covered -- capture at size 1 if the NaN survives it,\n"
            "           else step 5N (CAPTURE_UNREACHABLE) (step1-ask.md *vLLM*)"
            % short(tp, 2, root=root)
        )
    forced = _dedup(grep(r"enforce_eager\s*=\s*True", both, include=("*.py", "*.sh")))
    switch = _dedup(grep(r"enforce.eager", both, include=("*.py", "*.sh")))
    if forced:
        print(
            "VLLM     enforce_eager=True at %s: eager ops, visible to the capture"
            % short(forced, 2, root=root)
        )
    else:
        print(
            "VLLM     default mode: torch.compile + HIP graphs -- a NaN born in compiled code "
            "reaches the capture\n           only PROPAGATED (step5n-compiled.md *vLLM*)%s"
            % (
                "; an enforce_eager switch at %s is C0's compile-off flag"
                % short(switch, 2, root=root)
                if switch
                else "; C0's compile-off flag: enforce_eager=True"
            )
        )


def sglang_check(files, rs, root):
    """SGLANG: an sglang.Engine in the model source.  The engine always runs the model in a
    spawned scheduler subprocess, so the capture is installed there by
    nan_doctor_rt/sglang_capture (a sitecustomize patching Scheduler.run_batch), not wired
    into the user's request loop.  `files`: the model source(s)."""
    server = re.search(
        r"sglang\.launch_server|sglang\s+serve\b", Path(rs).read_text(errors="replace")
    )
    uses = (
        _dedup(grep(r"^\s*(from sglang\b|import sglang\b)", files, include=("*.py",)))
        if files
        else []
    )
    if not uses and not server:
        return
    if server and not uses:
        print(
            "SGLANG   the run script starts an SGLang server (sglang.launch_server): drive the "
            "same model through\n           offline sglang.Engine in one script and debug that "
            "(step1-ask.md *SGLang*)"
        )
        return
    print(
        "SGLANG   %s: sglang.Engine runs the model in a scheduler SUBPROCESS -- don't wire the "
        "request loop;\n           capture lines get nan_doctor_rt/sglang_capture on "
        "PYTHONPATH (step1-ask.md *SGLang*)" % short(uses, 1, root=root)
    )
    both = files + [rs]
    tp = [
        h
        for h in _dedup(
            grep(
                r"tp_size\s*=|--tp(-size)?\b|dp_size\s*=|--dp(-size)?\b",
                both,
                include=("*.py", "*.sh"),
            )
        )
        if not re.search(r"=\s*1\b|\s1\b", h[2])
    ]
    if tp:
        print(
            "SGLANG   parallel size set at %s: > 1 is not supported by the capture -- capture "
            "at size 1 if the\n           NaN survives it" % short(tp, 2, root=root)
        )


def image_check(rs):
    """IMAGE: the docker image(s) the run script names, checked before any job is spent."""
    text = Path(rs).read_text(errors="replace")
    tags = sorted(set(re.findall(r"IMAGE=\$\{IMAGE:-([^}]+)\}", text)))
    if not tags:
        print(
            "IMAGE    ?  no IMAGE=${IMAGE:-...} in the run script -- find the image by hand "
            "(for a madengine run: the docker_image in build_manifest.json)"
        )
        return
    have_docker = shutil.which("docker") is not None
    for t in tags:
        if not have_docker:
            print(
                "IMAGE    %s  docker tag (docker is not on PATH here: check it on the GPU host)"
                % t
            )
        elif (
            subprocess.run(
                ["docker", "image", "inspect", t], capture_output=True
            ).returncode
            == 0
        ):
            print("IMAGE    %s  OK" % t)
        else:
            print(
                "IMAGE    %s  MISSING -- build or pull it before step 4:  docker pull %s"
                % (t, t)
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_script")
    ap.add_argument(
        "--source", action="append", default=[], help="model file(s) to wire"
    )
    ap.add_argument(
        "--signal", default=None, help="NaN signal regex (default: from the run script)"
    )
    ap.add_argument(
        "--root", default=None, help="model repo (default: run script's git root)"
    )
    ap.add_argument(
        "--venv", default=None, help="model venv (default: .venv* under root)"
    )
    a = ap.parse_args()

    rs = os.path.realpath(a.run_script)  # realpath: root (git) is resolved too
    root = (
        a.root
        or sh(
            ["git", "-C", os.path.dirname(rs), "rev-parse", "--show-toplevel"]
        ).strip()
        or os.path.dirname(rs)
    )
    venv = a.venv
    if venv is None:
        cands = (
            sorted(d for d in os.listdir(root) if d.startswith(".venv"))
            if os.path.isdir(root)
            else []
        )
        venv = os.path.join(root, cands[0]) if cands else None
    site = None
    if venv:
        libs = sh(
            ["bash", "-c", "ls -d %s/lib/python*/site-packages 2>/dev/null" % venv]
        ).split()
        site = libs[0] if libs else None
    scan = [root]
    print("preflight: run=%s root=%s venv=%s" % (rs, root, venv or "(none)"))

    image_check(rs)

    # the model sources: --source, else the .py files the run script names
    named = [
        os.path.join(os.path.dirname(rs), f)
        for f in re.findall(r"([\w./-]+\.py)\b", Path(rs).read_text(errors="replace"))
    ]
    srcs = [os.path.realpath(f) for f in a.source] or [
        f for f in named if os.path.isfile(f)
    ]

    # SIGNAL: is each alternative raised anywhere?
    alts, notes = signal_alternatives(rs, a.signal, srcs)
    for n in notes:
        print("SIGNAL   (%s)" % n)
    if not alts:
        print("SIGNAL   ?  no quoted grep pattern in the run script; pass --signal")
    elif not a.signal and not any(re.search(_SIGNAL_WORDS, x[0], re.I) for x in alts):
        # e.g. a run script whose only grep builds a per-step table: the NaN is the
        # framework's own raise, and searching that table regex would report a false alarm.
        print(
            "SIGNAL   ?  the run script greps no NaN line (only %s): the NaN surfaces in the "
            "model/framework itself -- pass its error line as --signal"
            % ", ".join("'%s'" % x[0][:32] for x in alts)
        )
        alts = []
    real_srcs = {os.path.realpath(f) for f in srcs}
    # the --source's own lines first (else the folder's: a sibling op repro's `isnan` is
    # noise), and a print/raise before a docstring sentence that only mentions non-finite
    raises = raised_messages(scan)
    raises = [h for h in raises if os.path.realpath(h[0]) in real_srcs] or raises
    raises.sort(
        key=lambda h: (
            os.path.realpath(h[0]) not in real_srcs,
            not re.search(r"\b(print|raise|log\w*|warn\w*)\b\W", h[3]),
            # "X is/are not finite" states the failure; a PROBE line only reports it
            not re.search(
                r"\b(is|are) (not finite|non-finite|nan)\b|NaN detected", h[2], re.I
            ),
        )
    )
    msgs = []
    for f, ln, text, _ in raises:
        if text not in [m[0] for m in msgs]:
            msgs.append((text, "%s:%s" % (os.path.relpath(f, root), ln)))
    msgs_txt = "; ".join("'%s' (%s)" % m for m in msgs[:3]) + (
        " (+%d)" % (len(msgs) - 3) if len(msgs) > 3 else ""
    )
    if not alts:
        if raises:
            print(
                "SIGNAL   the source raises/logs: %s -- candidates for <NAN_SIGNAL>"
                % msgs_txt
            )
        # a finite signal: the model's own wrong-value / parity / determinism line, no NaN
        wrong = grep(
            r"""["'][^"']*(wrong=|RESULT |mismatch|differ|parity)""",
            scan,
            include=("*.py",),
        )
        if wrong:
            print(
                "SIGNAL   the source prints: %s -- finite-signal candidates (wrong values, "
                "no NaN: step4-explode.md)" % short(wrong, root=root)
            )
    for alt, _, lab, kind in alts:
        tag = "  [%s]" % lab if lab else ""
        clean = matches_clean(alt)
        if clean:
            print(
                "SIGNAL   %-34s MATCHES CLEAN OUTPUT ('%s') -- not a NaN signal; use the line "
                "printed on the NaN path%s" % ("'%s'" % alt[:32], clean, tag)
            )
            continue
        # a pattern with regex syntax (grep -o 'step: *[0-9]*', grep -E) is searched as one;
        # read literally it can never match the source
        rx = bool(re.search(r"[\\\[\]*+?{}]", alt))

        def look(pat):
            if (
                rx
            ):  # ^...$ anchor a log line; the source prints it from inside a print(f"...")
                pat = pat.lstrip("^").rstrip("$") or pat
            hits = [
                h
                for h in grep(pat, scan, fixed=not rx)
                if os.path.abspath(h[0]) != rs and not h[0].endswith("~")
            ]
            # the --source, then the model's raise before sibling scripts
            hits.sort(
                key=lambda h: (
                    os.path.realpath(h[0]) not in real_srcs,
                    h[0].endswith(".sh"),
                )
            )
            return hits or (
                grep(pat, [site], fixed=not rx, include=("*.py",), venv_ok=True)
                if site
                else []
            )

        where, verdict = look(alt), "found"
        # `RESULT .*wrong=yes`: the source prints `wrong={wrong}` -- the field is static,
        # its value is not
        field = (
            re.match(r"(.*\w=)\w+$", alt)
            if rx and (kind == "field" or not where)
            else None
        )
        if field and look(field.group(1)):
            where = look(field.group(1))
            verdict = "found as a field (its value is set at runtime)"
        if kind == "field":
            verdict += " -- the run script's verdict field: a wrong-value signal (step4-explode.md)"
        if not where:
            verdict = (
                "? not in the source as a regex -- a pattern over formatted output cannot be "
                "checked statically: confirm it on one model log"
                if rx
                else "NOT RAISED ANYWHERE -- the run script cannot see this NaN"
            )
        print(
            "SIGNAL   %-34s %s%s%s"
            % (
                "'%s'" % alt[:32],
                verdict,
                (": " + short(where, root=root)) if where else "",
                tag,
            )
        )
    if alts and all(k == "field" for *_, k in alts) and msgs:
        # the run script only reads a RESULT verdict, the source also prints the NaN
        # itself, mid-run -- the line a capture can stop on
        print(
            "SIGNAL   the source also raises/logs: %s -- a NaN-specific line: prefer it as "
            "<NAN_SIGNAL>;\n           the verdict field above is the wrong-value fallback"
            % msgs_txt
        )

    # MASK: values that would hide a NaN before any check
    mask = grep(
        r"nan_to_num|masked_fill\(.*isnan|where\(.*isfinite", scan, include=("*.py",)
    )
    loss = [h for h in mask if re.search(r"loss|total", h[2])]
    print(
        "MASK     %d nan_to_num/mask site(s)%s"
        % (len(mask), (", on a loss: " + short(loss, root=root)) if loss else "")
    )

    # HOOK: already wired?
    hook = grep(
        r"capture_from_env|OpNanCapture|op_nan_capture|cap\.step\(\)",
        scan,
        include=("*.py",),
    )
    titan = re.search(r"torchtitan", Path(rs).read_text(errors="replace"))
    print(
        "HOOK     %s"
        % (
            "wired: " + short(hook, root=root)
            if hook
            else (
                "not wired in the source -- a torchtitan run is wired by "
                "nan_doctor_rt.torchtitan_capture instead (step4-distributed.md): "
                "nothing to edit"
                if titan
                else "not wired (madengine nan wire apply <source>)"
            )
        )
    )

    # ENV: does the run script's env reach the trainer?
    src = Path(rs).read_text()
    called = re.findall(r"(?:bash|sh)\s+(\S+\.sh)", src)
    texts = [src] + [
        Path(os.path.join(root, c)).read_text()
        for c in called
        if os.path.isfile(os.path.join(root, c))
    ]
    mech = []
    for t in texts:
        if re.search(r"set -a", t) and re.search(r"^\s*(source|\.)\s", t, re.M):
            mech.append("set -a + source")
        if re.search(r"docker run[^\n]*(-e |--env-file)", t):
            mech.append("docker -e/--env-file")
        if re.search(r"\benv -i\b", t):
            mech.append("env -i (CLEARS the environment)")
    sanit = [h for h in grep(r"env\s*=\s*\{", scan, include=("*.py",))]
    sanit = [h for h in sanit if "os.environ" not in h[2]]
    print(
        "ENV      %s%s"
        % (
            ", ".join(sorted(set(mech))) or "no mechanism seen -- check by hand",
            (
                ("; subprocess env dict(s) to read: " + short(sanit, root=root))
                if sanit
                else ""
            ),
        )
    )

    # LAZY
    lazy = grep(
        r"torch\.compile\(|^\s*compile:\s*true",
        scan,
        include=("*.py", "*.yaml", "*.yml"),
    )
    print(
        "LAZY     %s"
        % (
            (
                "torch.compile/compile:true at "
                + short(lazy, root=root)
                + " -> NAN_CAPTURE_ARM_AFTER_STEPS=1"
            )
            if lazy
            else "none seen"
        )
    )

    vllm_check(srcs, rs, root)
    sglang_check(srcs, rs, root)

    # LOOP
    for s in a.source:
        out = sh(
            [
                sys.executable,
                os.path.join(HERE, "capture_points.py"),
                s,
                "--windows",
                "0",
            ]
        )
        top = next((x for x in out.splitlines() if x.strip().startswith("LOOP")), None)
        print(
            "LOOP     %s: %s"
            % (
                s,
                (
                    " ".join(top.split()[1:])
                    if top
                    else "no LOOP candidate -- read the entry point"
                ),
            )
        )


if __name__ == "__main__":
    main()

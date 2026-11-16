#!/usr/bin/env python3
"""Can this run script see a NaN, and can a capture be wired into its model?
One report instead of a dozen greps (mad-debug-nan steps 1 and 4.1).

    capture_preflight.py <run script> [--source <model.py>] [--signal REGEX]
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
           messages as --signal candidates instead.
  MASK     nan_to_num / masked loss in the repo: a NaN that never reaches a check.
  HOOK     is a nan-doctor capture already wired (capture_from_env /
           OpNanCapture / cap.step) -- and where.
  ENV      how the run script's env reaches the trainer (set -a + source, docker
           -e / --env-file, env -i), and subprocess env dicts in the repo that do
           not start from os.environ (they would drop NAN_* settings).
  LAZY     torch.compile / compile: true in specs -> NAN_CAPTURE_ARM_AFTER_STEPS=1.
  LOOP     find_capture_points.py's top loop for --source (where cap.step() goes).

Static: reads files, imports nothing, runs no GPU work.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

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
    args = [
        "grep",
        "-rnI",
        "-F" if fixed else "-E",
        "--exclude-dir=.git",
        "--exclude-dir=__pycache__",
        "--exclude-dir=qualification-runs*",
        "--exclude-dir=artifacts",
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


def signal_alternatives(run_script, given):
    """The literal strings the run script's NaN signal can match.

    `grep -E 'a|b'` is split on `|`; any other grep (`grep -q "..."`, `grep -F`, plain)
    is a single literal -- splitting it would invent alternatives that were never there.
    """
    if given:
        pats = [(given, True)]
    else:
        src = open(run_script).read()
        pats = [
            (pat, "E" in flags and "F" not in flags)
            for flags, pat in re.findall(
                r"""(?<![\w-])z?grep\b((?:\s+-[a-zA-Z]+)*)\s+['"]([^'"]+)['"]""", src
            )
        ]
        # (?<![\w-]): `pgrep -af 'worker|torchrun'` (an idle-GPU check) is not a log test
    # A run script greps for more than its NaN line (exit codes, step numbers): keep the
    # patterns that mention a NaN, unless none does.
    pats = [x for x in pats if re.search(r"nan|finite", x[0], re.I)] or pats
    alts = []
    for p, regex in pats:
        for a in (p.split("|") if regex else [p]):
            a = a.strip("()^$ ") if regex else a
            if len(a) >= 6 and a not in alts:
                alts.append(a)
    return alts


def image_check(rs):
    """IMAGE: the docker image(s) the run script names, checked before any job is spent."""
    text = open(rs, errors="replace").read()
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

    # SIGNAL: is each alternative raised anywhere?
    alts = signal_alternatives(rs, a.signal)
    if not alts:
        print("SIGNAL   ?  no quoted grep pattern in the run script; pass --signal")
    elif not a.signal and not any(re.search(r"nan|finite", x, re.I) for x in alts):
        # e.g. a run script whose only grep builds a per-step table: the NaN is the
        # framework's own raise, and searching that table regex would report a false alarm.
        print(
            "SIGNAL   ?  the run script greps no NaN line (only %s): the NaN surfaces in the "
            "model/framework itself -- pass its error line as --signal"
            % ", ".join("'%s'" % x[:32] for x in alts)
        )
        alts = []
    if not alts:
        raises = grep(
            r"""["'][^"']*(not finite|non-finite|[Ii]s ?[Nn]a[Nn]|NaN detected)""",
            scan,
            include=("*.py",),
        )
        if raises:
            print(
                "SIGNAL   the source raises/logs: %s -- candidates for <NAN_SIGNAL>"
                % short(raises, root=root)
            )
    for alt in alts:
        # a pattern with regex syntax (grep -o 'step: *[0-9]*', grep -E) is searched as one;
        # read literally it can never match the source
        rx = bool(re.search(r"[\\\[\]*+?{}]", alt))
        in_repo = [
            h
            for h in grep(alt, scan, fixed=not rx)
            if os.path.abspath(h[0]) != rs and not h[0].endswith("~")
        ]
        in_venv = (
            grep(alt, [site], fixed=not rx, include=("*.py",), venv_ok=True)
            if site
            else []
        )
        in_repo.sort(
            key=lambda h: h[0].endswith(".sh")
        )  # the model's raise before sibling scripts
        where = in_repo or in_venv
        verdict = (
            "found"
            if where
            else (
                "? not in the source as a regex -- a pattern over formatted output cannot be "
                "checked statically: confirm it on one model log"
                if rx
                else "NOT RAISED ANYWHERE -- the run script cannot see this NaN"
            )
        )
        print(
            "SIGNAL   %-34s %s%s"
            % (
                "'%s'" % alt[:32],
                verdict,
                (": " + short(where, root=root)) if where else "",
            )
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
    print(
        "HOOK     %s"
        % (
            "wired: " + short(hook, root=root)
            if hook
            else "not wired (wire_capture.py apply <source>)"
        )
    )

    # ENV: does the run script's env reach the trainer?
    src = open(rs).read()
    called = re.findall(r"(?:bash|sh)\s+(\S+\.sh)", src)
    texts = [src] + [
        open(os.path.join(root, c)).read()
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

#!/usr/bin/env python3
"""Wire the nan-doctor capture into a model source, and take it out again
(mad-debug-nan step 4.1 and step 9's "restore the source").

    madengine nan wire apply   <model.py> --backup-dir <ARTIFACT_DIR> [--loop-line N]
    madengine nan wire restore <model.py> --backup-dir <ARTIFACT_DIR>
    madengine nan wire status  <model.py> --backup-dir <ARTIFACT_DIR>

apply inserts the three-line integration of nan_doctor_rt.op_nan_capture.capture_from_env
(`madengine nan exec` mounts the package at /opt/nan_doctor; elsewhere set NAN_DOCTOR_RT_PARENT):

    import sys / sys.path.insert(... NAN_DOCTOR_RT_PARENT ...) / from nan_doctor_rt.op_nan_capture import capture_from_env
                                          after the last top-level import
    cap = capture_from_env()              just before the loop at --loop-line
    cap.step()                            first statement of the loop body
    cap.close()                           just after the loop

--loop-line defaults to find_capture_points.py's top LOOP.  It first copies the
file to <backup-dir>/<name>.orig with its sha256, refuses a file that already
has a capture, and checks the result compiles.  restore copies the backup back
and verifies the bytes match the recorded sha256.  Arming, filters and the
capsule directory stay in the environment (NAN_CAPTURE_*), so the wired source
also serves as the control arm with NAN_CAPTURE_DIR unset.
"""

import argparse
import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "# nan-doctor capture (madengine nan wire)"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def paths(src, backup_dir):
    base = os.path.join(backup_dir, os.path.basename(src) + ".orig")
    return base, base + ".sha256"


def top_loop(src):
    out = subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "capture_points.py"),
            src,
            "--windows",
            "0",
        ],
        capture_output=True,
        text=True,
    ).stdout
    m = re.search(r"LOOP\s+L(\d+)\s+#1", out)
    if not m:
        sys.exit("no LOOP candidate in %s: pass --loop-line" % src)
    return int(m.group(1))


def apply(src, backup_dir, loop_line):
    text = open(src).read()
    if "capture_from_env" in text or "OpNanCapture" in text:
        sys.exit("%s already has a capture; not wiring twice" % src)
    orig, shaf = paths(src, backup_dir)
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(src, orig)
    open(shaf, "w").write(sha(src) + "\n")

    tree = ast.parse(text)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imports:
        sys.exit("no top-level import in %s" % src)
    imp_end = max(n.end_lineno for n in imports)
    loop_line = loop_line or top_loop(src)
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.For, ast.While)) and n.lineno == loop_line
    ]
    if not loops:
        sys.exit("no for/while loop starts at line %d of %s" % (loop_line, src))
    loop = loops[0]
    lines = text.splitlines(keepends=True)
    ind = " " * loop.col_offset
    body_ind = " " * loop.body[0].col_offset
    # Insert bottom-up so earlier line numbers stay valid.
    lines.insert(loop.end_lineno, "%scap.close()  %s\n" % (ind, MARK))
    lines.insert(loop.body[0].lineno - 1, "%scap.step()  %s\n" % (body_ind, MARK))
    lines.insert(loop.lineno - 1, "%scap = capture_from_env()  %s\n" % (ind, MARK))
    lines.insert(
        imp_end,
        "import os as _nd_os, sys as _nd_sys  %s\n"
        '_nd_sys.path.insert(0, _nd_os.environ.get("NAN_DOCTOR_RT_PARENT", "/opt/nan_doctor"))\n'
        "from nan_doctor_rt.op_nan_capture import capture_from_env\n" % MARK,
    )
    open(src, "w").write("".join(lines))
    try:
        compile(open(src).read(), src, "exec")
    except SyntaxError as e:
        shutil.copy2(orig, src)
        sys.exit("wired source does not compile (%s); original restored" % e)
    print(
        "WIRED %s: import after L%d, loop L%d-%d; backup %s (sha256 %s)"
        % (
            src,
            imp_end,
            loop.lineno,
            loop.end_lineno,
            orig,
            open(shaf).read().strip()[:16],
        )
    )


def restore(src, backup_dir):
    orig, shaf = paths(src, backup_dir)
    if not os.path.exists(orig):
        sys.exit("no backup %s" % orig)
    want = open(shaf).read().strip()
    if sha(orig) != want:
        sys.exit("backup %s does not match its recorded sha256; not restoring" % orig)
    shutil.copy2(orig, src)
    print("RESTORED %s from %s (sha256 %s matches)" % (src, orig, want[:16]))


def status(src, backup_dir):
    orig, shaf = paths(src, backup_dir)
    wired = MARK in open(src).read()
    same = os.path.exists(shaf) and sha(src) == open(shaf).read().strip()
    print(
        "%s: %s; backup %s%s"
        % (
            src,
            "WIRED" if wired else "not wired",
            orig if os.path.exists(orig) else "(none)",
            "; identical to the original" if same else "",
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["apply", "restore", "status"])
    ap.add_argument("source")
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--loop-line", type=int, default=None)
    a = ap.parse_args()
    {
        "apply": lambda: apply(a.source, a.backup_dir, a.loop_line),
        "restore": lambda: restore(a.source, a.backup_dir),
        "status": lambda: status(a.source, a.backup_dir),
    }[a.action]()


if __name__ == "__main__":
    main()

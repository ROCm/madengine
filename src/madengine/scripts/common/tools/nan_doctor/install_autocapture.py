#!/usr/bin/env python3
"""Install (or remove) the nan-doctor capture runtime into a site-packages dir.

    python3 install_autocapture.py                 # first site-packages of this python3
    python3 install_autocapture.py --target DIR    # any directory on sys.path
    python3 install_autocapture.py --uninstall [--target DIR]

Copies op_nan_capture.py and nan_doctor_autocapture.py next to this script into
the target, and writes nan_doctor_autocapture.pth there.  Nothing happens in a
process until NAN_CAPTURE_DIR is set and NAN_CAPTURE_AUTO=optimizer (see
nan_doctor_autocapture.py).  Standard library only: it runs before torch is
known to work.  Prints the target directory on success, which is what an
uninstall step needs.

This is the entry point the madengine `nan_doctor` tool calls from its pre_script
(ADR 0002); the files it copies are the whole capture runtime.
"""
import argparse
import os
import shutil
import site
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ("op_nan_capture.py", "nan_doctor_autocapture.py")
PTH = "nan_doctor_autocapture.pth"
PTH_LINE = "import nan_doctor_autocapture\n"


def default_target():
    for d in site.getsitepackages():
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    user = site.getusersitepackages()
    os.makedirs(user, exist_ok=True)
    return user


def install(target):
    os.makedirs(target, exist_ok=True)
    for m in MODULES:
        shutil.copyfile(os.path.join(HERE, m), os.path.join(target, m))
    with open(os.path.join(target, PTH), "w") as f:
        f.write(PTH_LINE)


def uninstall(target):
    for name in MODULES + (PTH,):
        p = os.path.join(target, name)
        if os.path.exists(p):
            os.remove(p)
    cache = os.path.join(target, "__pycache__")
    if os.path.isdir(cache):
        for f in os.listdir(cache):
            if f.split(".")[0] in ("op_nan_capture", "nan_doctor_autocapture"):
                os.remove(os.path.join(cache, f))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", help="site-packages (or any sys.path) directory")
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args(argv)
    target = os.path.abspath(a.target) if a.target else default_target()
    (uninstall if a.uninstall else install)(target)
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())

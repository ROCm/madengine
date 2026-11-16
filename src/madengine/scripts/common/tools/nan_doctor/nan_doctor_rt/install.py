#!/usr/bin/env python3
"""Install (or remove) the nan_doctor_rt runtime into a site-packages directory.

    python3 install.py                 # the first writable site-packages of this python3
    python3 install.py --target DIR    # any directory on sys.path
    python3 install.py --uninstall [--target DIR]

Copies this package to <target>/nan_doctor_rt and writes <target>/nan_doctor_rt.pth,
whose single line imports nan_doctor_rt.autocapture.  Nothing happens in a process
until NAN_CAPTURE_DIR is set and NAN_CAPTURE_AUTO=optimizer (see autocapture.py), and
importing the package imports no torch.  Standard library only: it runs before torch
is known to work.  Prints the target directory, which is what an uninstall needs.

This is the entry point of madengine's `nan_doctor` tool (its pre-script).
"""
import argparse
import os
import shutil
import site
import sys

PKG = "nan_doctor_rt"
HERE = os.path.dirname(os.path.abspath(__file__))
PTH = PKG + ".pth"
PTH_LINE = "import %s.autocapture\n" % PKG


def default_target():
    for d in site.getsitepackages():
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    user = site.getusersitepackages()
    os.makedirs(user, exist_ok=True)
    return user


def install(target):
    os.makedirs(target, exist_ok=True)
    dest = os.path.join(target, PKG)
    if os.path.abspath(dest) != HERE:
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(HERE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    with open(os.path.join(target, PTH), "w") as f:
        f.write(PTH_LINE)


def uninstall(target):
    dest = os.path.join(target, PKG)
    if os.path.abspath(dest) != HERE and os.path.isdir(dest):
        shutil.rmtree(dest)
    pth = os.path.join(target, PTH)
    if os.path.exists(pth):
        os.remove(pth)


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

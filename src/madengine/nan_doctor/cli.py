"""Dispatch `nan` subcommands to the tool modules (standard library only)."""

import importlib
import sys

# name -> (module, calling convention, one-line help)
#   argv:  module.main(argv)            (argv excludes the program name)
#   prog:  module.main([prog] + argv)
#   sys:   sys.argv is set, module.main() parses it
COMMANDS = {
    "exec": (
        "runner",
        "argv",
        "run a command in the model's image on the GPUs (the <GPU_EXEC>)",
    ),
    "group": ("group", "argv", "5a: group capsules and order the groups for replay"),
    "replay": (
        "replay",
        "argv",
        "5b: replay a capsule under load on every GPU; the concurrency ladder",
    ),
    "summary": ("summary", "argv", "summarise one replay attempt directory"),
    "package": ("package", "argv", "6: build a standalone reproducer directory"),
    "readme": ("repro_readme", "argv", "6: write the reproducer README, pack, re-pack"),
    "report": (
        "report",
        "sys",
        "9: write the investigation report from the artifact directory",
    ),
    "intake": ("intake", "argv", "validate an intake file; print the step-1 answers"),
    "ledger": (
        "ledger",
        "sys",
        "append-only investigation ledger: start|note|job|wait|end|table",
    ),
    "verdict": ("verdict", "sys", "replay verdict rules: attempt <dir> | kernel <log>"),
    "waves": ("waves", "sys", "summarise a wave of model trials"),
    "handoff": (
        "handoff",
        "sys",
        "5N: what the capsule path tried, for the standalone fallback",
    ),
    "framework-check": (
        "framework_check",
        "sys",
        "0: is the workload a PyTorch program?",
    ),
    "gpu-check": ("gpu_check", "sys", "0/1: which GPU is this, and is it tested?"),
    "preflight": (
        "preflight",
        "sys",
        "1: check the run script, image and NaN signal before any job",
    ),
    "capture-points": (
        "capture_points",
        "sys",
        "4: where would capture hook into this model source?",
    ),
    "wire": (
        "wire",
        "sys",
        "4: apply/restore the three-line capture wiring in a source tree",
    ),
    "digest": ("digest", "sys", "digest capture logs"),
    "power": (
        "power",
        "argv",
        "7/8: runs an experiment needs; is a clean result powered?",
    ),
    "kernel-match": (
        "kernel_match",
        "prog",
        "match a replay's kernel id against the model's kernel",
    ),
    "arms": (
        "configs",
        "argv",
        "2 (no signal): differential-mode arms: detect | arms | list",
    ),
    "diff": (
        "output_diff",
        "argv",
        "2 (no signal): compare the arms' outputs -> ND_DIFF verdict / switch",
    ),
}


def usage(out=sys.stderr):
    w = out.write
    w("usage: madengine nan <command> [options]\n\ncommands:\n")
    for name, (_, _, h) in COMMANDS.items():
        w("  %-16s %s\n" % (name, h))
    w("\nRun `madengine nan <command> --help` for a command's options.\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage(sys.stdout if argv else sys.stderr)
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        sys.stderr.write("unknown command %r\n" % name)
        usage()
        return 2
    modname, conv, _ = COMMANDS[name]
    mod = importlib.import_module("." + modname, __package__)
    prog = "madengine nan " + name
    old = sys.argv
    try:
        if conv == "argv":
            rc = mod.main(rest)
        elif conv == "prog":
            rc = mod.main([prog] + rest)
        else:
            sys.argv = [prog] + rest
            rc = mod.main()
    except SystemExit as e:
        rc = e.code
    finally:
        sys.argv = old
    if rc is None:
        return 0
    return rc if isinstance(rc, int) else 1

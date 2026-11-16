"""5a: group NaN capsules, then order the groups for replay (debug-nan).

    madengine nan group <capsule_dir> --image <image> --out-dir <dir> [--recursive] [--job]
    madengine nan group <capsule_dir> --out-dir <dir> --from-json groups.json    # no container

Runs `python3 -m nan_doctor_rt.compare_nan_capsules` in the model's image (it needs torch) and
writes, under --out-dir:
  groups.json            the grouping, as compare_nan_capsules writes it
  groups.txt             the grouping as printed
  groups_priority.txt    one line per group in replay order:  G<n> <count> <capsule> <op>[ TAG]
TAG is FALSE_POSITIVE? (an allocator/view op, or only uninitialised memory: not replayed),
ORIGIN (the world-size>1 job's origin rank), or PROPAGATED? (its inputs were already
non-finite at capture: the bug is upstream).

Priority = capsule count; with a job manifest, the group holding the origin rank's capsule
goes first, represented by that rank's own capsule.
"""

import argparse
import ast
import glob
import json
import os
import shlex
import sys

FALSE_POSITIVE_OPS = (
    "aten.empty",
    "aten.new_empty",
    "aten.detach",
    "aten.alias",
    "aten.view",
    "aten.as_strided",
)


def _members(g):
    for s in g.get("operand_sets", []):
        infos = s.get("member_info") or [{}] * len(s.get("members", []))
        for p, mi in zip(s.get("members", []), infos):
            yield p, mi


def priority(groups_json, capsule_dir):
    """Lines of groups_priority.txt, in replay order."""
    groups = json.load(open(groups_json))["groups"]
    origin = set()
    for m in glob.glob(
        os.path.join(capsule_dir, "**", "manifest*.json"), recursive=True
    ):
        try:
            origin |= set(
                json.load(open(m)).get("verdict", {}).get("origin_ranks") or []
            )
        except (OSError, ValueError):
            pass

    def origin_member(g):
        return next(
            (
                p
                for p, mi in _members(g)
                if mi.get("rank") in origin
                and not mi.get("arrived")
                and mi.get("nan_origin") != "uninitialized"
            ),
            None,
        )

    lines = []
    for i, g in enumerate(
        sorted(groups, key=lambda g: (origin_member(g) is None, -g["count"])), 1
    ):
        ident = g["identity"]
        if isinstance(ident, str):
            try:
                ident = ast.literal_eval(ident)
            except (ValueError, SyntaxError):
                ident = {"op": ident}
        op = ident.get("op", "?")
        fin = [
            o.get("inputs_finite")
            for s in g.get("operand_sets", [])
            for o in s.get("outcomes", [])
        ]
        mi = [x for _, x in _members(g)]
        om = origin_member(g)
        if op.startswith(FALSE_POSITIVE_OPS) or (
            mi and all(x.get("nan_origin") == "uninitialized" for x in mi)
        ):
            tag = " FALSE_POSITIVE?"
        elif om:
            tag = " ORIGIN"
        elif (fin and all(f is False for f in fin)) or (
            mi and all(x.get("arrived") for x in mi)
        ):
            tag = " PROPAGATED?"
        else:
            tag = ""
        lines.append(
            "G%d %d %s %s%s" % (i, g["count"], om or g["representative"], op, tag)
        )
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="madengine nan group",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("capsule_dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image")
    ap.add_argument("--manifest")
    ap.add_argument("--model")
    ap.add_argument(
        "--recursive", action="store_true", help="capsules in trial_<n>/ subdirectories"
    )
    ap.add_argument(
        "--job", action="store_true", help="a multi-rank job: group with its manifest"
    )
    ap.add_argument(
        "--from-json", help="skip the container: order an existing groups.json"
    )
    ap.add_argument("--runtime-dir")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    gj = os.path.join(out, "groups.json")
    if a.from_json:
        gj = a.from_json
    else:
        from . import runner

        if not (a.image or a.manifest):
            ap.error("--image or --manifest is required (or --from-json)")
        try:
            image = a.image or runner.image_from_manifest(a.manifest, a.model)
            runner.ensure_image(image)
        except (ValueError, RuntimeError) as e:
            print("nan group: %s" % e, file=sys.stderr)
            return 2
        cap = os.path.abspath(a.capsule_dir)
        cmd = "python3 -m nan_doctor_rt.compare_nan_capsules %s --json %s%s%s" % (
            shlex.quote(cap),
            shlex.quote(gj),
            " --recursive" if a.recursive else "",
            " --job" if a.job else "",
        )
        mounts = []
        cwd = os.path.realpath(os.getcwd())
        for d in (cap, out):
            if not (
                os.path.realpath(d) == cwd or os.path.realpath(d).startswith(cwd + "/")
            ):
                mounts.append((d, d, "rw"))
        os.chmod(out, 0o777)
        rc = runner.run(
            image,
            cmd,
            log=os.path.join(out, "groups.txt"),
            mounts=mounts,
            runtime_dir=a.runtime_dir,
        )
        if rc != 0 and not os.path.exists(gj):
            print(
                "GROUPING_FAILED -- see %s" % os.path.join(out, "groups.txt"),
                file=sys.stderr,
            )
            return 1
    if not os.path.isfile(gj) or os.path.getsize(gj) == 0:
        print("GROUPING_FAILED -- no %s" % gj, file=sys.stderr)
        return 1
    lines = priority(gj, os.path.abspath(a.capsule_dir))
    with open(os.path.join(out, "groups_priority.txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print("\n".join(lines))
    print(
        "GROUPS=%d (%s)"
        % (
            len(lines),
            "; ".join(" ".join(ln.split()[:2] + ln.split()[3:4]) for ln in lines),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

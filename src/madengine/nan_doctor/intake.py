#!/usr/bin/env python3
"""Validate an intake file and print the step-1 values it settles.

    python3 intake.py <intake.json>            # validate, print KEY=VALUE lines
    python3 intake.py <intake.json> --json     # validate, print the normalised intake
    python3 intake.py --schema                 # print the schema as documentation

An **intake** answers step 1 ("how does this workload run") without asking: MAD's
/mad-debug-numeric skill and mad-nan-hunt workflow write one from `madengine discover`
(GLOSSARY.md, reference/step1-intake.md).  nan-doctor owns this schema; producers
only fill it.  Exit 0 = valid; exit 2 = invalid, one `INTAKE_ERROR <field>: <why>`
line per problem.  Standard library only.
"""

import json
import re
import sys

SCHEMA_VERSION = 1

DEFAULT_NAN_SIGNAL = r"(?i)\b(loss|grad_norm)\b[^\n]*\b(nan|inf)\b"

# field: (required, type(s), description, default)
FIELDS = {
    "schema": (True, int, "intake schema version; must be %d" % SCHEMA_VERSION, None),
    "source": (True, str, "producer: 'mad' | 'manual'", None),
    "env": (True, str, "environment profile: 'madengine' | 'docker'", None),
    "model_tag": (
        False,
        str,
        "madengine tag resolving to exactly one model (env=madengine)",
        None,
    ),
    "run": (
        True,
        str,
        "<RUN_SCRIPT>: the command that runs the model as-is (no capture)",
        None,
    ),
    "additional_context": (
        False,
        dict,
        "madengine --additional-context, minus 'tools' (env=madengine)",
        {},
    ),
    "deploy_target": (
        False,
        str,
        "'local' | 'slurm' (env=madengine; k8s is not supported yet)",
        "local",
    ),
    "nan_signal": (
        False,
        dict,
        "{'regex': ...}: <NAN_SIGNAL>",
        {"regex": DEFAULT_NAN_SIGNAL},
    ),
    "world_size": (True, int, "<WORLD_SIZE>: ranks of ONE job", None),
    "nnodes": (False, int, "nodes of one job", 1),
    "image": (
        False,
        str,
        "image the model runs in (madengine: build_manifest docker_image)",
        None,
    ),
    "artifact_dir": (True, str, "<ARTIFACT_DIR>, created by the producer", None),
    "preferences": (False, dict, "step-1 questions 2-6, see PREFERENCES", {}),
}

# preference: (allowed values or type, default, step-1 question it answers)
PREFERENCES = {
    "measure_rate": ((bool,), False, "Q2 run step 2 (one wave, measures the NaN rate)"),
    "capture_data": (
        ("shapes", "values"),
        "shapes",
        "Q3 capsule data mode (values = NAN_CAPTURE_DATA=1)",
    ),
    "rootcause": ((bool,), False, "Q4 run step 7 within budget_minutes"),
    "fix": ((bool,), False, "Q4 run step 8 (needs rootcause; interactive only)"),
    "groups": (("largest", "all"), "largest", "Q5 <GROUPS_TO_REPLAY>"),
    "budget_minutes": ((int,), 0, "Q6/step 7 wall-clock budget; 0 = none given"),
    "priors": (
        ("use", "none"),
        "use",
        "Q7 <PRIORS>: use earlier investigations of this script (none = fresh eyes)",
    ),
}


def _type_name(t):
    return t.__name__ if isinstance(t, type) else "/".join(x.__name__ for x in t)


def validate(raw):
    """Return (normalised intake, [error strings])."""
    errs = []
    if not isinstance(raw, dict):
        return None, ["(root): must be a JSON object"]
    out = {}
    for k in raw:
        if k not in FIELDS:
            errs.append("%s: unknown field" % k)
    for k, (req, typ, _, default) in FIELDS.items():
        if k not in raw or raw[k] is None:
            if req:
                errs.append("%s: required" % k)
            elif default is not None:
                out[k] = json.loads(json.dumps(default))
            continue
        v = raw[k]
        if typ is int and isinstance(v, bool) or not isinstance(v, typ):
            errs.append("%s: must be %s" % (k, _type_name(typ)))
            continue
        out[k] = v
    if errs:
        return None, errs

    if out["schema"] != SCHEMA_VERSION:
        errs.append(
            "schema: %r is not supported (this nan-doctor reads %d)"
            % (out["schema"], SCHEMA_VERSION)
        )
    if out["source"] not in ("mad", "manual"):
        errs.append("source: must be 'mad' or 'manual'")
    if out["env"] not in ("madengine", "docker"):
        errs.append("env: must be 'madengine' or 'docker'")
    if out["world_size"] < 1:
        errs.append("world_size: must be >= 1")
    if out["nnodes"] < 1:
        errs.append("nnodes: must be >= 1")
    if not out["run"].strip():
        errs.append("run: empty")

    if out["env"] == "madengine":
        if not out.get("model_tag"):
            errs.append("model_tag: required when env is 'madengine'")
        if out["deploy_target"] not in ("local", "slurm"):
            errs.append(
                "deploy_target: must be 'local' or 'slurm' (k8s is not supported yet)"
            )
        if "tools" in out["additional_context"]:
            errs.append(
                "additional_context: must not carry 'tools'; the capture step adds nan_doctor itself"
            )
        if "madengine run" not in out["run"]:
            errs.append(
                "run: must be the 'madengine run ...' command when env is 'madengine'"
            )

    sig = out["nan_signal"]
    if set(sig) != {"regex"} or not isinstance(sig.get("regex"), str):
        errs.append("nan_signal: must be {'regex': <string>}")
    else:
        try:
            re.compile(sig["regex"])
        except re.error as e:
            errs.append("nan_signal.regex: does not compile: %s" % e)

    prefs = dict((k, d) for k, (_, d, _) in PREFERENCES.items())
    for k, v in out.get("preferences", {}).items():
        if k not in PREFERENCES:
            errs.append("preferences.%s: unknown preference" % k)
            continue
        allowed = PREFERENCES[k][0]
        if all(isinstance(a, type) for a in allowed):
            if not isinstance(v, allowed) or (
                bool not in allowed and isinstance(v, bool)
            ):
                errs.append("preferences.%s: must be %s" % (k, _type_name(allowed)))
                continue
        elif v not in allowed:
            errs.append("preferences.%s: must be one of %s" % (k, ", ".join(allowed)))
            continue
        prefs[k] = v
    if prefs["fix"] and not prefs["rootcause"]:
        errs.append("preferences.fix: needs preferences.rootcause")
    if prefs["budget_minutes"] < 0:
        errs.append("preferences.budget_minutes: must be >= 0")
    out["preferences"] = prefs
    return (None if errs else out), errs


def kv_lines(intake):
    p = intake["preferences"]
    lines = [
        ("RUN_SCRIPT", intake["run"]),
        ("NAN_SIGNAL", intake["nan_signal"]["regex"]),
        ("WORLD_SIZE", intake["world_size"]),
        ("NNODES", intake["nnodes"]),
        ("ENV", intake["env"]),
        ("ARTIFACT_DIR", intake["artifact_dir"]),
        ("IMAGE", intake.get("image") or "?"),
    ]
    if intake["env"] == "madengine":
        lines += [
            ("MODEL_TAG", intake["model_tag"]),
            ("DEPLOY_TARGET", intake["deploy_target"]),
            (
                "ADDITIONAL_CONTEXT",
                json.dumps(intake["additional_context"], sort_keys=True),
            ),
        ]
    lines += [
        ("PREF_%s" % k.upper(), json.dumps(v) if isinstance(v, bool) else v)
        for k, v in p.items()
    ]
    return ["%s=%s" % (k, v) for k, v in lines]


def schema_text():
    rows = ["field | required | type | meaning | default", "---|---|---|---|---"]
    for k, (req, typ, desc, d) in FIELDS.items():
        rows.append(
            "%s | %s | %s | %s | %s"
            % (
                k,
                "yes" if req else "no",
                _type_name(typ),
                desc,
                "" if d is None else json.dumps(d),
            )
        )
    rows += ["", "preference | values | default | answers", "---|---|---|---"]
    for k, (allowed, d, desc) in PREFERENCES.items():
        vals = (
            _type_name(allowed)
            if all(isinstance(a, type) for a in allowed)
            else " / ".join(allowed)
        )
        rows.append("%s | %s | %s | %s" % (k, vals, json.dumps(d), desc))
    return "\n".join(rows)


def main(argv):
    if argv[:1] == ["--schema"]:
        print(schema_text())
        return 0
    if not argv or argv[0].startswith("-"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        with open(argv[0]) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        print("INTAKE_ERROR (file): %s" % e)
        return 2
    intake, errs = validate(raw)
    if errs:
        for e in errs:
            print("INTAKE_ERROR %s" % e)
        return 2
    if "--json" in argv[1:]:
        print(json.dumps(intake, indent=2, sort_keys=True))
    else:
        print("\n".join(kv_lines(intake)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

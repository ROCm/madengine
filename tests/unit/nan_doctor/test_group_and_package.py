"""5a priority ordering and the step-6 reproducer directory."""

import json
import os
import subprocess
import sys

from madengine.nan_doctor import group, package


def groups_json(tmp_path, groups):
    p = tmp_path / "groups.json"
    p.write_text(json.dumps({"groups": groups}))
    return str(p)


def g(rep, op, count, **extra):
    return {
        "representative": rep,
        "identity": {"op": op},
        "count": count,
        "operand_sets": extra.get("sets", []),
    }


def test_groups_are_ordered_by_capsule_count_and_false_positives_are_tagged(tmp_path):
    j = groups_json(
        tmp_path,
        [
            g("/c/a.pt", "aten.empty.memory_format", 9),
            g("/c/b.pt", "aten.mm.default", 2),
            g("/c/c.pt", "aten._grouped_mm.default", 5),
        ],
    )
    lines = group.priority(j, str(tmp_path))
    assert [ln.split()[3] for ln in lines] == [
        "aten.empty.memory_format",
        "aten._grouped_mm.default",
        "aten.mm.default",
    ]
    assert lines[0].endswith("FALSE_POSITIVE?") and not lines[1].endswith("?")
    assert lines[0].startswith("G1 9 /c/a.pt")


def test_propagated_when_every_input_was_already_non_finite(tmp_path):
    sets = [
        {
            "outcomes": [{"inputs_finite": False}],
            "members": ["/c/p.pt"],
            "member_info": [{"rank": 0}],
        }
    ]
    j = groups_json(tmp_path, [g("/c/p.pt", "aten.mm.default", 1, sets=sets)])
    assert group.priority(j, str(tmp_path))[0].endswith("PROPAGATED?")


def test_the_origin_rank_group_goes_first_with_its_own_capsule(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"verdict": {"origin_ranks": [3]}})
    )
    sets = [
        {
            "members": ["/c/r0.pt", "/c/r3.pt"],
            "member_info": [{"rank": 0, "arrived": True}, {"rank": 3}],
            "outcomes": [],
        }
    ]
    j = groups_json(
        tmp_path,
        [
            g("/c/big.pt", "aten.mm.default", 9),
            g("/c/r0.pt", "aten.bmm.default", 2, sets=sets),
        ],
    )
    lines = group.priority(j, str(tmp_path))
    assert lines[0] == "G1 2 /c/r3.pt aten.bmm.default ORIGIN"
    assert lines[1].startswith("G2 9")


def test_group_from_json_writes_the_priority_file(tmp_path, capsys):
    j = groups_json(tmp_path, [g("/c/b.pt", "aten.mm.default", 2)])
    out = tmp_path / "art"
    assert group.main([str(tmp_path), "--out-dir", str(out), "--from-json", j]) == 0
    assert (
        (out / "groups_priority.txt")
        .read_text()
        .startswith("G1 2 /c/b.pt aten.mm.default")
    )
    assert "GROUPS=1" in capsys.readouterr().out


def make_attempt(tmp_path):
    d = tmp_path / "art" / "replay" / "G1" / "c3.77.host"
    d.mkdir(parents=True)
    (d / "g0_k0.host.log").write_text(
        "[C1] chain walk\n[verdict] ok\nchain loop: non-finite on 4/100 chain runs\n"
        "env: matches capture\n"
    )
    (d / "summary.txt").write_text(
        "=== totals: copies NAN 1 / CLEAN 0 / OOM 0 / ERROR 0; non-finite 4/100 ===\n"
        "  g0_k0.host.log NAN\nATTEMPT_RESULT=REPRODUCED\n"
    )
    return d


def test_package_builds_a_standalone_directory(tmp_path):
    cap = tmp_path / "rank0_abc.pt"
    cap.write_bytes(b"capsule")
    repro = tmp_path / "repro-G1"
    package.build(
        str(repro),
        str(cap),
        "ci-model",
        5000,
        3,
        "chain",
        "python3 -m nan_doctor_rt.transformer_hog",
        "--synthesize",
        {"LD_LIBRARY_PATH": "/opt/lib"},
    )
    for rel in (
        "run.sh",
        "capsule/rank0_abc.pt",
        "host/nan_doctor/__main__.py",
        "host/nan_doctor/replay.py",
        "runtime/nan_doctor_rt/replay_nan_op.py",
        "runtime/nan_doctor_rt/kernel_patch.py",
        "runtime/nan_doctor_rt/shim/hsa_patch_shim.c",
    ):
        assert (repro / rel).exists(), rel
    run = (repro / "run.sh").read_text()
    assert "IMAGE=${IMAGE:-ci-model}" in run
    assert "CAPSULE_ITER_COUNT:-5000" in run and "CAPSULE_CONCURRENCY:-3" in run
    assert "--cotenant 'python3 -m nan_doctor_rt.transformer_hog'" in run
    assert (
        "--extra-args --synthesize" in run and "--env LD_LIBRARY_PATH=/opt/lib" in run
    )
    assert subprocess.run(["bash", "-n", str(repro / "run.sh")]).returncode == 0
    assert os.access(repro / "run.sh", os.X_OK)


def test_the_vendored_host_package_runs_without_madengine(tmp_path):
    """A reproducer must work where only python3 and docker exist."""
    cap = tmp_path / "c.pt"
    cap.write_bytes(b"x")
    repro = tmp_path / "repro-G1"
    package.build(str(repro), str(cap), "img", 100, 2, "chain", "", "", {})
    out = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.path.insert(0, 'host'); "
            "import nan_doctor.cli as c; "
            "assert 'madengine' not in sys.modules; "
            "import nan_doctor.replay, nan_doctor.runner, nan_doctor.summary, nan_doctor.package; "
            "assert 'madengine' not in sys.modules; print(c.main(['--help']))",
        ],
        cwd=repro,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "replay" in out.stdout and "madengine" not in out.stderr


def test_package_main_writes_the_readme_and_packs(tmp_path):
    d = make_attempt(tmp_path)
    cap = tmp_path / "rank0_abc.pt"
    cap.write_bytes(b"capsule")
    gj = groups_json(tmp_path, [g(str(cap), "aten.mm.default", 1)])
    rc = package.main(
        [
            "--group",
            "G1",
            "--artifact-dir",
            str(tmp_path / "art"),
            "--image",
            "ci-model",
            "--capsule",
            str(cap),
            "--attempt",
            str(d),
            "--groups",
            gj,
            "--pack",
        ]
    )
    assert rc in (0, None)
    readme = (tmp_path / "art" / "repro-G1" / "README.md").read_text()
    assert "aten.mm.default" in readme and "CHECK" in readme and "bash run.sh" in readme
    assert "apptainer" not in readme.lower() and "sbatch" not in readme.lower()
    assert (tmp_path / "art" / "repro-G1.tar.gz").exists()

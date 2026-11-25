"""`madengine nan verify-nodes`: the package run on other nodes, and each node's score."""

import os
import subprocess

import pytest

from madengine.nan_doctor import verify_nodes


@pytest.mark.parametrize(
    "text,want",
    [
        ("x\nLADDER_RESULT=REPRODUCED at concurrency 1\n", "REPRODUCED"),
        ("LADDER_RESULT=NOT_REPRODUCED -- 0/100\n", "NOT_REPRODUCED"),
        ("LADDER_RESULT=NOT_SET_UP ran clean\n", "NOT_REPRODUCED"),
        ("NATIVE_TOTAL=3/40 copies=8x1 out=o\n", "REPRODUCED"),
        ("NATIVE_TOTAL=0/40 copies=8x1 out=o\n", "NOT_REPRODUCED"),
        ("fake package: missing image\nPACKAGE_RC=3\n", "ERROR"),
        ("LADDER_RESULT=ERROR reading the capsule\n", "ERROR"),
    ],
)
def test_score_line(text, want):
    assert verify_nodes.score_line(text)[0] == want


def logs(tmp_path, **hosts):
    for h, text in hosts.items():
        (tmp_path / ("%s.log" % h)).write_text(text)


def test_some_nodes_is_node_dependent(tmp_path, capsys):
    logs(
        tmp_path,
        a="LADDER_RESULT=REPRODUCED at 1\n",
        b="LADDER_RESULT=NOT_REPRODUCED\n",
    )
    verify_nodes.score(str(tmp_path))
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert (
        last
        == "NODES_REPRODUCED=1/2 hosts=a:REPRODUCED,b:NOT_REPRODUCED NODE_DEPENDENT"
    )


def test_an_error_is_never_node_dependent(tmp_path, capsys):
    logs(tmp_path, a="LADDER_RESULT=REPRODUCED at 1\n", b="PACKAGE_RC=3\n")
    verify_nodes.score(str(tmp_path))
    assert capsys.readouterr().out.strip().endswith("hosts=a:REPRODUCED,b:ERROR")


def test_no_node_reproduces(tmp_path, capsys):
    logs(tmp_path, a="LADDER_RESULT=NOT_REPRODUCED\n")
    verify_nodes.score(str(tmp_path))
    assert capsys.readouterr().out.strip().endswith("REPRODUCES_NOWHERE_ELSE")


def test_one_node_runs_its_copy_in_an_empty_environment(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "run.sh").write_text(
        'echo "LADDER_RESULT=REPRODUCED secret=${SECRET:-unset} image=$IMAGE pwd=$PWD"\n'
    )
    out = tmp_path / "out"
    verify_nodes.main(
        ["script", "--package", str(pkg), "--out", str(out), "--env", "IMAGE=x:1"]
    )
    sb = (out / "verify_nodes.slurm").read_text()
    assert "#SBATCH --nodes=2" in sb and "srun" in sb
    env = dict(os.environ, SECRET="1", SLURMD_NODENAME="node-08")
    r = subprocess.run(
        ["bash", str(out / "verify_node.sh")], env=env, capture_output=True, text=True
    )
    assert r.stdout.strip() == "NODE_DONE node-08 rc=0"
    log = (out / "node-08.log").read_text()
    assert "secret=unset image=x:1 pwd=%s" % (out / "pkg.node-08") in log
    assert log.rstrip().endswith("PACKAGE_RC=0")


def test_submit_without_slurm_says_so(tmp_path, monkeypatch, capsys):
    (tmp_path / "run.sh").write_text("true\n")
    monkeypatch.setattr(verify_nodes.shutil, "which", lambda name: None)
    args = ["script", "--package", str(tmp_path), "--out", str(tmp_path / "o")]
    assert verify_nodes.main(args + ["--submit"]) == 3
    assert "VERIFY_NODES=NO_SLURM" in capsys.readouterr().out


def test_script_rejects_a_missing_script(tmp_path):
    with pytest.raises(SystemExit):
        verify_nodes.main(
            ["script", "--package", str(tmp_path), "--out", str(tmp_path)]
        )

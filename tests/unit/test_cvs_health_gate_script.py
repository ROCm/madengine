"""Tests for the cvs_health_gate.sh pre_script example.

These tests exercise the shell script directly with a fake `cvs` binary
placed on PATH -- no real cvs installation or cluster is required.
"""
import os
import shutil
import stat
import subprocess

import pytest

SCRIPT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "madengine",
        "scripts",
        "common",
        "pre_scripts",
        "cvs_health_gate.sh",
    )
)


def _make_fake_cvs(tmp_path, exit_code):
    """Write a fake `cvs` executable onto a fresh PATH directory."""
    fake_bin_dir = tmp_path / "bin"
    fake_bin_dir.mkdir()
    fake_cvs = fake_bin_dir / "cvs"
    fake_cvs.write_text(
        "#!/usr/bin/env bash\n"
        'echo "fake cvs invoked with: $*"\n'
        f"exit {exit_code}\n"
    )
    fake_cvs.chmod(fake_cvs.stat().st_mode | stat.S_IEXEC)
    return str(fake_bin_dir)


def test_script_exists_and_is_executable():
    assert os.path.isfile(SCRIPT)
    assert os.access(SCRIPT, os.X_OK)


def test_passes_through_cvs_success(tmp_path):
    fake_bin_dir = _make_fake_cvs(tmp_path, 0)
    env = {"PATH": fake_bin_dir + os.pathsep + os.environ["PATH"]}

    result = subprocess.run(
        ["bash", SCRIPT, "monitor", "check_cluster_health", "--cluster_file", "cluster.json"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (
        "fake cvs invoked with: monitor check_cluster_health --cluster_file cluster.json"
        in result.stdout
    )


def test_propagates_cvs_failure_exit_code(tmp_path):
    fake_bin_dir = _make_fake_cvs(tmp_path, 3)
    env = {"PATH": fake_bin_dir + os.pathsep + os.environ["PATH"]}

    result = subprocess.run(
        ["bash", SCRIPT, "monitor", "check_cluster_health", "--cluster_file", "cluster.json"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    assert "aborting run" in result.stderr


def test_fails_clearly_when_cvs_not_installed(tmp_path):
    empty_bin_dir = tmp_path / "emptybin"
    empty_bin_dir.mkdir()
    env = {"PATH": str(empty_bin_dir)}

    result = subprocess.run(
        # Use an absolute path to bash so the child process doesn't need to
        # resolve "bash" itself via the deliberately-empty PATH we're testing.
        [shutil.which("bash"), SCRIPT, "monitor", "check_cluster_health"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "not found in PATH" in result.stderr


def test_requires_at_least_one_argument(tmp_path):
    fake_bin_dir = _make_fake_cvs(tmp_path, 0)
    env = {"PATH": fake_bin_dir + os.pathsep + os.environ["PATH"]}

    result = subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True)

    assert result.returncode == 1
    assert "usage:" in result.stderr

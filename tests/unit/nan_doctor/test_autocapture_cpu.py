"""CPU tests of the source-free capture: the installed .pth hook arms op capture in an
UNMODIFIED training script, single process and 2-rank torchrun (gloo). Needs torch, no GPU.

The runtime is installed into a temp dir, never the real site-packages; a sitecustomize.py on
PYTHONPATH makes that dir a site dir, so the .pth is honoured by every interpreter the workload
starts, torchrun ranks included -- the same path a container's site-packages takes.
"""

import glob
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = (
    ROOT / "src/madengine/scripts/common/tools/nan_doctor/nan_doctor_rt/install.py"
)
TRAIN = ROOT / "tests/fixtures/nan_doctor/autocapture_train.py"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("autocapture")
    site_dir = tmp / "site"
    out = subprocess.run(
        [sys.executable, str(INSTALLER), "--target", str(site_dir)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == str(site_dir)
    boot = tmp / "boot"
    boot.mkdir()
    (boot / "sitecustomize.py").write_text(
        "import site\nsite.addsitedir(%r)\n" % str(site_dir)
    )
    return {"dir": site_dir, "boot": boot, "tmp": tmp}


def run(site, cmd, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("NAN_CAPTURE")}
    e["PYTHONPATH"] = str(site["boot"])
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=600)


def capsules(d):
    return glob.glob(os.path.join(str(d), "**", "*.pt"), recursive=True)


def test_installer_wrote_the_package_and_the_pth(site):
    assert (
        site["dir"] / "nan_doctor_rt.pth"
    ).read_text() == "import nan_doctor_rt.autocapture\n"
    for rel in (
        "__init__.py",
        "autocapture.py",
        "op_nan_capture.py",
        "replay_nan_op.py",
        "shim/hsa_patch_shim.c",
    ):
        assert (site["dir"] / "nan_doctor_rt" / rel).exists(), rel


def test_the_pth_imports_the_module_without_importing_torch(site):
    r = run(
        site,
        [
            sys.executable,
            "-c",
            "import sys, nan_doctor_rt.autocapture as m; print('torch' in sys.modules, m.enabled())",
        ],
    )
    assert r.stdout.split() == ["False", "False"], r.stderr


def test_disabled_workload_runs_untouched(site):
    r = run(site, [sys.executable, str(TRAIN)], PLANT_STEP=4)
    assert "TRAINING_DONE" in r.stdout and "[op-nan]" not in r.stdout + r.stderr


def test_the_capture_dir_alone_does_not_arm_it(site):
    d = site["tmp"] / "caps_dir_only"
    r = run(site, [sys.executable, str(TRAIN)], PLANT_STEP=4, NAN_CAPTURE_DIR=d)
    assert "TRAINING_DONE" in r.stdout and not capsules(d)


def test_armed_clean_run_writes_no_capsule(site):
    d = site["tmp"] / "caps_clean"
    r = run(
        site,
        [sys.executable, str(TRAIN)],
        NAN_CAPTURE_DIR=d,
        NAN_CAPTURE_AUTO="optimizer",
    )
    assert "TRAINING_DONE" in r.stdout, r.stderr
    assert (
        "optimizer-step hook registered" in r.stderr
        and "nan-capture armed" in r.stdout + r.stderr
    )
    assert not capsules(d)


def test_planted_nan_single_process_writes_a_capsule_and_stops(site):
    d = site["tmp"] / "caps_single"
    r = run(
        site,
        [sys.executable, str(TRAIN)],
        PLANT_STEP=4,
        NAN_CAPTURE_DIR=d,
        NAN_CAPTURE_AUTO="optimizer",
    )
    assert len(capsules(d)) >= 1, r.stdout + r.stderr
    assert "TRAINING_DONE" not in r.stdout and "[op-nan]" in r.stderr


def test_only_the_first_optimizer_is_counted(site):
    d = site["tmp"] / "caps_two"
    r = run(
        site,
        [sys.executable, str(TRAIN)],
        TWO_OPTIMIZERS=1,
        NAN_CAPTURE_DIR=d,
        NAN_CAPTURE_AUTO="optimizer",
        NAN_PROGRESS_EVERY=1,
    )
    assert r.stdout.count("[op-nan] Progress step") == 8, r.stdout + r.stderr


def test_planted_nan_on_one_of_two_torchrun_ranks_stops_both(site):
    d = site["tmp"] / "caps_dist"
    r = run(
        site,
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "2",
            "--master_port",
            str(free_port()),
            str(TRAIN),
        ],
        PLANT_STEP=4,
        PLANT_RANK=1,
        NAN_CAPTURE_DIR=d,
        NAN_CAPTURE_AUTO="optimizer",
    )
    log = r.stdout + r.stderr
    assert log.count("optimizer-step hook registered") >= 2, log
    assert len(capsules(d)) >= 1, log
    assert "TRAINING_DONE" not in r.stdout and "ranks that captured" in log


def test_uninstall_removes_everything(site):
    subprocess.run(
        [sys.executable, str(INSTALLER), "--uninstall", "--target", str(site["dir"])],
        check=True,
        capture_output=True,
    )
    assert (
        not (site["dir"] / "nan_doctor_rt").exists()
        and not (site["dir"] / "nan_doctor_rt.pth").exists()
    )

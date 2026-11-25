"""`madengine nan exec`: the docker command that runs a payload in a model's image."""

import json

import pytest

from madengine.nan_doctor import runner


def cmd(**kw):
    kw.setdefault("runtime_dir", "/rt")
    kw.setdefault("cwd", "/work")
    return runner.build_command("img:1", "echo hi", **kw)


def test_gpu_flags_and_timing_flags_are_always_present():
    argv = cmd()
    for flag in (
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--ipc=host",
        "--shm-size=64g",
        "--network=host",
        "--cap-add=SYS_PTRACE",
        "--cap-add=IPC_LOCK",
    ):
        assert flag in argv
    assert argv[argv.index("--security-opt") + 1] == "seccomp=unconfined"
    assert argv[-4:] == ["img:1", "bash", "-lc", "echo hi"]


def test_one_gpu_pins_the_device_and_a_distinct_master_port():
    argv = cmd(gpus="3")
    assert "HIP_VISIBLE_DEVICES=3" in argv and "MASTER_PORT=29503" in argv
    assert "HIP_VISIBLE_DEVICES=0,1" in cmd(gpus="0,1")
    assert not any(a.startswith("HIP_VISIBLE_DEVICES") for a in cmd(gpus="all"))


def test_multi_gpu_trials_share_the_host_pid_namespace():
    assert "--pid=host" not in cmd(gpus="0")
    assert "--pid=host" in cmd(gpus="0,1")
    assert "--pid=host" in cmd(gpus="all")


@pytest.mark.parametrize("bad", ["0;rm", "x", "-1", "0,", ""])
def test_gpus_must_be_all_or_a_list(bad):
    with pytest.raises(ValueError):
        cmd(gpus=bad)


def test_runtime_is_mounted_read_only_and_importable():
    argv = cmd()
    assert "/rt:/opt/nan_doctor:ro" in argv
    assert "PYTHONPATH=/opt/nan_doctor:/opt/nan_doctor_host" in argv
    assert any(a.endswith(":/opt/nan_doctor_host/nan_doctor:ro") for a in argv)
    assert "NAN_DOCTOR_RT=/opt/nan_doctor/nan_doctor_rt" in argv


def test_working_directory_is_mounted_at_its_own_path_and_extra_mounts_follow():
    argv = cmd(mounts=[("/data", "/data", "ro"), ("/a", "/b", "rw")])
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert mounts[0] == "/work:/work:rw"
    assert "/data:/data:ro" in mounts and "/a:/b:rw" in mounts
    assert argv[argv.index("-w") + 1] == "/work"


def test_env_names_are_validated_and_values_pass_through():
    assert "FOO=a b" in cmd(env={"FOO": "a b"})
    with pytest.raises(ValueError):
        cmd(env={"BAD NAME": "1"})


def test_as_user_runs_as_the_host_user_with_a_private_home():
    argv = cmd(as_user=True)
    assert "--user" in argv and "HOME=/tmp" in argv


def test_parse_mount():
    assert runner.parse_mount("/x") == ("/x", "/x", "rw")
    assert runner.parse_mount("/x:/y:ro") == ("/x", "/y", "ro")
    with pytest.raises(ValueError):
        runner.parse_mount("/x:/y:rx")


def test_image_from_manifest(tmp_path):
    m = tmp_path / "build_manifest.json"
    m.write_text(
        json.dumps(
            {
                "built_images": {
                    "a": {"docker_image": "ci-a"},
                    "b": {"docker_image": "ci-b"},
                },
                "built_models": {
                    "a": {"name": "dir/model_a"},
                    "b": {"name": "model_b"},
                },
            }
        )
    )
    assert runner.image_from_manifest(str(m), "model_a") == "ci-a"
    assert runner.image_from_manifest(str(m), "model_b") == "ci-b"
    with pytest.raises(ValueError, match="2 image"):
        runner.image_from_manifest(str(m))
    m.write_text(
        json.dumps(
            {
                "built_images": {"a": {"docker_image": "ci-a"}},
                "built_models": {"a": {"name": "m"}},
            }
        )
    )
    assert runner.image_from_manifest(str(m)) == "ci-a"


def test_print_command_needs_no_docker(capsys, monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    rc = runner.main(
        [
            "--image",
            "img:1",
            "--runtime-dir",
            "/rt",
            "--print-command",
            "--",
            "python3",
            "-m",
            "x",
            "a b",
        ]
    )
    out = capsys.readouterr().out
    assert (
        rc == 0
        and "docker run --rm" in out
        and "'python3 -m x '\"'\"'a b'\"'\"''" in out.replace("\n", "")
        or "python3 -m x" in out
    )


def test_missing_image_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(runner, "image_present", lambda _: False)
    rc = runner.main(["--image", "nope:1", "--runtime-dir", "/rt", "--", "true"])
    assert rc == 2 and "not on this host" in capsys.readouterr().err

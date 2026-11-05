"""DockerBuilder.pin_set_build_args: a card naming a pin set (PINS) gets the rest of
its build args from the pin-set tool next to the Dockerfile (MAD docker/pins/pins.py),
for the architecture of this build.
"""
from unittest.mock import MagicMock

import pytest

from madengine.execution.docker_builder import DockerBuilder

FAKE_TOOL = r'''
import json, sys
a = sys.argv[1:]
name, arch = a[1], a[a.index("--arch") + 1]
if arch not in ("gfx942", "gfx950"):
    print(f"ERROR: {name} is built for gfx942, gfx950, not {arch}", file=sys.stderr); sys.exit(1)
print(json.dumps({"PINS": name, "BASE_IMAGE": f"base-for-{arch}@sha256:" + "0" * 64,
                  "MAD_SYSTEM_GPU_ARCHITECTURE": arch, "MAX_JOBS": "32"}))
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "docker" / "pins").mkdir(parents=True)
    (tmp_path / "docker" / "pins" / "pins.py").write_text(FAKE_TOOL)
    df = tmp_path / "docker" / "di_vllm.ubuntu.amd.Dockerfile"
    df.write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    return df


def _builder(ctx_build_args=None):
    ctx = MagicMock()
    ctx.ctx = {"docker_build_arg": ctx_build_args} if ctx_build_args else {}
    return DockerBuilder(ctx, MagicMock())


MODEL = {"name": "m"}


def test_no_pin_set_no_change(repo):
    assert _builder().pin_set_build_args(MODEL, str(repo), {"X": "1"}) == {}


def test_adds_base_and_settings_for_the_arch(repo):
    got = _builder({"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx950"}).pin_set_build_args(
        MODEL, str(repo), {"PINS": "vllm-kimi-k3"})
    assert got == {"BASE_IMAGE": "base-for-gfx950@sha256:" + "0" * 64, "MAX_JOBS": "32"}


def test_card_arch_when_the_run_names_none(repo):
    got = _builder().pin_set_build_args(MODEL, str(repo), {"PINS": "p", "MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"})
    assert got["BASE_IMAGE"].startswith("base-for-gfx942")


def test_dockerfile_default_arch(repo):
    repo.write_text("ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942\nARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    got = _builder().pin_set_build_args(MODEL, str(repo), {"PINS": "p"})
    assert got["BASE_IMAGE"].startswith("base-for-gfx942")


def test_no_arch_is_an_error(repo):
    with pytest.raises(RuntimeError, match="needs MAD_SYSTEM_GPU_ARCHITECTURE"):
        _builder().pin_set_build_args(MODEL, str(repo), {"PINS": "p"})


def test_an_arch_the_set_is_not_for_is_an_error(repo):
    with pytest.raises(RuntimeError, match="not gfx999"):
        _builder({"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx999"}).pin_set_build_args(MODEL, str(repo), {"PINS": "p"})


def test_a_disagreeing_base_is_an_error(repo):
    with pytest.raises(RuntimeError, match="BASE_IMAGE=other disagrees"):
        _builder({"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942", "BASE_IMAGE": "other"}).pin_set_build_args(
            MODEL, str(repo), {"PINS": "p"})


def test_an_agreeing_value_is_kept_not_repeated(repo):
    got = _builder({"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942", "MAX_JOBS": "32"}).pin_set_build_args(
        MODEL, str(repo), {"PINS": "p"})
    assert "MAX_JOBS" not in got and "BASE_IMAGE" in got


def test_missing_tool_is_an_error(tmp_path):
    df = tmp_path / "x.Dockerfile"
    df.write_text("FROM scratch\n")
    with pytest.raises(RuntimeError, match="does not exist"):
        _builder({"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx942"}).pin_set_build_args(MODEL, str(df), {"PINS": "p"})


def test_the_base_for_registry_login_is_the_pin_sets(repo):
    b = _builder()
    assert b._resolve_base_docker(str(repo), {"BASE_IMAGE": "lmsysorg/sglang:v1@sha256:" + "1" * 64}).startswith(
        "lmsysorg/sglang:v1")

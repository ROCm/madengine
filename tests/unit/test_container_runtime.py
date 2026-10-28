"""Podman uses the existing Docker command path through a local shim."""

import os
import subprocess
from pathlib import Path

import pytest

from madengine.core.container_runtime import (
    configure_container_cli,
    reset_container_cli,
)
from madengine.core.errors import ConfigurationError
from madengine.orchestration.platform import resolve_context_platform


@pytest.fixture(autouse=True)
def _restore_container_cli():
    reset_container_cli()
    yield
    reset_container_cli()


class TestPodmanDockerShim:
    def test_podman_places_a_docker_shim_first_on_path(self, monkeypatch, tmp_path):
        podman = tmp_path / "podman"
        podman.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        podman.chmod(0o755)
        monkeypatch.setattr(
            "madengine.core.container_runtime.shutil.which",
            lambda name: str(podman) if name == "podman" else None,
        )

        configure_container_cli("podman")

        shim = Path(os.environ["PATH"].split(os.pathsep)[0]) / "docker"
        assert shim.is_file()
        assert shim.stat().st_mode & 0o111
        shim_text = shim.read_text(encoding="utf-8")
        assert str(podman) in shim_text
        assert "unix:///run/podman/podman.sock" in shim_text

    def test_shim_handles_podman_name_and_apparmor_differences(
        self, monkeypatch, tmp_path
    ):
        captured = tmp_path / "arguments"
        podman = tmp_path / "podman"
        podman.write_text(
            f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {captured}\n",
            encoding="utf-8",
        )
        podman.chmod(0o755)
        monkeypatch.setattr(
            "madengine.core.container_runtime.shutil.which",
            lambda name: str(podman) if name == "podman" else None,
        )

        configure_container_cli("podman")
        shim = Path(os.environ["PATH"].split(os.pathsep)[0]) / "docker"
        subprocess.run(
            [
                str(shim),
                "container",
                "ps",
                "--filter",
                "name=^/job.1$",
                "--security-opt",
                "apparmor=unconfined",
            ],
            check=True,
        )

        assert captured.read_text(encoding="utf-8").splitlines() == [
            "container",
            "ps",
            "--filter",
            "name=^/?job.1$",
        ]

    def test_manifest_inspect_prints_a_digest_the_existing_parser_reads(
        self, monkeypatch, tmp_path
    ):
        podman = tmp_path / "podman"
        podman.write_text(
            """#!/bin/sh
if [ "$1" = "manifest" ]; then
    printf '%s\\n' '{"manifests": null}'
    exit 0
fi
printf '%s\\n' 'sha256:abc'
""",
            encoding="utf-8",
        )
        podman.chmod(0o755)
        monkeypatch.setattr(
            "madengine.core.container_runtime.shutil.which",
            lambda name: str(podman) if name == "podman" else None,
        )

        configure_container_cli("podman")
        shim = Path(os.environ["PATH"].split(os.pathsep)[0]) / "docker"
        digest = subprocess.check_output(
            f"{shim} manifest inspect rocm/pytorch | grep digest | head -n 1 | "
            """cut -d '"' -f 4""",
            shell=True,
            text=True,
        ).strip()
        assert digest == "sha256:abc"

    def test_docker_and_bare_metal_do_not_install_a_shim(self):
        configure_container_cli("docker")
        assert "madengine-podman-" not in os.environ["PATH"]
        configure_container_cli("bare_metal")
        assert "madengine-podman-" not in os.environ["PATH"]

    def test_missing_podman_cli_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            "madengine.core.container_runtime.shutil.which", lambda _name: None
        )
        with pytest.raises(ConfigurationError, match="podman"):
            configure_container_cli("podman")


def test_podman_with_nvidia_is_rejected_until_supported():
    with pytest.raises(ConfigurationError, match="NVIDIA"):
        resolve_context_platform({"platform": "podman", "gpu_vendor": "NVIDIA"})

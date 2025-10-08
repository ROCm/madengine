import subprocess
import sys
from pathlib import Path
import pytest


def _run(command, cwd=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, "Command failed (exit code {}):\n{}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
        result.returncode,
        " ".join(command),
        result.stdout,
        result.stderr,
    )


class TestPackaging:

    @pytest.mark.packaging
    def test_build_install_and_import(self, tmp_path):
        """Build a wheel, install it in isolation, then import madengine."""

        project_root = Path(__file__).resolve().parents[1]
        dist_dir = tmp_path / "wheel"
        site_dir = tmp_path / "site-packages"
        dist_dir.mkdir()
        site_dir.mkdir()

        # build a wheel into the temporary dist folder.
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-cache-dir",
                "-w",
                str(dist_dir),
                str(project_root),
            ],
            cwd=project_root,
        )

        wheels = sorted(dist_dir.glob("madengine-*.whl"))
        assert wheels, "Expected pip wheel to create a madengine wheel"
        wheel_path = wheels[0]

        # install that wheel into an isolated folder.
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-cache-dir",
                "--target",
                str(site_dir),
                str(wheel_path),
            ]
        )

        # import madengine from the isolated folder.
        _run(
            [
                sys.executable,
                "-c",
                "import sys; " f"sys.path.insert(0, {repr(str(site_dir))}); " "import madengine; print(madengine.__version__)",
            ]
        )

"""Unit tests for shell injection hardening via shlex.quote().

Validates that Docker, DockerBuilder, ContainerRunner, and RunOrchestrator
properly quote user-controlled values interpolated into shell commands.
"""

import os
import shlex
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from madengine.core.console import Console


MALICIOUS_IMAGE = "img:latest; rm -rf /"
MALICIOUS_PATH = "/data/path; echo pwned"
SAFE_IMAGE = "registry.io/org/model:ci-tag"


class TestDockerInitQuoting:
    """Docker.__init__ quotes container_name, image, and mount paths."""

    @patch.object(Console, "sh", return_value="")
    def test_container_name_and_image_are_quoted(self, mock_sh):
        from madengine.core.docker import Docker

        mock_sh.side_effect = self._make_sh_side_effect()
        try:
            Docker(
                image=MALICIOUS_IMAGE,
                container_name="evil;name",
                dockerOpts="",
                console=Console(shellVerbose=False),
            )
        except Exception:
            pass

        docker_run_calls = [
            c for c in mock_sh.call_args_list if "docker run" in str(c)
        ]
        assert docker_run_calls, "Expected at least one docker run call"
        run_cmd = docker_run_calls[0].args[0]
        assert shlex.quote("evil;name") in run_cmd
        assert shlex.quote(MALICIOUS_IMAGE) in run_cmd

    @patch.object(Console, "sh", return_value="")
    def test_mount_paths_are_quoted(self, mock_sh):
        from madengine.core.docker import Docker

        mock_sh.side_effect = self._make_sh_side_effect()
        try:
            Docker(
                image="ubuntu:22.04",
                container_name="test-container",
                dockerOpts="",
                mounts=[MALICIOUS_PATH],
                console=Console(shellVerbose=False),
            )
        except Exception:
            pass

        docker_run_calls = [
            c for c in mock_sh.call_args_list if "docker run" in str(c)
        ]
        assert docker_run_calls
        run_cmd = docker_run_calls[0].args[0]
        assert shlex.quote(MALICIOUS_PATH) in run_cmd

    @patch.object(Console, "sh", return_value="")
    def test_cwd_is_quoted(self, mock_sh):
        from madengine.core.docker import Docker

        mock_sh.side_effect = self._make_sh_side_effect()
        try:
            with patch("os.getcwd", return_value="/path with spaces/project"):
                Docker(
                    image="ubuntu:22.04",
                    container_name="test",
                    dockerOpts="",
                    console=Console(shellVerbose=False),
                )
        except Exception:
            pass

        docker_run_calls = [
            c for c in mock_sh.call_args_list if "docker run" in str(c)
        ]
        assert docker_run_calls
        run_cmd = docker_run_calls[0].args[0]
        assert shlex.quote("/path with spaces/project") in run_cmd

    @staticmethod
    def _make_sh_side_effect():
        def side_effect(cmd, **kwargs):
            if "id -u" in cmd:
                return "1000"
            if "id -g" in cmd:
                return "1000"
            if "docker container ps" in cmd:
                return ""
            if "docker run" in cmd:
                return ""
            if "docker ps" in cmd:
                return "abc123"
            return ""

        return side_effect


class TestDockerBuilderQuoting:
    """DockerBuilder.build_image quotes dockerfile, image, and context."""

    def test_build_command_quotes_image_dockerfile_context(self):
        from madengine.execution.docker_builder import DockerBuilder

        ctx = MagicMock()
        ctx.ctx = {}
        builder = DockerBuilder(ctx)
        mock_console = MagicMock()
        mock_console.sh = MagicMock(return_value="")
        builder.console = mock_console
        builder.rich_console = MagicMock()
        builder.live_output = False

        dockerfile = "docker/evil;file.Dockerfile"
        docker_image = "img:$(whoami)"
        docker_context = "/ctx;path"
        model_info = {"name": "test/model"}

        builder.get_context_path = MagicMock(return_value=docker_context)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".live.log", delete=False) as f:
            log_path = f.name

        try:
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__ = MagicMock()
                mock_open.return_value.__exit__ = MagicMock(return_value=False)

                builder.build_image(
                    model_info=model_info,
                    dockerfile=dockerfile,
                    override_image_name=docker_image,
                )
        except Exception:
            pass

        build_calls = [
            c for c in mock_console.sh.call_args_list
            if "docker build" in str(c)
        ]
        assert build_calls, "Expected a docker build call"
        build_cmd = build_calls[0].args[0]
        assert shlex.quote(docker_image) in build_cmd
        assert shlex.quote(dockerfile) in build_cmd
        assert shlex.quote(docker_context) in build_cmd


class TestContainerRunnerPullQuoting:
    """ContainerRunner.pull_image quotes registry_image and local_name."""

    @patch.dict(os.environ, {"MAD_DEPLOYMENT_TYPE": "local"}, clear=False)
    def test_pull_image_quotes_registry_and_local(self):
        from madengine.execution.container_runner import ContainerRunner

        ctx = MagicMock()
        ctx.ctx = {}
        mock_console = MagicMock()
        mock_console.sh = MagicMock(return_value="")
        runner = ContainerRunner(context=ctx, console=mock_console)
        runner.rich_console = MagicMock()

        registry_image = "registry/img:$(evil)"
        local_name = "local;name"

        try:
            runner.pull_image(registry_image, local_name=local_name)
        except Exception:
            pass

        all_cmds = [c.args[0] for c in mock_console.sh.call_args_list]

        pull_cmds = [c for c in all_cmds if "docker pull" in c]
        assert pull_cmds
        assert shlex.quote(registry_image) in pull_cmds[0]

        tag_cmds = [c for c in all_cmds if "docker tag" in c]
        assert tag_cmds
        assert shlex.quote(registry_image) in tag_cmds[0]
        assert shlex.quote(local_name) in tag_cmds[0]

    @patch.dict(os.environ, {"MAD_DEPLOYMENT_TYPE": "local"}, clear=False)
    def test_rmi_quotes_image_on_slurm(self):
        from madengine.execution.container_runner import ContainerRunner

        ctx = MagicMock()
        ctx.ctx = {}
        mock_console = MagicMock()
        mock_console.sh = MagicMock(return_value="")
        runner = ContainerRunner(context=ctx, console=mock_console)
        runner.rich_console = MagicMock()

        registry_image = "registry/img:$(evil)"

        with patch.dict(
            os.environ,
            {"MAD_DEPLOYMENT_TYPE": "slurm", "MAD_IN_SLURM_JOB": "1"},
        ):
            try:
                runner.pull_image(registry_image)
            except Exception:
                pass

        all_cmds = [c.args[0] for c in mock_console.sh.call_args_list]
        rmi_cmds = [c for c in all_cmds if "docker rmi" in c]
        assert rmi_cmds
        assert shlex.quote(registry_image) in rmi_cmds[0]


class TestContainerRunnerMountQuoting:
    """ContainerRunner.get_mount_arg quotes mount paths."""

    def test_datapath_mounts_are_quoted(self):
        from madengine.execution.container_runner import ContainerRunner

        ctx = MagicMock()
        ctx.ctx = {}
        runner = ContainerRunner(context=ctx, console=MagicMock())

        mount_datapaths = [
            {"path": "/data;evil", "home": "/container;evil"},
        ]

        result = runner.get_mount_arg(mount_datapaths)
        assert shlex.quote("/data;evil") in result
        assert shlex.quote("/container;evil") in result

    def test_context_docker_mounts_are_quoted(self):
        from madengine.execution.container_runner import ContainerRunner

        ctx = MagicMock()
        ctx.ctx = {
            "docker_mounts": {"/container;dst": "/host;src"},
        }
        runner = ContainerRunner(context=ctx, console=MagicMock())

        result = runner.get_mount_arg([])
        assert shlex.quote("/host;src") in result
        assert shlex.quote("/container;dst") in result


class TestRunOrchestratorImageQuoting:
    """RunOrchestrator quotes image_name in docker inspect and pull."""

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_image_inspect_is_quoted(self, mock_context):
        from madengine.orchestration.run_orchestrator import RunOrchestrator

        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False

        orchestrator = RunOrchestrator(mock_args)
        mock_console = MagicMock()
        mock_console.sh = MagicMock(return_value="")
        orchestrator.console = mock_console
        orchestrator.rich_console = MagicMock()

        image_name = "img:$(whoami)"

        try:
            orchestrator._create_manifest_from_local_image(
                image_name=image_name, tags=["test"]
            )
        except Exception:
            pass

        all_cmds = [c.args[0] for c in mock_console.sh.call_args_list]
        inspect_cmds = [c for c in all_cmds if "docker image inspect" in c]
        assert inspect_cmds
        assert shlex.quote(image_name) in inspect_cmds[0]

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_docker_pull_is_quoted_on_fallback(self, mock_context):
        from madengine.orchestration.run_orchestrator import RunOrchestrator

        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False

        orchestrator = RunOrchestrator(mock_args)
        mock_console = MagicMock()

        call_count = [0]
        def sh_side_effect(cmd, **kwargs):
            call_count[0] += 1
            if "docker image inspect" in cmd:
                raise RuntimeError("not found")
            return ""

        mock_console.sh = MagicMock(side_effect=sh_side_effect)
        orchestrator.console = mock_console
        orchestrator.rich_console = MagicMock()

        image_name = "img:$(whoami)"

        try:
            orchestrator._create_manifest_from_local_image(
                image_name=image_name, tags=["test"]
            )
        except Exception:
            pass

        all_cmds = [c.args[0] for c in mock_console.sh.call_args_list]
        pull_cmds = [c for c in all_cmds if "docker pull" in c]
        assert pull_cmds
        assert shlex.quote(image_name) in pull_cmds[0]


class TestSafeInputsUnchanged:
    """Normal inputs (no metacharacters) produce working commands with the value present."""

    @patch.dict(os.environ, {"MAD_DEPLOYMENT_TYPE": "local"}, clear=False)
    def test_safe_image_name_still_works(self):
        from madengine.execution.container_runner import ContainerRunner

        ctx = MagicMock()
        ctx.ctx = {}
        mock_console = MagicMock()
        mock_console.sh = MagicMock(return_value="")
        runner = ContainerRunner(context=ctx, console=mock_console)
        runner.rich_console = MagicMock()

        try:
            runner.pull_image(SAFE_IMAGE)
        except Exception:
            pass

        all_cmds = [c.args[0] for c in mock_console.sh.call_args_list]
        pull_cmds = [c for c in all_cmds if "docker pull" in c]
        assert pull_cmds
        assert SAFE_IMAGE in pull_cmds[0]

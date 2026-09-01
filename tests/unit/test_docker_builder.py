"""Unit tests for madengine.execution.docker_builder (DockerBuilder).

Currently covers registry image naming for single-arch builds (credentials → repository:tag).
"""

from unittest.mock import MagicMock, patch

import pytest

from madengine.execution.docker_builder import DockerBuilder

DIGEST = "sha256:" + "df36ef7e" * 8


@pytest.fixture
def docker_builder():
    ctx = MagicMock()
    ctx.ctx = {}
    return DockerBuilder(ctx)


def test_create_registry_image_name_uses_dockerhub_repository(docker_builder):
    creds = {
        "dockerhub": {
            "repository": "myorg/ci",
            "username": "u",
            "password": "p",
        }
    }
    out = docker_builder._create_registry_image_name(
        "ci-dummy_dummy.ubuntu.amd",
        "dockerhub",
        None,
        {"name": "dummy"},
        creds,
    )
    assert out == "myorg/ci:ci-dummy_dummy.ubuntu.amd"


def test_create_registry_image_name_without_credentials_matches_local_tag(docker_builder):
    out = docker_builder._create_registry_image_name(
        "ci-dummy_dummy.ubuntu.amd",
        "dockerhub",
        None,
        {"name": "dummy"},
        None,
    )
    assert out == "ci-dummy_dummy.ubuntu.amd"


class TestPushImageRecordsDigest:
    """push_image records the pushed image digest in builder.pushed_digests."""

    def _builder(self, sh_side_effect):
        ctx = MagicMock()
        ctx.ctx = {}
        console = MagicMock()
        console.sh = MagicMock(side_effect=sh_side_effect)
        builder = DockerBuilder(ctx, console)
        builder.rich_console = MagicMock()
        return builder

    def test_digest_parsed_from_push_output(self):
        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return f"mymodel: digest: {DIGEST} size: 4738"
            return ""

        builder = self._builder(sh)
        result = builder.push_image(
            "ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy"
        )

        assert result == "localhost:5000/ci-dummy"
        assert builder.pushed_digests["localhost:5000/ci-dummy"] == DIGEST

    def test_push_runs_without_a_timeout(self):
        """Multi-GB pushes routinely exceed Console.sh's 60s default."""

        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return f"mymodel: digest: {DIGEST} size: 4738"
            return ""

        builder = self._builder(sh)
        builder.push_image("ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy")

        push_calls = [
            c for c in builder.console.sh.call_args_list if "docker push" in c.args[0]
        ]
        assert len(push_calls) == 1
        assert push_calls[0].kwargs.get("timeout") is None

    def test_falls_back_to_image_inspect_when_push_output_has_no_digest(self):
        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return "The push refers to repository [localhost:5000/ci-dummy]\nlayer: Pushed"
            if "docker image inspect" in command:
                return f"localhost:5000/ci-dummy@{DIGEST}"
            return ""

        builder = self._builder(sh)
        builder.push_image(
            "ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy"
        )

        assert builder.pushed_digests["localhost:5000/ci-dummy"] == DIGEST
        inspect_calls = [
            c
            for c in builder.console.sh.call_args_list
            if "docker image inspect" in c.args[0]
        ]
        assert len(inspect_calls) == 1
        assert "RepoDigests" in inspect_calls[0].args[0]

    def test_no_digest_anywhere_leaves_entry_absent_and_push_still_succeeds(self):
        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return "layer: Pushed"
            if "docker image inspect" in command:
                return "<no value>"
            return ""

        builder = self._builder(sh)
        result = builder.push_image(
            "ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy"
        )

        assert result == "localhost:5000/ci-dummy"
        assert "localhost:5000/ci-dummy" not in builder.pushed_digests
        # The gap is noted at dim level, not as a user-facing warning.
        printed = " ".join(str(c) for c in builder.rich_console.print.call_args_list)
        assert "[dim]" in printed
        assert "no pushed digest" in printed.lower()

    def test_inspect_failure_is_swallowed(self):
        def sh(command, *args, **kwargs):
            if "docker push" in command:
                return "layer: Pushed"
            if "docker image inspect" in command:
                raise RuntimeError("no such image")
            return ""

        builder = self._builder(sh)
        result = builder.push_image(
            "ci-dummy", "localhost:5000", None, "localhost:5000/ci-dummy"
        )

        assert result == "localhost:5000/ci-dummy"
        assert builder.pushed_digests == {}

    def test_no_registry_records_nothing(self):
        builder = self._builder(lambda command, *a, **k: "")
        result = builder.push_image("ci-dummy")

        assert result == "ci-dummy"
        assert builder.pushed_digests == {}


class TestBuildInfoCarriesImageDigest:
    """Both push call sites copy the recorded digest into build_info."""

    def _builder(self):
        ctx = MagicMock()
        ctx.ctx = {}
        builder = DockerBuilder(ctx, MagicMock())
        builder.rich_console = MagicMock()
        return builder

    def _run_single_arch(self, builder):
        """Drive _build_model_single_arch with everything below push_image stubbed."""
        return builder._build_model_single_arch(
            model_info={"name": "dummy", "dockerfile": "docker/dummy"},
            credentials={},
            clean_cache=False,
            registry="localhost:5000",
            phase_suffix="",
            batch_build_metadata=None,
        )

    def test_single_arch_push_sets_image_digest(self):
        builder = self._builder()

        def fake_push(docker_image, registry, credentials, explicit_registry_image):
            builder.pushed_digests[explicit_registry_image] = DIGEST
            return explicit_registry_image

        with patch.object(
            builder, "_get_dockerfiles_for_model", return_value=["docker/dummy.ubuntu"]
        ), patch.object(
            builder,
            "build_image",
            return_value={"docker_image": "ci-dummy", "model": "dummy"},
        ), patch.object(
            builder, "_get_effective_gpu_architecture", return_value=""
        ), patch.object(
            builder, "_create_registry_image_name", return_value="localhost:5000/ci-dummy"
        ), patch.object(
            builder, "push_image", side_effect=fake_push
        ):
            results = self._run_single_arch(builder)

        assert results[0]["registry_image"] == "localhost:5000/ci-dummy"
        assert results[0]["image_digest"] == DIGEST
        # A push failure must still be recorded the way it is today.
        assert "push_error" not in results[0]

    def test_single_arch_push_without_digest_omits_key(self):
        builder = self._builder()

        with patch.object(
            builder, "_get_dockerfiles_for_model", return_value=["docker/dummy.ubuntu"]
        ), patch.object(
            builder,
            "build_image",
            return_value={"docker_image": "ci-dummy", "model": "dummy"},
        ), patch.object(
            builder, "_get_effective_gpu_architecture", return_value=""
        ), patch.object(
            builder, "_create_registry_image_name", return_value="localhost:5000/ci-dummy"
        ), patch.object(
            builder, "push_image", return_value="localhost:5000/ci-dummy"
        ):
            results = self._run_single_arch(builder)

        assert results[0]["registry_image"] == "localhost:5000/ci-dummy"
        assert "image_digest" not in results[0]

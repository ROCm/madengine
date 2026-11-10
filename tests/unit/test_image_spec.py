"""Unit tests for madengine.execution.image_spec and its DockerBuilder hooks."""

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml

from madengine.execution import image_spec
from madengine.execution.docker_builder import DockerBuilder
from madengine.execution.image_spec import ImageSpecError

MATRIX = {
    "bare_os_base": "ubuntu:24.04",
    "default_release": "7.2",
    "indexes": {"therock": "https://therock.example/whl/",
                "nightly": "https://nightly.example/whl/"},
    "releases": {
        "7.2": {"default_install": "therock", "index": "therock", "pip_spec": ">=7.2,<7.3"},
        "7.1": {"default_install": "legacy"},
    },
    "components": {
        "torch": {"2.10.0+rocm7.2": {
            "index": "therock", "requirement": "torch[device-{gpu_arch}]==2.10.0+rocm7.2",
            "requires": ["rocm>=7.2,<7.3"]}},
    },
}
SPEC = {"rocm": "7.2", "components": {"torch": "2.10.0+rocm7.2"}}


def write_mad(root, folder_spec=None):
    (root / "compat").mkdir(exist_ok=True)
    (root / "compat" / "rocm.yaml").write_text(yaml.safe_dump(MATRIX))
    if folder_spec is not None:
        (root / "scripts" / "w").mkdir(parents=True, exist_ok=True)
        (root / "scripts" / "w" / "image_spec.yaml").write_text(yaml.safe_dump(folder_spec))


class TestCardSpec:
    @pytest.fixture
    def root(self, tmp_path):
        write_mad(tmp_path, SPEC)
        return tmp_path

    def card(self, value):
        return {"name": "w/m", "_fs_rel_path": "w", "image_spec": value}

    def test_no_spec_keeps_card_on_its_dockerfile(self, root):
        assert image_spec.effective_image_spec({"name": "m"}, {}, str(root)) is None
        assert image_spec.effective_image_spec(self.card({}), {}, str(root)) is None

    def test_reference_to_folder_spec(self, root):
        spec = image_spec.effective_image_spec(self.card("${image_spec}"), {}, str(root))
        assert spec == SPEC

    def test_extends_with_card_overrides(self, root):
        spec = image_spec.effective_image_spec(
            self.card({"extends": "${image_spec}", "gpu_arch": "gfx942",
                       "components": {"vllm": "x@y"}, "order": ["vllm", "torch"]}),
            {}, str(root))
        assert spec == {"rocm": "7.2", "gpu_arch": "gfx942", "order": ["vllm", "torch"],
                        "components": {"torch": "2.10.0+rocm7.2", "vllm": "x@y"}}

    def test_reference_into_compat(self, root):
        spec = image_spec.effective_image_spec(
            self.card({"extends": "${image_spec}",
                       "rocm": "${compat.default_release}"}), {}, str(root))
        assert spec["rocm"] == "7.2"

    def test_context_merges_over_card(self, root):
        spec = image_spec.effective_image_spec(
            self.card("${image_spec}"), {"image_spec": {"gpu_arch": "gfx950"}}, str(root))
        assert spec["gpu_arch"] == "gfx950" and spec["rocm"] == "7.2"

    def test_missing_folder_spec_is_reported(self, tmp_path):
        write_mad(tmp_path)
        with pytest.raises(ImageSpecError, match="no image_spec.yaml next to"):
            image_spec.effective_image_spec(self.card("${image_spec}"), {}, str(tmp_path))

    def test_unknown_key_is_rejected(self, root):
        with pytest.raises(ImageSpecError, match="rocm_version"):
            image_spec.effective_image_spec(
                self.card({"rocm_version": "7.2"}), {}, str(root))


class TestResolve:
    @pytest.fixture(autouse=True)
    def root(self, tmp_path, monkeypatch):
        write_mad(tmp_path)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_generates_dockerfile_in_dependency_order(self, root):
        plan = image_spec.resolve(SPEC, "AMD", "gfx942", name="w/m")
        assert plan.dockerfile_prefix.startswith(".madengine/build/spec-w_m-")
        assert plan.resolved["order"] == ["rocm", "torch"]
        assert plan.resolved["pulled_in"] == {}
        text = (root / f"{plan.dockerfile_prefix}.ubuntu.amd.Dockerfile").read_text()
        assert text.index("# rocm 7.2") < text.index("# torch 2.10.0+rocm7.2")
        assert "rocm[libraries,devel,device-gfx942]>=7.2,<7.3" in text
        # only the indexes the plan installs from, never the whole matrix
        assert "nightly.example" not in text
        recorded = json.loads(plan.build_args[image_spec.BUILD_PLAN_ARG])
        assert recorded["digest"] == plan.digest()

    def test_spec_without_components(self, root):
        plan = image_spec.resolve({"rocm": "7.2"}, "AMD", name="m")
        assert plan.resolved["order"] == ["rocm"]
        assert plan.resolved["gpu_arch"] == "all"

    def test_unauthorized_build_is_rejected(self, root):
        with pytest.raises(ImageSpecError, match="not authorized"):
            image_spec.resolve({"rocm": "7.2", "components": {"torch": "9.9"}},
                               "AMD", name="m")

    def test_amd_only(self):
        with pytest.raises(ImageSpecError, match="vendor"):
            image_spec.resolve(SPEC, "OTHER")

    def test_digest_tracks_the_plan(self):
        a = image_spec.resolve(SPEC, "AMD", "gfx942")
        b = image_spec.resolve(SPEC, "AMD", "gfx950")
        assert a.digest() == image_spec.resolve(SPEC, "AMD", "gfx942").digest()
        assert a.digest() != b.digest()

    def test_missing_matrix_points_at_mad_root(self, tmp_path):
        with pytest.raises(ImageSpecError, match=r"compat[\\/]rocm\.yaml"):
            image_spec.load_matrix(str(tmp_path / "nowhere"))


class TestDockerBuilderHooks:
    @pytest.fixture
    def mad_root(self, tmp_path, monkeypatch):
        write_mad(tmp_path)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def _builder(self, ctx, sh):
        context = MagicMock()
        context.ctx = ctx
        console = MagicMock()
        console.sh.side_effect = sh
        builder = DockerBuilder(context, console)
        builder.rich_console = MagicMock()
        return builder, console

    def test_spec_card_builds_the_generated_dockerfile(self, mad_root):
        def sh(cmd, **kw):
            if cmd.startswith("ls .madengine/build/spec-m-"):
                prefix = cmd.split()[1].rstrip("*").rstrip(".")
                return f"{prefix}.ubuntu.amd.Dockerfile"
            return ""
        builder, console = self._builder({}, sh)
        builder.context.filter = lambda d: d
        builder._get_dockerfiles_for_model(
            {"name": "m", "dockerfile": "docker/ignored", "image_spec": {}}
        )
        # An empty card spec is "no spec": the hand-written Dockerfile is used.
        assert console.sh.call_args_list[0].args[0].startswith("ls docker/ignored.*")
        found = builder._get_dockerfiles_for_model({"name": "m", "image_spec": SPEC})
        assert len(found) == 1 and found[0].startswith(".madengine/build/spec-m-")
        assert (mad_root / found[0]).exists()

    def test_bad_spec_fails_instead_of_skipping(self, mad_root):
        builder, _ = self._builder({}, lambda cmd, **kw: "")
        with pytest.raises(ImageSpecError):
            builder._get_dockerfiles_for_model(
                {"name": "m", "image_spec": {"bogus": 1}}
            )

    def test_plan_and_image_info_are_recorded(self, mad_root):
        info = {"rocm_version": "7.2.0", "packages": {"rocm": "7.2.0"}}

        def sh(cmd, **kw):
            if cmd.startswith("docker run --rm --entrypoint cat"):
                return json.dumps(info)
            return ""

        ctx = {"docker_build_arg": {"BASE_DOCKER": "rocm/dev-ubuntu-24.04:7.2"}}
        builder, console = self._builder(ctx, sh)
        prefix = image_spec.dockerfile_prefix(SPEC, name="m")
        with patch.object(builder, "_local_image_size_bytes", return_value=None):
            build_info = builder.build_image(
                {"name": "m", "image_spec": SPEC}, f"{prefix}.ubuntu.amd.Dockerfile")

        build_cmd = next(c.args[0] for c in console.sh.call_args_list
                         if c.args[0].startswith("docker build"))
        # context docker_build_arg still wins (BASE_DOCKER is an ARG of the file)
        assert "--build-arg BASE_DOCKER=rocm/dev-ubuntu-24.04:7.2" in build_cmd
        assert "MAD_BUILD_PLAN=" in build_cmd
        plan = build_info["build_plan"]
        assert plan["vendor"] == "AMD"
        assert plan["resolved"]["order"] == ["rocm", "torch"]
        assert build_info["image_info"] == info

    def test_legacy_card_runs_no_probe_unless_asked(self, mad_root):
        def sh(cmd, **kw):
            if cmd.startswith("docker run --rm --entrypoint python3"):
                return '{"python": "3.12.3"}'
            return ""

        card = {"name": "m", "dockerfile": "docker/m"}
        probed = {"python": "3.12.3"}
        for ctx, expected in (({}, None), ({"image_probe": True}, probed)):
            builder, console = self._builder(ctx, sh)
            with patch.object(builder, "_local_image_size_bytes", return_value=None):
                build_info = builder.build_image(card, "docker/m.ubuntu.amd.Dockerfile")
            ran = [c.args[0] for c in console.sh.call_args_list
                   if c.args[0].startswith("docker run")]
            assert bool(ran) == bool(expected)
            assert build_info.get("image_info") == expected
            assert "build_plan" not in build_info

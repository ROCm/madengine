"""Unit tests for madengine.execution.image_spec and its DockerBuilder hooks."""

import json
from unittest.mock import MagicMock, patch

import pytest

from madengine.execution import image_spec
from madengine.execution.docker_builder import DockerBuilder
from madengine.execution.image_spec import ImageSpecError

MIGRAPHX_REPO = "https://github.com/ROCm/AMDMIGraphX"

ROCM = {
    "default_version": "7.2",
    "bare_os_base": "ubuntu:24.04",
    "releases": {
        "7.2": {
            "default_install": "therock",
            "therock_index_url": "https://therock.example/whl/",
            "rocm_pip_spec": ">=7.2,<7.3",
            "legacy_installer_url": "https://repo.example/amdgpu-install_7.2.deb",
            "torch_index_url": {"therock": "https://therock.example/whl/"},
        },
        "7.1": {"therock_index_url": "https://therock.example/7.1/"},
    },
}

CUDA = {
    "torch": [
        {"Torch": "2.6.0+cu118", "Torchvision": "0.21.0+cu118", "Torchaudio": "2.6.0",
         "ExtraIndexURL": "https://download.pytorch.org/whl/cu118/", "CUDA": "11.8"},
        {"Torch": "2.6.0+cu124", "Torchvision": "0.21.0+cu124", "Torchaudio": "2.6.0",
         "ExtraIndexURL": "https://download.pytorch.org/whl/cu124/", "CUDA": "12.4"},
        {"Torch": "2.6.0+cu126", "Torchvision": "0.21.0+cu126", "Torchaudio": "2.6.0",
         "ExtraIndexURL": "https://download.pytorch.org/whl/cu126/", "CUDA": "12.6"},
        {"Torch": "2.6.0+cpu", "CUDA": None},
    ],
    "base_images": [
        {"Tag": "12.4.1-cudnn-runtime-ubuntu22.04", "CUDA": "12.4.1", "CuDNN": "9",
         "IsDevel": False, "Ubuntu": "22.04"},
        {"Tag": "12.4.1-cudnn-devel-ubuntu22.04", "CUDA": "12.4.1", "CuDNN": "9",
         "IsDevel": True, "Ubuntu": "22.04"},
        {"Tag": "12.4.1-cudnn-devel-ubuntu20.04", "CUDA": "12.4.1", "CuDNN": "9",
         "IsDevel": True, "Ubuntu": "20.04"},
        {"Tag": "12.6.3-cudnn-devel-ubuntu24.04", "CUDA": "12.6.3", "CuDNN": "9",
         "IsDevel": True, "Ubuntu": "24.04"},
    ],
}


class TestEffectiveSpec:
    def test_no_spec_keeps_card_on_its_dockerfile(self):
        assert image_spec.effective_image_spec({"name": "m"}, {}) is None

    def test_context_deep_merges_over_card(self):
        card = {"image_spec": {"rocm": {"version": "7.1", "install": "legacy"}}}
        ctx = {"image_spec": {"rocm": {"version": "7.2"}, "torch": "2.10"}}
        spec = image_spec.effective_image_spec(card, ctx)
        assert spec == {
            "rocm": {"version": "7.2", "install": "legacy"},
            "torch": "2.10",
        }

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ImageSpecError, match="rocm_version"):
            image_spec.effective_image_spec({"image_spec": {"rocm_version": "7.2"}}, {})

    def test_non_object_is_rejected(self):
        with pytest.raises(ImageSpecError, match="must be an object"):
            image_spec.effective_image_spec({}, {"image_spec": "rocm7.2"})


class TestResolveAmd:
    def test_therock_bare_os(self):
        spec = {
            "components": [
                {"name": "migraphx", "url": "https://x/migraphx.whl", "sha256": "ab"},
                {"name": "cuda-only", "nvidia": {"url": "https://x/cu.whl"}},
            ],
            "sources": [{"repo": MIGRAPHX_REPO, "commit": "abc1234"}],
            "pip": ["numpy==1.26.4"],
        }
        plan = image_spec.resolve(spec, "AMD", "gfx942", matrix=ROCM)
        args = plan.build_args
        assert plan.dockerfile_prefix == "docker/global_univ2"
        assert args["INSTALL_TYPE"] == "therock"
        assert args["ROCM_RELEASE_URL_THEROCK"] == "https://therock.example/whl/"
        assert args["BASE_DOCKER"] == "ubuntu:24.04"
        assert args["DOCKER_TYPE"] == "BARE_OS"
        assert args["ROCM_EXTRAS"] == "rocm[libraries,devel,device-gfx942]>=7.2,<7.3"
        assert args["GPU_ARCH"] == "gfx942"
        assert args["COMPONENT_RELEASE_URL_THEROCK"] == "https://x/migraphx.whl"
        assert args["COMPONENT_SHA256"] == "ab"
        assert args["COMPONENT_REPOS"] == f"{MIGRAPHX_REPO}/commit/abc1234"
        assert args["PIP_PREREQS_EXTRA"] == "numpy==1.26.4"
        assert [c["name"] for c in plan.resolved["components"]] == ["migraphx"]
        recorded = json.loads(args[image_spec.BUILD_PLAN_ARG])
        assert recorded["digest"] == plan.digest()
        assert recorded["resolved"]["rocm_version"] == "7.2"

    def test_legacy_with_explicit_base_autodetects(self):
        spec = {"base": "rocm/pytorch:latest", "rocm": {"install": "legacy"},
                "components": [{"name": "m", "url": "https://x/m.deb"}]}
        args = image_spec.resolve(spec, "AMD", matrix=ROCM).build_args
        legacy = ROCM["releases"]["7.2"]["legacy_installer_url"]
        assert args["ROCM_RELEASE_URL_LEGACY"] == legacy
        assert args["COMPONENT_RELEASE_URL_LEGACY"] == "https://x/m.deb"
        assert args["BASE_DOCKER"] == "rocm/pytorch:latest"
        assert args["DOCKER_TYPE"] == "auto"
        assert "GPU_ARCH" not in args

    def test_torch_uses_matrix_index(self):
        args = image_spec.resolve({"torch": "2.10"}, "AMD", matrix=ROCM).build_args
        assert args["TORCH_PACKAGES"] == "torch[device-all]==2.10.* rocm>=7.2,<7.3"
        assert args["ROCM_EXTRAS"] == "rocm[libraries,devel,device-all]>=7.2,<7.3"
        assert args["TORCH_INDEX_URL"] == "https://therock.example/whl/"

    def test_torch_without_index_for_install_type_fails(self):
        with pytest.raises(ImageSpecError, match="no torch index"):
            image_spec.resolve({"torch": "2.10", "rocm": {"install": "legacy"}},
                               "AMD", matrix=ROCM)

    def test_unknown_rocm_version_lists_known(self):
        with pytest.raises(ImageSpecError, match="known: 7.1, 7.2"):
            image_spec.resolve({"rocm": {"version": "6.0"}}, "AMD", matrix=ROCM)

    def test_source_commit_must_be_a_sha(self):
        with pytest.raises(ImageSpecError, match="commit SHA"):
            image_spec.resolve({"sources": [{"repo": "r", "commit": "main"}]},
                               "AMD", matrix=ROCM)


class TestResolveNvidia:
    def test_highest_cuda_not_above_request_and_devel_base(self):
        plan = image_spec.resolve(
            {"torch": "2.6", "cuda": {"version": "12.5"}, "python": "3.12"},
            "NVIDIA", matrix=CUDA,
        )
        args = plan.build_args
        assert args["TORCH_PACKAGES"] == (
            "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"
        )
        assert args["TORCH_INDEX_URL"] == "https://download.pytorch.org/whl/cu124/"
        assert args["BASE_DOCKER"] == "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
        assert args["PYTHON_VERSION"] == "3.12"
        assert plan.resolved["cuda_version"] == "12.4"
        assert plan.resolved["cuda_requested"] == "12.5"

    def test_torch_alone_picks_latest_cuda(self):
        plan = image_spec.resolve({"torch": "2.6"}, "NVIDIA", matrix=CUDA)
        assert plan.resolved["cuda_version"] == "12.6"
        base = plan.build_args["BASE_DOCKER"]
        assert base == "nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04"

    def test_needs_cuda_or_torch(self):
        with pytest.raises(ImageSpecError, match="cuda.version or torch"):
            image_spec.resolve({}, "NVIDIA", matrix=CUDA)

    def test_vendor_components_only(self):
        spec = {"cuda": {"version": "12.4"},
                "components": [{"name": "a", "amd": {"url": "https://x/a.whl"}},
                               {"name": "b", "url": "https://x/b.whl"}]}
        args = image_spec.resolve(spec, "NVIDIA", matrix=CUDA).build_args
        assert args["COMPONENT_WHEEL_URLS"] == "https://x/b.whl"


def test_digest_tracks_build_args():
    a = image_spec.resolve({"rocm": {"version": "7.2"}}, "AMD", matrix=ROCM)
    b = image_spec.resolve({"rocm": {"version": "7.1"}}, "AMD", matrix=ROCM)
    assert a.digest() == image_spec.resolve({}, "AMD", matrix=ROCM).digest()
    assert a.digest() != b.digest()


def test_missing_matrix_points_at_mad_root(tmp_path):
    with pytest.raises(ImageSpecError, match=r"compat[\\/]rocm\.json"):
        image_spec.load_matrix("AMD", str(tmp_path))


def test_unsupported_vendor():
    with pytest.raises(ImageSpecError, match="vendor"):
        image_spec.resolve({}, "INTEL", matrix={})


class TestDockerBuilderHooks:
    @pytest.fixture
    def mad_root(self, tmp_path, monkeypatch):
        (tmp_path / "compat").mkdir()
        (tmp_path / "compat" / "rocm.json").write_text(json.dumps(ROCM))
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

    def test_spec_card_selects_backend_dockerfile(self, mad_root):
        def sh(cmd, **kw):
            if cmd.startswith("ls docker/global_univ2.*"):
                return ("docker/global_univ2.ubuntu.amd.Dockerfile\n"
                        "docker/global_univ2.ubuntu.nvidia.Dockerfile")
            return ""
        builder, console = self._builder({}, sh)
        builder.context.filter = lambda d: d
        found = builder._get_dockerfiles_for_model(
            {"name": "m", "dockerfile": "docker/ignored", "image_spec": {}}
        )
        # An empty card spec is "no spec": the hand-written Dockerfile is used.
        assert console.sh.call_args_list[0].args[0].startswith("ls docker/ignored.*")
        found = builder._get_dockerfiles_for_model(
            {"name": "m", "image_spec": {"rocm": {"version": "7.2"}}}
        )
        assert found == ["docker/global_univ2.ubuntu.amd.Dockerfile",
                         "docker/global_univ2.ubuntu.nvidia.Dockerfile"]

    def test_bad_spec_fails_instead_of_skipping(self, mad_root):
        builder, _ = self._builder({}, lambda cmd, **kw: "")
        with pytest.raises(ImageSpecError):
            builder._get_dockerfiles_for_model(
                {"name": "m", "image_spec": {"bogus": 1}}
            )

    def test_spec_args_are_lowest_precedence_and_recorded(self, mad_root):
        info = {"rocm_version": "7.2.0", "packages": {"rocm": "7.2.0"}}

        def sh(cmd, **kw):
            if cmd.startswith("docker run --rm --entrypoint cat"):
                return json.dumps(info)
            return ""

        ctx = {"docker_build_arg": {"BASE_DOCKER": "rocm/dev-ubuntu-24.04:7.2"}}
        builder, console = self._builder(ctx, sh)
        with patch.object(builder, "_local_image_size_bytes", return_value=None):
            build_info = builder.build_image(
                {"name": "m", "image_spec": {"rocm": {"version": "7.2"}},
                 "docker_build_arg": {"INSTALL_TYPE": "legacy"}},
                "docker/global_univ2.ubuntu.amd.Dockerfile",
            )

        build_cmd = next(c.args[0] for c in console.sh.call_args_list
                         if c.args[0].startswith("docker build"))
        assert "--build-arg BASE_DOCKER=rocm/dev-ubuntu-24.04:7.2" in build_cmd
        assert "--build-arg INSTALL_TYPE=legacy" in build_cmd
        assert "INSTALL_TYPE=therock" not in build_cmd
        assert "--build-arg DOCKER_TYPE=BARE_OS" in build_cmd
        assert "MAD_BUILD_PLAN=" in build_cmd
        plan = build_info["build_plan"]
        assert plan["vendor"] == "AMD"
        assert plan["overridden_args"] == ["BASE_DOCKER", "INSTALL_TYPE"]
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

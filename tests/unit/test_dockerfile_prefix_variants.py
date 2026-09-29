"""A card's dockerfile is a prefix for its OS/vendor variants, not for every file
that starts with it.

`ls <prefix>.*` also matched model-specific siblings: every card on
docker/vllm_disagg_inference built vllm_disagg_inference.glmv5.1 and .kimik3 as
well, and ran in the GLM-5.1 image.
"""

from unittest.mock import MagicMock

import pytest

from madengine.execution.docker_builder import DockerBuilder, _is_prefix_variant


@pytest.mark.parametrize(
    "path,want",
    [
        ("docker/x.Dockerfile", True),
        ("docker/x.ubuntu.amd.Dockerfile", True),
        ("docker/x.ubuntu.nvidia.Dockerfile", True),
        ("docker/x.glmv5.1.ubuntu.amd.Dockerfile", False),
        ("docker/x.kimik3.ubuntu.amd.Dockerfile", False),
        ("docker/xy.ubuntu.amd.Dockerfile", False),
        ("docker/x.ubuntu.amd.Dockerfile.bak", False),
    ],
)
def test_only_os_vendor_variants_belong_to_a_prefix(path, want):
    assert _is_prefix_variant("docker/x", path) is want


def test_a_model_specific_prefix_still_finds_its_own_file():
    assert _is_prefix_variant(
        "docker/vllm_disagg_inference.glmv5.1",
        "docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile",
    )


def test_base_card_builds_only_the_base_image():
    builder = DockerBuilder.__new__(DockerBuilder)
    listing = "\n".join([
        "docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile",
        "docker/vllm_disagg_inference.kimik3.ubuntu.amd.Dockerfile",
        "docker/vllm_disagg_inference.ubuntu.amd.Dockerfile",
    ])
    builder.console = MagicMock()
    builder.console.sh.side_effect = lambda cmd, **kw: listing if cmd.startswith("ls ") else ""
    builder.context = MagicMock()
    builder.context.filter.side_effect = lambda d: d
    builder.rich_console = MagicMock()
    got = builder._get_dockerfiles_for_model(
        {"name": "pyt_vllm_disagg_nixl_llama-3.3-70b-fp8", "dockerfile": "docker/vllm_disagg_inference"}
    )
    assert got == ["docker/vllm_disagg_inference.ubuntu.amd.Dockerfile"]

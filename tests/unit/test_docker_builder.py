"""Unit tests for madengine.execution.docker_builder (DockerBuilder).

Currently covers registry image naming for single-arch builds (credentials → repository:tag).
"""

from unittest.mock import MagicMock

import pytest

from madengine.execution.docker_builder import DockerBuilder


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

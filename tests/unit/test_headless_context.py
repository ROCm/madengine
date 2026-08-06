#!/usr/bin/env python3
"""
Tests for building a runtime context on a node without GPUs.

A SLURM submit node is usually a login node: no GPUs, no ROCm, no /dev/dri. Runtime
context initialisation used to raise there — "Unable to determine gpu vendor" — even though
the job being prepared runs somewhere else entirely. When a cluster profile states what the
compute nodes are, those facts are the honest answer, and probing the login node is not.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json

import pytest

from madengine.core.context import Context
from madengine.deployment.presets.cluster_profiles import PROFILE_PATH_ENV


@pytest.fixture
def profile_dir(tmp_path, monkeypatch):
    """A directory of site profiles on the search path."""
    monkeypatch.setenv(PROFILE_PATH_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def headless_profile(profile_dir):
    """A profile describing compute nodes this machine is not one of."""
    (profile_dir / "compute.json").write_text(
        json.dumps(
            {
                "facts": {
                    "gpu_vendor": "AMD",
                    "gpus_per_node": 8,
                    "gpu_architecture": "gfx942",
                    "gpu_product_name": "AMD Instinct MI300X",
                    "hip_version": "6.4",
                }
            }
        )
    )
    return "compute"


@pytest.fixture
def no_local_gpu(monkeypatch):
    """A machine where every GPU probe fails, like a login node."""
    def unavailable(*args, **kwargs):
        raise RuntimeError("no GPU on this node")

    for name in (
        "get_gpu_vendor",
        "get_system_ngpus",
        "get_system_gpu_architecture",
        "get_system_gpu_product_name",
        "get_system_hip_version",
        "get_docker_gpus",
        "get_gpu_renderD_nodes",
    ):
        monkeypatch.setattr(Context, name, unavailable)


def build_context(profile, **kwargs):
    """Construct a runtime Context that selects a cluster profile."""
    return Context(additional_context=repr({"cluster_profile": profile, **kwargs}))


class TestHeadlessSubmitNode:
    """A node with no GPUs can still prepare a job for nodes that have them."""

    def test_context_builds_without_local_gpus(self, headless_profile, no_local_gpu):
        """This is the failure the profile exists to remove."""
        context = build_context(headless_profile)

        assert context.ctx["gpu_vendor"] == "AMD"

    def test_facts_populate_the_container_environment(self, headless_profile, no_local_gpu):
        """The values a container would have been given by probing come from the profile."""
        context = build_context(headless_profile)
        env = context.ctx["docker_env_vars"]

        assert env["MAD_SYSTEM_NGPUS"] == 8
        assert env["MAD_SYSTEM_GPU_ARCHITECTURE"] == "gfx942"
        assert env["MAD_SYSTEM_GPU_PRODUCT_NAME"] == "AMD Instinct MI300X"
        assert env["MAD_SYSTEM_HIP_VERSION"] == "6.4"

    def test_architecture_reaches_build_args(self, headless_profile, no_local_gpu):
        """A build kicked off from the submit node targets the compute nodes' architecture."""
        context = build_context(headless_profile)

        assert context.ctx["docker_build_arg"]["MAD_SYSTEM_GPU_ARCHITECTURE"] == "gfx942"

    def test_machine_specific_probes_are_not_invented(self, headless_profile, no_local_gpu):
        """Render node numbers describe one machine, so the submit node reports none."""
        context = build_context(headless_profile)

        assert context.ctx["gpu_renderDs"] is None

    def test_profile_from_the_slurm_block(self, profile_dir, headless_profile, no_local_gpu):
        """A manifest names the profile under slurm, which is where deployments read it."""
        context = Context(
            additional_context=repr({"slurm": {"cluster_profile": headless_profile}})
        )

        assert context.ctx["gpu_vendor"] == "AMD"

    def test_partial_facts_still_probe_what_they_omit(self, profile_dir, no_local_gpu):
        """A profile that names only the vendor does not silently invent an architecture."""
        (profile_dir / "vendor-only.json").write_text(
            json.dumps({"facts": {"gpu_vendor": "AMD"}})
        )

        context = build_context("vendor-only")

        assert context.ctx["docker_env_vars"]["MAD_SYSTEM_GPU_ARCHITECTURE"] == ""


class TestProbingIsStillTheDefault:
    """Nothing changes for a node that can answer for itself."""

    def test_no_profile_means_detection_failure_is_fatal(self, no_local_gpu):
        """Without a profile there is no second source, so the run stops."""
        with pytest.raises(RuntimeError, match="GPU detection failed"):
            Context()

    def test_probed_values_win_when_the_node_has_gpus(self, headless_profile, monkeypatch):
        """On a compute node the local answer describes the machine that will run the work."""
        monkeypatch.setattr(Context, "get_gpu_vendor", lambda self: "AMD")
        monkeypatch.setattr(Context, "get_system_ngpus", lambda self: 4)
        monkeypatch.setattr(Context, "get_system_gpu_architecture", lambda self: "gfx950")
        monkeypatch.setattr(Context, "get_system_gpu_product_name", lambda self: "local")
        monkeypatch.setattr(Context, "get_system_hip_version", lambda self: "7.0")
        monkeypatch.setattr(Context, "get_docker_gpus", lambda self: "all")
        monkeypatch.setattr(Context, "get_gpu_renderD_nodes", lambda self: [128, 129])

        context = build_context(headless_profile)

        # The profile speaks for the cluster, this node speaks for itself, and it is the
        # one about to run the container — including on a heterogeneous partition where
        # the profile's architecture would be wrong.
        assert context.ctx["docker_env_vars"]["MAD_SYSTEM_GPU_ARCHITECTURE"] == "gfx950"
        assert context.ctx["docker_env_vars"]["MAD_SYSTEM_NGPUS"] == 4
        assert context.ctx["gpu_renderDs"] == [128, 129]

    def test_user_context_still_wins_over_facts(self, headless_profile, no_local_gpu):
        """An explicit override beats both the profile and the node."""
        context = build_context(headless_profile, gpu_vendor="NVIDIA")

        assert context.ctx["gpu_vendor"] == "NVIDIA"

    def test_unknown_profile_is_reported(self, profile_dir, no_local_gpu):
        """A mistyped profile name must not look like "no profile"."""
        from madengine.core.errors import ValidationError

        with pytest.raises((ValidationError, RuntimeError)):
            build_context("not-a-profile")

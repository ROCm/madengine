#!/usr/bin/env python3
"""
Tests for per-cluster fact profiles.

The behaviour under test is the separation the profiles exist for: the shape presets say
how big the job is, the cluster profile says what the cluster is, and the user has the last
word on both. The regression that motivated this is concrete — the shipped multi-node preset
puts every run on TCP over eth0, which is wrong on any RoCE cluster and on any cluster whose
management interface is not called eth0.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json

import pytest

from madengine.core.errors import ValidationError
from madengine.deployment.config_loader import ConfigLoader
from madengine.deployment.presets.cluster_profiles import (
    CLUSTER_DIR,
    PROFILE_NAME_ENV,
    PROFILE_PATH_ENV,
    apply_cluster_profiles,
    available_profiles,
    load_profile,
    resolve_profile_path,
    selected_profiles,
)


@pytest.fixture
def site_profiles(tmp_path, monkeypatch):
    """A directory of site profiles on the search path."""
    monkeypatch.setenv(PROFILE_PATH_ENV, str(tmp_path))
    return tmp_path


def write_profile(directory, name, body):
    """Write a profile file and return its path."""
    path = directory / f"{name}.json"
    path.write_text(json.dumps(body))
    return path


class TestBundledProfiles:
    """The archetypes madengine ships."""

    def test_bundled_profiles_are_discoverable(self):
        """A user who mistypes a name gets the list, so the list has to be real."""
        assert "roce-broadcom-thor2" in available_profiles()
        assert "no-gpu-gres" in available_profiles()

    @pytest.mark.parametrize(
        "name", sorted(path.stem for path in CLUSTER_DIR.glob("*.json"))
    )
    def test_bundled_profile_matches_the_schema(self, name):
        """Every shipped profile validates; a typo here breaks a cluster, not a test."""
        assert load_profile(name)

    def test_roce_profile_enables_rdma(self):
        """The point of the RoCE archetype is that IB is not disabled."""
        profile = load_profile("roce-broadcom-thor2")

        assert profile["env_vars"]["NCCL_IB_DISABLE"] == "0"
        assert profile["env_vars"]["NCCL_IB_HCA"] == "bnxt_re"

    def test_no_gres_profile_is_scheduler_only(self):
        """Whether GRES exists says nothing about the fabric, so the profile says nothing."""
        profile = load_profile("no-gpu-gres")

        assert profile["slurm"]["skip_gpus_directive"] is True
        assert "env_vars" not in profile

    def test_bundled_profiles_name_no_cluster(self):
        """Archetypes describe hardware; a site's own cluster stays out of the repository."""
        for path in CLUSTER_DIR.glob("*.json"):
            body = path.read_text()
            assert "partition" not in body, path.name
            assert "account" not in body, path.name


class TestProfileResolution:
    """Finding the file behind a reference."""

    def test_site_directory_is_searched_first(self, site_profiles):
        """A site can shadow an archetype it disagrees with."""
        override = write_profile(site_profiles, "roce-broadcom-thor2", {"facts": {"fabric": "roce"}})

        assert resolve_profile_path("roce-broadcom-thor2") == override

    def test_path_reference_is_taken_as_a_path(self, tmp_path):
        """A profile kept next to the manifests needs no installation."""
        path = write_profile(tmp_path, "our-cluster", {"facts": {"gpus_per_node": 8}})

        assert resolve_profile_path(str(path)) == path

    def test_unknown_name_lists_what_exists(self):
        """The error a mistyped name produces should be the answer to it."""
        with pytest.raises(ValidationError) as exc_info:
            load_profile("no-such-cluster")

        message = str(exc_info.value)
        assert "no-such-cluster" in message

    def test_missing_path_is_reported(self, tmp_path):
        """A path that does not exist fails as a path, not as an unknown name."""
        with pytest.raises(ValidationError) as exc_info:
            load_profile(str(tmp_path / "absent.json"))

        assert "absent.json" in str(exc_info.value)

    def test_malformed_profile_is_reported(self, site_profiles):
        """Broken JSON names the file it could not parse."""
        (site_profiles / "broken.json").write_text("{not json")

        with pytest.raises(ValidationError) as exc_info:
            load_profile("broken")

        assert "broken.json" in str(exc_info.value)

    def test_profile_violating_the_schema_is_rejected(self, site_profiles):
        """A profile is validated before it can misconfigure a run."""
        write_profile(site_profiles, "bad", {"env_vars": {"NCCL_IB_HCA": ["mlx5"]}})

        with pytest.raises(ValidationError) as exc_info:
            load_profile("bad")

        assert "NCCL_IB_HCA" in str(exc_info.value)

    def test_unknown_top_level_key_is_rejected(self, site_profiles):
        """Facts land where the loader looks for them, or not at all."""
        write_profile(site_profiles, "bad", {"enviroment": {"NCCL_IB_HCA": "mlx5"}})

        with pytest.raises(ValidationError):
            load_profile("bad")


class TestSelection:
    """Which profiles a configuration asks for."""

    def test_single_name(self):
        """The common case."""
        config = {"slurm": {"cluster_profile": "ethernet-tcp"}}

        assert selected_profiles(config) == ["ethernet-tcp"]

    def test_list_of_names(self):
        """Orthogonal facts are separate profiles rather than a combinatorial file set."""
        config = {"slurm": {"cluster_profile": ["roce-broadcom-thor2", "no-gpu-gres"]}}

        assert selected_profiles(config) == ["roce-broadcom-thor2", "no-gpu-gres"]

    def test_environment_names_a_profile(self, monkeypatch):
        """A site can point every run at its profile without touching manifests."""
        monkeypatch.setenv(PROFILE_NAME_ENV, "ethernet-tcp")

        assert selected_profiles({"slurm": {}}) == ["ethernet-tcp"]

    def test_manifest_wins_over_environment(self, monkeypatch):
        """An explicit choice in the manifest is not overridden by the environment."""
        monkeypatch.setenv(PROFILE_NAME_ENV, "ethernet-tcp")
        config = {"slurm": {"cluster_profile": "infiniband-mellanox"}}

        assert selected_profiles(config) == ["infiniband-mellanox"]

    def test_nothing_selected(self):
        """Profiles are opt-in; without one, nothing changes."""
        assert selected_profiles({"slurm": {}}) == []


class TestMerging:
    """How facts land on a configuration."""

    def test_profile_overrides_shape_preset(self, site_profiles):
        """This is the regression: a RoCE cluster must not inherit NCCL_IB_DISABLE=1."""
        write_profile(site_profiles, "ours", {"env_vars": {"NCCL_IB_DISABLE": "0"}})
        config = {"env_vars": {"NCCL_IB_DISABLE": "1", "NCCL_DEBUG": "WARN"}}

        merged = apply_cluster_profiles(config, ["ours"])

        assert merged["env_vars"]["NCCL_IB_DISABLE"] == "0"
        assert merged["env_vars"]["NCCL_DEBUG"] == "WARN"

    def test_null_removes_an_inherited_variable(self, site_profiles):
        """An interface name that does not exist here is worse than none."""
        write_profile(site_profiles, "ours", {"env_vars": {"NCCL_SOCKET_IFNAME": None}})
        config = {"env_vars": {"NCCL_SOCKET_IFNAME": "eth0"}}

        merged = apply_cluster_profiles(config, ["ours"])

        assert "NCCL_SOCKET_IFNAME" not in merged["env_vars"]

    def test_profiles_merge_left_to_right(self, site_profiles):
        """Later profiles refine earlier ones."""
        write_profile(site_profiles, "fabric", {"env_vars": {"NCCL_IB_HCA": "mlx5"}})
        write_profile(site_profiles, "site", {"env_vars": {"NCCL_IB_HCA": "mlx5_0"}})

        merged = apply_cluster_profiles({}, ["fabric", "site"])

        assert merged["env_vars"]["NCCL_IB_HCA"] == "mlx5_0"

    def test_documentation_keys_do_not_leak_into_config(self, site_profiles):
        """`_description` describes the file, not the cluster."""
        write_profile(site_profiles, "ours", {"_description": "ours", "facts": {"fabric": "roce"}})

        merged = apply_cluster_profiles({}, ["ours"])

        assert "_description" not in merged

    def test_input_configuration_is_not_mutated(self, site_profiles):
        """Merging returns a new configuration; callers keep theirs."""
        write_profile(site_profiles, "ours", {"env_vars": {"NCCL_IB_DISABLE": "0"}})
        config = {"env_vars": {"NCCL_IB_DISABLE": "1"}}

        apply_cluster_profiles(config, ["ours"])

        assert config["env_vars"]["NCCL_IB_DISABLE"] == "1"


class TestConfigLoaderIntegration:
    """The layer as the deployment path sees it."""

    def test_cluster_facts_beat_the_multi_node_preset(self):
        """A two-node RoCE run keeps RDMA on, which the shipped preset would have disabled."""
        config = ConfigLoader.load_slurm_config(
            {"slurm": {"nodes": 2, "cluster_profile": "roce-broadcom-thor2"}}
        )

        assert config["env_vars"]["NCCL_IB_DISABLE"] == "0"
        assert config["env_vars"]["NCCL_IB_HCA"] == "bnxt_re"
        assert "NCCL_SOCKET_IFNAME" not in config["env_vars"]

    def test_user_configuration_still_wins(self):
        """A profile is a default, not a policy."""
        config = ConfigLoader.load_slurm_config(
            {
                "slurm": {"nodes": 2, "cluster_profile": "roce-broadcom-thor2"},
                "env_vars": {"NCCL_IB_HCA": "bnxt_re0"},
            }
        )

        assert config["env_vars"]["NCCL_IB_HCA"] == "bnxt_re0"

    def test_scheduler_fact_reaches_the_slurm_block(self):
        """skip_gpus_directive is what keeps sbatch from rejecting the job."""
        config = ConfigLoader.load_slurm_config(
            {"slurm": {"nodes": 2, "cluster_profile": ["roce-broadcom-thor2", "no-gpu-gres"]}}
        )

        assert config["slurm"]["skip_gpus_directive"] is True
        assert config["env_vars"]["NCCL_IB_DISABLE"] == "0"

    def test_no_profile_leaves_behaviour_unchanged(self):
        """Existing deployments see exactly what they saw before."""
        config = ConfigLoader.load_slurm_config({"slurm": {"nodes": 2}})

        assert config["env_vars"]["NCCL_IB_DISABLE"] == "1"
        assert config["env_vars"]["NCCL_SOCKET_IFNAME"] == "eth0"

    def test_unknown_profile_fails_the_run(self):
        """Better a startup error than a silent fallback to the wrong fabric."""
        with pytest.raises(ValidationError):
            ConfigLoader.load_slurm_config(
                {"slurm": {"nodes": 2, "cluster_profile": "not-a-cluster"}}
            )

"""
SLURM preset env_vars reach templated launchers, never a slurm_multi card.

load_slurm_config merges profiles/multi-node.json's env_vars whenever nodes > 1,
and once merged they looked exactly like the user's. _prepare_slurm_multi_script
then exported them all into the wrapper, so a card run under madengine got
HSA_ENABLE_SDMA=0, NCCL_IB_DISABLE=1 and NCCL_SOCKET_IFNAME=eth0 that the same
card submitted with plain sbatch never saw -- and MAD's run_xPyD_models.slurm forwards
`-e K=${K:-default}`, so the preset beat the card's own SDMA=1.
"""

import json
from pathlib import Path

import pytest

from madengine.deployment.base import DeploymentConfig
from madengine.deployment.config_loader import PRESET_ENV_KEYS, ConfigLoader
from madengine.deployment.slurm import SlurmDeployment

MULTI_NODE_PRESET_ENV = ConfigLoader.load_preset("slurm/profiles/multi-node.json")["env_vars"]
DEFAULTS_PRESET_ENV = ConfigLoader.load_preset("slurm/defaults.json")["env_vars"]
ALL_PRESET_KEYS = set(MULTI_NODE_PRESET_ENV) | set(DEFAULTS_PRESET_ENV)

CARD = {
    "name": "sglang/pyt_disagg_card",
    "scripts": "scripts/card/run.slurm",
    "n_gpus": "8",
    "tags": ["pyt"],
    "args": "",
    "distributed": {"launcher": "slurm_multi", "nnodes": 2},
    "env_vars": {"DOCKER_IMAGE_NAME": "rocm/card:tag", "CARD_ONLY": "from-card"},
}


def _deployment(tmp_path: Path, launcher="slurm_multi", env_vars=None, card_env=None):
    card = json.loads(json.dumps(CARD))
    card["distributed"]["launcher"] = launcher
    if card_env is not None:
        card["env_vars"].update(card_env)
    script = tmp_path / card["scripts"]
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\n")

    manifest = {
        "built_images": {"img": {"docker_image": "rocm/card:tag"}},
        "built_models": {"img": card},
        "context": {"docker_env_vars": {}, "gpu_vendor": "AMD", "guest_os": "UBUNTU"},
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    additional_context = {
        "gpu_vendor": "AMD",
        "guest_os": "UBUNTU",
        "slurm": {
            "partition": "p",
            "nodes": 2,
            "gpus_per_node": 8,
            "output_dir": str(tmp_path / "out"),
        },
        "distributed": {"launcher": launcher, "nnodes": 2, "nproc_per_node": 8},
    }
    if env_vars is not None:
        additional_context["env_vars"] = env_vars
    return SlurmDeployment(
        DeploymentConfig(
            target="slurm", manifest_file=str(manifest_path), additional_context=additional_context
        )
    )


def _wrapper(deployment) -> str:
    assert deployment.prepare() is True
    return Path(deployment.script_path).read_text()


def _exports(script: str) -> dict:
    out = {}
    for line in script.splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            key, _, value = line[len("export "):].partition("=")
            out[key] = value.strip("'\"")
    return out


class TestLoadSlurmConfigMarksPresetEnv:
    def test_multi_node_preset_keys_are_marked(self):
        cfg = ConfigLoader.load_slurm_config({"slurm": {"nodes": 2}})
        assert set(cfg[PRESET_ENV_KEYS]) == ALL_PRESET_KEYS

    def test_user_set_key_is_not_marked(self):
        cfg = ConfigLoader.load_slurm_config(
            {"slurm": {"nodes": 2}, "env_vars": {"NCCL_IB_DISABLE": "0"}}
        )
        assert "NCCL_IB_DISABLE" not in cfg[PRESET_ENV_KEYS]
        assert cfg["env_vars"]["NCCL_IB_DISABLE"] == "0"

    def test_reloading_a_loaded_config_keeps_the_marks(self):
        once = ConfigLoader.load_slurm_config({"slurm": {"nodes": 2}})
        twice = ConfigLoader.load_slurm_config(once)
        assert twice[PRESET_ENV_KEYS] == once[PRESET_ENV_KEYS]


class TestSlurmMultiDoesNotExportPresetEnv:
    def test_no_preset_key_in_wrapper_without_user_env(self, tmp_path):
        exported = _exports(_wrapper(_deployment(tmp_path)))
        leaked = sorted(ALL_PRESET_KEYS & set(exported))
        assert leaked == [], f"preset env leaked into slurm_multi wrapper: {leaked}"

    def test_user_value_for_a_preset_name_is_exported(self, tmp_path):
        exported = _exports(
            _wrapper(_deployment(tmp_path, env_vars={"NCCL_IB_DISABLE": "0"}))
        )
        assert exported["NCCL_IB_DISABLE"] == "0"
        assert "HSA_ENABLE_SDMA" not in exported

    def test_user_value_equal_to_preset_is_still_exported(self, tmp_path):
        # Deliberately setting the preset's own value is still a user choice.
        exported = _exports(
            _wrapper(_deployment(tmp_path, env_vars={"HSA_ENABLE_SDMA": "0"}))
        )
        assert exported["HSA_ENABLE_SDMA"] == "0"

    def test_card_env_vars_still_exported(self, tmp_path):
        exported = _exports(
            _wrapper(_deployment(tmp_path, card_env={"HSA_ENABLE_SDMA": "1"}))
        )
        assert exported["CARD_ONLY"] == "from-card"
        assert exported["HSA_ENABLE_SDMA"] == "1"


class TestTemplatedLauncherKeepsPresetEnv:
    def test_torchrun_job_script_receives_preset_env(self, tmp_path):
        deployment = _deployment(tmp_path, launcher="torchrun")
        context = deployment._prepare_template_context(deployment.manifest["built_models"]["img"])
        script = deployment.jinja_env.get_template("job.sh.j2").render(**context)
        for key, value in MULTI_NODE_PRESET_ENV.items():
            assert key in context["env_vars"]
            assert key in script
        assert context["env_vars"]["HSA_ENABLE_SDMA"] == "0"


class TestBuildManifestSavesOnlyUserEnv:
    def test_preset_env_not_persisted_into_deployment_config(self, tmp_path):
        """Saved into the manifest, preset values came back at run time as
        additional_context.env_vars and were indistinguishable from the user's."""
        from unittest.mock import MagicMock
        from madengine.orchestration.build_orchestrator import BuildOrchestrator

        orch = BuildOrchestrator.__new__(BuildOrchestrator)
        orch.rich_console = MagicMock()
        orch.additional_context = ConfigLoader.load_slurm_config(
            {"slurm": {"nodes": 2, "partition": "p"}, "env_vars": {"NCCL_IB_DISABLE": "0"}}
        )
        manifest = tmp_path / "build_manifest.json"
        manifest.write_text(json.dumps({"built_images": {}, "built_models": {}}))

        orch._save_deployment_config(str(manifest))

        saved = json.loads(manifest.read_text())["deployment_config"]["env_vars"]
        assert saved == {"NCCL_IB_DISABLE": "0"}

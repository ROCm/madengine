"""The submit layer of a slurm_multi card (deployment/slurm_multi_env.py).

One function builds it for every path that runs a slurm_multi card, with where each
value came from, and the launcher receives the sources as MAD_ENV_SOURCES (MAD's
scripts/common/config/layers.sh records them per job).
"""
import json
import shlex
from pathlib import Path

import yaml

from madengine.deployment.config_loader import PRESET_ENV_KEYS
from madengine.deployment.slurm_multi_env import SOURCES_VAR, submit_env

from .test_k8s_slurm_multi import CARD as K8S_CARD, _prepared
from .test_slurm_multi import PR186_MODEL_ENTRY, _slurm_multi_deployment


def _sources(encoded):
    return dict(p.split("=", 1) for p in encoded.split(",") if p)


class TestSubmitEnv:
    def test_layers_and_their_sources(self):
        card = {"env_vars": {"A": "card", "B": "card"},
                "distributed": {"sglang_disagg": {"prefill_nodes": 2, "decode_nodes": 3}}}
        ctx = {"docker_env_vars": {"A": "dev", "C": "dev"},
               "env_vars": {"B": "user", "NCCL_IB_DISABLE": "1", "xP": "4"},
               PRESET_ENV_KEYS: ["NCCL_IB_DISABLE"]}
        env, warnings, dropped = submit_env(card, ctx, None)
        assert env.env == {"A": "card", "C": "dev", "B": "user", "xP": "4", "yD": "3"}
        assert env.sources == {"A": "card", "C": "docker-env-vars", "B": "additional-context",
                               "xP": "additional-context", "yD": "distributed"}
        assert dropped == ["NCCL_IB_DISABLE"]

    def test_encoded_for_the_launcher(self):
        env, _, _ = submit_env({"env_vars": {"X": "1"}}, {"env_vars": {"Y": "2"}}, None)
        final = env.final()
        assert _sources(final[SOURCES_VAR]) == {"X": "card", "Y": "additional-context"}


class TestSlurmWrapper:
    def test_sources_exported(self, tmp_path):
        d = _slurm_multi_deployment(tmp_path, PR186_MODEL_ENTRY)
        d.config.additional_context["env_vars"] = {"BENCHMARK_CONCURRENCY": "16"}
        assert d.prepare() is True
        text = Path(d.script_path).read_text()
        line = next(l for l in text.splitlines() if l.startswith(f"export {SOURCES_VAR}="))
        src = _sources(shlex.split(line.split("=", 1)[1])[0])
        assert src["MODEL_NAME"] == "card"
        assert src["BENCHMARK_CONCURRENCY"] == "additional-context"
        assert "export BENCHMARK_CONCURRENCY=16" in text


class TestKubernetesGetsTheSameSubmitLayer:
    """Before, the k8s path built only card env_vars + the run's env_vars."""

    def test_docker_env_vars_layered_config_and_sglang_counts_reach_the_pod(self, tmp_path, monkeypatch):
        card = json.loads(json.dumps(K8S_CARD))
        card["env_vars"].pop("xP")
        card["distributed"]["sglang_disagg"] = {"prefill_nodes": 2, "decode_nodes": 1}
        d_dir = tmp_path / "scripts" / "demo"
        d_dir.mkdir(parents=True, exist_ok=True)
        (d_dir / "mad-config.yaml").write_text(yaml.safe_dump(
            {"version": 1, "model": {"env": {"FROM_MAD_CONFIG": "1"}}}))

        def with_ctx(cfg):
            cfg.additional_context["docker_env_vars"] = {"FROM_DOCKER_ENV": "1"}
            return cfg

        from madengine.deployment import kubernetes as k8s_mod
        orig = k8s_mod.KubernetesDeployment.__init__

        def init(self, config):
            orig(self, with_ctx(config))

        monkeypatch.setattr(k8s_mod.KubernetesDeployment, "__init__", init)
        _, _, _, env = _prepared(tmp_path, monkeypatch, card=card)
        src = _sources(env[SOURCES_VAR])
        assert (env["xP"], env["yD"]) == ("2", "1")
        assert src["xP"] == "distributed"
        assert env["FROM_DOCKER_ENV"] == "1" and src["FROM_DOCKER_ENV"] == "docker-env-vars"
        assert src["MODEL_NAME"] == "card"
        assert src["DOCKER_IMAGE_NAME"] == "madengine"
        assert env["FROM_MAD_CONFIG"] == "1" and src["FROM_MAD_CONFIG"] == "mad-config"

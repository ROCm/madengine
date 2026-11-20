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


class TestRecipePrecheck:
    def _checkout(self, tmp_path, rc, msg="refused"):
        d = tmp_path / "scripts"
        (d / "common" / "recipe").mkdir(parents=True)
        (d / "vllm_dissag").mkdir()
        (d / "common" / "recipe" / "precheck.py").write_text(
            f"import sys\nprint({msg!r}, file=sys.stderr)\nsys.exit({rc})\n")
        script = d / "vllm_dissag" / "run_xPyD_models.slurm"
        script.write_text("#!/bin/bash\n")
        return script

    def test_refusal_text_when_the_gate_refuses(self, tmp_path):
        from madengine.deployment.slurm_multi_env import recipe_precheck
        assert recipe_precheck({"MODEL_NAME": "M"}, self._checkout(tmp_path, 1, "no gfx950"), "gfx950") == "no gfx950"

    def test_nothing_without_a_known_gpu_or_the_tool(self, tmp_path):
        from madengine.deployment.slurm_multi_env import recipe_precheck
        script = self._checkout(tmp_path, 1)
        assert recipe_precheck({"MODEL_NAME": "M"}, script, None) is None
        bare = tmp_path / "other" / "x" / "run.slurm"
        bare.parent.mkdir(parents=True)
        bare.write_text("")
        assert recipe_precheck({"MODEL_NAME": "M"}, bare, "gfx950") is None
        assert recipe_precheck({"MODEL_NAME": "M"}, self._checkout(tmp_path / "ok", 0), "gfx950") is None


class TestPlacementDirectives:
    def test_segment_from_the_checkouts_topology_tool(self, tmp_path):
        from madengine.deployment.slurm_multi_env import placement_directives
        tool = tmp_path / "scripts" / "common" / "topology" / "topology.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("import json\nprint(json.dumps({'options': ['--segment=18'], 'notes': []}))\n")
        script = tmp_path / "scripts" / "vllm_dissag" / "run_xPyD_models.slurm"
        script.parent.mkdir(parents=True)
        script.write_text("")
        assert placement_directives({"MAD_TOPOLOGY": "x.json"}, script) == ["--segment=18"]
        assert placement_directives({}, script) == []


class TestPlatformGpus:
    def test_from_the_checkouts_catalog(self, tmp_path):
        from madengine.deployment.slurm_multi_env import platform_gpus_per_node
        (tmp_path / "compat").mkdir()
        (tmp_path / "compat" / "rocm.yaml").write_text(
            "default_platform: a\nplatforms:\n  a: {gpus_per_node: 8}\n  helios: {gpus_per_node: 4}\n")
        script = tmp_path / "scripts" / "vllm_dissag" / "run.slurm"
        assert platform_gpus_per_node({}, script) == 8
        assert platform_gpus_per_node({"MAD_PLATFORM": "helios"}, script) == 4
        # A checkout with the hardware split out of the matrix (compat/hardware.yaml) is read the same.
        (script.parent.parent.parent / "compat" / "hardware.yaml").write_text(
            "default_platform: h\nplatforms:\n  h: {gpus_per_node: 2}\n")
        assert platform_gpus_per_node({}, script) == 2
        assert platform_gpus_per_node({}, tmp_path / "x" / "y" / "z" / "run.slurm") is None


class TestK8sDomainAffinity:
    def _tool(self, tmp_path, body):
        tool = tmp_path / "scripts" / "common" / "topology" / "topology.py"
        tool.parent.mkdir(parents=True)
        tool.write_text(body)
        script = tmp_path / "scripts" / "vllm_multinode" / "run_multinode.slurm"
        script.parent.mkdir(parents=True)
        script.write_text("")
        return script

    def test_the_label_when_the_topology_asks_for_one_domain(self, tmp_path):
        from madengine.deployment.slurm_multi_env import k8s_domain_affinity
        script = self._tool(tmp_path, "import json\nprint(json.dumps({'options': [{'pod_affinity_topology_key': 'r'}]}))\n")
        assert k8s_domain_affinity({"MAD_TOPOLOGY": "s.json"}, script, "r") == "r"
        assert k8s_domain_affinity({}, script, "r") is None
        assert k8s_domain_affinity({"MAD_TOPOLOGY": "s.json"}, script, None) is None

    def test_a_placement_one_job_cannot_hold_is_refused(self, tmp_path):
        import pytest
        from madengine.deployment.slurm_multi_env import k8s_domain_affinity
        script = self._tool(tmp_path, "import sys\nprint('needs a Job per role', file=sys.stderr)\nsys.exit(1)\n")
        with pytest.raises(ValueError, match="Job per role"):
            k8s_domain_affinity({"MAD_TOPOLOGY": "s.json"}, script, "r")


class TestEmitConfig:
    """`madengine build --emit-config`: the card checkout's emit.py writes each card's file."""

    def _checkout(self, tmp_path):
        d = tmp_path / "scripts"
        (d / "common" / "config").mkdir(parents=True)
        (d / "vllm_dissag").mkdir()
        # A stand-in for MAD's emit.py: writes the card it was given, as YAML-ish text.
        (d / "common" / "config" / "emit.py").write_text(
            "import json, sys\n"
            "a = sys.argv\n"
            "card = json.load(open(a[a.index('--card-json') + 1]))\n"
            "out = a[a.index('-o') + 1]\n"
            "import os; os.makedirs(os.path.dirname(out), exist_ok=True)\n"
            "open(out, 'w').write('card: %s\\ncatalog: %s\\n' % (card['name'], a[a.index('--catalog') + 1]))\n")
        (d / "vllm_dissag" / "run_xPyD_models.slurm").write_text("#!/bin/bash\n")
        return d

    def test_writes_the_file_and_its_digest(self, tmp_path):
        from madengine.deployment.slurm_multi_env import emit_config
        d = self._checkout(tmp_path)
        got = emit_config({"name": "vllm_dissag/c1", "scripts": str(d / "vllm_dissag" / "run_xPyD_models.slurm")}, tmp_path / "out")
        assert got["path"].endswith("out/vllm_dissag/c1.yaml")
        assert open(got["path"]).read() == "card: vllm_dissag/c1\ncatalog: vllm_dissag\n"
        assert len(got["sha256"]) == 64

    def test_nothing_for_other_cards_or_without_the_tool(self, tmp_path):
        from madengine.deployment.slurm_multi_env import emit_config
        assert emit_config({"name": "x", "scripts": "scripts/dummy/run.sh"}, tmp_path) is None
        bare = tmp_path / "bare" / "scripts" / "vllm_dissag" / "run.slurm"
        bare.parent.mkdir(parents=True)
        bare.write_text("")
        assert emit_config({"name": "x", "scripts": str(bare)}, tmp_path) is None

    def test_the_image_is_generalized_with_the_config(self, tmp_path):
        from madengine.deployment.slurm_multi_env import emit_config
        from madengine.stack.wheels import Wheel
        d = self._checkout(tmp_path)
        # The stand-in emit.py writes the generalized image it was given.
        (d / "common" / "config" / "emit.py").write_text(
            "import json, sys, os\n"
            "a = sys.argv\n"
            "card = json.load(open(a[a.index('--card-json') + 1]))\n"
            "out = a[a.index('-o') + 1]\n"
            "os.makedirs(os.path.dirname(out), exist_ok=True)\n"
            "json.dump({'spec': card.get('_image_spec'), 'unresolved': card.get('_image_unresolved')}, open(out, 'w'))\n")
        (tmp_path / "compat").mkdir()
        (tmp_path / "compat" / "rocm.yaml").write_text(
            "# the checkout's matrix\nindexes: {fw: 'https://fw/'}\ncomponents:\n  torch:\n"
            "    '2.13.0': {index: fw, requirement: torch==2.13.0}\n"
            "  vllm:\n    '0.29.1@fw': {url: 'https://fw/old.whl'}\n")

        def find(name, version):
            if (name, version) == ("vllm", "0.30.0"):
                return "fw", Wheel(name=name, version=version, python="cp312", platform="x", url="https://fw/v.whl",
                                   requires=["torch==2.13.0"])
            return None

        build = {"base_docker": "rocm/base:1",
                 "image_info": {"python": "3.12.1", "packages": {"vllm": "0.30.0", "torch": "2.13.0", "mylib": "1+g1234567"}}}
        got = emit_config({"name": "c1", "scripts": str(d / "vllm_dissag" / "run_xPyD_models.slurm")},
                          tmp_path / "out", build, find)
        import json as _json, yaml as _yaml
        written = _json.load(open(got["path"]))
        assert written["spec"] == {"base": "rocm/base:1", "python": "3.12",
                                   "components": {"torch": "2.13.0", "vllm": "0.30.0@fw"}}
        assert [u["component"] for u in written["unresolved"]] == ["mylib"]
        matrix = _yaml.safe_load(open(got["compat"]))
        assert matrix["components"]["vllm"]["0.30.0@fw"]["requires"] == ["torch==2.13.0"]
        assert open(got["compat"]).read().startswith("# the checkout's matrix")
        assert "0.30.0" not in (tmp_path / "compat" / "rocm.yaml").read_text()   # the checkout is untouched

    def test_the_manifest_records_each_file(self, tmp_path):
        import json
        from madengine.cli.commands.build import write_emitted_configs
        d = self._checkout(tmp_path)
        m = tmp_path / "build_manifest.json"
        m.write_text(json.dumps({"built_models": {
            "c1": {"name": "c1", "scripts": str(d / "vllm_dissag" / "run_xPyD_models.slurm")},
            "other": {"name": "other", "scripts": "scripts/dummy/run.sh"}}}))
        assert write_emitted_configs(str(m), str(tmp_path / "cfg")) == 1
        got = json.loads(m.read_text())["built_models"]
        assert got["c1"]["emitted_config"]["path"].endswith("cfg/vllm_dissag/c1.yaml")
        assert "emitted_config" not in got["other"]


class TestImageComponents:
    def test_an_image_spec_images_builds_reach_the_run(self):
        from madengine.deployment.slurm_multi_env import submit_env
        build = {"build_plan": {"resolved": {"components": {"mori": "fe12a11a@di", "vllm": "9a7ae4b8+cd0d2656@vllm-kimi-k3"}}}}
        sub, _, _ = submit_env({"name": "c", "env_vars": {}}, {}, None, {}, image_build=build)
        assert sub.env["MAD_IMAGE_COMPONENTS"] == "mori=fe12a11a@di,vllm=9a7ae4b8+cd0d2656@vllm-kimi-k3"   # by name
        assert sub.sources["MAD_IMAGE_COMPONENTS"] == "madengine"

    def test_none_for_a_hand_written_dockerfile_and_the_run_wins(self):
        from madengine.deployment.slurm_multi_env import image_components, submit_env
        assert image_components({"docker_image": "x"}) is None and image_components(None) is None
        build = {"build_plan": {"resolved": {"components": {"vllm": "a@b"}}}}
        sub, _, _ = submit_env({"name": "c", "env_vars": {}}, {"env_vars": {"MAD_IMAGE_COMPONENTS": "vllm=mine"}},
                               None, {}, image_build=build)
        assert sub.env["MAD_IMAGE_COMPONENTS"] == "vllm=mine"

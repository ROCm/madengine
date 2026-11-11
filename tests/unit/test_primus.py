"""
Primus-related unit tests: backend/overlay inference and K8s launcher command generation.

Keeps Primus coverage in one module to avoid several small `test_*primus*.py` files.
"""

from types import SimpleNamespace

import pytest

from madengine.deployment.k8s_names import (
    sanitize_k8s_container_name,
    sanitize_k8s_object_name,
)
from madengine.deployment.kubernetes_launcher_mixin import KubernetesLauncherMixin
from madengine.deployment.primus_backend import (
    card_sets_config_path,
    infer_primus_backend_from_model_name,
    infer_primus_examples_overlay_subdirs,
    merged_primus_config,
)


# --- K8s mixin: _generate_primus_command -------------------------------------


class _PrimusCommandHarness(KubernetesLauncherMixin):
    """Minimal object with attributes _generate_primus_command expects."""

    def __init__(self, additional_context, job_name="madengine-test", namespace="default"):
        self.config = SimpleNamespace(additional_context=additional_context)
        self.job_name = job_name
        self.namespace = namespace


@pytest.mark.unit
class TestGeneratePrimusCommand:
    def test_single_node_no_backend(self):
        h = _PrimusCommandHarness(
            {"distributed": {"primus": {"config_path": "examples/torchtitan/x.yaml"}}}
        )
        cmd = h._generate_primus_command(
            1, 8, 1234, "scripts/primus_pretrain/run.sh", "--config_path z"
        )
        assert "PRIMUS_ROOT=" in cmd
        assert "PRIMUS_CONFIG_PATH=" in cmd
        assert "export BACKEND=" not in cmd
        assert "MASTER_ADDR" in cmd
        assert "NNODES=1" in cmd
        assert "PRIMUS_SHARED_FS" not in cmd

    def test_backend_inferred_from_model_name(self):
        h = _PrimusCommandHarness(
            {"distributed": {"primus": {"config_path": "examples/torchtitan/x.yaml"}}}
        )
        cmd = h._generate_primus_command(
            1,
            8,
            1234,
            "scripts/primus_pretrain/run.sh",
            "",
            model_name="primus_pretrain/torchtitan_MI300X_qwen3_4B-pretrain",
        )
        assert 'export BACKEND="torchtitan"' in cmd

    def test_single_node_with_backend_override(self):
        h = _PrimusCommandHarness(
            {
                "distributed": {
                    "primus": {
                        "config_path": "examples/maxtext/foo.yaml",
                        "backend": "MaxText",
                    }
                }
            }
        )
        cmd = h._generate_primus_command(1, 4, 1234, "scripts/primus_pretrain/run.sh", "")
        assert 'export BACKEND="MaxText"' in cmd
        assert "PRIMUS_CONFIG_PATH=" in cmd

    def test_multi_node_service_dns(self):
        h = _PrimusCommandHarness(
            {"distributed": {"primus": {}}}, job_name="madengine-j", namespace="ns1"
        )
        cmd = h._generate_primus_command(2, 8, 1234, "scripts/primus_pretrain/run.sh", "")
        assert "madengine-j-0.madengine-j.ns1.svc.cluster.local" in cmd
        assert "JOB_COMPLETION_INDEX" in cmd
        assert "NNODES=2" in cmd
        assert "export PRIMUS_SHARED_FS=0" in cmd

    def test_multi_node_master_dns_uses_short_headless_subdomain(self):
        """Headless Service / Pod subdomain must be a DNS label (≤63); hostname stays Job name-index."""
        long_job = sanitize_k8s_object_name("madengine", "z" * 400)
        sub = sanitize_k8s_container_name(long_job)
        assert len(sub) <= 63
        h = _PrimusCommandHarness(
            {"distributed": {"primus": {}}}, job_name=long_job, namespace="ns1"
        )
        h.service_name = sub
        cmd = h._generate_primus_command(2, 8, 1234, "scripts/primus_pretrain/run.sh", "")
        assert f"{long_job}-0.{sub}.ns1.svc.cluster.local" in cmd

    def test_configmap_files_overlay_primus_root_outside_workspace(self):
        h = _PrimusCommandHarness({"distributed": {"primus": {}}})
        cmd = h._generate_primus_command(2, 8, 1234, "scripts/primus_train/run.sh", "")
        assert '[ "$PRIMUS_ROOT" != /workspace/Primus ]' in cmd
        assert 'cp -r /workspace/Primus/. "$PRIMUS_ROOT/"' in cmd

    def test_card_config_path_is_not_replaced_by_fallback(self):
        h = _PrimusCommandHarness({"distributed": {"primus": {}}})
        cmd = h._generate_primus_command(
            2, 8, 1234, "scripts/primus_train/run.sh", "--config_path examples/m.yaml"
        )
        assert "PRIMUS_CONFIG_PATH" not in cmd
        assert "bash run.sh --config_path examples/m.yaml" in cmd

    def test_explicit_config_path_still_wins_over_card(self):
        h = _PrimusCommandHarness(
            {"distributed": {"primus": {"config_path": "examples/x.yaml"}}}
        )
        cmd = h._generate_primus_command(
            1, 8, 1234, "scripts/primus_train/run.sh", "--config_path examples/m.yaml"
        )
        assert 'export PRIMUS_CONFIG_PATH="examples/x.yaml"' in cmd

    def test_fallback_config_path_without_card_arg(self):
        h = _PrimusCommandHarness({"distributed": {"primus": {}}})
        cmd = h._generate_primus_command(1, 8, 1234, "scripts/primus_train/run.sh", "")
        assert (
            'export PRIMUS_CONFIG_PATH="examples/torchtitan/configs/MI300X/'
            'qwen3_1.7B-pretrain.yaml"' in cmd
        )


# --- primus_backend helpers --------------------------------------------------


@pytest.mark.unit
class TestInferPrimusBackendFromModelName:
    def test_torchtitan(self):
        assert (
            infer_primus_backend_from_model_name(
                "primus_pretrain/torchtitan_MI300X_qwen3_4B-pretrain"
            )
            == "torchtitan"
        )

    def test_megatron(self):
        assert (
            infer_primus_backend_from_model_name(
                "primus_pretrain/megatron_MI300X_some_experiment"
            )
            == "megatron"
        )

    def test_maxtext_capitalization(self):
        assert (
            infer_primus_backend_from_model_name("primus_pretrain/maxtext_MI300X_foo")
            == "MaxText"
        )

    def test_non_primus_prefix(self):
        assert infer_primus_backend_from_model_name("other/torchtitan_x") is None

    def test_unknown_launcher(self):
        assert (
            infer_primus_backend_from_model_name(
                "primus_pretrain/unknownThing_MI300X_x"
            )
            is None
        )


@pytest.mark.unit
class TestCardSetsConfigPath:
    def test_flag_in_args(self):
        assert card_sets_config_path("--config_path examples/a.yaml --foo 1")

    def test_absent_or_other_flag(self):
        assert not card_sets_config_path("")
        assert not card_sets_config_path("--config_path_extra x")

    def test_equals_form(self):
        assert card_sets_config_path("--config_path=examples/a.yaml --foo 1")


@pytest.mark.unit
class TestMergedPrimusConfig:
    def test_runtime_overrides_manifest(self):
        manifest = {
            "deployment_config": {
                "distributed": {
                    "primus": {
                        "config_path": "examples/torchtitan/a.yaml",
                        "backend": "megatron",
                    }
                }
            }
        }
        ac = {"distributed": {"primus": {"config_path": "examples/torchtitan/b.yaml"}}}
        m = merged_primus_config(manifest, ac)
        assert m["config_path"] == "examples/torchtitan/b.yaml"
        assert m["backend"] == "megatron"


@pytest.mark.unit
class TestInferPrimusExamplesOverlaySubdirs:
    def test_from_torchtitan_path(self):
        assert infer_primus_examples_overlay_subdirs(
            "examples/torchtitan/configs/MI300X/qwen3_4B-pretrain.yaml"
        ) == ["torchtitan"]

    def test_megatron_bridge_before_megatron(self):
        assert infer_primus_examples_overlay_subdirs(
            "examples/megatron_bridge/foo.yaml"
        ) == ["megatron_bridge"]

    def test_megatron_path(self):
        assert infer_primus_examples_overlay_subdirs(
            "examples/megatron/configs/MI300X/x.yaml"
        ) == ["megatron"]

    def test_backend_hint_when_path_ambiguous(self):
        assert infer_primus_examples_overlay_subdirs(
            "custom/exp.yaml", backend_hint="maxtext"
        ) == ["maxtext"]

    def test_model_name_fallback(self):
        assert infer_primus_examples_overlay_subdirs(
            "unknown.yaml",
            model_name="primus_pretrain/megatron_MI300X_x",
        ) == ["megatron"]

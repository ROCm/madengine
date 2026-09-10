#!/usr/bin/env python3
"""
Unit tests for the spur (Crusoe) deployment backend.

Spur ships SLURM-compatible CLI shims but cannot fan out with `srun`, so
madengine drives multi-node runs with a job ARRAY of single-node tasks that
self-form the cluster through a shared-filesystem rendezvous. These tests lock
in the contract points that make that work:

1. `slurm.scheduler == "spur"` selects the spur backend (target inference and
   ConfigLoader), and bad combinations raise.
2. `_expand_nodelist` handles both the spur (expanded) and stock SLURM
   (compressed) nodelist forms.
3. The rendered job script and the slurm_multi wrapper emit array directives,
   `%A_%a` log names, and the fail-fast rendezvous.
4. `SpurDeployment.monitor()` reports completion from the per-rank markers and
   cannot hang when squeue is empty or unavailable.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import shlex
import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from madengine.deployment.base import DeploymentConfig, DeploymentStatus
from madengine.deployment.config_loader import ConfigLoader
from madengine.deployment.factory import DeploymentFactory
from madengine.deployment.slurm import SlurmDeployment
from madengine.deployment.spur import (
    DEFAULT_RENDEZVOUS_TIMEOUT,
    SpurDeployment,
    render_rendezvous_block,
)
from madengine.orchestration.run_orchestrator import RunOrchestrator

# ---------------------------------------------------------------------------
# 1. Target inference


class TestSpurTargetInference:
    """`slurm.scheduler` distinguishes spur from stock SLURM."""

    @pytest.fixture
    def orchestrator(self) -> RunOrchestrator:
        args = MagicMock()
        args.additional_context = None
        args.live_output = True
        return RunOrchestrator(args)

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({}, "local"),
            ({"slurm": {"nodes": 2}}, "slurm"),
            ({"slurm": {"nodes": 2, "scheduler": "slurm"}}, "slurm"),
            ({"slurm": {"nodes": 2, "scheduler": "spur"}}, "spur"),
            ({"slurm": {"nodes": 2, "scheduler": "SPUR"}}, "spur"),
            ({"k8s": {"namespace": "default"}}, "k8s"),
        ],
    )
    def test_infer_deployment_target(self, orchestrator, config, expected):
        assert orchestrator._infer_deployment_target(config) == expected

    def test_runtime_context_still_wins_over_manifest(self, orchestrator):
        """Regression: the manifest's build-time target must not override the
        runtime --additional-context (Convention over Configuration)."""
        assert orchestrator._infer_deployment_target({"k8s": {}}) == "k8s"

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({}, "local"),
            ({"slurm": {"nodes": 2}}, "slurm"),
            ({"slurm": {"nodes": 2, "scheduler": "spur"}}, "spur"),
            ({"deploy": "spur", "slurm": {"nodes": 2}}, "spur"),
            ({"k8s": {}}, "k8s"),
        ],
    )
    def test_config_loader_infers_spur(self, config, expected):
        assert ConfigLoader.infer_and_validate_deploy_type(config) == expected

    def test_deploy_spur_without_slurm_config_raises(self):
        with pytest.raises(ValueError, match="no 'slurm' config"):
            ConfigLoader.infer_and_validate_deploy_type({"deploy": "spur"})

    def test_deploy_spur_conflicting_scheduler_raises(self):
        with pytest.raises(ValueError, match="slurm.scheduler"):
            ConfigLoader.infer_and_validate_deploy_type(
                {"deploy": "spur", "slurm": {"scheduler": "slurm"}}
            )

    def test_unknown_scheduler_raises(self):
        with pytest.raises(ValueError, match="Unknown slurm.scheduler"):
            ConfigLoader.infer_and_validate_deploy_type({"slurm": {"scheduler": "pbs"}})

    def test_spur_reuses_slurm_presets(self):
        """load_config must apply the SLURM presets for spur, not fall through to local."""
        merged = ConfigLoader.load_config({"slurm": {"nodes": 2, "scheduler": "spur"}})
        assert merged["slurm"]["scheduler"] == "spur"
        assert merged["slurm"]["nodes"] == 2
        # Presets contribute keys the user did not supply.
        assert len(merged["slurm"]) > 2


# ---------------------------------------------------------------------------
# 2. Nodelist expansion


class TestExpandNodelist:
    """`_expand_nodelist` must work on both spur and stock SLURM forms."""

    def test_empty(self):
        assert SlurmDeployment._expand_nodelist("") == []

    def test_spur_expanded_comma_list(self):
        """Spur exposes an already-expanded list and has no `scontrol show hostnames`."""
        assert SlurmDeployment._expand_nodelist("nodeA,nodeB,nodeC") == [
            "nodeA",
            "nodeB",
            "nodeC",
        ]

    def test_single_host(self):
        assert SlurmDeployment._expand_nodelist("nodeA") == ["nodeA"]

    def test_compressed_form_uses_scontrol(self):
        with patch("madengine.deployment.slurm.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="node01\nnode02\n")
            assert SlurmDeployment._expand_nodelist("node[01-02]") == [
                "node01",
                "node02",
            ]

    def test_compressed_form_falls_back_when_scontrol_missing(self):
        with patch(
            "madengine.deployment.slurm.subprocess.run", side_effect=FileNotFoundError
        ):
            # No expansion possible; must not raise.
            assert SlurmDeployment._expand_nodelist("node[01-02]") == ["node[01-02]"]


# ---------------------------------------------------------------------------
# 3. Rendezvous block


class TestRendezvousBlock:
    """The rendezvous must fail fast instead of continuing with an empty MASTER_ADDR."""

    def test_rank0_publishes_and_peers_wait(self):
        block = "\n".join(render_rendezvous_block("/shared/rendezvous", 600))
        assert "/shared/rendezvous/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" in block
        assert 'echo "$_MAD_MY_IP" > "$_MAD_REND_DIR/master_addr"' in block
        assert "_MAD_REND_TIMEOUT=600" in block

    def test_timeout_writes_failure_marker_and_exits(self):
        """A peer that never sees master_addr must report a failure, not hang or
        continue silently: monitor() keys off the done_rank markers."""
        block = "\n".join(render_rendezvous_block("/shared/rendezvous", 600))
        assert 'if [ -z "$MASTER_ADDR" ]; then' in block
        assert 'echo "1" > "$_MAD_REND_DIR/done_rank${SLURM_PROCID}"' in block
        assert "exit 1" in block
        assert ">&2" in block

    def test_timeout_is_configurable(self):
        assert "_MAD_REND_TIMEOUT=42" in "\n".join(
            render_rendezvous_block("/shared/rendezvous", 42)
        )


# ---------------------------------------------------------------------------
# Shared fixtures for deployment-level tests

SPUR_MODEL_ENTRY = {
    "name": "dummy_spur_model",
    "url": "",
    "dockerfile": "docker/dummy",
    "scripts": "scripts/dummy/run.sh",
    "n_gpus": "8",
    "owner": "mad.support@amd.com",
    "tags": ["dummy", "spur"],
    "timeout": -1,
    "args": "",
    "env_vars": {"DOCKER_IMAGE_NAME": "registry.example.com/rocm/dummy:latest"},
}


def _make_manifest(tmp_path: Path, distributed: dict, image: str = None) -> Path:
    script_abs = tmp_path / SPUR_MODEL_ENTRY["scripts"]
    script_abs.parent.mkdir(parents=True, exist_ok=True)
    script_abs.write_text("#!/bin/bash\n# placeholder model script for unit test\n")

    image_key = image or SPUR_MODEL_ENTRY["env_vars"]["DOCKER_IMAGE_NAME"]
    model_entry = dict(
        SPUR_MODEL_ENTRY,
        distributed=distributed,
        env_vars={"DOCKER_IMAGE_NAME": image_key},
    )
    manifest = {
        "built_images": {
            image_key: {
                "image_name": image_key,
                "docker_image": image_key,
                "registry_image": image_key,
            },
        },
        "built_models": {image_key: model_entry},
        "context": {
            "docker_env_vars": {},
            "docker_mounts": {},
            "docker_build_arg": {},
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "docker_gpus": "all",
        },
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _make_deployment(
    tmp_path: Path, cls, launcher: str, nodes: int = 3, image: str = None, **slurm_extra
):
    distributed = {
        "launcher": launcher,
        "nnodes": nodes,
        "nproc_per_node": 8,
        "backend": "nccl",
        "port": 29500,
    }
    manifest_path = _make_manifest(tmp_path, distributed, image=image)
    slurm = dict(
        partition="amd-rccl",
        nodes=nodes,
        gpus_per_node=8,
        time="01:00:00",
        output_dir=str(tmp_path / "slurm_results"),
        exclusive=True,
        **slurm_extra,
    )
    additional_context = {
        "gpu_vendor": "AMD",
        "guest_os": "UBUNTU",
        "slurm": slurm,
        "distributed": distributed,
    }
    cfg = DeploymentConfig(
        target=cls.DEPLOYMENT_TYPE,
        manifest_file=str(manifest_path),
        additional_context=additional_context,
    )
    return cls(cfg)


def _render_job_script(deployment) -> str:
    model_info = next(iter(deployment.manifest["built_models"].values()))
    context = deployment._prepare_template_context(model_info)
    return deployment.jinja_env.get_template("job.sh.j2").render(**context)


# ---------------------------------------------------------------------------
# 4. Rendered sbatch template


class TestSpurJobTemplate:
    """job.sh.j2 must switch to job-array fan-out under scheduler == 'spur'."""

    def test_spur_uses_job_array(self, tmp_path):
        script = _render_job_script(
            _make_deployment(tmp_path, SpurDeployment, "torchrun")
        )
        assert "#SBATCH --array=0-2" in script
        assert "#SBATCH --nodes=1" in script
        # srun cannot fan out on spur; each array task runs its own task script.
        assert 'srun bash "$TASK_SCRIPT"' not in script
        assert 'bash "$TASK_SCRIPT"' in script

    def test_stock_slurm_still_uses_srun(self, tmp_path):
        script = _render_job_script(
            _make_deployment(tmp_path, SlurmDeployment, "torchrun")
        )
        assert "#SBATCH --array" not in script
        assert "#SBATCH --nodes=3" in script

    def test_spur_log_names_use_array_job_id(self, tmp_path):
        """%j is each array task's own job id; result collection globs on the
        array job id that sbatch returns, which is %A."""
        script = _render_job_script(
            _make_deployment(tmp_path, SpurDeployment, "torchrun")
        )
        assert "_%A_%a.out" in script
        assert "_%A_%a.err" in script
        assert "_%j_%t.out" not in script

    def test_stock_slurm_log_names_unchanged(self, tmp_path):
        script = _render_job_script(
            _make_deployment(tmp_path, SlurmDeployment, "torchrun")
        )
        assert "_%j_%t.out" in script
        assert "_%A_%a.out" not in script

    def test_spur_exports_node_rank(self, tmp_path):
        """Each array task is its own single-node allocation, so SLURM_NODEID
        would otherwise be 0 on every node."""
        script = _render_job_script(
            _make_deployment(tmp_path, SpurDeployment, "torchrun")
        )
        assert 'export SLURM_NODEID="${SLURM_ARRAY_TASK_ID:-0}"' in script
        assert 'export SLURM_PROCID="${SLURM_ARRAY_TASK_ID:-0}"' in script

    def test_nodelist_helper_is_exported_to_task_script(self, tmp_path):
        """The generated TASK_SCRIPT runs as a separate bash process, which does
        not inherit shell functions without `export -f`."""
        script = _render_job_script(
            _make_deployment(tmp_path, SlurmDeployment, "torchrun")
        )
        assert "export -f mad_expand_nodelist" in script

    def test_spur_rendezvous_is_rendered(self, tmp_path):
        script = _render_job_script(
            _make_deployment(tmp_path, SpurDeployment, "torchrun")
        )
        assert "[spur-rendezvous]" in script
        assert f"_MAD_REND_TIMEOUT={DEFAULT_RENDEZVOUS_TIMEOUT}" in script

    def test_rendezvous_timeout_is_configurable(self, tmp_path):
        deployment = _make_deployment(
            tmp_path, SpurDeployment, "torchrun", rendezvous_timeout=1234
        )
        assert "_MAD_REND_TIMEOUT=1234" in _render_job_script(deployment)


# ---------------------------------------------------------------------------
# 5. slurm_multi wrapper


class TestSpurSlurmMultiScript:
    """The slurm_multi wrapper needs the same array/rendezvous treatment."""

    @pytest.fixture
    def wrapper(self, tmp_path) -> str:
        deployment = _make_deployment(tmp_path, SpurDeployment, "slurm_multi")
        assert deployment.prepare() is True
        return Path(deployment.script_path).read_text()

    def test_emits_array_directives(self, wrapper):
        assert "#SBATCH --array=0-2" in wrapper
        assert "#SBATCH --nodes=1" in wrapper

    def test_log_names_use_array_job_id(self, wrapper):
        assert "_%A_%a.out" in wrapper
        assert "_%j_%t.out" not in wrapper

    def test_stock_slurm_wrapper_keeps_task_log_names(self, tmp_path):
        deployment = _make_deployment(tmp_path, SlurmDeployment, "slurm_multi")
        assert deployment.prepare() is True
        wrapper = Path(deployment.script_path).read_text()
        assert "_%j_%t.out" in wrapper
        assert "#SBATCH --array" not in wrapper

    def test_pulls_locally_through_a_quoted_variable(self, wrapper):
        """One array task per node, so no srun fan-out for the pull."""
        assert "MAD_PULL_IMAGE=registry.example.com/rocm/dummy:latest" in wrapper
        assert 'docker pull "$MAD_PULL_IMAGE"' in wrapper
        assert "srun --nodes=$SLURM_NNODES" not in wrapper

    def test_docker_image_is_shell_quoted(self, tmp_path):
        """Registry image names are interpolated into generated bash."""
        hostile = "registry.example.com/rocm/dummy:latest; touch /tmp/pwned"
        deployment = _make_deployment(
            tmp_path, SpurDeployment, "slurm_multi", image=hostile
        )
        assert deployment.prepare() is True
        wrapper = Path(deployment.script_path).read_text()
        assert f"MAD_PULL_IMAGE={shlex.quote(hostile)}" in wrapper
        assert "; touch /tmp/pwned" not in wrapper.replace(shlex.quote(hostile), "")

    def test_completion_marker_is_per_rank(self, wrapper):
        """SLURM_JOB_ID is pinned to the shared array id, so the marker path
        needs the rank or all N tasks race on one file."""
        assert "_rank${SLURM_ARRAY_TASK_ID:-0}.complete" in wrapper

    def test_writes_per_rank_done_marker(self, wrapper):
        assert "done_rank${NODE_RANK}" in wrapper

    def test_uses_the_shared_rendezvous_block(self, wrapper):
        assert "[spur-rendezvous]" in wrapper
        assert 'echo "1" > "$_MAD_REND_DIR/done_rank${SLURM_PROCID}"' in wrapper


# ---------------------------------------------------------------------------
# 6. monitor()


class TestSpurMonitor:
    """Marker-based completion detection, and no way to poll forever."""

    @pytest.fixture
    def deployment(self, tmp_path):
        dep = _make_deployment(tmp_path, SpurDeployment, "torchrun")
        dep._show_log_summary = MagicMock()
        dep._stream_job_output = MagicMock()
        return dep

    @staticmethod
    def _write_markers(deployment, job_id: str, codes: dict):
        marker_dir = Path(deployment.rendezvous_dir) / job_id
        marker_dir.mkdir(parents=True, exist_ok=True)
        for rank, code in codes.items():
            (marker_dir / f"done_rank{rank}").write_text(str(code))

    def test_all_ranks_succeeded(self, deployment):
        self._write_markers(deployment, "111", {0: 0, 1: 0, 2: 0})
        result = deployment.monitor("111")
        assert result.status == DeploymentStatus.SUCCESS
        deployment._show_log_summary.assert_called_once_with("111", success=True)

    def test_one_rank_failed(self, deployment):
        self._write_markers(deployment, "111", {0: 0, 1: 7, 2: 0})
        result = deployment.monitor("111")
        assert result.status == DeploymentStatus.FAILED
        assert "1: 7" in result.message
        deployment._show_log_summary.assert_called_once_with("111", success=False)

    def test_unreadable_marker_counts_as_failure(self, deployment):
        self._write_markers(deployment, "111", {0: 0, 1: "garbage", 2: 0})
        assert deployment.monitor("111").status == DeploymentStatus.FAILED

    def test_partial_with_live_tasks_keeps_running(self, deployment):
        self._write_markers(deployment, "111", {0: 0})
        deployment._live_task_count = MagicMock(return_value=2)
        result = deployment.monitor("111")
        assert result.status == DeploymentStatus.RUNNING
        assert "1/3 ranks done" in result.message

    def test_startup_grace_before_tasks_are_registered(self, deployment):
        """Right after sbatch, spur's squeue lists nothing; that must not be
        mistaken for a dead array."""
        deployment._live_task_count = MagicMock(return_value=0)
        for _ in range(deployment._SPUR_DEAD_POLLS + 2):
            assert deployment.monitor("111").status == DeploymentStatus.RUNNING

    def test_startup_grace_is_bounded(self, deployment):
        """An array that dies before squeue ever lists it writes no marker and
        never appears: the startup window must still end, or monitor() (which the
        caller polls without a timeout) never returns."""
        deployment._live_task_count = MagicMock(return_value=0)
        for _ in range(deployment._SPUR_STARTUP_POLLS - 1):
            assert deployment.monitor("111").status == DeploymentStatus.RUNNING
        result = deployment.monitor("111")
        assert result.status == DeploymentStatus.FAILED
        assert "was ever seen in the queue" in result.message
        deployment._show_log_summary.assert_called_once_with("111", success=False)

    def test_startup_counter_resets_once_tasks_appear(self, deployment):
        """A slow queue must not accumulate toward the startup bound."""
        deployment._live_task_count = MagicMock(
            side_effect=[0] * (deployment._SPUR_STARTUP_POLLS - 1)
            + [2]
            + [0] * (deployment._SPUR_STARTUP_POLLS + 2)
        )
        for _ in range(deployment._SPUR_STARTUP_POLLS):
            assert deployment.monitor("111").status == DeploymentStatus.RUNNING
        # Seen alive, then empty again: now the (shorter) dead-array window applies.
        for _ in range(deployment._SPUR_DEAD_POLLS - 1):
            assert deployment.monitor("111").status == DeploymentStatus.RUNNING
        assert deployment.monitor("111").status == DeploymentStatus.FAILED

    def test_dead_array_fails_after_grace_window(self, deployment):
        self._write_markers(deployment, "111", {0: 0})
        deployment._live_task_count = MagicMock(side_effect=[3] + [0] * 10)
        assert deployment.monitor("111").status == DeploymentStatus.RUNNING  # seen live
        for _ in range(deployment._SPUR_DEAD_POLLS - 1):
            assert deployment.monitor("111").status == DeploymentStatus.RUNNING
        result = deployment.monitor("111")
        assert result.status == DeploymentStatus.FAILED
        assert "1/3 ranks reported completion" in result.message

    def test_persistent_squeue_outage_gives_up(self, deployment):
        """live == -1 must not reset the loop forever: the caller polls without
        a timeout, so an unreachable control plane would hang the run."""
        deployment._live_task_count = MagicMock(return_value=-1)
        for _ in range(deployment._SPUR_UNKNOWN_POLLS - 1):
            assert deployment.monitor("111").status == DeploymentStatus.RUNNING
        assert deployment.monitor("111").status == DeploymentStatus.UNKNOWN

    def test_markers_win_over_squeue_outage(self, deployment):
        deployment._live_task_count = MagicMock(return_value=-1)
        self._write_markers(deployment, "111", {0: 0, 1: 0, 2: 0})
        assert deployment.monitor("111").status == DeploymentStatus.SUCCESS

    def test_live_output_streams_instead_of_summary(self, deployment):
        deployment.config.additional_context["live_output"] = True
        deployment._live_task_count = MagicMock(return_value=3)
        deployment.monitor("111")
        deployment._stream_job_output.assert_called_with("111")

        self._write_markers(deployment, "111", {0: 0, 1: 0, 2: 0})
        deployment.monitor("111")
        deployment._stream_job_output.assert_called_with("111", final=True)
        deployment._show_log_summary.assert_not_called()


class TestLiveTaskCount:
    """squeue parsing for the liveness guard."""

    @pytest.fixture
    def deployment(self, tmp_path):
        return _make_deployment(tmp_path, SpurDeployment, "torchrun")

    def _run_squeue(self, deployment, stdout, returncode=0):
        with patch("madengine.deployment.spur.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=returncode, stdout=stdout)
            count = deployment._live_task_count("111", "madengine-dummy_spur_model")
            return count, mock_run.call_args[0][0]

    def test_counts_only_my_array_tasks(self, deployment):
        """A concurrent run of the same model must not inflate the count."""
        stdout = (
            "111_0 madengine-dummy_spur_model RUNNING\n"
            "111_1 madengine-dummy_spur_model RUNNING\n"
            "222_0 madengine-dummy_spur_model RUNNING\n"
        )
        count, _ = self._run_squeue(deployment, stdout)
        assert count == 2

    def test_ignores_finished_states(self, deployment):
        stdout = (
            "111_0 madengine-dummy_spur_model COMPLETED\n"
            "111_1 madengine-dummy_spur_model RUNNING\n"
        )
        count, _ = self._run_squeue(deployment, stdout)
        assert count == 1

    def test_pending_array_range(self, deployment):
        count, _ = self._run_squeue(
            deployment, "111_[0-2] madengine-dummy_spur_model PENDING\n"
        )
        assert count == 1

    def test_falls_back_to_name_when_no_id_matches(self, deployment):
        count, _ = self._run_squeue(
            deployment, "999 madengine-dummy_spur_model RUNNING\n"
        )
        assert count == 1

    def test_non_zero_exit_is_unknown(self, deployment):
        count, _ = self._run_squeue(deployment, "", returncode=1)
        assert count == -1

    def test_exception_is_unknown(self, deployment):
        with patch("madengine.deployment.spur.subprocess.run", side_effect=OSError):
            assert deployment._live_task_count("111", "madengine-x") == -1

    def test_unset_user_omits_the_flag(self, deployment):
        """`squeue -u ""` is an error, so drop -u entirely."""
        with patch.dict("os.environ", {}, clear=True):
            _, cmd = self._run_squeue(deployment, "")
        assert "-u" not in cmd

    def test_user_is_passed_when_set(self, deployment):
        with patch.dict("os.environ", {"USER": "someone"}, clear=True):
            _, cmd = self._run_squeue(deployment, "")
        assert cmd[cmd.index("-u") + 1] == "someone"


# ---------------------------------------------------------------------------
# 7. Backend selection wiring


class TestFactoryRegistration:
    """The inferred target string must reach the right class."""

    def test_spur_target_creates_spur_deployment(self, tmp_path):
        dep = _make_deployment(tmp_path, SpurDeployment, "torchrun")
        assert isinstance(DeploymentFactory.create(dep.config), SpurDeployment)

    def test_slurm_target_still_creates_slurm_deployment(self, tmp_path):
        dep = _make_deployment(tmp_path, SlurmDeployment, "torchrun")
        created = DeploymentFactory.create(dep.config)
        assert isinstance(created, SlurmDeployment)
        assert not isinstance(created, SpurDeployment)

    def test_stock_slurm_is_not_flagged_as_spur(self):
        """IS_SPUR gates every spur branch inside the shared SLURM code."""
        assert SlurmDeployment.IS_SPUR is False
        assert SpurDeployment.IS_SPUR is True


class TestInferenceConsistency:
    """The three inference sites (ConfigLoader, build, run) must agree, or a
    config builds for one backend and runs on the other."""

    @pytest.fixture
    def orchestrator(self) -> RunOrchestrator:
        args = MagicMock()
        args.additional_context = None
        args.live_output = True
        return RunOrchestrator(args)

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"slurm": {"nodes": 2}},
            {"slurm": {"nodes": 2, "scheduler": "slurm"}},
            {"slurm": {"nodes": 2, "scheduler": "spur"}},
            {"deploy": "spur", "slurm": {"nodes": 2}},
            {"deploy": "slurm", "slurm": {"nodes": 2}},
            {"k8s": {"namespace": "default"}},
        ],
    )
    def test_config_loader_and_run_orchestrator_agree(self, orchestrator, config):
        assert orchestrator._infer_deployment_target(
            config
        ) == ConfigLoader.infer_and_validate_deploy_type(config)


class TestSpurValidate:
    """spur's scontrol is only partially implemented, so it must not be required."""

    def test_validate_does_not_probe_scontrol(self, tmp_path):
        dep = _make_deployment(tmp_path, SpurDeployment, "torchrun")
        with patch("madengine.deployment.slurm.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert dep.validate() is True
        probed = [call.args[0] for call in mock_run.call_args_list]
        assert ["which", "scontrol"] not in probed
        assert ["which", "sbatch"] in probed

    def test_stock_slurm_still_requires_scontrol(self, tmp_path):
        dep = _make_deployment(tmp_path, SlurmDeployment, "torchrun")
        with patch("madengine.deployment.slurm.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert dep.validate() is True
        probed = [call.args[0] for call in mock_run.call_args_list]
        assert ["which", "scontrol"] in probed


# ---------------------------------------------------------------------------
# 8. Node health preflight


class TestNodePreflight:
    """The srun-based preflight cannot work on spur, and the nodelist it pins is
    actively harmful there: every array task requests --nodes=1, so a multi-node
    #SBATCH --nodelist would make each task demand all of them."""

    @staticmethod
    def _submit(deployment):
        with patch("madengine.deployment.slurm.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Submitted batch job 4242", stderr=""
            )
            return deployment.deploy()

    def test_spur_skips_preflight_and_submits(self, tmp_path):
        dep = _make_deployment(tmp_path, SpurDeployment, "torchrun")
        assert dep.prepare() is True
        with patch("madengine.deployment.slurm.SlurmNodeSelector") as selector:
            result = self._submit(dep)
        selector.assert_not_called()
        assert result.deployment_id == "4242"
        assert "nodelist" not in dep.slurm_config
        assert "#SBATCH --nodelist" not in Path(dep.script_path).read_text()

    def test_stock_slurm_preflight_is_unchanged(self, tmp_path):
        """Regression: SLURM still health-checks and still gates multi-node
        submission when there are not enough clean nodes."""
        dep = _make_deployment(tmp_path, SlurmDeployment, "torchrun")
        assert dep.prepare() is True
        with patch("madengine.deployment.slurm.SlurmNodeSelector") as selector:
            selector.return_value.select_nodes.return_value = (["nodeA"], "")
            result = self._submit(dep)
        selector.return_value.select_nodes.assert_called_once()
        assert result.status == DeploymentStatus.FAILED
        assert "Not enough clean nodes" in result.message

    def test_stock_slurm_preflight_still_pins_clean_nodes(self, tmp_path):
        dep = _make_deployment(tmp_path, SlurmDeployment, "torchrun")
        assert dep.prepare() is True
        with patch("madengine.deployment.slurm.SlurmNodeSelector") as selector:
            selector.return_value.select_nodes.return_value = (
                ["nodeA", "nodeB", "nodeC"],
                "",
            )
            result = self._submit(dep)
        assert result.deployment_id == "4242"
        assert dep.slurm_config["nodelist"] == "nodeA,nodeB,nodeC"
        assert (
            "#SBATCH --nodelist=nodeA,nodeB,nodeC" in Path(dep.script_path).read_text()
        )


# ---------------------------------------------------------------------------
# 9. Generated scripts are valid bash


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
class TestGeneratedScriptsAreValidBash:
    """The jinja branches must not produce a script the shell rejects - a syntax
    error would only surface as a failed job on the cluster."""

    @pytest.mark.parametrize("cls", [SlurmDeployment, SpurDeployment])
    def test_rendered_template_parses(self, tmp_path, cls):
        script = tmp_path / "job.sh"
        script.write_text(
            _render_job_script(_make_deployment(tmp_path, cls, "torchrun"))
        )
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0

    @pytest.mark.parametrize("cls", [SlurmDeployment, SpurDeployment])
    def test_slurm_multi_wrapper_parses(self, tmp_path, cls):
        deployment = _make_deployment(tmp_path, cls, "slurm_multi")
        assert deployment.prepare() is True
        assert subprocess.run(["bash", "-n", deployment.script_path]).returncode == 0


# ---------------------------------------------------------------------------
# 10. Log/artifact collection keys on the id sbatch returned


class TestLogCollectionCompatibility:
    """collect_results()/_show_log_summary() glob on the deployment id, which for
    an array is the array job id (%A), not each task's own job id (%j)."""

    @staticmethod
    def _sbatch_log_names(script: str, job_id: str) -> list:
        return [
            line.split("=", 1)[1]
            .replace("%A", job_id)
            .replace("%a", "0")
            .replace("%j", "999")
            .replace("%t", "0")
            for line in script.splitlines()
            if line.startswith("#SBATCH --output=")
        ]

    def test_spur_sbatch_log_names_match_the_collection_glob(self, tmp_path):
        dep = _make_deployment(tmp_path, SpurDeployment, "torchrun")
        names = self._sbatch_log_names(_render_job_script(dep), "4242")
        assert names
        for name in names:
            assert fnmatch(Path(name).name, "madengine-*_4242_*.out")

    def test_show_log_summary_finds_spur_array_logs(self, tmp_path):
        dep = _make_deployment(tmp_path, SpurDeployment, "torchrun")
        dep.output_dir.mkdir(parents=True, exist_ok=True)
        log = dep.output_dir / "madengine-dummy_spur_model_4242_1.out"
        log.write_text("done\n")
        dep.console = MagicMock()
        dep._show_log_summary("4242", success=True)
        printed = " ".join(str(c.args[0]) for c in dep.console.print.call_args_list)
        assert str(log) in printed

    @pytest.mark.parametrize("cls", [SlurmDeployment, SpurDeployment])
    def test_node_logs_key_on_the_collection_job_id(self, tmp_path, cls):
        """The per-node logs the task script writes are the ones collect_results
        reads for multi-node runs, so they must carry the same id."""
        script = _render_job_script(_make_deployment(tmp_path, cls, "torchrun"))
        assert (
            'export MAD_COLLECT_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"' in script
        )
        assert "_${MAD_COLLECT_JOB_ID}_node_${SLURM_PROCID}.out" in script
        assert "_${SLURM_JOB_ID}_node_${SLURM_PROCID}.out" not in script


# ---------------------------------------------------------------------------
# 11. Rendezvous configuration plumbing


class TestRendezvousConfig:
    def test_rendezvous_dir_is_under_the_shared_output_dir(self, tmp_path):
        dep = _make_deployment(tmp_path, SpurDeployment, "torchrun")
        assert Path(dep.rendezvous_dir) == dep.output_dir.resolve() / "spur_rendezvous"

    def test_default_timeout(self, tmp_path):
        dep = _make_deployment(tmp_path, SpurDeployment, "torchrun")
        assert dep.rendezvous_timeout == DEFAULT_RENDEZVOUS_TIMEOUT

    def test_timeout_comes_from_the_slurm_block(self, tmp_path):
        dep = _make_deployment(
            tmp_path, SpurDeployment, "torchrun", rendezvous_timeout="60"
        )
        assert dep.rendezvous_timeout == 60

    def test_template_context_carries_the_scheduler_flavor(self, tmp_path):
        def model(d):
            return next(iter(d.manifest["built_models"].values()))

        spur = _make_deployment(tmp_path / "spur", SpurDeployment, "torchrun")
        slurm = _make_deployment(tmp_path / "slurm", SlurmDeployment, "torchrun")
        assert spur._prepare_template_context(model(spur))["scheduler"] == "spur"
        assert slurm._prepare_template_context(model(slurm))["scheduler"] == "slurm"

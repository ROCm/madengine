"""Context blocks for each Deployment, through the CLI and Hydra seams.

Canonical examples set one copy of each fact. A file that sets only one copy
of an old key still runs. Two copies that disagree fail before a build.
"""

import json
from pathlib import Path

import pytest
import typer

from madengine.cli.constants import ExitCode
from madengine.cli.validators import validate_additional_context
from madengine.config import load_config


def _context(**extra) -> str:
    return json.dumps({"gpu_vendor": "AMD", "guest_os": "UBUNTU", **extra})


def _captured(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


class TestCanonicalWorkflows:
    """Each Deployment accepts the canonical block and keeps today's short form."""

    def test_local_docker_single_node_torchrun(self):
        result = validate_additional_context(
            additional_context=_context(
                docker_gpus="0,1,2,3,4,5,6,7",
                distributed={
                    "launcher": "torchrun",
                    "nnodes": 1,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                },
            )
        )
        assert result["distributed"]["launcher"] == "torchrun"
        assert result["distributed"]["nnodes"] == 1

    def test_local_docker_without_a_distributed_block_stays_valid(self):
        result = validate_additional_context(
            additional_context=_context(docker_gpus="0")
        )
        assert "distributed" not in result

    @pytest.mark.parametrize(
        "launcher",
        [
            "torchrun",
            "deepspeed",
            "megatron-lm",
            "torchtitan",
            "vllm",
            "sglang",
            "primus",
        ],
    )
    def test_local_single_node_launchers(self, launcher):
        result = validate_additional_context(
            additional_context=_context(
                distributed={"launcher": launcher, "nnodes": 1, "nproc_per_node": 8}
            )
        )
        assert result["distributed"]["launcher"] == launcher

    def test_bare_metal_single_node(self):
        result = validate_additional_context(
            additional_context=_context(
                platform="bare_metal",
                conda={"conda_bin": "/path/to/micromamba", "reuse_env": True},
                distributed={
                    "launcher": "torchrun",
                    "nnodes": 1,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                },
            )
        )
        assert result["platform"] == "bare_metal"
        assert result["distributed"]["launcher"] == "torchrun"

    def test_bare_metal_with_a_scheduler_is_rejected(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_context(
                    platform="bare_metal",
                    slurm={"partition": "gpu", "nodes": 1},
                    distributed={"launcher": "torchrun", "nnodes": 1},
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_slurm_single_node_vllm(self):
        result = validate_additional_context(
            additional_context=_context(
                slurm={
                    "partition": "gpu",
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "time": "04:00:00",
                },
                distributed={
                    "launcher": "vllm",
                    "nnodes": 1,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                },
            )
        )
        assert result["slurm"]["nodes"] == 1
        assert result["distributed"]["launcher"] == "vllm"

    def test_slurm_multi_node_torchrun(self):
        result = validate_additional_context(
            additional_context=_context(
                slurm={"partition": "gpu", "nodes": 4, "gpus_per_node": 8},
                distributed={
                    "launcher": "torchrun",
                    "nnodes": 4,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                },
            )
        )
        assert result["distributed"]["nnodes"] == 4

    def test_slurm_vllm_disagg_two_nodes(self):
        result = validate_additional_context(
            additional_context=_context(
                slurm={"partition": "gpu", "nodes": 2, "gpus_per_node": 8},
                distributed={
                    "launcher": "vllm-disagg",
                    "nnodes": 2,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                    "vllm_disagg": {"prefill_nodes": 1, "decode_nodes": 1},
                },
            )
        )
        assert result["distributed"]["launcher"] == "vllm-disagg"

    def test_spur_multi_node_torchrun(self):
        result = validate_additional_context(
            additional_context=_context(
                slurm={
                    "scheduler": "spur",
                    "partition": "gpu",
                    "nodes": 4,
                    "gpus_per_node": 8,
                },
                distributed={
                    "launcher": "torchrun",
                    "nnodes": 4,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                },
            )
        )
        assert result["slurm"]["scheduler"] == "spur"

    def test_k8s_single_node_torchrun(self):
        result = validate_additional_context(
            additional_context=_context(
                k8s={"namespace": "default", "gpu_count": 8},
                distributed={
                    "launcher": "torchrun",
                    "nnodes": 1,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                },
            )
        )
        assert result["distributed"]["nnodes"] == 1

    def test_k8s_multi_node_megatron(self):
        result = validate_additional_context(
            additional_context=_context(
                k8s={"namespace": "default", "gpu_count": 8},
                distributed={
                    "launcher": "megatron-lm",
                    "nnodes": 4,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                },
            )
        )
        assert result["distributed"]["launcher"] == "megatron-lm"

    def test_k8s_sglang_disagg_three_nodes(self):
        result = validate_additional_context(
            additional_context=_context(
                k8s={"namespace": "default", "gpu_count": 8},
                distributed={
                    "launcher": "sglang-disagg",
                    "nnodes": 3,
                    "nproc_per_node": 8,
                    "master_port": 29500,
                    "sglang_disagg": {"prefill_nodes": 1, "decode_nodes": 1},
                },
            )
        )
        assert result["distributed"]["sglang_disagg"]["prefill_nodes"] == 1

    def test_llm_d_aggregated_has_no_distributed_launcher(self):
        result = validate_additional_context(
            additional_context=_context(
                k8s={"namespace": "llm-d-bench", "gpu_count": 0},
                llm_d={
                    "endpoint_url": "http://gateway/v1",
                    "model": {"name": "Qwen/Qwen3-32B"},
                },
            )
        )
        assert "distributed" not in result
        assert result["llm_d"]["model"]["name"] == "Qwen/Qwen3-32B"

    def test_llm_d_disaggregated_scale_lives_in_prefill_and_decode(self):
        result = validate_additional_context(
            additional_context=_context(
                k8s={"namespace": "llm-d-bench", "gpu_count": 0},
                llm_d={
                    "model": {"name": "Qwen/Qwen3-32B"},
                    "prefill": {"replicas": 2, "tensor_parallel": 8, "gpu_count": 8},
                    "decode": {"replicas": 1, "tensor_parallel": 8, "gpu_count": 8},
                },
            )
        )
        assert result["llm_d"]["prefill"]["replicas"] == 2

    def test_summary_names_the_deployment(self, capsys):
        validate_additional_context(
            additional_context=_context(
                slurm={"partition": "gpu", "nodes": 4, "gpus_per_node": 8},
                distributed={
                    "launcher": "torchrun",
                    "nnodes": 4,
                    "nproc_per_node": 8,
                },
            )
        )
        text = _captured(capsys)
        assert "deployment=slurm" in text
        assert "launcher=torchrun" in text


class TestContextDisagreements:
    """Two copies of one fact fail. One copy, including an old key, still runs."""

    def test_port_and_master_port_disagree(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_context(
                    distributed={
                        "launcher": "torchrun",
                        "nnodes": 1,
                        "port": 29500,
                        "master_port": 30000,
                    }
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_port_alone_is_still_accepted(self):
        result = validate_additional_context(
            additional_context=_context(
                distributed={"launcher": "torchrun", "nnodes": 1, "port": 29500}
            )
        )
        assert result["distributed"]["port"] == 29500

    def test_slurm_multi_keeps_a_port_mismatch(self):
        result = validate_additional_context(
            additional_context=_context(
                slurm={"partition": "gpu"},
                distributed={
                    "launcher": "slurm_multi",
                    "port": 1,
                    "master_port": 2,
                },
            )
        )
        assert result["distributed"]["launcher"] == "slurm_multi"
        assert result["distributed"]["port"] == 1

    def test_launcher_type_and_distributed_launcher_disagree(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_context(
                    k8s={"namespace": "default"},
                    launcher={"type": "vllm", "nnodes": 1},
                    distributed={"launcher": "torchrun", "nnodes": 1},
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_sglang_blocks_disagree(self, capsys):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_context(
                    slurm={"partition": "gpu", "nodes": 4},
                    sglang_disagg={"prefill_nodes": 1, "decode_nodes": 3},
                    distributed={
                        "launcher": "sglang-disagg",
                        "nnodes": 4,
                        "sglang_disagg": {"prefill_nodes": 2, "decode_nodes": 2},
                    },
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS
        assert "sglang_disagg" in _captured(capsys)

    def test_top_level_sglang_block_alone_is_accepted(self):
        result = validate_additional_context(
            additional_context=_context(
                slurm={"partition": "gpu", "nodes": 2},
                sglang_disagg={"prefill_nodes": 1, "decode_nodes": 1},
                distributed={"launcher": "sglang-disagg", "nnodes": 2},
            )
        )
        assert result["sglang_disagg"]["decode_nodes"] == 1

    def test_k8s_and_kubernetes_disagree(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_context(
                    k8s={"namespace": "a", "gpu_count": 8},
                    kubernetes={"namespace": "b", "gpu_count": 8},
                    distributed={
                        "launcher": "torchrun",
                        "nnodes": 1,
                        "nproc_per_node": 8,
                    },
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_disagg_on_one_slurm_node_is_rejected(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_context(
                    slurm={"partition": "gpu", "nodes": 1},
                    distributed={"launcher": "vllm-disagg", "nnodes": 1},
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_k8s_disagg_with_two_nodes_is_rejected(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_context(
                    k8s={"namespace": "default", "gpu_count": 8},
                    distributed={"launcher": "sglang-disagg", "nnodes": 2},
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_llm_d_with_a_distributed_launcher_is_rejected(self):
        with pytest.raises(typer.Exit) as exc_info:
            validate_additional_context(
                additional_context=_context(
                    k8s={"namespace": "llm-d-bench", "gpu_count": 0},
                    llm_d={"model": {"name": "Qwen/Qwen3-32B"}},
                    distributed={"launcher": "vllm-disagg", "nnodes": 2},
                )
            )
        assert exc_info.value.exit_code == ExitCode.INVALID_ARGS

    def test_node_count_mismatch_warns_and_continues(self, capsys):
        result = validate_additional_context(
            additional_context=_context(
                slurm={"partition": "gpu", "nodes": 2, "gpus_per_node": 8},
                distributed={
                    "launcher": "torchrun",
                    "nnodes": 4,
                    "nproc_per_node": 8,
                },
            )
        )
        assert result["slurm"]["nodes"] == 2
        assert "distributed.nnodes=4" in _captured(capsys)

    def test_gpu_count_mismatch_warns_and_continues(self, capsys):
        validate_additional_context(
            additional_context=_context(
                k8s={"namespace": "default", "gpu_count": 8},
                distributed={
                    "launcher": "torchrun",
                    "nnodes": 1,
                    "nproc_per_node": 4,
                },
            )
        )
        assert "nproc_per_node=4" in _captured(capsys)


class TestHydraContext:
    def test_torchrun_group_sets_master_port_only(self):
        ctx, _meta = load_config(["launcher=torchrun"])
        assert ctx["distributed"]["master_port"] == 29500
        assert "port" not in ctx["distributed"]
        assert "backend" not in ctx["distributed"]

    def test_sglang_disagg_group_is_nested(self):
        ctx, _meta = load_config(["scheduler=slurm", "launcher=sglang_disagg"])
        assert ctx["distributed"]["sglang_disagg"]["prefill_nodes"] is None
        assert "sglang_disagg" not in ctx

    def test_vllm_group_has_no_unused_block(self):
        ctx, _meta = load_config(["launcher=vllm"])
        assert "vllm" not in ctx

    def test_platform_bare_metal_writes_the_string(self):
        ctx, _meta = load_config(["platform=bare_metal"])
        assert ctx["platform"] == "bare_metal"

    def test_user_yaml_launcher_without_enabled_launches(self, tmp_path: Path):
        job = tmp_path / "job.yaml"
        job.write_text("distributed:\n  launcher: torchrun\n  nnodes: 1\n")
        ctx, _meta = load_config([str(job)])
        assert ctx["distributed"]["launcher"] == "torchrun"
        assert ctx["distributed"]["enabled"] is True

    def test_explicit_enabled_false_stays_off(self, tmp_path: Path):
        job = tmp_path / "job.yaml"
        job.write_text(
            "distributed:\n  launcher: torchrun\n  enabled: false\n  nnodes: 1\n"
        )
        ctx, _meta = load_config([str(job)])
        assert ctx["distributed"]["enabled"] is False

    def test_enabled_override_false_stays_off(self):
        ctx, _meta = load_config(["launcher=torchrun", "distributed.enabled=false"])
        assert ctx["distributed"]["enabled"] is False

    def test_slurm_multi_group_is_unchanged(self):
        ctx, _meta = load_config(["launcher=slurm_multi"])
        assert ctx["distributed"]["launcher"] == "slurm_multi"
        assert ctx["distributed"]["backend"] == "nccl"


class TestSglangDisaggFallback:
    def test_top_level_block_reaches_the_slurm_command(self, tmp_path: Path):
        from madengine.deployment.base import DeploymentConfig
        from madengine.deployment.slurm import SlurmDeployment

        manifest = tmp_path / "build_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "built_images": {"dummy-image": {"docker_image": "dummy:latest"}},
                    "built_models": {
                        "dummy-image": {
                            "name": "dummy_multinode",
                            "scripts": "run.sh",
                            "n_gpus": "8",
                        }
                    },
                    "context": {},
                }
            )
        )
        deployment = SlurmDeployment(
            DeploymentConfig(
                target="slurm",
                manifest_file=str(manifest),
                additional_context={
                    "gpu_vendor": "AMD",
                    "guest_os": "UBUNTU",
                    "slurm": {
                        "partition": "gpu",
                        "nodes": 4,
                        "gpus_per_node": 8,
                        "output_dir": str(tmp_path / "out"),
                    },
                    "sglang_disagg": {"prefill_nodes": 2, "decode_nodes": 2},
                    "distributed": {"launcher": "sglang-disagg", "nnodes": 4},
                },
            )
        )
        script = deployment._generate_sglang_disagg_command(4, 8, 29500)
        assert "SGLANG_DISAGG_PREFILL_NODES=2" in script
        assert "SGLANG_DISAGG_DECODE_NODES=2" in script

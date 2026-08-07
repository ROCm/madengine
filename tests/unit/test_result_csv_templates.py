"""The job templates must copy a results CSV nobody declared.

The per-node copy used to sit inside ``{% if multiple_results %}``, so a card without the
field left nothing behind on the node and there was nothing for the collector to find --
discovery on the login node would have had no input. These tests pin the sweep into all
four blocks: the multi-node task script and the single-node tail of the SLURM job, and
both container blocks of the Kubernetes job.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from pathlib import Path

import madengine.deployment as deployment_package
from madengine.deployment.base import create_jinja_env
from madengine.deployment.slurm import SlurmDeployment


#: The shape test the shell does, in the one form it is written in both templates.
SHAPE_TEST = 'if($i=="model")m=1;if($i=="performance")p=1;if($i=="metric")t=1'
SWEEP_LOOP = "for _result_csv in"


def render_slurm(tmp_path, model_overrides=None, nodes=2):
    """Render job.sh.j2 the way prepare() does, with the model entry adjusted."""
    from tests.unit.test_slurm_job_template import MODEL_ENTRY, _build_deployment

    deployment = _build_deployment(tmp_path, {"nodes": nodes}, {"nnodes": nodes})
    model_entry = dict(MODEL_ENTRY)
    model_entry.update(model_overrides or {})
    context = deployment._prepare_template_context(model_entry)
    return deployment.jinja_env.get_template("job.sh.j2").render(**context)


def render_kubernetes(**context):
    """Render job.yaml.j2 on its own; undeclared variables render empty, as in Jinja."""
    templates = Path(deployment_package.__file__).parent / "templates" / "kubernetes"
    context.setdefault("env_vars", {})
    return create_jinja_env(templates).get_template("job.yaml.j2").render(**context)


class TestSlurmJobScript:
    """Both copy sites in the SLURM job script: the task script and the single-node tail."""

    def test_the_multi_node_task_script_carries_it(self, tmp_path):
        script = render_slurm(tmp_path, {"multiple_results": ""}, nodes=2)
        task_script = script.split("TASK_SCRIPT_EOF")[1]
        assert SWEEP_LOOP in task_script
        assert SHAPE_TEST in task_script
        assert 'cp "$_result_csv" "$NODE_COLLECTION_DIR"/' in task_script

    def test_the_single_node_tail_carries_it(self, tmp_path):
        script = render_slurm(tmp_path, {"multiple_results": ""}, nodes=1)
        assert "TASK_SCRIPT_EOF" not in script
        assert SWEEP_LOOP in script
        assert SHAPE_TEST in script
        assert 'mkdir -p "$NODE_COLLECTION_DIR"' in script

    def test_a_declared_file_is_still_copied_by_name(self, tmp_path):
        script = render_slurm(tmp_path, {"multiple_results": "perf_dummy.csv"}, nodes=1)
        assert '"$WORKSPACE/run_directory/perf_dummy.csv"' in script
        assert SWEEP_LOOP in script

    def test_madengine_own_outputs_are_skipped_by_the_sweep(self, tmp_path):
        script = render_slurm(tmp_path, {"multiple_results": ""})  # noqa: E501
        assert "perf.csv|perf_super*|perf_entry*" in script

    def test_the_sweep_looks_in_the_workspace_and_the_run_directory(self, tmp_path):
        script = render_slurm(tmp_path, {"multiple_results": ""})
        assert '"$WORKSPACE"/*.csv "$WORKSPACE"/run_directory/*.csv' in script


class TestKubernetesJob:
    """Both container blocks: the launcher arm and the direct-script arm."""

    def test_the_launcher_arm_carries_it(self):
        manifest = render_kubernetes(launcher_command="bash /tmp/run_launcher.sh")
        assert SWEEP_LOOP in manifest
        assert SHAPE_TEST in manifest

    def test_the_direct_script_arm_carries_it(self):
        manifest = render_kubernetes()
        assert SWEEP_LOOP in manifest
        assert SHAPE_TEST in manifest

    def test_it_copies_into_the_results_volume(self):
        manifest = render_kubernetes()
        assert 'cp "$_result_csv" /results/${HOSTNAME}/' in manifest

    def test_a_declared_file_is_still_copied_by_name(self):
        manifest = render_kubernetes(multiple_results="perf_dummy.csv")
        assert "/workspace/perf_dummy.csv" in manifest
        assert SWEEP_LOOP in manifest


class TestRankingIsShared:
    """The SLURM collector ranks candidates with the same code the Docker path uses."""

    def test_the_deployment_delegates_to_the_shared_ranking(self, tmp_path):
        from unittest.mock import MagicMock

        from tests.unit.test_result_csv_discovery import make_result_csv

        deployment = object.__new__(SlurmDeployment)
        deployment.console = MagicMock()
        thin = make_result_csv(tmp_path / "node_0" / "r.csv", count=1)
        rich = make_result_csv(tmp_path / "node_1" / "r.csv", count=8)
        assert deployment._select_best_multiple_results_csv([thin, rich]) == rich

    def test_no_candidates_yields_nothing(self):
        from unittest.mock import MagicMock

        deployment = object.__new__(SlurmDeployment)
        deployment.console = MagicMock()
        assert deployment._select_best_multiple_results_csv([]) is None

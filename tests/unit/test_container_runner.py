"""Unit tests for ContainerRunner: setup failure recording and perf CSV."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from madengine.execution.container_runner import ContainerRunner


class TestCreateSetupFailurePerfEntry:
    """_create_setup_failure_perf_entry builds valid perf entry for pre-run failures."""

    def test_returns_dict_with_status_failure(self):
        """Entry has status FAILURE and model name."""
        runner = ContainerRunner(context=MagicMock(), console=MagicMock())
        runner.context.ctx = {"docker_env_vars": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a"}}

        model_info = {"name": "org/model1", "tags": "v1", "n_gpus": "2"}
        build_info = {"dockerfile": "Dockerfile", "docker_image": "img:latest"}

        entry = runner._create_setup_failure_perf_entry(
            model_info=model_info,
            build_info=build_info,
            image_name="img:latest",
            error_message="pull failed",
        )

        assert entry["status"] == "FAILURE"
        assert entry["model"] == "org/model1"
        assert entry["docker_image"] == "img:latest"
        assert "tags" in entry
        assert entry["performance"] == ""
        assert entry["metric"] == ""

    def test_tags_list_flattened_to_string(self):
        """Tags list is flattened to comma-separated string."""
        runner = ContainerRunner(context=MagicMock(), console=MagicMock())
        runner.context.ctx = {}

        model_info = {"name": "m", "tags": ["a", "b"], "n_gpus": "1"}
        build_info = {}

        entry = runner._create_setup_failure_perf_entry(
            model_info=model_info,
            build_info=build_info,
            image_name="img",
            error_message="err",
        )
        assert entry["tags"] == "a,b"


class TestRunModelsFromManifestSetupFailureRecordsToPerfCsv:
    """When an exception occurs before run_container, failure is recorded to perf CSV."""

    @patch("madengine.execution.container_runner.update_perf_csv")
    def test_setup_failure_appends_to_failed_runs_and_records_to_csv(
        self, mock_update_perf_csv
    ):
        """Exception before run_container leads to failed_runs entry and update_perf_csv call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "build_manifest.json")
            perf_csv_path = os.path.join(tmpdir, "perf.csv")
            # Create empty perf CSV so runner can append
            with open(perf_csv_path, "w") as f:
                f.write(
                    "model,n_gpus,nnodes,gpus_per_node,training_precision,pipeline,args,tags,"
                    "docker_file,base_docker,docker_sha,docker_image,git_commit,machine_name,"
                    "deployment_type,launcher,gpu_architecture,performance,metric,relative_change,"
                    "status,build_duration,test_duration,dataname,data_provider_type,data_size,"
                    "data_download_duration,build_number,additional_docker_run_options\n"
                )

            manifest = {
                "built_images": {"img1": {"docker_image": "local/img1", "dockerfile": "D"}},
                "built_models": {
                    "img1": {"name": "test/model", "tags": "t1", "n_gpus": "1", "args": ""}
                },
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            ctx = MagicMock()
            ctx.ctx = {"docker_env_vars": {"MAD_SYSTEM_GPU_ARCHITECTURE": "gfx90a"}}
            ctx.ensure_runtime_context = MagicMock()
            mock_console = MagicMock()
            mock_console.sh.return_value = "testhost"
            runner = ContainerRunner(context=ctx, console=mock_console)
            runner.perf_csv_path = perf_csv_path
            runner.set_credentials({})

            # Make run_container raise so we hit the except block (simulates pull/setup failure)
            with patch.object(
                runner, "run_container", side_effect=RuntimeError("pull failed")
            ):
                result = runner.run_models_from_manifest(
                    manifest_file=manifest_path,
                    registry=None,
                    timeout=60,
                )

            assert len(result["failed_runs"]) == 1
            assert result["failed_runs"][0]["model"] == "test/model"
            assert "pull failed" in result["failed_runs"][0]["error"]

            # update_perf_csv should have been called with exception_result
            assert mock_update_perf_csv.called
            call_kw = mock_update_perf_csv.call_args[1]
            assert call_kw.get("perf_csv") == perf_csv_path
            assert "exception_result" in call_kw

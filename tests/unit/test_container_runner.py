"""Unit tests for ContainerRunner: setup failure recording and perf CSV."""

import json
import os
import re
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from madengine.deployment.base import PERFORMANCE_LOG_PATTERN
from madengine.execution.container_runner import ContainerRunner


PERF_PATTERN = PERFORMANCE_LOG_PATTERN


class TestPerformanceRegex:
    """Performance regex in container_runner matches all supported log formats."""

    def _match(self, log_line):
        m = re.search(PERF_PATTERN, log_line)
        return (m.group(1), m.group(2)) if m else (None, None)

    # --- formats that were already handled before the regex change ---

    def test_basic_integer(self):
        assert self._match("performance: 12345 samples_per_second") == ("12345", "samples_per_second")

    def test_decimal(self):
        assert self._match("performance: 100.5 samples_per_second") == ("100.5", "samples_per_second")

    def test_scientific_lowercase_e(self):
        assert self._match("performance: 1.23e+4 samples_per_second") == ("1.23e+4", "samples_per_second")

    def test_scientific_negative_exponent(self):
        assert self._match("performance: 1.23e-4 samples_per_second") == ("1.23e-4", "samples_per_second")

    def test_zero(self):
        assert self._match("performance: 0 samples_per_second") == ("0", "samples_per_second")

    def test_metric_with_digits(self):
        assert self._match("performance: 123 metric123") == ("123", "metric123")

    def test_metric_starting_with_underscore(self):
        assert self._match("performance: 123 _metric") == ("123", "_metric")

    # --- new formats added with the extended regex ---

    def test_unit_suffix_slash_s(self):
        """Value followed by /s unit suffix: suffix is stripped, metric parsed correctly."""
        assert self._match("performance: 14164/s samples_per_second") == ("14164", "samples_per_second")

    def test_unit_suffix_and_comma(self):
        """Value with /s suffix and comma separator."""
        assert self._match("performance: 14164.5/s, samples_per_second") == ("14164.5", "samples_per_second")

    def test_comma_separator_no_suffix(self):
        """Comma after value without a unit suffix."""
        assert self._match("performance: 100.5, samples_per_second") == ("100.5", "samples_per_second")

    def test_comma_before_suffix(self):
        """Comma immediately before /s suffix: 123,/s metric."""
        assert self._match("performance: 123,/s metric") == ("123", "metric")

    def test_comma_space_before_suffix(self):
        """Comma then space then /s suffix: 123, /s metric."""
        assert self._match("performance: 123, /s metric") == ("123", "metric")

    # --- formats inherited from madenginev1 ---

    def test_scientific_uppercase_e(self):
        """Uppercase E in scientific notation (v1 supported, old v2 broke on this)."""
        assert self._match("performance: 1.23E+4 samples_per_second") == ("1.23E+4", "samples_per_second")

    def test_positive_sign(self):
        """Explicitly signed positive value (v1 supported via [+|-]? prefix)."""
        assert self._match("performance: +123.45 samples_per_second") == ("+123.45", "samples_per_second")

    def test_negative_sign(self):
        """Signed negative value (v1 supported)."""
        assert self._match("performance: -123.45 samples_per_second") == ("-123.45", "samples_per_second")

    def test_leading_dot_decimal(self):
        """Leading-dot decimal without integer part (v1 supported via [0-9]*[.]?[0-9]*)."""
        assert self._match("performance: .5 samples_per_second") == (".5", "samples_per_second")

    # --- slash-containing metric names (e.g. samples/sec, tokens/sec) ---

    def test_metric_samples_per_sec_slash(self):
        """samples/sec metric (used by _determine_aggregation_method) is parsed."""
        assert self._match("performance: 1234.5 samples/sec") == ("1234.5", "samples/sec")

    def test_metric_tokens_per_sec_slash(self):
        """tokens/sec metric (used by _determine_aggregation_method) is parsed."""
        assert self._match("performance: 500.0 tokens/sec") == ("500.0", "tokens/sec")

    def test_metric_with_slash_and_suffix(self):
        """Slash metric combined with /s value suffix."""
        assert self._match("performance: 500.0/s tokens/sec") == ("500.0", "tokens/sec")

    # --- non-matching cases ---

    def test_no_match_missing_metric(self):
        """Line with value but no metric name does not match."""
        val, met = self._match("performance: 100.5")
        assert val is None and met is None

    def test_no_match_wrong_keyword(self):
        """Unrelated log line does not match."""
        val, met = self._match("throughput: 100.5 samples_per_second")
        assert val is None and met is None


class TestResolveDockerImage:
    """_resolve_docker_image uses subprocess argv (no shell) for docker inspect."""

    @patch("madengine.execution.container_runner.subprocess.run")
    def test_inspect_passes_image_ref_as_single_argv_element(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            ["docker", "image", "inspect", "x"], 0
        )
        runner = ContainerRunner(context=MagicMock(), console=MagicMock())
        ref = "registry/ns/name:ci-model_df; touch evil"
        assert runner._resolve_docker_image(ref, "other/model") == ref
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["docker", "image", "inspect", ref]
        assert kwargs.get("check") is True
        assert kwargs.get("stdout") == subprocess.DEVNULL


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

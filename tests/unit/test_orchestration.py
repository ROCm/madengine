"""Unit tests for orchestration: image_filtering and orchestrator init/validation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from madengine.core.additional_context_defaults import (
    DEFAULT_GPU_VENDOR,
    DEFAULT_GUEST_OS,
)
from madengine.core.errors import ConfigurationError
from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.orchestration.image_filtering import (
    filter_images_by_gpu_compatibility,
    filter_images_by_skip_gpu_arch,
)
from madengine.orchestration.run_orchestrator import RunOrchestrator

# ---- image_filtering ----


class TestFilterImagesByGpuCompatibility:
    """filter_images_by_gpu_compatibility behavior."""

    def test_empty_input(self):
        compat, skipped = filter_images_by_gpu_compatibility({}, "AMD", "gfx90a")
        assert compat == {}
        assert skipped == []

    def test_no_vendor_treated_as_compatible(self):
        built = {"m1": {"gpu_vendor": "", "gpu_architecture": ""}}
        compat, skipped = filter_images_by_gpu_compatibility(built, "AMD", "gfx90a")
        assert compat == {"m1": built["m1"]}
        assert skipped == []

    def test_vendor_match_included_with_or_without_arch(self):
        """Vendor match with empty arch or matching arch both include the image."""
        for gpu_arch in ["", "gfx90a"]:
            built = {"m1": {"gpu_vendor": "AMD", "gpu_architecture": gpu_arch}}
            compat, skipped = filter_images_by_gpu_compatibility(built, "AMD", "gfx90a")
            assert compat == {"m1": built["m1"]}
            assert skipped == []

    def test_vendor_match_arch_mismatch_skipped(self):
        built = {"m1": {"gpu_vendor": "AMD", "gpu_architecture": "gfx90a"}}
        compat, skipped = filter_images_by_gpu_compatibility(built, "AMD", "sm_90")
        assert compat == {}
        assert len(skipped) == 1
        assert skipped[0][0] == "m1"
        assert "architecture mismatch" in skipped[0][1]

    def test_vendor_mismatch_skipped(self):
        built = {"m1": {"gpu_vendor": "NVIDIA", "gpu_architecture": "sm_90"}}
        compat, skipped = filter_images_by_gpu_compatibility(built, "AMD", "gfx90a")
        assert compat == {}
        assert len(skipped) == 1
        assert "vendor mismatch" in skipped[0][1]

    def test_none_runtime_vendor_accepts_any_vendor(self):
        built = {"m1": {"gpu_vendor": "AMD", "gpu_architecture": "gfx90a"}}
        compat, skipped = filter_images_by_gpu_compatibility(built, "NONE", "gfx90a")
        assert compat == {"m1": built["m1"]}
        assert skipped == []


class TestFilterImagesBySkipGpuArch:
    """filter_images_by_skip_gpu_arch behavior."""

    def test_disable_skip_returns_all(self):
        built = {"m1": {}}
        models = {"m1": {"skip_gpu_arch": "A100"}}
        compat, skipped = filter_images_by_skip_gpu_arch(
            built, models, "A100", disable_skip=True
        )
        assert compat == built
        assert skipped == []

    def test_no_skip_gpu_arch_included(self):
        built = {"m1": {"img": "x"}}
        models = {"m1": {}}
        compat, skipped = filter_images_by_skip_gpu_arch(built, models, "A100")
        assert compat == {"m1": built["m1"]}
        assert skipped == []

    def test_skip_gpu_arch_match_skipped(self):
        built = {"m1": {"img": "x"}}
        models = {"m1": {"skip_gpu_arch": "A100"}}
        compat, skipped = filter_images_by_skip_gpu_arch(built, models, "A100")
        assert compat == {}
        assert len(skipped) == 1
        assert skipped[0] == ("m1", built["m1"], "A100")

    def test_skip_gpu_arch_nvidia_prefix_normalized(self):
        built = {"m1": {}}
        models = {"m1": {"skip_gpu_arch": "A100"}}
        compat, skipped = filter_images_by_skip_gpu_arch(built, models, "NVIDIA A100")
        assert compat == {}
        assert skipped[0][2] == "NVIDIA A100"

    def test_skip_gpu_arch_no_match_included(self):
        built = {"m1": {}}
        models = {"m1": {"skip_gpu_arch": "A100"}}
        compat, skipped = filter_images_by_skip_gpu_arch(built, models, "gfx90a")
        assert compat == {"m1": built["m1"]}
        assert skipped == []


# ---- orchestrator init and validation ----


@pytest.mark.unit
class TestBuildOrchestratorInit:
    """Test Build Orchestrator initialization."""

    @patch("madengine.orchestration.build_orchestrator.Context")
    @patch("os.path.exists", return_value=False)
    def test_initializes_with_minimal_args(self, mock_exists, mock_context):
        """Should initialize with minimal arguments and build defaults for Dockerfile filtering."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.additional_context_file = None
        mock_args.live_output = True

        orchestrator = BuildOrchestrator(mock_args)

        assert orchestrator.args == mock_args
        assert orchestrator.additional_context == {
            "gpu_vendor": DEFAULT_GPU_VENDOR,
            "guest_os": DEFAULT_GUEST_OS,
        }
        assert orchestrator.credentials is None

    @patch("madengine.orchestration.build_orchestrator.Context")
    @patch("os.path.exists", return_value=False)
    def test_parses_additional_context_json(self, mock_exists, mock_context):
        """Should parse JSON additional context and merge build defaults."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"key": "value"}'
        mock_args.additional_context_file = None
        mock_args.live_output = True

        orchestrator = BuildOrchestrator(mock_args)

        assert orchestrator.additional_context == {
            "key": "value",
            "gpu_vendor": DEFAULT_GPU_VENDOR,
            "guest_os": DEFAULT_GUEST_OS,
        }


@pytest.mark.unit
class TestRunOrchestratorInit:
    """Test Run Orchestrator initialization."""

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_initializes_with_args(self, mock_context):
        """Should initialize with provided arguments."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        assert orchestrator.args == mock_args
        assert orchestrator.additional_context == {}

    def test_parses_deploy_type_from_context(self):
        """Should extract deploy type from additional context."""
        mock_args = MagicMock()
        mock_args.additional_context = '{"deploy": "slurm"}'
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        assert orchestrator.additional_context["deploy"] == "slurm"


@pytest.mark.unit
class TestManifestValidation:
    """Test manifest validation logic."""

    @patch("os.path.exists", return_value=False)
    def test_run_without_manifest_or_tags_raises_error(self, mock_exists):
        """Should raise ConfigurationError without manifest or tags."""
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = True

        orchestrator = RunOrchestrator(mock_args)

        with pytest.raises(ConfigurationError):
            orchestrator.execute(manifest_file=None, tags=None)


@pytest.mark.unit
class TestSkipModelRunPolicyA:
    """Policy A: --skip-model-run only skips execution after an internal build."""

    @patch.object(RunOrchestrator, "_cleanup_model_dir_copies")
    def test_skip_after_build_skips_execute_local(self, mock_cleanup, tmp_path):
        """Full workflow: skip_model_run + build phase skips _execute_local."""
        perf = tmp_path / "perf.csv"
        manifest_path = tmp_path / "build_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "deployment_config": {"target": "local"},
                    "context": {},
                    "built_images": {},
                }
            )
        )

        mock_args = MagicMock()
        mock_args.skip_model_run = True
        mock_args.additional_context = None
        mock_args.live_output = False
        mock_args.output = str(perf)

        orchestrator = RunOrchestrator(mock_args)

        with patch.object(
            RunOrchestrator, "_build_phase", return_value=str(manifest_path)
        ):
            with patch.object(
                RunOrchestrator, "_load_and_merge_manifest", side_effect=lambda f: f
            ):
                with patch.object(RunOrchestrator, "_execute_local") as mock_local:
                    with patch.object(
                        RunOrchestrator, "_combine_build_and_run_logs"
                    ) as mock_combine:
                        orchestrator.execute(
                            manifest_file=None, tags=["dummy"], timeout=60
                        )

        mock_local.assert_not_called()
        mock_combine.assert_not_called()
        mock_cleanup.assert_called()

    @patch.object(RunOrchestrator, "_cleanup_model_dir_copies")
    def test_skip_ignored_when_run_only_still_calls_execute_local(
        self, mock_cleanup, tmp_path
    ):
        """Run-only: skip_model_run is ignored; _execute_local runs."""
        perf = tmp_path / "perf.csv"
        manifest_path = tmp_path / "build_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "deployment_config": {"target": "local"},
                    "context": {},
                    "built_images": {},
                }
            )
        )

        mock_args = MagicMock()
        mock_args.skip_model_run = True
        mock_args.additional_context = None
        mock_args.live_output = False
        mock_args.output = str(perf)

        orchestrator = RunOrchestrator(mock_args)

        with patch.object(RunOrchestrator, "_execute_local") as mock_local:
            mock_local.return_value = {
                "successful_runs": [],
                "failed_runs": [],
            }
            orchestrator.execute(
                manifest_file=str(manifest_path), tags=None, timeout=60
            )

        mock_local.assert_called_once()
        mock_cleanup.assert_called()

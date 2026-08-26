"""Unit tests for orchestration: image_filtering and orchestrator init/validation."""

import json

import pytest
from unittest.mock import MagicMock, patch

from madengine.orchestration.image_filtering import (
    filter_images_by_gpu_compatibility,
    filter_images_by_skip_gpu_arch,
)
from madengine.core.additional_context_defaults import (
    DEFAULT_GPU_VENDOR,
    DEFAULT_GUEST_OS,
)
from madengine.orchestration.build_orchestrator import BuildOrchestrator
from madengine.orchestration.run_orchestrator import RunOrchestrator
from madengine.core.errors import ConfigurationError


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
        compat, skipped = filter_images_by_skip_gpu_arch(
            built, models, "NVIDIA A100"
        )
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
class TestSkipModelRun:
    """--skip-model-run is forwarded to the container runner; it never short-circuits _execute_local."""

    @patch.object(RunOrchestrator, "_cleanup_model_dir_copies")
    def test_skip_after_build_calls_execute_local(self, mock_cleanup, tmp_path):
        """Full workflow: skip_model_run + build phase still calls _execute_local (skip handled inside container runner)."""
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

        with patch.object(RunOrchestrator, "_build_phase", return_value=str(manifest_path)):
            with patch.object(
                RunOrchestrator, "_load_and_merge_manifest", side_effect=lambda f: f
            ):
                with patch.object(RunOrchestrator, "_execute_local") as mock_local:
                    mock_local.return_value = {
                        "successful_runs": [],
                        "failed_runs": [],
                    }
                    with patch.object(
                        RunOrchestrator, "_combine_build_and_run_logs"
                    ):
                        orchestrator.execute(
                            manifest_file=None, tags=["dummy"], timeout=60
                        )

        mock_local.assert_called_once()
        mock_cleanup.assert_called()

    @patch.object(RunOrchestrator, "_cleanup_model_dir_copies")
    def test_skip_run_only_still_calls_execute_local(
        self, mock_cleanup, tmp_path
    ):
        """Run-only (existing manifest): skip_model_run still calls _execute_local (skip handled inside container runner)."""
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
            orchestrator.execute(manifest_file=str(manifest_path), tags=None, timeout=60)

        mock_local.assert_called_once()
        mock_cleanup.assert_called()

    @patch.object(RunOrchestrator, "_cleanup_model_dir_copies")
    def test_skip_model_run_calls_execute_local(self, mock_cleanup, tmp_path):
        """skip_model_run no longer short-circuits before _execute_local."""
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

        with patch.object(RunOrchestrator, "_build_phase", return_value=str(manifest_path)):
            with patch.object(
                RunOrchestrator, "_load_and_merge_manifest", side_effect=lambda f: f
            ):
                with patch.object(RunOrchestrator, "_execute_local") as mock_local:
                    mock_local.return_value = {
                        "successful_runs": [],
                        "failed_runs": [],
                    }
                    orchestrator.execute(
                        manifest_file=None, tags=["dummy"], timeout=60
                    )

        mock_local.assert_called_once()
        mock_cleanup.assert_called()


@pytest.mark.unit
class TestRunOrchestrator:
    """Test RunOrchestrator methods."""

    def test_distributed_warns_on_local_only_flags(self, tmp_path):
        """_execute_distributed warns when local-only flags are set."""
        from unittest.mock import MagicMock, patch
        from madengine.orchestration.run_orchestrator import RunOrchestrator

        mock_args = MagicMock()
        mock_args.keep_alive = True
        mock_args.keep_model_dir = False
        mock_args.skip_model_run = True
        mock_args.timeout = 60
        mock_args.additional_context = None
        mock_args.live_output = False

        orchestrator = RunOrchestrator(mock_args)
        orchestrator.additional_context = {}

        # Replace rich_console with a mock so we can inspect print calls
        mock_rich_console = MagicMock()
        orchestrator.rich_console = mock_rich_console

        fake_result = MagicMock()
        fake_result.is_success = True
        fake_result.deployment_id = "test-id"
        fake_result.logs_path = None
        fake_result.metrics = {"successful_runs": [], "failed_runs": []}

        with patch("madengine.deployment.factory.DeploymentFactory.create") as mock_create:
            mock_deploy = MagicMock()
            mock_deploy.execute.return_value = fake_result
            mock_create.return_value = mock_deploy

            orchestrator._execute_distributed("slurm", str(tmp_path / "manifest.json"))

        # Verify warning was printed mentioning the active flags
        printed = " ".join(
            str(call) for call in mock_rich_console.print.call_args_list
        )
        assert "--keep-alive" in printed
        assert "--skip-model-run" in printed
        assert "--keep-model-dir" not in printed  # was False, must not appear


@pytest.mark.unit
class TestCreateManifestFromLocalImage:
    """MAD_CONTAINER_IMAGE (local image) mode must carry every models.json field
    that ContainerRunner relies on -- including multiple_results, whose absence
    silently drops perf-CSV-based result reporting (falls back to scraping the
    log for a 'performance: NUMBER METRIC' line and reports FAILURE)."""

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_multiple_results_field_is_preserved(self, mock_context, tmp_path):
        mock_context.return_value.ctx = {}
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False

        orchestrator = RunOrchestrator(mock_args)
        orchestrator.console = MagicMock()
        orchestrator.rich_console = MagicMock()

        fake_model = {
            "name": "uber_storefront/v1t1",
            "tags": ["inference"],
            "scripts": "run.sh",
            "n_gpus": "1",
            "data": "uber_storefront_models",
            "args": "--model-dir v1t1",
            "multiple_results": "perf_uber_storefront_v1t1.csv",
        }

        manifest_output = str(tmp_path / "build_manifest.json")

        with patch(
            "madengine.utils.discover_models.DiscoverModels.run",
            return_value=[fake_model],
        ):
            orchestrator._create_manifest_from_local_image(
                image_name="registry.io/org/model:ci-tag",
                tags=["inference"],
                manifest_output=manifest_output,
            )

        with open(manifest_output) as f:
            manifest = json.load(f)

        built_model = next(iter(manifest["built_models"].values()))
        assert built_model["multiple_results"] == "perf_uber_storefront_v1t1.csv"

    @patch("madengine.orchestration.run_orchestrator.Context")
    def test_multiple_results_defaults_to_empty_string(self, mock_context, tmp_path):
        """Models without multiple_results in models.json still get the key (empty),
        so ContainerRunner's model_info.get("multiple_results") lookups never KeyError."""
        mock_context.return_value.ctx = {}
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False

        orchestrator = RunOrchestrator(mock_args)
        orchestrator.console = MagicMock()
        orchestrator.rich_console = MagicMock()

        fake_model = {
            "name": "dummy/model",
            "tags": ["inference"],
            "scripts": "run.sh",
            "n_gpus": "1",
            "data": "",
            "args": "",
        }

        manifest_output = str(tmp_path / "build_manifest.json")

        with patch(
            "madengine.utils.discover_models.DiscoverModels.run",
            return_value=[fake_model],
        ):
            orchestrator._create_manifest_from_local_image(
                image_name="registry.io/org/model:ci-tag",
                tags=["inference"],
                manifest_output=manifest_output,
            )

        with open(manifest_output) as f:
            manifest = json.load(f)

        built_model = next(iter(manifest["built_models"].values()))
        assert built_model["multiple_results"] == ""


class TestPlaceholderImageRejection:
    """Model cards ship DOCKER_IMAGE_NAME as a "<supply-your-image>" marker.

    The implicit --use-image path used to accept any single distinct card value, so
    the placeholder became the image name and every compute node failed on
    `docker pull <supply-your-image>` instead of the user getting told at submit time.
    """

    @pytest.mark.parametrize("value", [
        "<supply-your-image>",
        "<your-image-here>",
        "  <supply-your-image>  ",
        "",
        None,
    ])
    def test_placeholders_detected(self, value):
        assert BuildOrchestrator._is_placeholder_image(value) is True

    @pytest.mark.parametrize("value", [
        "rocm/vllm:latest",
        "docker.io/myorg/img:tag",
        "ci-pyt_vllm_kimi_k3_mi300x",
        "localhost:5000/img",
    ])
    def test_real_images_accepted(self, value):
        assert BuildOrchestrator._is_placeholder_image(value) is False

    def test_reject_raises_configuration_error(self):
        orchestrator = BuildOrchestrator.__new__(BuildOrchestrator)
        with pytest.raises(ConfigurationError) as exc:
            orchestrator._reject_placeholder_image("<supply-your-image>", ["m1"])
        assert "placeholder" in str(exc.value).lower()

    def test_reject_passes_through_real_image(self):
        orchestrator = BuildOrchestrator.__new__(BuildOrchestrator)
        orchestrator._reject_placeholder_image("rocm/vllm:latest", ["m1"])


class TestSelfManagedLauncherImpliesSlurm:
    """A slurm_multi model card without a `slurm` block still deploys to SLURM.

    Target inference keys on the presence of a `slurm`/`k8s` block. Model cards
    routinely declare only `distributed.launcher: slurm_multi`, which inferred
    "local" and handed the job to the container runner — so the slurm_multi path
    was never reached and the model's .slurm script was run as a local workload.
    """

    @pytest.mark.parametrize("config,expected", [
        ({}, "local"),
        ({"slurm": {}}, "slurm"),
        ({"k8s": {}}, "k8s"),
        ({"distributed": {"launcher": "slurm_multi"}}, "slurm"),
        ({"distributed": {"launcher": "slurm-multi"}}, "slurm"),
        ({"distributed": {"launcher": "torchrun"}}, "local"),
        ({"distributed": {"launcher": "vllm"}}, "local"),
        ({"distributed": {}}, "local"),
        # An explicit k8s block still wins; slurm_multi is SLURM-only by construction
        # but the explicit target is the user's stated intent.
        ({"k8s": {}, "distributed": {"launcher": "slurm_multi"}}, "k8s"),
    ])
    def test_inference(self, config, expected):
        assert RunOrchestrator._infer_deployment_target(None, config) == expected

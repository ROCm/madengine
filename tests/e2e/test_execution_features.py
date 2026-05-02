"""Test the timeouts in madengine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import re
import time

import pytest

from tests.fixtures.utils import (
    BASE_DIR,
    DEFAULT_CLEAN_FILES,
    MODEL_DIR,
    build_run_command,
    clean_test_temp_files,
    generate_additional_context_for_machine,
    get_run_live_log_path,
    get_timeout_seconds_from_log,
    global_data,
    is_nvidia,
)

# ============================================================================
# Timeout Feature Tests
# ============================================================================


class TestCustomTimeoutsFunctionality:

    @pytest.mark.parametrize(
        "clean_test_temp_files", [DEFAULT_CLEAN_FILES], indirect=True
    )
    @pytest.mark.parametrize(
        "tags,log_base_name,expected_seconds,extra_args",
        [
            ("dummy", "dummy_dummy", "7200", ""),
            ("dummy_timeout", "dummy_timeout_dummy", "360", ""),
            ("dummy", "dummy_dummy", "120", "--timeout 120"),
            ("dummy_timeout", "dummy_timeout_dummy", "120", "--timeout 120"),
        ],
    )
    def test_timeout_value_in_log(
        self,
        global_data,
        clean_test_temp_files,
        tags,
        log_base_name,
        expected_seconds,
        extra_args,
    ):
        """
        Timeout is set as expected (default 2h, model override, CLI override).
        Only checks the value in the log; does not actually time the model.
        """
        global_data["console"].sh(build_run_command(tags, extra_args=extra_args))
        log_path = get_run_live_log_path(log_base_name)
        found = get_timeout_seconds_from_log(log_path)
        if found != expected_seconds:
            pytest.fail(
                f"expected timeout {expected_seconds}s in log, got {found}s (log: {log_path})."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["run_directory"]],
        indirect=True,
    )
    def test_timeout_in_commandline_timesout_correctly(
        self, global_data, clean_test_temp_files
    ):
        """
        timeout command-line argument times model out correctly
        """
        start_time = time.time()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_sleep --timeout 60",
            canFail=True,
            timeout=180,
        )

        test_duration = time.time() - start_time

        assert test_duration == pytest.approx(60, 10)

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["run_directory"]],
        indirect=True,
    )
    def test_timeout_in_model_timesout_correctly(
        self, global_data, clean_test_temp_files
    ):
        """
        timeout in models.json times model out correctly
        """
        start_time = time.time()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_sleep",
            canFail=True,
            timeout=180,
        )

        test_duration = time.time() - start_time

        assert test_duration == pytest.approx(120, 20)


# ============================================================================
# Debugging Feature Tests
# ============================================================================


class TestDebuggingFunctionality:
    """"""

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["run_directory"]],
        indirect=True,
    )
    def test_keepAlive_keeps_docker_alive(self, global_data, clean_test_temp_files):
        """
        keep-alive command-line argument keeps the docker container alive
        UPDATED: Now uses python3 -m madengine.cli.app with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"python3 -m madengine.cli.app run --live-output --tags dummy --keep-alive --additional-context '{json.dumps(context)}'"
        )
        output = global_data["console"].sh(
            "docker ps -aqf 'name=container_ci-dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
            + "'"
        )

        if not output:
            pytest.fail("docker container not found after keep-alive argument.")

        global_data["console"].sh(
            "docker container stop --time=1 container_ci-dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
        )
        global_data["console"].sh(
            "docker container rm -f container_ci-dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
        )

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["run_directory"]],
        indirect=True,
    )
    def test_no_keepAlive_does_not_keep_docker_alive(
        self, global_data, clean_test_temp_files
    ):
        """
        without keep-alive command-line argument, the docker container is not kept alive
        UPDATED: Now uses python3 -m madengine.cli.app with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"python3 -m madengine.cli.app run --live-output --tags dummy --additional-context '{json.dumps(context)}'"
        )
        output = global_data["console"].sh(
            "docker ps -aqf 'name=container_ci-dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
            + "'"
        )

        if output:
            global_data["console"].sh(
                "docker container stop --time=1 container_ci-dummy_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
            )
            global_data["console"].sh(
                "docker container rm -f container_ci-dummy_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
            )
            pytest.fail(
                "docker container found after not specifying keep-alive argument."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["run_directory"]],
        indirect=True,
    )
    def test_keepAlive_preserves_model_dir(self, global_data, clean_test_temp_files):
        """
        keep-alive command-line argument will keep model directory after run
        UPDATED: Now uses python3 -m madengine.cli.app with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"python3 -m madengine.cli.app run --live-output --tags dummy --keep-alive --additional-context '{json.dumps(context)}'"
        )

        global_data["console"].sh(
            "docker container stop --time=1 container_ci-dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
        )
        global_data["console"].sh(
            "docker container rm -f container_ci-dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
        )
        if not os.path.exists(os.path.join(BASE_DIR, "run_directory")):
            pytest.fail("model directory not left over after keep-alive argument.")

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["run_directory"]],
        indirect=True,
    )
    def test_keepModelDir_keeps_model_dir(self, global_data, clean_test_temp_files):
        """
        keep-model-dir command-line argument keeps model directory after run
        UPDATED: Now uses python3 -m madengine.cli.app with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"python3 -m madengine.cli.app run --live-output --tags dummy --keep-model-dir --additional-context '{json.dumps(context)}'"
        )

        if not os.path.exists(os.path.join(BASE_DIR, "run_directory")):
            pytest.fail("model directory not left over after keep-model-dir argument.")

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [DEFAULT_CLEAN_FILES + ["run_directory"]],
        indirect=True,
    )
    def test_no_keepModelDir_does_not_keep_model_dir(
        self, global_data, clean_test_temp_files
    ):
        """
        keep-model-dir command-line argument keeps model directory after run
        UPDATED: Now uses python3 -m madengine.cli.app with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"python3 -m madengine.cli.app run --live-output --tags dummy --additional-context '{json.dumps(context)}'"
        )

        if os.path.exists(os.path.join(BASE_DIR, "run_directory")):
            pytest.fail(
                "model directory left over after not specifying keep-model-dir (or keep-alive) argument."
            )


# ============================================================================
# Live Output Feature Tests
# ============================================================================


class TestLiveOutputFunctionality:
    """Test the live output functionality."""

    @pytest.mark.parametrize(
        "clean_test_temp_files", [DEFAULT_CLEAN_FILES], indirect=True
    )
    def test_default_silent_run(self, global_data, clean_test_temp_files):
        """
        default run is silent
        UPDATED: Now uses python3 -m madengine.cli.app instead of legacy mad.py
        """
        context = generate_additional_context_for_machine()
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"python3 -m madengine.cli.app run --tags dummy --additional-context '{json.dumps(context)}'"
        )

        regexp = re.compile(r"performance: [0-9]* samples_per_second")
        if regexp.search(output):
            pytest.fail("default run is not silent")

        if "ARG BASE_DOCKER=" in output:
            pytest.fail("default run is not silent")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [DEFAULT_CLEAN_FILES], indirect=True
    )
    def test_liveOutput_prints_output_to_screen(
        self, global_data, clean_test_temp_files
    ):
        """
        live_output prints output to screen
        UPDATED: Now uses python3 -m madengine.cli.app instead of legacy mad.py
        """
        context = generate_additional_context_for_machine()
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"python3 -m madengine.cli.app run --live-output --tags dummy --live-output --additional-context '{json.dumps(context)}'"
        )

        regexp = re.compile(r"performance: [0-9]* samples_per_second")
        if not regexp.search(output):
            pytest.fail("default run is silent")

        if "ARG BASE_DOCKER=" not in output:
            pytest.fail("default run is silent")

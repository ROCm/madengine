"""Test the timeouts in MADEngine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import json
import os
import re
import csv
import time

from tests.fixtures.utils import BASE_DIR, MODEL_DIR
from tests.fixtures.utils import global_data
from tests.fixtures.utils import clean_test_temp_files
from tests.fixtures.utils import is_nvidia
from tests.fixtures.utils import generate_additional_context_for_machine



# ============================================================================
# Timeout Feature Tests
# ============================================================================

class TestCustomTimeoutsFunctionality:

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_default_model_timeout_2hrs(self, global_data, clean_test_temp_files):
        """
        default model timeout is 2 hrs
        This test only checks if the timeout is set; it does not actually time the model.
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy"
        )

        regexp = re.compile(r"⏰ Setting timeout to ([0-9]*) seconds.")
        foundTimeout = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
                + ".run.live.log",
            ),
            "r",
        ) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                match = regexp.search(line)
                if match:
                    foundTimeout = match.groups()[0]
        if foundTimeout != "7200":
            pytest.fail("default model timeout is not 2 hrs (" + str(foundTimeout) + "s).")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_can_override_timeout_in_model(self, global_data, clean_test_temp_files):
        """
        timeout can be overridden in model
        This test only checks if the timeout is set; it does not actually time the model.
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_timeout"
        )

        regexp = re.compile(r"⏰ Setting timeout to ([0-9]*) seconds.")
        foundTimeout = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_timeout_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
                + ".run.live.log",
            ),
            "r",
        ) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                match = regexp.search(line)
                if match:
                    foundTimeout = match.groups()[0]
        if foundTimeout != "360":
            pytest.fail(
                "timeout in models.json (360s) could not override actual timeout ("
                + str(foundTimeout)
                + "s)."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_can_override_timeout_in_commandline(
        self, global_data, clean_test_temp_files
    ):
        """
        timeout command-line argument overrides default timeout
        This test only checks if the timeout is set; it does not actually time the model.
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy --timeout 120"
        )

        regexp = re.compile(r"⏰ Setting timeout to ([0-9]*) seconds.")
        foundTimeout = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
                + ".run.live.log",
            ),
            "r",
        ) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                match = regexp.search(line)
                if match:
                    foundTimeout = match.groups()[0]
        if foundTimeout != "120":
            pytest.fail(
                "timeout command-line argument (120s) could not override actual timeout ("
                + foundTimeout
                + "s)."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_commandline_timeout_overrides_model_timeout(
        self, global_data, clean_test_temp_files
    ):
        """
        timeout command-line argument overrides model timeout
        This test only checks if the timeout is set; it does not actually time the model.
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_timeout --timeout 120"
        )

        regexp = re.compile(r"⏰ Setting timeout to ([0-9]*) seconds.")
        foundTimeout = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_timeout_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
                + ".run.live.log",
            ),
            "r",
        ) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                match = regexp.search(line)
                if match:
                    foundTimeout = match.groups()[0]
        if foundTimeout != "120":
            pytest.fail(
                "timeout in command-line argument (360s) could not override model.json timeout ("
                + foundTimeout
                + "s)."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "run_directory"]],
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
        [["perf.csv", "perf.html", "run_directory"]],
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
        [["perf.csv", "perf.html", "run_directory"]],
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
        [["perf.csv", "perf.html", "run_directory"]],
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
        [["perf.csv", "perf.html", "run_directory"]],
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
        [["perf.csv", "perf.html", "run_directory"]],
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
        [["perf.csv", "perf.html", "run_directory"]],
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
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
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
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
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



"""Test the functionality of live output in MADEngine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import re
import pytest

# project modules
from .fixtures.utils import global_data
from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import clean_test_temp_files


class TestLiveOutputFunctionality:
    """Test the live output functionality."""

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_default_silent_run(self, global_data, clean_test_temp_files):
        """
        default run is silent
        """
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy"
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
        """
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --live-output"
        )

        regexp = re.compile(r"performance: [0-9]* samples_per_second")
        if not regexp.search(output):
            pytest.fail("default run is silent")

        if "ARG BASE_DOCKER=" not in output:
            pytest.fail("default run is silent")

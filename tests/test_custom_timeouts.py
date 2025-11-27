"""Test the timeouts in MADEngine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import os
import re
import csv
import time

from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files
from .fixtures.utils import is_nvidia


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
            + "python3 src/madengine/mad.py run --tags dummy"
        )

        regexp = re.compile(r"Setting timeout to ([0-9]*) seconds.")
        foundTimeout = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
                + ".live.log",
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
            pytest.fail("default model timeout is not 2 hrs (" + foundTimeout + "s).")

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
            + "python3 src/madengine/mad.py run --tags dummy_timeout"
        )

        regexp = re.compile(r"Setting timeout to ([0-9]*) seconds.")
        foundTimeout = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_timeout_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
                + ".live.log",
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
                + foundTimeout
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
            + "python3 src/madengine/mad.py run --tags dummy --timeout 120"
        )

        regexp = re.compile(r"Setting timeout to ([0-9]*) seconds.")
        foundTimeout = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
                + ".live.log",
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
            + "python3 src/madengine/mad.py run --tags dummy_timeout --timeout 120"
        )

        regexp = re.compile(r"Setting timeout to ([0-9]*) seconds.")
        foundTimeout = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_timeout_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
                + ".live.log",
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
            + "python3 src/madengine/mad.py run --tags dummy_sleep --timeout 60",
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
            + "python3 src/madengine/mad.py run --tags dummy_sleep",
            canFail=True,
            timeout=180,
        )

        test_duration = time.time() - start_time

        assert test_duration == pytest.approx(120, 20)

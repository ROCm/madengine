"""Test the scripts for pre and post processing.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import re
import csv
import time

# 3rd party modules
import pytest

# project modules
from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files
from .fixtures.utils import is_nvidia


class TestPrePostScriptsFunctionality:

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_pre_scripts_run_before_model(self, global_data, clean_test_temp_files):
        """
        pre_scripts are run in docker container before model execution
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --additional-context \"{ 'pre_scripts':[{'path':'scripts/common/pre_scripts/pre_test.sh'}] }\" "
        )

        regexp = re.compile(r"Pre-Script test called ([0-9]*)")
        foundLine = None
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
                    foundLine = match.groups()[0]
        if foundLine != "0":
            pytest.fail(
                "pre_scripts specification did not run the selected pre-script."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_post_scripts_run_after_model(self, global_data, clean_test_temp_files):
        """
        post_scripts are run in docker container after model execution
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --additional-context \"{ 'post_scripts':[{'path':'scripts/common/post_scripts/post_test.sh'}] }\" "
        )

        regexp = re.compile(r"Post-Script test called ([0-9]*)")
        foundLine = None
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
                    foundLine = match.groups()[0]
        if foundLine != "0":
            pytest.fail(
                "post_scripts specification did not run the selected post-script."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_pre_scripts_accept_arguments(self, global_data, clean_test_temp_files):
        """
        pre_scripts are run in docker container before model execution and accept arguments
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --additional-context \"{ 'pre_scripts':[{'path':'scripts/common/pre_scripts/pre_test.sh', 'args':'1'}] }\" "
        )

        regexp = re.compile(r"Pre-Script test called ([0-9]*)")
        foundLine = None
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
                    foundLine = match.groups()[0]
        if foundLine != "1":
            pytest.fail(
                "pre_scripts specification did not run the selected pre-script."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_post_scripts_accept_arguments(self, global_data, clean_test_temp_files):
        """
        post_scripts are run in docker container after model execution and accept arguments
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --additional-context \"{ 'post_scripts':[{'path':'scripts/common/post_scripts/post_test.sh', 'args':'1'}] }\" "
        )

        regexp = re.compile(r"Post-Script test called ([0-9]*)")
        foundLine = None
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
                    foundLine = match.groups()[0]
        if foundLine != "1":
            pytest.fail(
                "post_scripts specification did not run the selected post-script."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_both_pre_and_post_scripts_run_before_and_after_model(
        self, global_data, clean_test_temp_files
    ):
        """
        post_scripts are run in docker container after model execution
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --additional-context \"{ 'pre_scripts':[{'path':'scripts/common/pre_scripts/pre_test.sh'}], 'post_scripts':[{'path':'scripts/common/post_scripts/post_test.sh'}] }\" "
        )

        regexp = re.compile(r"Pre-Script test called ([0-9]*)")
        foundLine = None
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
                    foundLine = match.groups()[0]
        if foundLine != "0":
            pytest.fail(
                "pre_scripts specification did not run the selected pre-script."
            )

        regexp = re.compile(r"Post-Script test called ([0-9]*)")
        foundLine = None
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
                    foundLine = match.groups()[0]
        if foundLine != "0":
            pytest.fail(
                "post_scripts specification did not run the selected post-script."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_all_pre_scripts_run_in_order(self, global_data, clean_test_temp_files):
        """
        all pre_scripts are run in order
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --additional-context \"{ 'pre_scripts':[{'path':'scripts/common/pre_scripts/pre_test.sh', 'args':'1'}, {'path':'scripts/common/pre_scripts/pre_test.sh', 'args':'2'} ] }\" "
        )

        regexp = re.compile(r"Pre-Script test called ([0-9]*)")
        foundLine = None
        pre_post_script_count = 0
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
                    foundLine = match.groups()[0]
                    pre_post_script_count += 1
                    if foundLine != str(pre_post_script_count):
                        pytest.fail(
                            "pre_scripts run in order. Did not find "
                            + str(pre_post_script_count)
                        )

        if foundLine != "2":
            pytest.fail(
                "pre_scripts specification did not run the selected pre-script."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_all_post_scripts_run_in_order(self, global_data, clean_test_temp_files):
        """
        all post_scripts are run in order
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --additional-context \"{ 'post_scripts':[{'path':'scripts/common/post_scripts/post_test.sh', 'args':'1'}, {'path':'scripts/common/post_scripts/post_test.sh', 'args':'2'} ] }\" "
        )

        regexp = re.compile(r"Post-Script test called ([0-9]*)")
        foundLine = None
        pre_post_script_count = 0
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
                    foundLine = match.groups()[0]
                    pre_post_script_count += 1
                    if foundLine != str(pre_post_script_count):
                        pytest.fail(
                            "post_scripts run in order. Did not find "
                            + str(pre_post_script_count)
                        )

        if foundLine != "2":
            pytest.fail(
                "post_scripts specification did not run the selected post-script."
            )

"""Test various Build workflows and command-line arguments.

This module tests various command-line argument behaviors including:
- Output file path specification (-o flag)
- GPU architecture checking and skip flags
- Multiple results output handling

UPDATED: Refactored to use python3 -m madengine.cli.app instead of legacy mad.py

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import sys
import csv
import json
import pandas as pd

# 3rd party modules
import pytest

# project modules
from tests.fixtures.utils import BASE_DIR, MODEL_DIR
from tests.fixtures.utils import global_data
from tests.fixtures.utils import clean_test_temp_files
from tests.fixtures.utils import generate_additional_context_for_machine



# ============================================================================
# Build CLI Features Tests
# ============================================================================

class TestCLIFeatures:
    """Test various CLI features and command-line argument behaviors."""

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf_test.csv", "perf_test.html"]], indirect=True
    )
    def test_output_commandline_argument_writes_csv_correctly(
        self, global_data, clean_test_temp_files
    ):
        """
        Test that -o/--output command-line argument writes CSV file to specified path.
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
            + f"python3 -m madengine.cli.app run --tags dummy -o perf_test.csv --live-output --additional-context '{json.dumps(context)}'"
        )
        success = False
        with open(os.path.join(BASE_DIR, "perf_test.csv"), "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row["model"] == "dummy":
                    if row["status"] == "SUCCESS":
                        success = True
                        break
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("model, dummy, not found in perf_test.csv.")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf_test.csv", "perf_test.html"]], indirect=True
    )
    def test_commandline_argument_skip_gpu_arch(
        self, global_data, clean_test_temp_files
    ):
        """
        Test that skip_gpu_arch command-line argument skips GPU architecture check.
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
            + f"python3 -m madengine.cli.app run --tags dummy_skip_gpu_arch --live-output --additional-context '{json.dumps(context)}'"
        )
        if "Skipping model" not in output:
            pytest.fail("Enable skipping gpu arch for running model is failed.")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf_test.csv", "perf_test.html"]], indirect=True
    )
    def test_commandline_argument_disable_skip_gpu_arch_fail(
        self, global_data, clean_test_temp_files
    ):
        """
        Test that --disable-skip-gpu-arch fails GPU architecture check as expected.
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
            + f"python3 -m madengine.cli.app run --tags dummy_skip_gpu_arch --disable-skip-gpu-arch --live-output --additional-context '{json.dumps(context)}'"
        )
        # Check if exception with message 'Skipping model' is thrown
        if "Skipping model" in output:
            pytest.fail("Disable skipping gpu arch for running model is failed.")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf_test.csv", "perf_test.html"]], indirect=True
    )
    def test_output_multi_results(self, global_data, clean_test_temp_files):
        """
        Test that multiple results are correctly written and merged into output CSV.
        UPDATED: Now uses python3 -m madengine.cli.app instead of legacy mad.py
        """
        context = generate_additional_context_for_machine()
        output = global_data['console'].sh(
            "cd " + BASE_DIR + "; " + 
            "MODEL_DIR=" + MODEL_DIR + " " + 
            f"python3 -m madengine.cli.app run --tags dummy_multi --live-output --additional-context '{json.dumps(context)}'"
        )
        # Check if multiple results are written to perf_dummy.csv
        success = False
        # Read the csv file to a dataframe using pandas
        multi_df = pd.read_csv(os.path.join(BASE_DIR, 'perf_dummy.csv'))
        # Check the number of rows in the dataframe is 4, and columns is 4
        if multi_df.shape == (4, 4):
            success = True
        if not success:
            pytest.fail("The generated multi results is not correct.")
        # Check if multiple results from perf_dummy.csv get copied over to perf.csv
        perf_df = pd.read_csv(os.path.join(BASE_DIR, 'perf.csv'))
        # Get the corresponding rows and columns from perf.csv
        perf_df = perf_df[multi_df.columns]
        perf_df = perf_df.iloc[-4:, :]
        # Drop model columns from both dataframes; these will not match
        # if multiple results csv has {model}, then perf csv has {tag_name}_{model}
        multi_df = multi_df.drop('model', axis=1)
        perf_df = perf_df.drop('model', axis=1)
        if all(perf_df.columns == multi_df.columns):
            success = True
        if not success:
            pytest.fail("The columns of the generated multi results do not match perf.csv.")




# ============================================================================
# Model Discovery Tests
# ============================================================================

class TestDiscover:
    """Test the model discovery feature."""

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_static(self, global_data, clean_test_temp_files):
        """
        test a tag from a models.json file
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy2/model2 "
        )

        success = False
        with open(os.path.join(BASE_DIR, "perf.csv"), "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row["model"] == "dummy2/model2" and row["status"] == "SUCCESS":
                    success = True
        if not success:
            pytest.fail("dummy2/model2 did not run successfully.")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_dynamic(self, global_data, clean_test_temp_files):
        """
        test a tag from a get_models_json.py file
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy3/model4 "
        )

        success = False
        with open(os.path.join(BASE_DIR, "perf.csv"), "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row["model"] == "dummy3/model4" and row["status"] == "SUCCESS":
                    success = True
        if not success:
            pytest.fail("dummy3/model4 did not run successfully.")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_additional_args(self, global_data, clean_test_temp_files):
        """
        passes additional args specified in the command line to the model
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy2/model2:batch-size=32 "
        )

        success = False
        with open(os.path.join(BASE_DIR, "perf.csv"), "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if (
                    row["model"] == "dummy2/model2"
                    and row["status"] == "SUCCESS"
                    and "--batch-size 32" in row["args"]
                ):
                    success = True
        if not success:
            pytest.fail("dummy2/model2:batch-size=32 did not run successfully.")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_multiple(self, global_data, clean_test_temp_files):
        """
        test multiple tags from top-level models.json, models.json in a script subdir, and get_models_json.py
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_test_group_1,dummy_test_group_2,dummy_test_group_3 "
        )

        success = False
        with open(os.path.join(BASE_DIR, "perf.csv"), "r") as csv_file:
            csv_reader = pd.read_csv(csv_file)
            if len(csv_reader) == 5:
                if csv_reader["model"].tolist() == [
                    "dummy",
                    "dummy2/model1",
                    "dummy2/model2",
                    "dummy3/model3",
                    "dummy3/model4",
                ]:
                    if csv_reader["status"].tolist() == [
                        "SUCCESS",
                        "SUCCESS",
                        "SUCCESS",
                        "SUCCESS",
                        "SUCCESS",
                    ]:
                        success = True
        if not success:
            pytest.fail("multiple tags did not run successfully.")



"""Test the tags feature.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import csv
import pandas as pd

# third-party modules
import pytest
import json

# project modules
from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files
from .fixtures.utils import generate_additional_context_for_machine


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
            + "madengine-cli run --live-output --tags dummy2/model2 "
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
            + "madengine-cli run --live-output --tags dummy3/model4 "
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
            + "madengine-cli run --live-output --tags dummy2/model2:batch-size=32 "
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
            + "madengine-cli run --live-output --tags dummy_test_group_1 dummy_test_group_2 dummy_test_group_3 "
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

"""Test the data provider module.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import sys
import csv
import re
import json
import tempfile

# third-party modules
import pytest

# project modules
from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files
from madengine.core.dataprovider import Data


class TestDataProviders:

    def test_reorder_data_provider_config(self):
        """
        Test the reorder_data_provider_config function to ensure it correctly orders data provider types
        """
        # Create a temporary data.json file with shuffled data provider types
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as temp_file:
            test_data = {
                "test_data": {
                    "aws": {"path": "s3://bucket/path"},
                    "local": {"path": "/local/path"},
                    "nas": {"path": "/nas/path"},
                    "custom": {"path": "scripts/custom.sh"},
                    "minio": {"path": "minio://bucket/path"},
                }
            }
            json.dump(test_data, temp_file)
            temp_file_path = temp_file.name

        try:
            # Create Data object with the test file
            data_obj = Data(filename=temp_file_path)

            # Check the initial order (should be as defined in the test_data)
            original_keys = list(data_obj.data_provider_config["test_data"].keys())

            # Call the reorder function
            data_obj.reorder_data_provider_config("test_data")

            # Check the order after reordering
            reordered_keys = list(data_obj.data_provider_config["test_data"].keys())
            expected_order = ["custom", "local", "minio", "nas", "aws"]

            # Filter expected_order to only include keys that exist in original_keys
            expected_filtered = [k for k in expected_order if k in original_keys]

            # Assert that the reordering happened correctly
            assert (
                reordered_keys == expected_filtered
            ), f"Expected order {expected_filtered}, got {reordered_keys}"

            # Specifically check that custom comes first, if it exists
            if "custom" in original_keys:
                assert (
                    reordered_keys[0] == "custom"
                ), "Custom should be first in the order"

            # Check that the order matches the expected priority
            for i, key in enumerate(reordered_keys):
                expected_index = expected_order.index(key)
                for j, other_key in enumerate(reordered_keys[i + 1 :], i + 1):
                    other_expected_index = expected_order.index(other_key)
                    assert (
                        expected_index < other_expected_index
                    ), f"{key} should come before {other_key}"

        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_local_data_provider_runs_successfully(
        self, global_data, clean_test_temp_files
    ):
        """
        local data provider gets data from local disk
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy_data_local "
        )

        success = False
        with open(os.path.join(BASE_DIR, "perf.csv"), "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row["model"] == "dummy_data_local":
                    if row["status"] == "SUCCESS":
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("local data provider test failed")

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "run_directory"]],
        indirect=True,
    )
    def test_model_executes_even_if_data_provider_fails(
        self, global_data, clean_test_temp_files
    ):
        """
        model executes even if data provider fails
        """
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy_data_local_fail --additional-context \"{'docker_env_vars':{'MAD_DATAHOME':'/data'} }\" --live-output ",
            canFail=True,
        )

        success = False
        with open(os.path.join(BASE_DIR, "perf.csv"), "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row["model"] == "dummy_data_local_fail":
                    if row["status"] == "FAILURE":
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("local data provider fail test passed")

        # Search for "/data is NOT mounted" to ensure model script ran
        regexp = re.compile(r"is NOT mounted")
        if not regexp.search(output):
            pytest.fail("model did not execute after data provider failed")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html", "dataLocal"]], indirect=True
    )
    def test_local_data_provider_mirrorlocal_does_not_mirror_data(
        self, global_data, clean_test_temp_files
    ):
        """
        In local data provider, mirrorlocal field in data.json does not mirror data in local disk
        """
        mirrorPath = os.path.join(BASE_DIR, "dataLocal")
        os.mkdir(mirrorPath)
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy_data_local --force-mirror-local "
            + mirrorPath
        )

        success = False
        with open(os.path.join(BASE_DIR, "perf.csv"), "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row["model"] == "dummy_data_local":
                    if row["status"] == "SUCCESS":
                        success = True
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("local data provider test failed")

        if os.path.exists(os.path.join(mirrorPath, "dummy_data_local")):
            pytest.fail("custom data provider did mirror data locally")

"""
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import os
import sys
import json

from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files


class TestTagsFunctionality:

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_can_select_model_subset_with_commandline_tag_argument(
        self, global_data, clean_test_temp_files
    ):
        """
        can select subset of models with tag with command-line argument
        """
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy_group_1"
        )

        if "Running model dummy" not in output:
            pytest.fail("dummy tag not selected with commandline --tags argument")

        if "Running model dummy2" not in output:
            pytest.fail("dummy2 tag not selected with commandline --tags argument")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_all_models_matching_any_tag_selected_with_multiple_tags(
        self, global_data, clean_test_temp_files
    ):
        """
        if multiple tags are specified, all models that match any tag will be selected
        """
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy_group_1 dummy_group_2"
        )

        if "Running model dummy" not in output:
            pytest.fail("dummy tag not selected with commandline --tags argument")

        if "Running model dummy2" not in output:
            pytest.fail("dummy2 tag not selected with commandline --tags argument")

        if "Running model dummy3" not in output:
            pytest.fail("dummy3 tag not selected with commandline --tags argument")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_model_names_are_automatically_tags(
        self, global_data, clean_test_temp_files
    ):
        """
        Each model name is automatically a tag
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

        if "Running model dummy" not in output:
            pytest.fail("dummy tag not selected with commandline --tags argument")

"""Test tag functionality in MADEngine.

UPDATED: Refactored to use madengine-cli instead of legacy mad.py

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import os
import sys
import json

from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files
from .fixtures.utils import generate_additional_context_for_machine


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
        context = generate_additional_context_for_machine()
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            +             f"madengine-cli run --tags dummy_group_1 --live-output --additional-context '{json.dumps(context)}'"
        )

        # Check for model execution (handles ANSI codes in output)
        if "dummy" not in output or "ci-dummy_dummy" not in output:
            pytest.fail("dummy tag not selected with commandline --tags argument")

        if "dummy2" not in output or "ci-dummy2_dummy" not in output:
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
        context = generate_additional_context_for_machine()
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"madengine-cli run --tags dummy_group_1,dummy_group_2 --live-output --additional-context '{json.dumps(context)}'"
        )

        # Check for model execution (handles ANSI codes in output)
        if "dummy" not in output or "ci-dummy_dummy" not in output:
            pytest.fail("dummy tag not selected with commandline --tags argument")

        if "dummy2" not in output or "ci-dummy2_dummy" not in output:
            pytest.fail("dummy2 tag not selected with commandline --tags argument")

        if "dummy3" not in output or "ci-dummy3_dummy" not in output:
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
        context = generate_additional_context_for_machine()
        output = global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            +             f"madengine-cli run --tags dummy --live-output --additional-context '{json.dumps(context)}'"
        )

        # Check for model execution (handles ANSI codes in output)
        if "dummy" not in output or "ci-dummy_dummy" not in output:
            pytest.fail("dummy tag not selected with commandline --tags argument")

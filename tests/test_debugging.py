"""Test the debugging in MADEngine.

UPDATED: Refactored to use madengine-cli instead of legacy mad.py

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import os
import re
import json

from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files
from .fixtures.utils import is_nvidia
from .fixtures.utils import generate_additional_context_for_machine


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
        UPDATED: Now uses madengine-cli with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"madengine-cli run --live-output --tags dummy --keep-alive --additional-context '{json.dumps(context)}'"
        )
        output = global_data["console"].sh(
            "docker ps -aqf 'name=container_dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
            + "'"
        )

        if not output:
            pytest.fail("docker container not found after keep-alive argument.")

        global_data["console"].sh(
            "docker container stop --time=1 container_dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
        )
        global_data["console"].sh(
            "docker container rm -f container_dummy_dummy.ubuntu."
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
        UPDATED: Now uses madengine-cli with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"madengine-cli run --live-output --tags dummy --additional-context '{json.dumps(context)}'"
        )
        output = global_data["console"].sh(
            "docker ps -aqf 'name=container_dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
            + "'"
        )

        if output:
            global_data["console"].sh(
                "docker container stop --time=1 container_dummy_dummy.ubuntu."
                + ("amd" if not is_nvidia() else "nvidia")
            )
            global_data["console"].sh(
                "docker container rm -f container_dummy_dummy.ubuntu."
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
        UPDATED: Now uses madengine-cli with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"madengine-cli run --live-output --tags dummy --keep-alive --additional-context '{json.dumps(context)}'"
        )

        global_data["console"].sh(
            "docker container stop --time=1 container_dummy_dummy.ubuntu."
            + ("amd" if not is_nvidia() else "nvidia")
        )
        global_data["console"].sh(
            "docker container rm -f container_dummy_dummy.ubuntu."
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
        UPDATED: Now uses madengine-cli with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"madengine-cli run --live-output --tags dummy --keep-model-dir --additional-context '{json.dumps(context)}'"
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
        UPDATED: Now uses madengine-cli with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"madengine-cli run --live-output --tags dummy --additional-context '{json.dumps(context)}'"
        )

        if os.path.exists(os.path.join(BASE_DIR, "run_directory")):
            pytest.fail(
                "model directory left over after not specifying keep-model-dir (or keep-alive) argument."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "run_directory"]],
        indirect=True,
    )
    def test_skipModelRun_does_not_run_model(self, global_data, clean_test_temp_files):
        """
        skip-model-run command-line argument does not run model
        UPDATED: Now uses madengine-cli with additional-context
        """
        context = generate_additional_context_for_machine()
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + f"madengine-cli run --live-output --tags dummy --skip-model-run --additional-context '{json.dumps(context)}'"
        )

        regexp = re.compile(r"performance: [0-9]* samples_per_second")
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
                if regexp.search(line):
                    pytest.fail("skip-model-run argument ran model.")

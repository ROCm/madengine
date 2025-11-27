"""Test the debugging in MADEngine.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import os
import re

from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files
from .fixtures.utils import is_nvidia


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
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --keep-alive"
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
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --keep-alive"
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
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --keep-model-dir"
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
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 src/madengine/mad.py run --tags dummy --skip-model-run"
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

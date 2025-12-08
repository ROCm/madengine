"""Test the profiling functionality.

UPDATED: Refactored to use python3 -m madengine.cli.app instead of legacy mad.py

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
import re
import sys
import csv
import json

# third-party modules
import pytest

# project modules
from tests.fixtures.utils import (
    BASE_DIR,
    MODEL_DIR,
    global_data,
    clean_test_temp_files,
    requires_gpu,
    is_nvidia,
    generate_additional_context_for_machine,
)


class TestProfilingFunctionality:

    @pytest.mark.skipif(is_nvidia(), reason="test does not run on NVIDIA")
    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "rocprof_output"]],
        indirect=True,
    )
    def test_rocprof_profiling_tool_runs_correctly(
        self, global_data, clean_test_temp_files
    ):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        # canFail is set to True because rocProf mode is failing the full DLM run; this test will test if the correct output files are generated
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"rocprof\"}]}' ",
            canFail=True,
        )

        # Check for both legacy rocprof (results.csv) and rocprofv3 (.db files) output
        rocprof_output_dir = os.path.join(BASE_DIR, "rocprof_output")
        legacy_output = os.path.join(rocprof_output_dir, "results.csv")
        
        # Check for rocprofv3 .db files in subdirectories
        rocprofv3_output_found = False
        if os.path.exists(rocprof_output_dir):
            for root, dirs, files in os.walk(rocprof_output_dir):
                for file in files:
                    if file.endswith("_results.db"):
                        rocprofv3_output_found = True
                        break
                if rocprofv3_output_found:
                    break
        
        if not os.path.exists(legacy_output) and not rocprofv3_output_found:
            pytest.fail(
                "Neither rocprof_output/results.csv (legacy) nor *_results.db (rocprofv3) generated with rocprof profiling run."
            )

    @pytest.mark.skipif(is_nvidia(), reason="test does not run on NVIDIA")
    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "rpd_output"]],
        indirect=True,
    )
    def test_rpd_profiling_tool_runs_correctly(
        self, global_data, clean_test_temp_files
    ):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        # canFail is set to True because rpd mode is failing the full DLM run; this test will test if the correct output files are generated
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"rpd\"}]}' ",
            canFail=True,
        )

        if not os.path.exists(os.path.join(BASE_DIR, "rpd_output", "trace.rpd")):
            pytest.fail("rpd_output/trace.rpd not generated with rpd profiling run.")

    @requires_gpu("gpu_info_power_profiler requires GPU hardware")
    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "gpu_info_power_profiler_output.csv"]],
        indirect=True,
    )
    def test_gpu_info_power_profiling_tool_runs_correctly(
        self, global_data, clean_test_temp_files
    ):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"gpu_info_power_profiler\"}]}' ",
            canFail=False,
        )

        if not os.path.exists(
            os.path.join(BASE_DIR, "gpu_info_power_profiler_output.csv")
        ):
            pytest.fail(
                "gpu_info_power_profiler_output.csv not generated with gpu_info_power_profiler run."
            )

    @requires_gpu("gpu_info_vram_profiler requires GPU hardware")
    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "gpu_info_vram_profiler_output.csv"]],
        indirect=True,
    )
    def test_gpu_info_vram_profiling_tool_runs_correctly(
        self, global_data, clean_test_temp_files
    ):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"gpu_info_vram_profiler\"}]}' ",
            canFail=False,
        )

        if not os.path.exists(
            os.path.join(BASE_DIR, "gpu_info_vram_profiler_output.csv")
        ):
            pytest.fail(
                "gpu_info_vram_profiler_output.csv not generated with gpu_info_vram_profiler run."
            )

    @pytest.mark.skipif(is_nvidia(), reason="test does not run on NVIDIA")
    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "library_trace.csv"]],
        indirect=True,
    )
    def test_rocblas_trace_runs_correctly(self, global_data, clean_test_temp_files):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"rocblas_trace\"}]}' ",
            canFail=False,
        )

        regexp = re.compile(r"rocblas-bench")
        foundMatch = None
        with open(os.path.join(BASE_DIR, "library_trace.csv"), "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                match = regexp.search(line)
                if match:
                    foundMatch = True
        if not foundMatch:
            pytest.fail(
                "could not detect rocblas-bench in output log file with rocblas trace tool."
            )

    @pytest.mark.skipif(is_nvidia(), reason="test does not run on NVIDIA")
    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "library_trace.csv"]],
        indirect=True,
    )
    def test_tensile_trace_runs_correctly(self, global_data, clean_test_temp_files):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"tensile_trace\"}]}' ",
            canFail=False,
        )

        regexp = re.compile(r"tensile,Cijk")
        foundMatch = None
        with open(os.path.join(BASE_DIR, "library_trace.csv"), "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                match = regexp.search(line)
                if match:
                    foundMatch = True
        if not foundMatch:
            pytest.fail(
                "could not detect tensile call in output log file with tensile trace tool."
            )

    @pytest.mark.skipif(is_nvidia(), reason="test does not run on NVIDIA")
    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "library_trace.csv"]],
        indirect=True,
    )
    def test_miopen_trace_runs_correctly(self, global_data, clean_test_temp_files):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"miopen_trace\"}]}' ",
            canFail=False,
        )

        regexp = re.compile(r"MIOpenDriver")
        foundMatch = None
        with open(os.path.join(BASE_DIR, "library_trace.csv"), "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                match = regexp.search(line)
                if match:
                    foundMatch = True
        if not foundMatch:
            pytest.fail(
                "could not detect miopen call in output log file with miopen trace tool."
            )

    @pytest.mark.skipif(is_nvidia(), reason="test does not run on NVIDIA")
    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_rccl_trace_runs_correctly(self, global_data, clean_test_temp_files):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof_rccl --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"rccl_trace\"}]}' ",
            canFail=False,
        )

        regexp = re.compile(r"NCCL INFO AllReduce:")
        foundMatch = None
        with open(
            os.path.join(
                BASE_DIR,
                "dummy_prof_rccl_dummy.ubuntu."
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
                    foundMatch = True
        if not foundMatch:
            pytest.fail(
                "could not detect rccl call in output log file with rccl trace tool."
            )

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_toolA_runs_correctly(self, global_data, clean_test_temp_files):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"test_tools_A\"}]}' ",
            canFail=False,
        )

        match_str_array = ["^pre_script A$", "^cmd_A$", "^post_script A$"]

        match_str_idx = 0
        regexp = re.compile(match_str_array[match_str_idx])
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
                    print("MATCH = ", line)
                    match_str_idx = match_str_idx + 1
                    if match_str_idx == len(match_str_array):
                        break
                    regexp = re.compile(match_str_array[match_str_idx])
        if match_str_idx != len(match_str_array):
            print("Matched up to ", match_str_idx)
            pytest.fail("all strings were not matched in toolA test.")

    @pytest.mark.parametrize(
        "clean_test_temp_files", [["perf.csv", "perf.html"]], indirect=True
    )
    def test_stackable_design_runs_correctly(self, global_data, clean_test_temp_files):
        """
        specifying a profiling tool runs respective pre and post scripts
        """
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"test_tools_A\"}, {\"name\": \"test_tools_B\"}]}' ",
            canFail=False,
        )

        match_str_array = [
            "^pre_script B$",
            "^pre_script A$",
            "^cmd_B$",
            "^cmd_A$",
            "^post_script A$",
            "^post_script B$",
        ]

        match_str_idx = 0
        regexp = re.compile(match_str_array[match_str_idx])
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
                    print("MATCH = ", line)
                    match_str_idx = match_str_idx + 1
                    if match_str_idx == len(match_str_array):
                        break
                    regexp = re.compile(match_str_array[match_str_idx])
        if match_str_idx != len(match_str_array):
            print("Matched up to ", match_str_idx)
            pytest.fail(
                "all strings were not matched in the stacked test using toolA and toolB."
            )

    @pytest.mark.skipif(is_nvidia(), reason="test does not run on NVIDIA")
    @pytest.mark.parametrize(
        "clean_test_temp_files",
        [["perf.csv", "perf.html", "rocprof_output"]],
        indirect=True,
    )
    def test_can_change_default_behavior_of_profiling_tool_with_additionalContext(
        self, global_data, clean_test_temp_files
    ):
        """
        default behavior of a profiling tool can be changed from additional-context
        """
        # Test overriding with --sys-trace (works with both rocprof and rocprofv3)
        # canFail is set to True because rocProf is failing; this test will test if the correct output files are generated
        global_data["console"].sh(
            "cd "
            + BASE_DIR
            + "; "
            + "MODEL_DIR="
            + MODEL_DIR
            + " "
            + "python3 -m madengine.cli.app run --live-output --tags dummy_prof --additional-context '{\"gpu_vendor\": \"AMD\", \"guest_os\": \"UBUNTU\", \"tools\": [{\"name\": \"rocprof\", \"cmd\": \"bash ../scripts/common/tools/rocprof_wrapper.sh --sys-trace\"}]}' ",
            canFail=True,
        )

        # Check for profiling output (either legacy or rocprofv3 format)
        rocprof_output_dir = os.path.join(BASE_DIR, "rocprof_output")
        
        # For rocprofv3 with --sys-trace, check for .db files
        rocprofv3_output_found = False
        if os.path.exists(rocprof_output_dir):
            for root, dirs, files in os.walk(rocprof_output_dir):
                for file in files:
                    if file.endswith("_results.db"):
                        rocprofv3_output_found = True
                        break
                if rocprofv3_output_found:
                    break
        
        # Legacy check for results files
        legacy_output = os.path.exists(os.path.join(BASE_DIR, "rocprof_output", "results.csv"))
        
        if not legacy_output and not rocprofv3_output_found:
            pytest.fail(
                "No profiling output generated with custom rocprof command override."
            )

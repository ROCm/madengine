"""Test the misc modules.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import sys
import csv
import pandas as pd
# 3rd party modules
import pytest
# project modules
from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files


class TestMiscFunctionality:

    @pytest.mark.parametrize('clean_test_temp_files', [['perf_test.csv', 'perf_test.html']], indirect=True)
    def test_output_commandline_argument_writes_csv_correctly(self, global_data, clean_test_temp_files):
        """ 
        output command-line argument writes csv file to specified output path
        """
        output = global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy -o perf_test.csv") 
        success = False
        with open(os.path.join(BASE_DIR, 'perf_test.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['model'] == 'dummy':
                    if row['status'] == 'SUCCESS': 
                        success = True
                        break
                    else:
                        pytest.fail("model in perf_test.csv did not run successfully.")
        if not success:
            pytest.fail("model, dummy, not found in perf_test.csv.")

    @pytest.mark.parametrize('clean_test_temp_files', [['perf_test.csv', 'perf_test.html']], indirect=True)
    def test_commandline_argument_skip_gpu_arch(self, global_data, clean_test_temp_files):
        """
        skip_gpu_arch command-line argument skips GPU architecture check
        """
        output = global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_skip_gpu_arch")   
        if 'Skipping model' not in output:
            pytest.fail("Enable skipping gpu arch for running model is failed.")    

    @pytest.mark.parametrize('clean_test_temp_files', [['perf_test.csv', 'perf_test.html']], indirect=True)
    def test_commandline_argument_disable_skip_gpu_arch_fail(self, global_data, clean_test_temp_files):
        """
        skip_gpu_arch command-line argument fails GPU architecture check
        """
        output = global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_skip_gpu_arch --disable-skip-gpu-arch") 
        # Check if exception with message 'Skipping model' is thrown 
        if 'Skipping model' in output:
            pytest.fail("Disable skipping gpu arch for running model is failed.")

    @pytest.mark.parametrize('clean_test_temp_files', [['perf_test.csv', 'perf_test.html']], indirect=True)
    def test_output_multi_results(self, global_data, clean_test_temp_files):
        """
        test output multiple results
        """
        output = global_data['console'].sh("cd " + BASE_DIR + "; " + "MODEL_DIR=" + MODEL_DIR + " " + "python3 src/madengine/mad.py run --tags dummy_multi") 
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


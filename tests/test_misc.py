"""Test the misc modules.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import sys
import csv
import json
import tempfile
import shutil
import pandas as pd
# 3rd party modules
import pytest
# project modules
from .fixtures.utils import BASE_DIR, MODEL_DIR
from .fixtures.utils import global_data
from .fixtures.utils import clean_test_temp_files

# Add src to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from madengine.utils.config_parser import ConfigParser
from madengine.tools.update_perf_super import update_perf_super_json


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


class TestPerfEntrySuperGeneration:
    """Test cases for perf_entry_super.json generation."""
    
    @pytest.fixture
    def test_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def fixtures_dir(self):
        """Get path to dummy fixtures directory."""
        return os.path.join(
            os.path.dirname(__file__),
            'fixtures',
            'dummy',
            'scripts',
            'dummy'
        )
    
    @pytest.fixture
    def config_file(self, fixtures_dir):
        """Get path to config file."""
        return os.path.join(fixtures_dir, 'configs', 'default.csv')
    
    def test_config_file_exists(self, config_file):
        """Test that the dummy config file exists."""
        assert os.path.exists(config_file), \
            f"Config file should exist at {config_file}"
    
    def test_config_parser_loads_csv(self, config_file):
        """Test that ConfigParser can load the dummy CSV config."""
        parser = ConfigParser()
        configs = parser.load_config_file(config_file)
        
        assert configs is not None, "Configs should not be None"
        assert isinstance(configs, list), "Configs should be a list"
        assert len(configs) == 3, "Should have 3 config rows"
        
        # Check first config has expected fields
        first_config = configs[0]
        assert 'model' in first_config
        assert 'benchmark' in first_config
        assert 'config_value' in first_config
        assert 'batch_size' in first_config
        assert 'datatype' in first_config
        assert 'max_tokens' in first_config
        
        # Verify values
        assert first_config['model'] == 'dummy/model-1'
        assert first_config['benchmark'] == 'throughput'
        assert first_config['datatype'] == 'float16'
    
    def test_config_parser_from_args(self, fixtures_dir):
        """Test parsing config path from args string."""
        parser = ConfigParser(scripts_base_dir=fixtures_dir)
        args_string = "--config configs/default.csv"
        
        config_path = parser.parse_config_from_args(
            args_string,
            os.path.join(fixtures_dir, 'run_perf_super.sh')
        )
        
        assert config_path is not None, "Config path should be found"
        assert os.path.exists(config_path), \
            f"Config file should exist at {config_path}"
    
    def test_perf_entry_super_json_structure(self, test_dir, fixtures_dir):
        """Test that perf_entry_super.json has the correct structure."""
        # Create mock data
        common_info = {
            "pipeline": "dummy_test",
            "n_gpus": "1",
            "training_precision": "",
            "args": "--config configs/default.csv",
            "tags": "dummies,perf_super_test",
            "docker_file": "docker/dummy.Dockerfile",
            "git_commit": "test123",
            "machine_name": "test_machine",
            "gpu_architecture": "test_gpu",
            "build_duration": "10",
            "test_duration": "20"
        }
        
        # Create common_info.json
        common_info_path = os.path.join(test_dir, "common_info_super.json")
        with open(common_info_path, 'w') as f:
            json.dump(common_info, f)
        
        # Create results CSV
        results_csv = os.path.join(test_dir, "perf_dummy_super.csv")
        with open(results_csv, 'w') as f:
            f.write("model,performance,metric,status\n")
            f.write("dummy/model-1,1234.56,tokens/s,SUCCESS\n")
            f.write("dummy/model-2,2345.67,requests/s,SUCCESS\n")
            f.write("dummy/model-3,345.78,ms,SUCCESS\n")
        
        # Generate perf_entry_super.json
        perf_super_path = os.path.join(test_dir, "perf_entry_super.json")
        
        update_perf_super_json(
            perf_super_json=perf_super_path,
            multiple_results=results_csv,
            common_info=common_info_path,
            model_name="dummy_perf_super",
            scripts_base_dir=fixtures_dir
        )
        
        # Verify file was created
        assert os.path.exists(perf_super_path), \
            "perf_entry_super.json should be created"
        
        # Load and verify structure
        with open(perf_super_path, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, list), "Data should be a list"
        assert len(data) == 3, "Should have 3 result records"
        
        # Check first record structure
        first_record = data[0]
        
        # Verify all common fields are present
        required_fields = [
            'model', 'performance', 'metric', 'status', 'pipeline',
            'n_gpus', 'args', 'tags', 'gpu_architecture'
        ]
        for field in required_fields:
            assert field in first_record, f"Field '{field}' should be present"
        
        # Verify configs field is present
        assert 'configs' in first_record, "configs field should be present"
        
        # Verify configs is not None (config file was found and loaded)
        assert first_record['configs'] is not None, \
            "configs should not be None when config file exists"
        
        # Verify configs has expected structure
        configs = first_record['configs']
        assert isinstance(configs, dict), "configs should be a dict"
        assert 'model' in configs
        assert 'benchmark' in configs
        assert 'config_value' in configs
        assert 'batch_size' in configs
        assert 'datatype' in configs
        assert 'max_tokens' in configs
    
    def test_perf_entry_super_config_matching(self, test_dir, fixtures_dir):
        """Test that configs are present for all results."""
        # Create mock data
        common_info = {
            "pipeline": "dummy_test",
            "n_gpus": "1",
            "args": "--config configs/default.csv",
            "tags": "dummies"
        }
        
        common_info_path = os.path.join(test_dir, "common_info_super.json")
        with open(common_info_path, 'w') as f:
            json.dump(common_info, f)
        
        # Create results CSV
        results_csv = os.path.join(test_dir, "perf_dummy_super.csv")
        with open(results_csv, 'w') as f:
            f.write("model,performance,metric,benchmark\n")
            f.write("dummy/model-1,1234.56,tokens/s,throughput\n")
            f.write("dummy/model-2,2345.67,requests/s,serving\n")
            f.write("dummy/model-3,345.78,ms,latency\n")
        
        perf_super_path = os.path.join(test_dir, "perf_entry_super.json")
        
        update_perf_super_json(
            perf_super_json=perf_super_path,
            multiple_results=results_csv,
            common_info=common_info_path,
            model_name="dummy_perf_super",
            scripts_base_dir=fixtures_dir
        )
        
        # Load and verify matching
        with open(perf_super_path, 'r') as f:
            data = json.load(f)
        
        # Verify each result has configs
        assert len(data) == 3, "Should have 3 results"
        
        for record in data:
            configs = record.get('configs')
            assert configs is not None, "Each record should have configs"
            assert isinstance(configs, dict), "Configs should be a dict"
            
            # Verify configs have expected structure (from default.csv)
            assert 'model' in configs
            assert 'benchmark' in configs
            assert 'config_value' in configs
            assert 'batch_size' in configs
            assert 'datatype' in configs
            assert 'max_tokens' in configs
            
            # Verify configs values are from our config file
            assert configs['benchmark'] in ['throughput', 'serving', 'latency']
            assert configs['datatype'] in ['float16', 'float32', 'bfloat16']
    
    def test_perf_entry_super_no_config(self, test_dir, fixtures_dir):
        """Test handling when no config file is specified."""
        # Create mock data without config
        common_info = {
            "pipeline": "dummy_test",
            "n_gpus": "1",
            "args": "",  # No --config argument
            "tags": "dummies"
        }
        
        common_info_path = os.path.join(test_dir, "common_info_super.json")
        with open(common_info_path, 'w') as f:
            json.dump(common_info, f)
        
        # Create results CSV
        results_csv = os.path.join(test_dir, "perf_dummy_super.csv")
        with open(results_csv, 'w') as f:
            f.write("model,performance,metric\n")
            f.write("dummy-no-config,1234.56,tokens/s\n")
        
        perf_super_path = os.path.join(test_dir, "perf_entry_super.json")
        
        update_perf_super_json(
            perf_super_json=perf_super_path,
            multiple_results=results_csv,
            common_info=common_info_path,
            model_name="dummy_no_config",
            scripts_base_dir=fixtures_dir
        )
        
        # Load and verify
        with open(perf_super_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 1
        
        # Verify configs is None for models without config files
        assert 'configs' in data[0]
        assert data[0]['configs'] is None, \
            "configs should be None when no config file is specified"
    
    def test_perf_entry_super_json_format_validation(self, test_dir, fixtures_dir):
        """Test that the JSON format matches expected schema."""
        # Create complete mock data
        common_info = {
            "pipeline": "dummy_test",
            "n_gpus": "1",
            "training_precision": "fp16",
            "args": "--config configs/default.csv",
            "tags": "dummies,perf_super_test",
            "docker_file": "docker/dummy.Dockerfile",
            "base_docker": "rocm/pytorch:latest",
            "docker_sha": "sha256:abc123",
            "docker_image": "test-image",
            "git_commit": "commit123",
            "machine_name": "test-machine",
            "gpu_architecture": "gfx942",
            "build_duration": "120",
            "test_duration": "300",
            "dataname": "test_data",
            "data_provider_type": "local",
            "data_size": "1GB",
            "data_download_duration": "60",
            "build_number": "12345",
            "additional_docker_run_options": "--shm-size=16g"
        }
        
        common_info_path = os.path.join(test_dir, "common_info_super.json")
        with open(common_info_path, 'w') as f:
            json.dump(common_info, f)
        
        results_csv = os.path.join(test_dir, "perf_dummy_super.csv")
        with open(results_csv, 'w') as f:
            f.write("model,performance,metric,status\n")
            f.write("dummy/model-1,1234.56,tokens/s,SUCCESS\n")
        
        perf_super_path = os.path.join(test_dir, "perf_entry_super.json")
        
        update_perf_super_json(
            perf_super_json=perf_super_path,
            multiple_results=results_csv,
            common_info=common_info_path,
            model_name="dummy_perf_super",
            scripts_base_dir=fixtures_dir
        )
        
        # Load and validate complete format
        with open(perf_super_path, 'r') as f:
            data = json.load(f)
        
        record = data[0]
        
        # Expected fields from RunDetails
        expected_fields = [
            'model', 'pipeline', 'n_gpus', 'training_precision', 'args',
            'tags', 'docker_file', 'base_docker', 'docker_sha', 'docker_image',
            'git_commit', 'machine_name', 'gpu_architecture', 'performance',
            'metric', 'status', 'build_duration', 'test_duration', 'dataname',
            'data_provider_type', 'data_size', 'data_download_duration',
            'build_number', 'additional_docker_run_options', 'configs'
        ]
        
        for field in expected_fields:
            assert field in record, \
                f"Field '{field}' should be present in perf_entry_super.json"
        
        # Verify configs structure
        configs = record['configs']
        assert isinstance(configs, dict)
        
        expected_config_fields = [
            'model', 'benchmark', 'config_value', 'batch_size',
            'datatype', 'max_tokens'
        ]
        
        for field in expected_config_fields:
            assert field in configs, \
                f"Config field '{field}' should be present"


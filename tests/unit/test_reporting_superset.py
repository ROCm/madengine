"""Unit tests for Performance Superset Reporting.

Tests the reporting layer's superset functionality including:
1. ConfigParser for loading model configuration files (CSV, JSON, YAML)
2. perf_super.json generation (cumulative) with configs and multi_results
3. perf_entry_super.json generation (latest run) from perf_super.json
4. CSV export from perf_super.json to perf_entry_super.csv and perf_super.csv
5. Handling of complex fields (configs, multi_results) in CSV format

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import json
import tempfile
import shutil
# 3rd party modules
import pytest
import pandas as pd
# project modules
from madengine.utils.config_parser import ConfigParser
from madengine.reporting.update_perf_super import (
    update_perf_super_json,
    update_perf_super_csv,
    convert_super_json_to_csv,
)


class TestConfigParser:
    """Test cases for ConfigParser functionality."""
    
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
            '..',
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
        assert first_config['batch_size'] == 8
        assert first_config['config_value'] == 128
        assert first_config['max_tokens'] == 1024
    
    def test_config_parser_from_args(self, fixtures_dir):
        """Test parsing config path from args string."""
        parser = ConfigParser(scripts_base_dir=fixtures_dir)
        args_string = "--config configs/default.csv"
        
        config_path = parser.parse_config_from_args(
            args_string,
            os.path.join(fixtures_dir, 'run.sh')
        )
        
        assert config_path is not None, "Config path should be found"
        assert os.path.exists(config_path), \
            f"Config file should exist at {config_path}"
    
    def test_config_parser_parse_and_load(self, fixtures_dir):
        """Test parse_and_load convenience method."""
        parser = ConfigParser(scripts_base_dir=fixtures_dir)
        args_string = "--batch-size 32 --config configs/default.csv"
        
        configs = parser.parse_and_load(args_string, fixtures_dir)
        
        assert configs is not None, "Configs should be loaded"
        assert isinstance(configs, list), "Configs should be a list"
        assert len(configs) == 3, "Should have 3 config rows"
    
    def test_config_parser_no_config_arg(self, fixtures_dir):
        """Test handling when no --config argument is present."""
        parser = ConfigParser(scripts_base_dir=fixtures_dir)
        args_string = "--batch-size 32 --epochs 10"
        
        configs = parser.parse_and_load(args_string, fixtures_dir)
        
        assert configs is None, "Should return None when no config argument"
    
    def test_config_parser_match_config_to_result(self, config_file):
        """Test matching configs to results."""
        parser = ConfigParser()
        configs = parser.load_config_file(config_file)
        
        # Test matching with model name
        result_data = {
            'model': 'dummy/model-1',
            'benchmark': 'throughput'
        }
        
        matched = parser.match_config_to_result(configs, result_data, 'dummy/model-1')
        
        assert matched is not None, "Should match a config"
        assert matched['model'] == 'dummy/model-1'
        assert matched['benchmark'] == 'throughput'
    
    def test_config_parser_json_file(self, test_dir):
        """Test loading JSON config file."""
        # Create a JSON config file
        json_config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        }
        
        json_path = os.path.join(test_dir, "config.json")
        with open(json_path, 'w') as f:
            json.dump(json_config, f)
        
        parser = ConfigParser()
        configs = parser.load_config_file(json_path)
        
        assert configs is not None, "Configs should be loaded"
        assert isinstance(configs, dict), "JSON config should be a dict"
        assert configs['batch_size'] == 32
        assert configs['learning_rate'] == 0.001


class TestPerfEntrySuperGeneration:
    """Test cases for perf_super.json generation (cumulative)."""
    
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
            '..',
            'fixtures',
            'dummy',
            'scripts',
            'dummy'
        )
    
    def test_perf_entry_super_json_structure(self, test_dir, fixtures_dir):
        """Test that perf_super.json has the correct structure."""
        # Create mock data
        common_info = {
            "pipeline": "dummy_test",
            "n_gpus": "1",
            "nnodes": "1",
            "gpus_per_node": "1",
            "training_precision": "",
            "args": "--config configs/default.csv",
            "tags": "dummies,perf_super_test",
            "docker_file": "docker/dummy.Dockerfile",
            "base_docker": "rocm/pytorch:latest",
            "docker_sha": "abc123",
            "docker_image": "test:v1",
            "git_commit": "test123",
            "machine_name": "test_machine",
            "deployment_type": "local",
            "launcher": "torchrun",
            "gpu_architecture": "test_gpu",
            "relative_change": "",
            "build_duration": "10",
            "test_duration": "20",
            "dataname": "",
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": "1",
            "additional_docker_run_options": "",
        }
        
        # Create common_info.json
        common_info_path = os.path.join(test_dir, "common_info.json")
        with open(common_info_path, 'w') as f:
            json.dump(common_info, f)
        
        # Create results CSV
        results_csv = os.path.join(test_dir, "perf_dummy_super.csv")
        with open(results_csv, 'w') as f:
            f.write("model,performance,metric,status\n")
            f.write("dummy/model-1,1234.56,tokens/s,SUCCESS\n")
            f.write("dummy/model-2,2345.67,requests/s,SUCCESS\n")
            f.write("dummy/model-3,345.78,ms,SUCCESS\n")
        
        # Generate perf_super.json (cumulative)
        perf_super_path = os.path.join(test_dir, "perf_super.json")
        
        update_perf_super_json(
            perf_super_json=perf_super_path,
            multiple_results=results_csv,
            common_info=common_info_path,
            model_name="dummy_perf_super",
            scripts_base_dir=fixtures_dir
        )
        
        # Verify file was created
        assert os.path.exists(perf_super_path), \
            "perf_super.json should be created"
        
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
        """Test that configs are correctly matched for all results."""
        # Create mock data
        common_info = {
            "pipeline": "dummy_test",
            "n_gpus": "1",
            "nnodes": "1",
            "gpus_per_node": "1",
            "args": "--config configs/default.csv",
            "tags": "dummies",
            "training_precision": "",
            "docker_file": "",
            "base_docker": "",
            "docker_sha": "",
            "docker_image": "",
            "git_commit": "",
            "machine_name": "",
            "deployment_type": "local",
            "launcher": "torchrun",
            "gpu_architecture": "",
            "relative_change": "",
            "build_duration": "",
            "test_duration": "",
            "dataname": "",
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": "",
            "additional_docker_run_options": "",
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
        
        perf_super_path = os.path.join(test_dir, "perf_super.json")
        
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
            "nnodes": "1",
            "gpus_per_node": "1",
            "args": "",  # No --config argument
            "tags": "dummies",
            "training_precision": "",
            "docker_file": "",
            "base_docker": "",
            "docker_sha": "",
            "docker_image": "",
            "git_commit": "",
            "machine_name": "",
            "deployment_type": "local",
            "launcher": "",
            "gpu_architecture": "",
            "relative_change": "",
            "build_duration": "",
            "test_duration": "",
            "dataname": "",
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": "",
            "additional_docker_run_options": "",
        }
        
        common_info_path = os.path.join(test_dir, "common_info_super.json")
        with open(common_info_path, 'w') as f:
            json.dump(common_info, f)
        
        # Create results CSV
        results_csv = os.path.join(test_dir, "perf_dummy_super.csv")
        with open(results_csv, 'w') as f:
            f.write("model,performance,metric\n")
            f.write("dummy-no-config,1234.56,tokens/s\n")
        
        perf_super_path = os.path.join(test_dir, "perf_super.json")
        
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
        
        assert len(data) == 1, "Should have 1 result"
        
        # Verify configs is None when no config file
        assert data[0]['configs'] is None, \
            "configs should be None when no config file specified"
    
    def test_perf_entry_super_multi_results(self, test_dir, fixtures_dir):
        """Test handling of multiple result metrics."""
        common_info = {
            "pipeline": "dummy_test",
            "n_gpus": "8",
            "nnodes": "1",
            "gpus_per_node": "8",
            "args": "",
            "tags": "multi_metrics",
            "training_precision": "fp16",
            "docker_file": "",
            "base_docker": "",
            "docker_sha": "",
            "docker_image": "",
            "git_commit": "",
            "machine_name": "",
            "deployment_type": "local",
            "launcher": "vllm",
            "gpu_architecture": "gfx90a",
            "relative_change": "",
            "build_duration": "",
            "test_duration": "",
            "dataname": "",
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": "",
            "additional_docker_run_options": "",
        }
        
        common_info_path = os.path.join(test_dir, "common_info_super.json")
        with open(common_info_path, 'w') as f:
            json.dump(common_info, f)
        
        # Create results CSV with extra metrics
        results_csv = os.path.join(test_dir, "perf_multi_metrics.csv")
        with open(results_csv, 'w') as f:
            f.write("model,performance,metric,throughput,latency_mean_ms,latency_p50_ms,latency_p90_ms,gpu_memory_used_mb\n")
            f.write("model-1,1234.56,tokens/s,1234.56,8.1,7.9,12.3,12288\n")
            f.write("model-2,2345.67,requests/s,2345.67,4.3,4.1,6.8,16384\n")
        
        perf_super_path = os.path.join(test_dir, "perf_super.json")
        
        update_perf_super_json(
            perf_super_json=perf_super_path,
            multiple_results=results_csv,
            common_info=common_info_path,
            model_name="test_multi_metrics",
            scripts_base_dir=fixtures_dir
        )
        
        # Load and verify
        with open(perf_super_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 2, "Should have 2 results"
        
        # Check first result has multi_results with extra metrics
        first_result = data[0]
        assert 'multi_results' in first_result, "Should have multi_results field"
        assert first_result['multi_results'] is not None, "multi_results should not be None"
        
        multi_results = first_result['multi_results']
        assert isinstance(multi_results, dict), "multi_results should be a dict"
        
        # Verify extra metrics are in multi_results
        assert 'throughput' in multi_results
        assert 'latency_mean_ms' in multi_results
        assert 'latency_p50_ms' in multi_results
        assert 'latency_p90_ms' in multi_results
        assert 'gpu_memory_used_mb' in multi_results
        
        # Verify values
        assert multi_results['throughput'] == 1234.56
        assert multi_results['latency_mean_ms'] == 8.1
        assert multi_results['gpu_memory_used_mb'] == 12288
    
    def test_perf_entry_super_deployment_fields(self, test_dir, fixtures_dir):
        """Test that all deployment-related fields are present."""
        common_info = {
            "pipeline": "dummy_test",
            "n_gpus": "16",  # 2 nodes × 8 GPUs
            "nnodes": "2",
            "gpus_per_node": "8",
            "args": "",
            "tags": "multi_node",
            "training_precision": "fp16",
            "docker_file": "",
            "base_docker": "",
            "docker_sha": "",
            "docker_image": "",
            "git_commit": "",
            "machine_name": "node-1",
            "deployment_type": "slurm",
            "launcher": "torchrun",
            "gpu_architecture": "gfx90a",
            "relative_change": "",
            "build_duration": "",
            "test_duration": "",
            "dataname": "",
            "data_provider_type": "",
            "data_size": "",
            "data_download_duration": "",
            "build_number": "",
            "additional_docker_run_options": "",
        }
        
        common_info_path = os.path.join(test_dir, "common_info_super.json")
        with open(common_info_path, 'w') as f:
            json.dump(common_info, f)
        
        # Create results CSV
        results_csv = os.path.join(test_dir, "perf_deployment.csv")
        with open(results_csv, 'w') as f:
            f.write("model,performance,metric\n")
            f.write("multi-node-test,5000.0,tokens/s\n")
        
        perf_super_path = os.path.join(test_dir, "perf_super.json")
        
        update_perf_super_json(
            perf_super_json=perf_super_path,
            multiple_results=results_csv,
            common_info=common_info_path,
            model_name="test_deployment",
            scripts_base_dir=fixtures_dir
        )
        
        # Load and verify
        with open(perf_super_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 1, "Should have 1 result"
        
        result = data[0]
        
        # Verify all deployment fields are present
        deployment_fields = {
            "n_gpus": "16",
            "nnodes": "2",
            "gpus_per_node": "8",
            "deployment_type": "slurm",
            "launcher": "torchrun",
            "machine_name": "node-1",
        }
        
        for field, expected_value in deployment_fields.items():
            assert field in result, f"Field '{field}' should be present"
            assert result[field] == expected_value, \
                f"Field '{field}' should be '{expected_value}', got '{result[field]}'"


class TestPerfSuperCSVGeneration:
    """Test cases for CSV generation from perf_super.json."""
    
    @pytest.fixture
    def test_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_csv_generation_from_json(self, test_dir):
        """Test CSV generation from perf_super.json."""
        # Create a sample perf_super.json
        data = [
            {
                "model": "test_model_1",
                "n_gpus": "8",
                "performance": "1234.56",
                "metric": "tokens/s",
                "status": "SUCCESS",
                "configs": {"batch_size": 32, "learning_rate": 0.001},
                "multi_results": {"throughput": 1234.56, "latency_ms": 8.1},
            },
            {
                "model": "test_model_2",
                "n_gpus": "8",
                "performance": "2345.67",
                "metric": "requests/s",
                "status": "SUCCESS",
                "configs": {"batch_size": 64, "learning_rate": 0.002},
                "multi_results": None,
            }
        ]
        
        json_path = os.path.join(test_dir, "perf_super.json")
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        # Change to test directory
        original_dir = os.getcwd()
        os.chdir(test_dir)
        
        try:
            # Generate CSVs
            update_perf_super_csv(
                perf_super_json="perf_super.json",
                perf_super_csv="perf_super.csv"
            )
            
            # Verify files exist
            assert os.path.exists("perf_entry_super.csv"), \
                "perf_entry_super.csv should be created"
            assert os.path.exists("perf_super.csv"), \
                "perf_super.csv should be created"
            
            # Load and verify perf_entry_super.csv (latest entry only)
            entry_df = pd.read_csv("perf_entry_super.csv")
            assert len(entry_df) == 1, "Should have 1 entry (latest)"
            assert entry_df.iloc[0]['model'] == "test_model_2"
            
            # Load and verify perf_super.csv (all entries)
            super_df = pd.read_csv("perf_super.csv")
            assert len(super_df) == 2, "Should have 2 entries (all)"
            
            # Verify configs column is JSON string
            assert 'configs' in super_df.columns
            first_configs = json.loads(super_df.iloc[0]['configs'])
            assert first_configs['batch_size'] == 32
            
            # Verify multi_results column
            assert 'multi_results' in super_df.columns
            first_multi = json.loads(super_df.iloc[0]['multi_results'])
            assert first_multi['throughput'] == 1234.56
            
        finally:
            os.chdir(original_dir)
    
    def test_csv_handles_none_values(self, test_dir):
        """Test that CSV generation handles None values correctly."""
        data = [
            {
                "model": "test_model",
                "performance": "1234.56",
                "metric": "tokens/s",
                "configs": None,
                "multi_results": None,
            }
        ]
        
        json_path = os.path.join(test_dir, "perf_super.json")
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        original_dir = os.getcwd()
        os.chdir(test_dir)
        
        try:
            update_perf_super_csv(
                perf_super_json="perf_super.json",
                perf_super_csv="perf_super.csv"
            )
            
            # Load CSV
            df = pd.read_csv("perf_super.csv")
            
            # Verify None values are handled
            assert pd.isna(df.iloc[0]['configs']) or df.iloc[0]['configs'] == ''
            assert pd.isna(df.iloc[0]['multi_results']) or df.iloc[0]['multi_results'] == ''
            
        finally:
            os.chdir(original_dir)
    
    def test_csv_multiple_entries_in_entry_file(self, test_dir):
        """Test that perf_entry_super.csv can contain multiple entries from current run.
        
        This tests the fix for the issue where perf_entry.csv and perf_entry.json
        had 4 entries (for multiple results) but perf_entry_super.csv only had 1.
        Now perf_entry_super.csv should contain all entries from the current run.
        """
        # Simulate a cumulative JSON with old entries + new entries
        data = [
            # Old entry from a previous run
            {
                "model": "old_model",
                "n_gpus": "4",
                "performance": "999.99",
                "metric": "tokens/s",
                "status": "SUCCESS",
                "configs": None,
                "multi_results": None,
            },
            # New entries from current run (4 models from multiple results)
            {
                "model": "dummy_multi_1",
                "n_gpus": "1",
                "performance": "1234.56",
                "metric": "samples_per_sec",
                "status": "SUCCESS",
                "configs": None,
                "multi_results": {"temperature": 12345},
            },
            {
                "model": "dummy_multi_2",
                "n_gpus": "1",
                "performance": "2345.67",
                "metric": "samples_per_sec",
                "status": "SUCCESS",
                "configs": None,
                "multi_results": {"temperature": 23456},
            },
            {
                "model": "dummy_multi_3",
                "n_gpus": "1",
                "performance": "3456.78",
                "metric": "samples_per_sec",
                "status": "SUCCESS",
                "configs": None,
                "multi_results": {"temperature": 34567},
            },
            {
                "model": "dummy_multi_4",
                "n_gpus": "1",
                "performance": "4567.89",
                "metric": "samples_per_sec",
                "status": "SUCCESS",
                "configs": None,
                "multi_results": {"temperature": 45678},
            }
        ]
        
        json_path = os.path.join(test_dir, "perf_super.json")
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        original_dir = os.getcwd()
        os.chdir(test_dir)
        
        try:
            # Generate CSVs with num_entries=4 (simulating 4 entries added in current run)
            update_perf_super_csv(
                perf_super_json="perf_super.json",
                perf_super_csv="perf_super.csv",
                num_entries=4
            )
            
            # Verify perf_entry_super.csv has ALL 4 entries from current run
            entry_df = pd.read_csv("perf_entry_super.csv")
            assert len(entry_df) == 4, \
                f"perf_entry_super.csv should have 4 entries, got {len(entry_df)}"
            
            # Verify the models are the 4 from the current run (not the old one)
            models = entry_df['model'].tolist()
            expected_models = ['dummy_multi_1', 'dummy_multi_2', 'dummy_multi_3', 'dummy_multi_4']
            assert models == expected_models, \
                f"Expected {expected_models}, got {models}"
            
            # Verify perf_super.csv has ALL 5 entries (old + new)
            super_df = pd.read_csv("perf_super.csv")
            assert len(super_df) == 5, \
                f"perf_super.csv should have 5 entries (1 old + 4 new), got {len(super_df)}"
            
            # Verify all models are in perf_super.csv
            all_models = super_df['model'].tolist()
            assert 'old_model' in all_models, "Old model should be in perf_super.csv"
            assert all(m in all_models for m in expected_models), \
                "All new models should be in perf_super.csv"
            
        finally:
            os.chdir(original_dir)


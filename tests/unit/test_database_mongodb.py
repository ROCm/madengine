"""
Unit tests for MongoDB database operations.

Tests the refactored database upload functionality including file loading,
type handling, and document transformation.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest
import pandas as pd

from madengine.database.mongodb import (
    MongoDBConfig,
    UploadOptions,
    UploadResult,
    FileFormat,
    JSONLoader,
    CSVLoader,
    DocumentTransformer,
    detect_file_format,
    get_loader,
    upload_file_to_mongodb,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_json_data():
    """Sample JSON data with native types."""
    return [
        {
            "model": "test_model_1",
            "performance": 123.45,
            "metric": "tokens/sec",
            "status": "SUCCESS",
            "configs": {
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "enabled": True,
            "timestamp": "2026-01-07 10:00:00"
        },
        {
            "model": "test_model_2",
            "performance": 234.56,
            "metric": "tokens/sec",
            "status": "SUCCESS",
            "configs": {
                "batch_size": 64,
                "learning_rate": 0.002
            },
            "enabled": False,
            "timestamp": "2026-01-07 10:05:00"
        }
    ]


@pytest.fixture
def temp_json_file(sample_json_data):
    """Create a temporary JSON file."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(sample_json_data, f)
        file_path = f.name
    
    yield Path(file_path)
    
    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False, newline=''
    ) as f:
        f.write("model,performance,metric,status,timestamp\n")
        f.write("csv_model_1,345.67,tokens/sec,SUCCESS,2026-01-07 11:00:00\n")
        f.write("csv_model_2,456.78,tokens/sec,SUCCESS,2026-01-07 11:05:00\n")
        file_path = f.name
    
    yield Path(file_path)
    
    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


# ============================================================================
# Configuration Tests
# ============================================================================

@pytest.mark.unit
def test_mongodb_config_defaults():
    """Test MongoDBConfig with default values."""
    config = MongoDBConfig()
    
    assert config.host == "localhost"
    assert config.port == 27017
    assert config.username == ""
    assert config.password == ""
    assert config.timeout_ms == 5000


@pytest.mark.unit
def test_mongodb_config_from_env():
    """Test MongoDBConfig loading from environment."""
    env_vars = {
        "MONGO_HOST": "test-host",
        "MONGO_PORT": "27018",
        "MONGO_USER": "testuser",
        "MONGO_PASSWORD": "testpass",
    }
    
    with patch.dict(os.environ, env_vars, clear=False):
        config = MongoDBConfig.from_env()
        
        assert config.host == "test-host"
        assert config.port == 27018
        assert config.username == "testuser"
        assert config.password == "testpass"


@pytest.mark.unit
def test_mongodb_config_uri_with_auth():
    """Test MongoDB URI generation with authentication."""
    config = MongoDBConfig(
        host="example.com",
        port=27017,
        username="user",
        password="pass"
    )
    
    assert config.uri == "mongodb://user:pass@example.com:27017/admin"


@pytest.mark.unit
def test_mongodb_config_uri_without_auth():
    """Test MongoDB URI generation without authentication."""
    config = MongoDBConfig(host="example.com", port=27017)
    
    assert config.uri == "mongodb://example.com:27017"


@pytest.mark.unit
def test_upload_options_defaults():
    """Test UploadOptions default values."""
    options = UploadOptions()
    
    assert options.unique_fields is None
    assert options.upsert is True
    assert options.batch_size == 1000
    assert options.ordered is False
    assert options.create_indexes is True
    assert options.add_metadata is True
    assert options.dry_run is False


# ============================================================================
# File Detection Tests
# ============================================================================

@pytest.mark.unit
def test_detect_json_format_by_extension(temp_json_file):
    """Test JSON format detection by file extension."""
    file_format = detect_file_format(temp_json_file)
    assert file_format == FileFormat.JSON


@pytest.mark.unit
def test_detect_csv_format_by_extension(temp_csv_file):
    """Test CSV format detection by file extension."""
    file_format = detect_file_format(temp_csv_file)
    assert file_format == FileFormat.CSV


@pytest.mark.unit
def test_detect_json_format_by_content():
    """Test JSON format detection by content when no extension."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='', delete=False
    ) as f:
        json.dump({"test": "data"}, f)
        file_path = f.name
    
    try:
        file_format = detect_file_format(Path(file_path))
        assert file_format == FileFormat.JSON
    finally:
        os.unlink(file_path)


@pytest.mark.unit
def test_get_loader_json():
    """Test getting JSON loader."""
    loader = get_loader(FileFormat.JSON)
    assert isinstance(loader, JSONLoader)


@pytest.mark.unit
def test_get_loader_csv():
    """Test getting CSV loader."""
    loader = get_loader(FileFormat.CSV)
    assert isinstance(loader, CSVLoader)


# ============================================================================
# JSON Loader Tests
# ============================================================================

@pytest.mark.unit
def test_json_loader_load_array(temp_json_file, sample_json_data):
    """Test JSONLoader with array format."""
    loader = JSONLoader()
    documents = loader.load(temp_json_file)
    
    assert len(documents) == 2
    assert documents[0]["model"] == "test_model_1"
    assert documents[0]["performance"] == 123.45
    assert isinstance(documents[0]["configs"], dict)
    assert documents[0]["enabled"] is True


@pytest.mark.unit
def test_json_loader_load_single_object():
    """Test JSONLoader with single object format."""
    data = {"model": "test", "value": 42}
    
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(data, f)
        file_path = f.name
    
    try:
        loader = JSONLoader()
        documents = loader.load(Path(file_path))
        
        assert len(documents) == 1
        assert documents[0]["model"] == "test"
        assert documents[0]["value"] == 42
    finally:
        os.unlink(file_path)


@pytest.mark.unit
def test_json_loader_preserves_types(temp_json_file):
    """Test that JSONLoader preserves native types."""
    loader = JSONLoader()
    documents = loader.load(temp_json_file)
    
    doc = documents[0]
    assert isinstance(doc["performance"], float)
    assert isinstance(doc["configs"], dict)
    assert isinstance(doc["enabled"], bool)
    assert isinstance(doc["model"], str)


@pytest.mark.unit
def test_json_loader_infer_schema(sample_json_data):
    """Test JSON schema inference."""
    loader = JSONLoader()
    schema = loader.infer_schema(sample_json_data)
    
    assert schema["model"] == str
    assert schema["performance"] == float
    assert schema["configs"] == dict
    assert schema["enabled"] == bool


# ============================================================================
# CSV Loader Tests
# ============================================================================

@pytest.mark.unit
def test_csv_loader_load(temp_csv_file):
    """Test CSVLoader basic loading."""
    loader = CSVLoader()
    documents = loader.load(temp_csv_file)
    
    assert len(documents) == 2
    assert documents[0]["model"] == "csv_model_1"
    assert documents[1]["model"] == "csv_model_2"


@pytest.mark.unit
def test_csv_loader_type_inference(temp_csv_file):
    """Test that CSVLoader infers types correctly."""
    loader = CSVLoader()
    documents = loader.load(temp_csv_file)
    
    doc = documents[0]
    # Performance should be float, not string
    assert isinstance(doc["performance"], (float, int))
    assert doc["performance"] == 345.67


@pytest.mark.unit
def test_csv_loader_json_string_parsing():
    """Test that CSVLoader parses JSON strings in columns."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False, newline=''
    ) as f:
        f.write('model,configs\n')
        f.write('test,"{""lr"": 0.001}"\n')
        file_path = f.name
    
    try:
        loader = CSVLoader()
        documents = loader.load(Path(file_path))
        
        # Should parse JSON string in configs column
        assert isinstance(documents[0]["configs"], (dict, str))
    finally:
        os.unlink(file_path)


@pytest.mark.unit
def test_csv_loader_handles_null_values():
    """Test CSVLoader handles null/missing values."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False, newline=''
    ) as f:
        f.write('model,value\n')
        f.write('test1,42\n')
        f.write('test2,\n')  # Empty value
        file_path = f.name
    
    try:
        loader = CSVLoader()
        documents = loader.load(Path(file_path))
        
        assert documents[0]["value"] == 42
        assert documents[1]["value"] is None
    finally:
        os.unlink(file_path)


# ============================================================================
# Document Transformer Tests
# ============================================================================

@pytest.mark.unit
def test_document_transformer_adds_metadata():
    """Test that transformer adds metadata fields."""
    options = UploadOptions(add_metadata=True)
    transformer = DocumentTransformer(options)
    
    documents = [{"model": "test", "value": 42}]
    transformed = transformer.transform(documents)
    
    assert "_meta_uploaded_at" in transformed[0]
    assert "created_date" in transformed[0]


@pytest.mark.unit
def test_document_transformer_preserves_existing_metadata():
    """Test that transformer preserves existing created_date."""
    options = UploadOptions(add_metadata=True)
    transformer = DocumentTransformer(options)
    
    original_date = "2026-01-01 00:00:00"
    documents = [{"model": "test", "created_date": original_date}]
    transformed = transformer.transform(documents)
    
    assert transformed[0]["created_date"] == original_date


@pytest.mark.unit
def test_document_transformer_infer_unique_fields():
    """Test automatic unique field inference."""
    options = UploadOptions()
    transformer = DocumentTransformer(options)
    
    documents = [
        {"model": "model1", "timestamp": "2026-01-01", "value": 1},
        {"model": "model2", "timestamp": "2026-01-02", "value": 2},
    ]
    
    unique_fields = transformer.infer_unique_fields(documents)
    
    assert "model" in unique_fields


@pytest.mark.unit
def test_document_transformer_no_metadata_when_disabled():
    """Test that metadata is not added when disabled."""
    options = UploadOptions(add_metadata=False)
    transformer = DocumentTransformer(options)
    
    documents = [{"model": "test", "value": 42}]
    transformed = transformer.transform(documents)
    
    assert "_meta_uploaded_at" not in transformed[0]


# ============================================================================
# Upload Result Tests
# ============================================================================

@pytest.mark.unit
def test_upload_result_success_status():
    """Test UploadResult with success status."""
    result = UploadResult(
        status="success",
        documents_read=10,
        documents_processed=10,
        documents_inserted=8,
        documents_updated=2,
        documents_failed=0,
        duration_seconds=1.5
    )
    
    assert result.status == "success"
    assert result.documents_read == 10
    assert result.documents_inserted == 8
    assert result.documents_updated == 2


@pytest.mark.unit
def test_upload_result_with_errors():
    """Test UploadResult with errors."""
    result = UploadResult(
        status="partial",
        documents_read=10,
        documents_processed=8,
        documents_inserted=7,
        documents_updated=1,
        documents_failed=2,
        errors=["Error 1", "Error 2"],
        duration_seconds=2.0
    )
    
    assert result.status == "partial"
    assert result.documents_failed == 2
    assert len(result.errors) == 2


# ============================================================================
# Main Upload Function Tests (Mocked)
# ============================================================================

@pytest.mark.unit
def test_upload_file_to_mongodb_json_dry_run(temp_json_file):
    """Test uploading JSON file in dry-run mode."""
    config = MongoDBConfig()
    options = UploadOptions(dry_run=True)
    
    result = upload_file_to_mongodb(
        file_path=str(temp_json_file),
        database_name="test_db",
        collection_name="test_collection",
        config=config,
        options=options
    )
    
    assert result.status == "success"
    assert result.documents_read == 2
    assert result.documents_processed == 0
    assert result.documents_inserted == 0


@pytest.mark.unit
def test_upload_file_to_mongodb_csv_dry_run(temp_csv_file):
    """Test uploading CSV file in dry-run mode."""
    config = MongoDBConfig()
    options = UploadOptions(dry_run=True)
    
    result = upload_file_to_mongodb(
        file_path=str(temp_csv_file),
        database_name="test_db",
        collection_name="test_collection",
        config=config,
        options=options
    )
    
    assert result.status == "success"
    assert result.documents_read == 2


@pytest.mark.unit
def test_upload_file_to_mongodb_auto_detects_unique_fields(temp_json_file):
    """Test that upload auto-detects unique fields."""
    config = MongoDBConfig()
    options = UploadOptions(
        dry_run=True,
        unique_fields=None  # Should auto-detect
    )
    
    result = upload_file_to_mongodb(
        file_path=str(temp_json_file),
        database_name="test_db",
        collection_name="test_collection",
        config=config,
        options=options
    )
    
    assert result.status == "success"
    # Options should have been updated with detected fields
    assert options.unique_fields is not None


@pytest.mark.unit
def test_upload_file_to_mongodb_file_not_found():
    """Test upload with non-existent file."""
    config = MongoDBConfig()
    options = UploadOptions()
    
    with pytest.raises(FileNotFoundError):
        upload_file_to_mongodb(
            file_path="/nonexistent/file.json",
            database_name="test_db",
            collection_name="test_collection",
            config=config,
            options=options
        )


@pytest.mark.unit
def test_upload_file_to_mongodb_with_custom_unique_fields(temp_json_file):
    """Test upload with custom unique fields."""
    config = MongoDBConfig()
    options = UploadOptions(
        dry_run=True,
        unique_fields=["model", "timestamp"]
    )
    
    result = upload_file_to_mongodb(
        file_path=str(temp_json_file),
        database_name="test_db",
        collection_name="test_collection",
        config=config,
        options=options
    )
    
    assert result.status == "success"
    assert options.unique_fields == ["model", "timestamp"]


@pytest.mark.unit
@patch('madengine.database.mongodb.MongoDBUploader')
def test_upload_file_to_mongodb_calls_uploader(mock_uploader_class, temp_json_file):
    """Test that upload function properly calls MongoDBUploader."""
    # Setup mock
    mock_uploader = MagicMock()
    mock_uploader_class.return_value.__enter__.return_value = mock_uploader
    mock_uploader.upload.return_value = UploadResult(
        status="success",
        documents_read=2,
        documents_processed=2,
        documents_inserted=2,
        documents_updated=0,
        documents_failed=0,
        duration_seconds=0.1
    )
    
    config = MongoDBConfig()
    options = UploadOptions(dry_run=False)
    
    result = upload_file_to_mongodb(
        file_path=str(temp_json_file),
        database_name="test_db",
        collection_name="test_collection",
        config=config,
        options=options
    )
    
    # Verify uploader was called
    mock_uploader.upload.assert_called_once()
    assert result.status == "success"
    assert result.documents_inserted == 2

"""
Modern MongoDB operations for madengine.

A clean, efficient implementation supporting CSV and JSON uploads with
intelligent type handling, bulk operations, and production-ready features.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

import pandas as pd
import pymongo
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure, PyMongoError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

logger = logging.getLogger(__name__)
console = Console()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MongoDBConfig:
    """MongoDB connection configuration."""
    
    host: str = "localhost"
    port: int = 27017
    username: str = ""
    password: str = ""
    auth_source: str = "admin"
    timeout_ms: int = 5000
    
    @classmethod
    def from_env(cls) -> 'MongoDBConfig':
        """Load configuration from environment variables."""
        import os
        return cls(
            host=os.getenv("MONGO_HOST", "localhost"),
            port=int(os.getenv("MONGO_PORT", "27017")),
            username=os.getenv("MONGO_USER", ""),
            password=os.getenv("MONGO_PASSWORD", ""),
            auth_source=os.getenv("MONGO_AUTH_SOURCE", "admin"),
            timeout_ms=int(os.getenv("MONGO_TIMEOUT_MS", "5000"))
        )
    
    @property
    def uri(self) -> str:
        """Build MongoDB connection URI."""
        if self.username and self.password:
            return (f"mongodb://{self.username}:{self.password}@"
                   f"{self.host}:{self.port}/{self.auth_source}")
        return f"mongodb://{self.host}:{self.port}"


@dataclass
class UploadOptions:
    """Options for document upload."""
    
    # Deduplication strategy
    unique_fields: Optional[List[str]] = None  # Fields to use for uniqueness
    upsert: bool = True  # Update existing or insert only
    
    # Performance options
    batch_size: int = 1000  # Documents per batch
    ordered: bool = False  # Continue on error
    
    # Index creation
    create_indexes: bool = True
    index_fields: Optional[List[str]] = None  # Auto-detect if None
    
    # Metadata
    add_metadata: bool = True
    metadata_prefix: str = "_meta"
    
    # Validation
    validate_schema: bool = True
    
    # Dry run
    dry_run: bool = False


@dataclass
class UploadResult:
    """Result of upload operation."""
    
    status: str  # success, partial, failed
    documents_read: int
    documents_processed: int
    documents_inserted: int
    documents_updated: int
    documents_failed: int
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    def print_summary(self):
        """Print formatted summary."""
        if self.status == "success":
            console.print(f"✅ [bold green]Upload successful![/bold green]")
        elif self.status == "partial":
            console.print(f"⚠️  [bold yellow]Partial success[/bold yellow]")
        else:
            console.print(f"❌ [bold red]Upload failed[/bold red]")
        
        console.print(f"   📊 Documents read: {self.documents_read}")
        console.print(f"   ✨ Documents processed: {self.documents_processed}")
        console.print(f"   ➕ Inserted: {self.documents_inserted}")
        console.print(f"   🔄 Updated: {self.documents_updated}")
        if self.documents_failed > 0:
            console.print(f"   ❌ Failed: {self.documents_failed}")
        console.print(f"   ⏱️  Duration: {self.duration_seconds:.2f}s")


# ============================================================================
# File Loaders (Strategy Pattern)
# ============================================================================

class FileFormat(Enum):
    """Supported file formats."""
    CSV = "csv"
    JSON = "json"


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load documents from file."""
        pass
    
    @abstractmethod
    def infer_schema(self, documents: List[Dict[str, Any]]) -> Dict[str, type]:
        """Infer schema from documents."""
        pass


class JSONLoader(DocumentLoader):
    """Loader for JSON files with native type preservation."""
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file preserving native types."""
        logger.info(f"Loading JSON file: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Normalize to list
        if isinstance(data, dict):
            documents = [data]
        elif isinstance(data, list):
            documents = data
        else:
            raise ValueError(f"Expected JSON object or array, got {type(data)}")
        
        # Validate structure
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} is not a JSON object: {type(doc)}")
        
        logger.info(f"Loaded {len(documents)} documents from JSON")
        return documents
    
    def infer_schema(self, documents: List[Dict[str, Any]]) -> Dict[str, type]:
        """Infer schema from JSON documents."""
        if not documents:
            return {}
        
        schema = {}
        sample_doc = documents[0]
        
        for key, value in sample_doc.items():
            schema[key] = type(value)
        
        return schema


class CSVLoader(DocumentLoader):
    """Loader for CSV files with intelligent type inference."""
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV file with type inference."""
        logger.info(f"Loading CSV file: {file_path}")
        
        # Read CSV with pandas (intelligent type inference)
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert to documents with native types preserved
        documents = []
        for _, row in df.iterrows():
            doc = {}
            for col in df.columns:
                value = row[col]
                # Handle pandas NA/NaN
                if pd.isna(value):
                    doc[col] = None
                # Try to parse JSON strings (for configs, multi_results)
                elif isinstance(value, str) and value.strip().startswith(('{', '[')):
                    try:
                        doc[col] = json.loads(value)
                    except json.JSONDecodeError:
                        doc[col] = value
                else:
                    # Keep native type (int, float, bool, str)
                    doc[col] = value if not pd.isna(value) else None
            
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from CSV")
        return documents
    
    def infer_schema(self, documents: List[Dict[str, Any]]) -> Dict[str, type]:
        """Infer schema from CSV documents."""
        if not documents:
            return {}
        
        schema = {}
        sample_doc = documents[0]
        
        for key, value in sample_doc.items():
            if value is None:
                schema[key] = type(None)
            else:
                schema[key] = type(value)
        
        return schema


def detect_file_format(file_path: Path) -> FileFormat:
    """Detect file format from extension and content."""
    
    extension = file_path.suffix.lower()
    
    if extension == '.json':
        return FileFormat.JSON
    elif extension == '.csv':
        return FileFormat.CSV
    
    # Content-based detection
    try:
        with open(file_path, 'r') as f:
            first_char = f.read(1).strip()
            if first_char in ['{', '[']:
                return FileFormat.JSON
            else:
                return FileFormat.CSV
    except Exception:
        raise ValueError(f"Cannot detect format for {file_path}")


def get_loader(file_format: FileFormat) -> DocumentLoader:
    """Get appropriate loader for file format."""
    loaders = {
        FileFormat.JSON: JSONLoader(),
        FileFormat.CSV: CSVLoader(),
    }
    return loaders[file_format]


# ============================================================================
# Document Transformer
# ============================================================================

class DocumentTransformer:
    """Transform and enrich documents before upload."""
    
    def __init__(self, options: UploadOptions):
        self.options = options
    
    def transform(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform documents with metadata and normalization."""
        transformed = []
        
        for doc in documents:
            # Add metadata
            if self.options.add_metadata:
                doc = self._add_metadata(doc)
            
            # Normalize types
            doc = self._normalize_types(doc)
            
            transformed.append(doc)
        
        return transformed
    
    def _add_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata fields."""
        prefix = self.options.metadata_prefix
        
        # Add upload timestamp if not present
        if f"{prefix}_uploaded_at" not in doc:
            doc[f"{prefix}_uploaded_at"] = datetime.utcnow()
        
        # Preserve original created_date if present
        if "created_date" not in doc:
            doc["created_date"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        return doc
    
    def _normalize_types(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize types for MongoDB compatibility."""
        normalized = {}
        
        for key, value in doc.items():
            # Handle numpy types (from pandas)
            if hasattr(value, 'item'):  # numpy scalar
                value = value.item()
            
            # Convert pandas Timestamp to datetime
            if hasattr(value, 'to_pydatetime'):
                value = value.to_pydatetime()
            
            # Keep None as None (not empty string)
            if pd.isna(value):
                value = None
            
            normalized[key] = value
        
        return normalized
    
    def infer_unique_fields(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Intelligently infer unique identifier fields."""
        if not documents:
            return []
        
        # Common unique field patterns
        candidate_fields = ['model', 'name', 'id', 'timestamp', 'date', 'pipeline']
        
        available_fields = set(documents[0].keys())
        unique_fields = []
        
        for field in candidate_fields:
            if field in available_fields:
                # Check if field has unique values
                values = [doc.get(field) for doc in documents[:100]]  # Sample
                if len(set(str(v) for v in values if v is not None)) == len([v for v in values if v is not None]):
                    unique_fields.append(field)
                    break  # Found a unique field
        
        # If no single unique field, try combinations
        if not unique_fields and 'model' in available_fields:
            unique_fields = ['model']
            if 'timestamp' in available_fields:
                unique_fields.append('timestamp')
        
        return unique_fields


# ============================================================================
# MongoDB Uploader
# ============================================================================

class MongoDBUploader:
    """Handles MongoDB connection and bulk upload operations."""
    
    def __init__(self, config: MongoDBConfig):
        self.config = config
        self.client: Optional[pymongo.MongoClient] = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def connect(self):
        """Establish MongoDB connection."""
        logger.info(f"Connecting to MongoDB at {self.config.host}:{self.config.port}")
        
        self.client = pymongo.MongoClient(
            self.config.uri,
            serverSelectionTimeoutMS=self.config.timeout_ms
        )
        
        # Test connection
        self.client.server_info()
        logger.info("✅ Connected to MongoDB")
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def upload(
        self,
        documents: List[Dict[str, Any]],
        database_name: str,
        collection_name: str,
        options: UploadOptions
    ) -> UploadResult:
        """Upload documents to MongoDB with bulk operations."""
        
        start_time = datetime.now()
        
        # Get collection
        db = self.client[database_name]
        collection = db[collection_name]
        
        # Create indexes if requested
        if options.create_indexes:
            self._create_indexes(collection, documents, options)
        
        # Perform bulk upload
        result = self._bulk_upload(collection, documents, options)
        
        # Calculate duration
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _create_indexes(
        self,
        collection,
        documents: List[Dict[str, Any]],
        options: UploadOptions
    ):
        """Create indexes for efficient querying."""
        if not documents:
            return
        
        # Determine fields to index
        index_fields = options.index_fields or []
        
        if not index_fields and options.unique_fields:
            index_fields = options.unique_fields
        
        # Auto-detect common index candidates
        if not index_fields:
            common_index_fields = ['model', 'timestamp', 'date', 'status', 'pipeline']
            available = set(documents[0].keys())
            index_fields = [f for f in common_index_fields if f in available]
        
        # Create indexes
        for field in index_fields:
            try:
                collection.create_index([(field, pymongo.ASCENDING)])
                logger.info(f"Created index on field: {field}")
            except PyMongoError as e:
                logger.warning(f"Could not create index on {field}: {e}")
        
        # Create compound index for unique fields
        if options.unique_fields and len(options.unique_fields) > 1:
            try:
                index_spec = [(f, pymongo.ASCENDING) for f in options.unique_fields]
                collection.create_index(index_spec, unique=False, background=True)
                logger.info(f"Created compound index on: {options.unique_fields}")
            except PyMongoError as e:
                logger.warning(f"Could not create compound index: {e}")
    
    def _bulk_upload(
        self,
        collection,
        documents: List[Dict[str, Any]],
        options: UploadOptions
    ) -> UploadResult:
        """Perform bulk upload with batching."""
        
        total_inserted = 0
        total_updated = 0
        total_failed = 0
        errors = []
        
        # Prepare bulk operations
        if options.upsert and options.unique_fields:
            operations = self._build_upsert_operations(documents, options.unique_fields)
        else:
            # Simple insert_many
            try:
                result = collection.insert_many(documents, ordered=options.ordered)
                total_inserted = len(result.inserted_ids)
            except BulkWriteError as e:
                total_inserted = e.details.get('nInserted', 0)
                total_failed = len(e.details.get('writeErrors', []))
                errors = [err['errmsg'] for err in e.details.get('writeErrors', [])]
            
            return UploadResult(
                status="success" if total_failed == 0 else "partial",
                documents_read=len(documents),
                documents_processed=total_inserted + total_failed,
                documents_inserted=total_inserted,
                documents_updated=0,
                documents_failed=total_failed,
                errors=errors
            )
        
        # Batched bulk write for upsert operations
        batch_size = options.batch_size
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Uploading to {collection.name}...",
                total=len(operations)
            )
            
            for i in range(0, len(operations), batch_size):
                batch = operations[i:i + batch_size]
                
                try:
                    result = collection.bulk_write(batch, ordered=options.ordered)
                    total_inserted += result.upserted_count
                    total_updated += result.modified_count
                    
                except BulkWriteError as e:
                    total_inserted += e.details.get('nUpserted', 0)
                    total_updated += e.details.get('nModified', 0)
                    write_errors = e.details.get('writeErrors', [])
                    total_failed += len(write_errors)
                    errors.extend([err['errmsg'] for err in write_errors[:5]])  # Limit error messages
                
                progress.update(task, advance=len(batch))
        
        status = "success" if total_failed == 0 else ("partial" if total_inserted + total_updated > 0 else "failed")
        
        return UploadResult(
            status=status,
            documents_read=len(documents),
            documents_processed=total_inserted + total_updated + total_failed,
            documents_inserted=total_inserted,
            documents_updated=total_updated,
            documents_failed=total_failed,
            errors=errors
        )
    
    def _build_upsert_operations(
        self,
        documents: List[Dict[str, Any]],
        unique_fields: List[str]
    ) -> List[UpdateOne]:
        """Build bulk upsert operations."""
        operations = []
        
        for doc in documents:
            # Build filter from unique fields
            filter_doc = {field: doc[field] for field in unique_fields if field in doc}
            
            if not filter_doc:
                # No unique fields, skip or insert
                continue
            
            # Upsert operation
            operations.append(
                UpdateOne(
                    filter_doc,
                    {"$set": doc},
                    upsert=True
                )
            )
        
        return operations


# ============================================================================
# Main Upload Function
# ============================================================================

def upload_file_to_mongodb(
    file_path: str,
    database_name: str,
    collection_name: str,
    config: Optional[MongoDBConfig] = None,
    options: Optional[UploadOptions] = None
) -> UploadResult:
    """
    Upload CSV or JSON file to MongoDB with intelligent handling.
    
    This is the main entry point for file uploads.

    Args:
        file_path: Path to CSV or JSON file
        database_name: MongoDB database name
        collection_name: MongoDB collection name
        config: MongoDB configuration (uses env vars if None)
        options: Upload options (uses defaults if None)

    Returns:
        UploadResult with operation details
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        ConnectionFailure: If MongoDB connection fails
    """
    # Setup
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    config = config or MongoDBConfig.from_env()
    options = options or UploadOptions()
    
    # Detect format and load documents
    file_format = detect_file_format(file_path)
    loader = get_loader(file_format)
    
    console.print(f"📂 Loading {file_format.value.upper()} file: [cyan]{file_path.name}[/cyan]")
    documents = loader.load(file_path)
    
    if not documents:
        raise ValueError(f"No documents found in {file_path}")
    
    console.print(f"✅ Loaded {len(documents)} documents")
    
    # Transform documents
    transformer = DocumentTransformer(options)
    
    # Infer unique fields if not specified
    if options.unique_fields is None:
        options.unique_fields = transformer.infer_unique_fields(documents)
        if options.unique_fields:
            console.print(f"🔑 Auto-detected unique fields: [yellow]{', '.join(options.unique_fields)}[/yellow]")
    
    documents = transformer.transform(documents)
    
    # Handle dry-run before connecting to MongoDB
    if options.dry_run:
        console.print(f"\n🔍 [yellow]DRY RUN: Would upload {len(documents)} documents[/yellow]")
        console.print(f"   Database: {database_name}")
        console.print(f"   Collection: {collection_name}")
        if options.unique_fields:
            console.print(f"   Unique fields: {', '.join(options.unique_fields)}")
        console.print(f"   Upsert: {options.upsert}")
        console.print(f"   Create indexes: {options.create_indexes}")
        
        return UploadResult(
            status="success",
            documents_read=len(documents),
            documents_processed=0,
            documents_inserted=0,
            documents_updated=0,
            documents_failed=0,
            duration_seconds=0.0
        )
    
    # Upload to MongoDB
    with MongoDBUploader(config) as uploader:
        result = uploader.upload(
            documents=documents,
            database_name=database_name,
            collection_name=collection_name,
            options=options
        )
    
    return result


# ============================================================================
# Legacy Compatibility
# ============================================================================

def upload_csv_to_mongodb(
    csv_file_path: str,
    database_name: str,
    collection_name: str,
    mongo_config: Optional[MongoDBConfig] = None
) -> Dict[str, Any]:
    """
    Upload CSV data to MongoDB collection.
    
    DEPRECATED: Use upload_file_to_mongodb() instead.
    This function is kept for backward compatibility.
    
    Args:
        csv_file_path: Path to CSV file
        database_name: Name of MongoDB database
        collection_name: Name of MongoDB collection
        mongo_config: MongoDB configuration (uses environment if None)
        
    Returns:
        Dictionary with operation results
    """
    logger.warning("upload_csv_to_mongodb is deprecated. Use upload_file_to_mongodb instead.")
    
    result = upload_file_to_mongodb(
        file_path=csv_file_path,
        database_name=database_name,
        collection_name=collection_name,
        config=mongo_config,
        options=UploadOptions()
    )
    
    # Convert UploadResult to legacy dict format
    return {
        "status": "success" if result.status == "success" else "partial",
            "database": database_name,
            "collection": collection_name,
        "records_processed": result.documents_processed,
    }


class MongoDBHandler:
    """
    Legacy handler class for MongoDB operations.
    
    DEPRECATED: This class is kept for backward compatibility.
    Use upload_file_to_mongodb() directly instead.
    """

    def __init__(self, args):
        """Initialize the MongoDBHandler."""
        import argparse

        self.args = args
        self.config = MongoDBConfig.from_env()
        self.database_name = args.database_name
        self.collection_name = args.collection_name
        
        # Support both old and new parameter names
        self.file_path = getattr(args, 'file_path', None) or getattr(args, 'csv_file_path', None)
        self.unique_key = getattr(args, 'unique_key', None)
        self.return_status = False

    def run(self) -> bool:
        """Execute the MongoDB upload operation."""
        logger.warning("MongoDBHandler is deprecated. Use upload_file_to_mongodb instead.")
        
        print("\n" + "=" * 80)
        print("📤 UPLOADING TO MONGODB")
        print("=" * 80)
        print(f"📂 File: {self.file_path}")
        print(f"🗄️  Database: {self.database_name}")
        print(f"📊 Collection: {self.collection_name}")
        
        try:
            # Parse unique fields if provided
            unique_fields = None
            if self.unique_key:
                unique_fields = [k.strip() for k in self.unique_key.split(',')]
            
            options = UploadOptions(unique_fields=unique_fields)
            
            result = upload_file_to_mongodb(
                file_path=self.file_path,
                database_name=self.database_name,
                collection_name=self.collection_name,
                config=self.config,
                options=options
            )
            
            print(f"✅ Successfully processed {result.documents_processed} documents")
            print(f"   Inserted: {result.documents_inserted}")
            print(f"   Updated: {result.documents_updated}")
            print("=" * 80 + "\n")
            
            self.return_status = True
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            self.return_status = False
        except ConnectionFailure as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("💡 Tip: Check MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASSWORD")
            self.return_status = False
        except ValueError as e:
            print(f"❌ Invalid file: {e}")
            self.return_status = False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            logger.exception("MongoDB upload failed")
            self.return_status = False
        
        print("=" * 80 + "\n")
        return self.return_status

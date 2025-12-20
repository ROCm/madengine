"""MongoDB operations for madengine.

This module provides functions to handle MongoDB operations, including
checking for collection existence, creating collections, and updating datasets.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import argparse
import logging
from typing import Optional, Dict, Any

import pandas as pd
import pymongo
from pymongo.errors import ConnectionFailure, PyMongoError

logger = logging.getLogger(__name__)


class MongoDBConfig:
    """Configuration class for MongoDB operations."""
    
    def __init__(self):
        """Initialize MongoDB configuration from environment variables."""
        self.user = os.getenv("MONGO_USER", "username")
        self.password = os.getenv("MONGO_PASSWORD", "password")
        self.host = os.getenv("MONGO_HOST", "localhost")
        self.port = os.getenv("MONGO_PORT", "27017")
    
    @property
    def uri(self) -> str:
        """Get MongoDB connection URI.
        
        Returns:
            MongoDB connection string
        """
        return f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}"


def load_csv_to_dataframe(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame containing the CSV data.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")
    
    logger.info(f"Loading CSV file: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_dataframe_for_mongo(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for MongoDB insertion.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Processed DataFrame ready for MongoDB
    """
    # Replace NaN with empty string
    df = df.where(pd.notnull(df), "")
    
    # Convert all columns to string type except boolean columns
    for col in df.columns:
        if df[col].dtype != "bool":
            df[col] = df[col].astype(str)
    
    # Add created_date column
    df["created_date"] = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
    
    # Remove leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    
    return df


def upload_csv_to_mongodb(
    csv_file_path: str,
    database_name: str,
    collection_name: str,
    mongo_config: Optional[MongoDBConfig] = None
) -> Dict[str, Any]:
    """Upload CSV data to MongoDB collection.
    
    Args:
        csv_file_path: Path to CSV file
        database_name: Name of MongoDB database
        collection_name: Name of MongoDB collection
        mongo_config: MongoDB configuration (uses environment if None)
        
    Returns:
        Dictionary with operation results
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ConnectionFailure: If MongoDB connection fails
    """
    if mongo_config is None:
        mongo_config = MongoDBConfig()
    
    logger.info(f"Connecting to MongoDB at {mongo_config.host}:{mongo_config.port}")
    
    # Load and prepare data
    df = load_csv_to_dataframe(csv_file_path)
    df = prepare_dataframe_for_mongo(df)
    
    # Connect to MongoDB
    try:
        client = pymongo.MongoClient(mongo_config.uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.server_info()
        logger.info("Successfully connected to MongoDB")
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    
    try:
        db = client[database_name]
        collection = db[collection_name]
        
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            logger.info(f"Collection '{collection_name}' does not exist. Creating it.")
            db.create_collection(collection_name)
        
        # Insert records
        records = df.to_dict(orient="records")
        logger.info(f"Uploading {len(records)} records to '{collection_name}'")
        
        for record in records:
            # Use upsert to avoid duplicates
            collection.update_one(record, {"$set": record}, upsert=True)
        
        result = {
            "status": "success",
            "database": database_name,
            "collection": collection_name,
            "records_processed": len(records),
        }
        
        logger.info(f"Successfully uploaded {len(records)} records")
        return result
        
    except PyMongoError as e:
        logger.error(f"MongoDB operation failed: {e}")
        raise
    finally:
        client.close()


class MongoDBHandler:
    """Handler class for MongoDB operations.
    
    This class provides a command-line interface wrapper for MongoDB operations.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the MongoDBHandler.

        Args:
            args: Command-line arguments containing database config.
        """
        self.args = args
        self.config = MongoDBConfig()
        self.database_name = args.database_name
        self.collection_name = args.collection_name
        self.csv_file_path = args.csv_file_path
        self.return_status = False

    def run(self) -> bool:
        """Execute the MongoDB upload operation.
        
        Returns:
            True if successful, False otherwise.
        """
        print("\n" + "=" * 80)
        print("📤 UPLOADING TO MONGODB")
        print("=" * 80)
        print(f"📂 CSV file: {self.csv_file_path}")
        print(f"🗄️  Database: {self.database_name}")
        print(f"📊 Collection: {self.collection_name}")
        
        try:
            result = upload_csv_to_mongodb(
                csv_file_path=self.csv_file_path,
                database_name=self.database_name,
                collection_name=self.collection_name,
                mongo_config=self.config
            )
            
            print(f"✅ Successfully uploaded {result['records_processed']} records")
            print("=" * 80 + "\n")
            self.return_status = True
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("=" * 80 + "\n")
            self.return_status = False
        except ConnectionFailure as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("💡 Tip: Check MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASSWORD environment variables")
            print("=" * 80 + "\n")
            self.return_status = False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            logger.exception("MongoDB upload failed")
            print("=" * 80 + "\n")
            self.return_status = False
        
        return self.return_status


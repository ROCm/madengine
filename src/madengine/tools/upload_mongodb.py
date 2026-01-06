#!/usr/bin/env python
"""Module to update MongoDB collections with data from a CSV or JSON file.

This module provides functions to handle MongoDB operations, including
checking for collection existence, creating collections, and updating datasets.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import json
import argparse

# third-party modules
import pandas as pd
import pymongo
from pymongo.errors import ConnectionFailure
from typing import Optional, Union, List, Dict

# MAD Engine modules
from madengine.db.logger import setup_logger

# Create the logger
LOGGER = setup_logger()

class MongoDBHandler:
    """Class to handle MongoDB operations."""
    
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the MongoDBHandler.

        Args:
            args (argparse.Namespace): The arguments passed to the script.
        """
        # MongoDB connection details from environment variables
        mongo_user = os.getenv("MONGO_USER", "username")
        mongo_password = os.getenv("MONGO_PASSWORD", "password")
        mongo_host = os.getenv("MONGO_HOST", "localhost")
        mongo_port = os.getenv("MONGO_PORT", "27017")
        mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}"
        self.uri = mongo_uri
        self.database_name = args.database_name
        self.collection_name = args.collection_name
        self.csv_file_path = getattr(args, 'csv_file_path', None)
        self.json_file_path = getattr(args, 'json_file_path', None)
        self.client = None
        self.db = None

    def connect(self) -> None:
        """Connect to the MongoDB server."""
        try:
            self.client = pymongo.MongoClient(self.uri)
            self.db = self.client[self.database_name]
            LOGGER.info("Connected to MongoDB.")
        except ConnectionFailure as e:
            LOGGER.error(f"Failed to connect to MongoDB: {e}")
            raise

    def collection_exists(self) -> bool:
        """Check if a collection exists in the database.
        
        Returns:
            bool: True if the collection exists, False otherwise.
        """
        return self.collection_name in self.db.list_collection_names()

    def update_collection(self, data: Union[pd.DataFrame, List[Dict]]) -> None:
        """Update a MongoDB collection with data from a DataFrame or list of dicts.

        Args:
            data: DataFrame or list of dicts containing the data to update.
        """
        if not self.collection_exists():
            LOGGER.info(f"Collection '{self.collection_name}' does not exist. Creating it.")
            self.db.create_collection(self.collection_name)

        collection = self.db[self.collection_name]
        
        # Convert to list of records if DataFrame
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        else:
            records = data
        
        for record in records:
            # Use an appropriate unique identifier for upsert (e.g., "_id" or another field)
            collection.update_one(record, {"$set": record}, upsert=True)
        LOGGER.info(f"Updated collection '{self.collection_name}' with {len(records)} records.")

    def run(self) -> None:
        """Run the process of updating a MongoDB collection with data from a CSV or JSON file.
        """
        self.connect()
        
        if self.json_file_path:
            # Load JSON file (perf_entry_super.json)
            data = load_json_to_list(self.json_file_path)
            
            # Add created_date to each record
            from datetime import datetime
            created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for record in data:
                if "created_date" not in record:
                    record["created_date"] = created_date
            
            self.update_collection(data)
        elif self.csv_file_path:
            # Load CSV file (legacy perf.csv)
            data = load_csv_to_dataframe(self.csv_file_path)

            # if the value is NaN, replace it with empty string
            data = data.where(pd.notnull(data), "")
            # Convert all columns to string type except boolean columns
            for col in data.columns:
                if data[col].dtype != "bool":
                    data[col] = data[col].astype(str)

            # Added created_date column and set it to now
            data["created_date"] = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")

            # Remove any leading or trailing whitespace from column names
            data.columns = data.columns.str.strip()
           
            self.update_collection(data)
        else:
            raise ValueError("Either csv_file_path or json_file_path must be provided")


def load_csv_to_dataframe(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")
    return pd.read_csv(csv_path)


def load_json_to_list(json_path: str) -> List[Dict]:
    """Load a JSON file into a list of dictionaries.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        List[Dict]: List of dictionaries containing the JSON data.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file '{json_path}' not found.")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Ensure it's a list
    if isinstance(data, list):
        return data
    else:
        return [data]

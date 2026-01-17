"""Module of the MAD Engine database.

This module provides the functions to create and update tables in the database.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import os
from datetime import datetime, timezone
# third-party modules
from sqlalchemy import Column, Integer, String, DateTime, TEXT, MetaData, Table
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine
from sqlalchemy.orm import mapper, clear_mappers

# MAD Engine modules (dual import: package first, then standalone fallback for scp use)
try:  # Package import context
    from madengine.db.logger import setup_logger  # type: ignore
    from madengine.db.base_class import BASE, BaseMixin  # type: ignore
    from madengine.db.utils import get_env_vars  # type: ignore
except ImportError:  # Standalone (scp) execution context
    from logger import setup_logger  # type: ignore
    from base_class import BASE, BaseMixin  # type: ignore
    from utils import get_env_vars  # type: ignore


# Create the logger
LOGGER = setup_logger()
# Get the environment variables
ENV_VARS = get_env_vars()

# Global engine variable - will be lazily initialized
ENGINE = None

def get_engine():
    """Get database engine, creating it lazily when first needed.
    
    Returns:
        sqlalchemy.engine.Engine: Database engine
        
    Raises:
        ValueError: If required environment variables are not set
    """
    global ENGINE
    
    if ENGINE is None:
        # Check if the environment variables are set
        if not ENV_VARS["user_name"] or not ENV_VARS["user_password"]:
            raise ValueError("User name or password not set")

        if not ENV_VARS["db_hostname"] or not ENV_VARS["db_port"]:
            raise ValueError("DB hostname or port not set")

        if not ENV_VARS["db_name"]:
            raise ValueError("DB name not set")

        # Create the engine
        ENGINE = create_engine(
            "mysql+pymysql://{user_name}:{user_password}@{hostname}:{port}/{db_name}".format(
                user_name=ENV_VARS["user_name"],
                user_password=ENV_VARS["user_password"],
                hostname=ENV_VARS["db_hostname"],
                port=ENV_VARS["db_port"],
                db_name=ENV_VARS["db_name"],
            )
        )
        LOGGER.info("Database engine created for %s@%s:%s/%s", 
                   ENV_VARS["user_name"], ENV_VARS["db_hostname"], 
                   ENV_VARS["db_port"], ENV_VARS["db_name"])
    
    return ENGINE

# Check for eager initialization
if os.getenv("MADENGINE_DB_EAGER") == "1":
    LOGGER.info("MADENGINE_DB_EAGER=1 detected, creating engine immediately")
    ENGINE = get_engine()

# Define the path to the SQL file
SQL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'db_table_def.sql')
# Update TABLE_SCHEMA and TABLE_NAME variables
TABLE_SCHEMA = ENV_VARS["db_name"]
TABLE_NAME = None
# get table name from SQL file
with open(SQL_FILE_PATH, 'r') as file:
    for line in file:
        if 'CREATE TABLE' in line:
            TABLE_NAME = line.split(' ')[2].split('(')[0]
            TABLE_NAME = TABLE_NAME.replace('`', '')
            break

if TABLE_NAME is None:
    raise ValueError("Table name not found in SQL file")

def read_sql_file(file_path: str) -> str:
    """Read the SQL file and return its content."""
    with open(file_path, 'r') as file:
        return file.read()

def parse_table_definition(sql_content: str) -> Table:
    """Parse the SQL content and return the table definition."""
    metadata = MetaData()
    table = Table(TABLE_NAME, metadata, autoload_with=ENGINE, autoload_replace=True)
    return table

# Read and parse the SQL file
sql_content = read_sql_file(SQL_FILE_PATH)
db_table_definition = parse_table_definition(sql_content)

# Clear any existing mappers
clear_mappers()

# Define the DB_TABLE class dynamically
class DB_TABLE(BaseMixin, BASE):
    """Represents db job table"""
    __tablename__ = db_table_definition.name
    __table__ = db_table_definition


def connect_db() -> None:
    """Create DB if it doesnt exist

    This function creates the database if it does not exist.

    Raises:
        OperationalError: An error occurred while creating the database.
    """
    db_name = ENV_VARS["db_name"]
    user_name = ENV_VARS["user_name"]

    try:
        get_engine().execute("Use {}".format(db_name))
        return
    except OperationalError:  # as err:
        LOGGER.warning(
            "Database %s does not exist, attempting to create database", db_name
        )

    try:
        get_engine().execute("Create database if not exists {}".format(db_name))
    except OperationalError as err:
        LOGGER.error("Database creation failed %s for username: %s", err, user_name)

    get_engine().execute("Use {}".format(db_name))
    get_engine().execute("SET GLOBAL max_allowed_packet=4294967296")


def clear_db() -> None:
    """Clear DB

    This function clears the database.

    Raises:
        OperationalError: An error occurred while clearing the database
    """
    db_name = ENV_VARS["db_name"]

    try:
        get_engine().execute("DROP DATABASE IF EXISTS {}".format(db_name))
        return
    except OperationalError:  # as err:
        LOGGER.warning("Database %s could not be dropped", db_name)


def show_db() -> None:
    """Show DB

    This function shows the database.

    Raises:
        OperationalError: An error occurred while showing the database
    """
    db_name = ENV_VARS["db_name"]

    try:
        result = get_engine().execute(
            "SELECT * FROM {} \
                WHERE {}.created_date= \
                    (SELECT MAX(created_date) FROM {}) ;".format(DB_TABLE.__tablename__)
        )
        for row in result:
            LOGGER.info("Latest entry: %s", row)
        return
    except OperationalError:  # as err:
        LOGGER.warning("Database %s could not be shown", db_name)


def create_tables() -> bool:
    """Function to create or sync DB tables/triggers

    This function creates or syncs the database tables/triggers.

    Returns:
        bool: True if the tables are created successfully.

    Raises:
        OperationalError: An error occurred while creating the tables.
    """
    connect_db()
    all_tables = [DB_TABLE]

    for table in all_tables:
        if not table.__table__.exists(ENGINE):
            try:
                table.__table__.create(ENGINE)
                LOGGER.info("Created: %s", table.__tablename__)
            except OperationalError as err:
                LOGGER.warning("Error occurred %s", err)
                LOGGER.warning("Failed to create table %s \n", table.__tablename__)
                continue
        else:
            LOGGER.info("Table %s already exists", table.__tablename__)

    return True


def trim_column(col_name: str) -> None:
    """Trim column

    This function trims the column.

    Args:
        col_name: Name of the column to be trimmed.

    Raises:
        OperationalError: An error occurred while trimming the column.
    """
    get_engine().execute(
        "UPDATE {} \
        SET \
        {} = TRIM({});".format(
            DB_TABLE.__tablename__, col_name, col_name
        )
    )
    show_db()


def get_column_names() -> list:
    """Get column names

    This function gets the column names.

    Returns:
        list: List of column names.

    Raises:
        OperationalError: An error occurred while getting the column names.
    """
    db_name = ENV_VARS["db_name"]

    result = get_engine().execute(
        "SELECT `COLUMN_NAME` \
            FROM `INFORMATION_SCHEMA`.`COLUMNS` \
                WHERE `TABLE_SCHEMA`='{}' \
                AND `TABLE_NAME`='{}'".format(db_name, DB_TABLE.__tablename__)
    )
    ret = []
    for row in result:
        ret.append(row[0])
    return ret

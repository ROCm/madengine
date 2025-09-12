"""Script to upload csv files to the database, 
and create or update tables in the database.

This script uploads csv files to the database, and creates or updates tables in the database.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import sys
import argparse
import pandas as pd
import typing
from datetime import datetime
# third-party modules
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
# MAD Engine modules (dual import: prefer package, fallback to local)
try:
    from madengine.db.database import get_engine, create_tables, DB_TABLE, LOGGER  # type: ignore
    from madengine.db.utils import (
        dataFrame_to_list,
        load_perf_csv,
        replace_nans_with_None,
    )  # type: ignore
    from madengine.db.relative_perf import relative_perf_all_configs  # type: ignore
except ImportError:
    from database import get_engine, create_tables, DB_TABLE, LOGGER  # type: ignore
    from utils import (
        dataFrame_to_list,
        load_perf_csv,
        replace_nans_with_None,
    )  # type: ignore
    from relative_perf import relative_perf_all_configs  # type: ignore


def add_csv_to_db(data: pd.DataFrame) -> bool:
    """Add csv files to the database.

    This function adds csv files to the database.

    Args:
        data (pd.DataFrame): The data.

    Returns:
        bool: True if data was successfully added, False otherwise
    """
    LOGGER.info("adding csv to Database")
    # Create the session
    session = sessionmaker()
    session.configure(bind=get_engine())
    s = session()

    # change nans to None to upload to database
    data = replace_nans_with_None(data)

    # Add unique ID column if it doesn't exist
    if 'id' not in data.columns:
        # Get the max ID from the existing table to ensure uniqueness
        try:
            max_id_query = s.query(DB_TABLE.id).order_by(DB_TABLE.id.desc()).first()
            start_id = 1 if max_id_query is None else max_id_query[0] + 1
        except:
            LOGGER.warning('Failed to query max ID, starting from 1')
            start_id = 1

        # Add sequential unique IDs
        data['id'] = range(start_id, start_id + len(data))

    # Explicitly set created_date to current timestamp if not provided
    if 'created_date' not in data.columns:
        data['created_date'] = datetime.now()

    LOGGER.info("Data:")
    LOGGER.info(data)
    # add data to databases
    success_count = 0
    data_as_list = dataFrame_to_list(data)
    total_records = len(data_as_list)

    for model_perf_info in tqdm(data_as_list):
        try:
            # Ensure created_date is set for each record if not present
            if 'created_date' not in model_perf_info or model_perf_info['created_date'] is None:
                model_perf_info['created_date'] = datetime.now()

            record = DB_TABLE(**model_perf_info)
            s.add(record)
            success_count += 1
        except Exception as e:
            LOGGER.warning(
                'Failed to add record to table due to %s \n', str(e))
            LOGGER.info(model_perf_info)
            s.rollback()

    # commit changes and close sesstion
    try:
        s.commit()
        LOGGER.info('Successfully added %d out of %d records to the database', 
                   success_count, total_records)
        success = success_count > 0
    except Exception as e:
        LOGGER.error('Failed to commit changes: %s', str(e))
        s.rollback()
        success = False
    finally:
        s.close()

    return success


def main() -> None:
    """Main script function to upload csv files to the database."""
    # parse arg
    parser = argparse.ArgumentParser(description='Upload perf.csv to database')
    parser.add_argument("--csv-file-path", type=str)
    args = parser.parse_args()

    ret = create_tables()
    LOGGER.info('DB creation successful: %s', ret)

    if args.csv_file_path is None:
        LOGGER.info("Only creating tables in the database")
        return
    else:
        # load perf.csv to db
        LOGGER.info("Loading %s to database", args.csv_file_path)
        data = load_perf_csv(args.csv_file_path)
        data = relative_perf_all_configs(data)
        add_csv_to_db(data)

if __name__ == '__main__':
    main()

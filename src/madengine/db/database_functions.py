"""Functions of the MAD Engine database.

This module contains the functions to interact with the MAD Engine database.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import typing

# MAD Engine modules
from database import ENGINE


def get_all_gpu_archs() -> typing.List[str]:
    """Get all GPU architectures from the database.

    Returns:
        typing.List[str]: A list of all GPU architectures in the database.
    """
    matching_entries = ENGINE.execute(
        "SELECT DISTINCT(gpu_architecture) FROM dlm_table"
    )

    archs = []
    for arch in matching_entries.fetchall():
        arch = arch[0]  # return from database is in list
        if arch:
            archs.append("{}".format(arch))

    return archs


def get_matching_db_entries(
    recent_entry: typing.Dict[str, typing.Any], filters: typing.Dict[str, typing.Any]
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Get matching entries from the database.

    Args:
        recent_entry (typing.Dict[str, typing.Any]): The recent entry to compare.
        filters (typing.Dict[str, typing.Any]): The filters to apply.

    Returns:
        typing.List[typing.Dict[str, typing.Any]]: The matching entries.
    """
    print(
        "Looking for entries with {}, {} and {}".format(
            recent_entry["model"], recent_entry["gpu_architecture"], filters
        )
    )

    # find matching entries to current entry
    matching_entries = ENGINE.execute(
        "SELECT * FROM dlm_table \
        WHERE model='{}' \
        AND gpu_architecture='{}' \
        ".format(
            recent_entry["model"], recent_entry["gpu_architecture"]
        )
    )
    matching_entries = matching_entries.mappings().all()

    # filter db entries
    filtered_matching_entries = []
    for m in matching_entries:
        should_add = True
        for filter, value in filters.items():
            if m[filter] != value:
                should_add = False

        if should_add:
            filtered_matching_entries.append(m)

    print(
        "Found {} similar entries in database filtered down to {} entries".format(
            len(matching_entries), len(filtered_matching_entries)
        )
    )
    return filtered_matching_entries

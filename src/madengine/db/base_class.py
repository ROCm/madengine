#!/usr/bin/env python3
"""Module for creating DB tables interfaces

This module provides the base class for our own common functionalities among tables

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# third-party modules
from sqlalchemy.ext.declarative import declarative_base


BASE = declarative_base()


class BaseMixin:
    """Base class for our own common functionalities among tables

    This class provides the common functionalities among tables

    Attributes:
        __tablename__ (str): The name of the table
        __table__ (str): The table object
    """

    @classmethod
    def obj_as_list_dict(cls, obj):
        """Function to help with printing"""
        dict_list = []
        for elem in obj:
            # extra elem at top of dict
            elem.__dict__.pop("_sa_instance_state", None)
            dict_list.append(elem.__dict__)
        return dict_list

    @classmethod
    def obj_columns(cls, obj):
        """Helper function"""
        return obj[0].__table__.columns.keys()

    @classmethod
    def obj_as_dict(cls, obj, ommit_ts=False):
        """Helper function"""
        if "_sa_instance_state" in obj.__dict__.keys():
            obj.__dict__.pop("_sa_instance_state")
        if ommit_ts:
            obj.__dict__.pop("update_ts")
            obj.__dict__.pop("insert_ts")
        return obj.__dict__

    def __repr__(self):
        return "Table name: {0}\nTable columns: {1}".format(
            self.__table__, self.__table__.columns
        )

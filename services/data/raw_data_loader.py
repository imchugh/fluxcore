#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:54:46 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

import csv
import pandas as pd
import pathlib

from infrastructure import file_io
from domain.constants import TIME_INDEX_NAME, DATA_TIME_FORMAT

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN INITS ###
###############################################################################

# TOA5 and EddyPro are fixed industry-standard formats; these constants
# capture only the fields consumed by this module (header line positions,
# separator, non-numeric columns, NA sentinel, and CSV quoting mode).
_FILE_FORMATS = {
    'TOA5': {
        'header_lines':    {'info': 0, 'variable': 1, 'units': 2, 'sampling': 3},
        'separator':       ',',
        'non_numeric_cols': ['TIMESTAMP'],
        'na_values':       'NAN',
        'quoting':         csv.QUOTE_NONNUMERIC,
    },
    'EddyPro': {
        'header_lines':    {'variable': 0, 'units': 1},
        'separator':       '\t',
        'non_numeric_cols': ['DATAH', 'filename', 'date', 'time'],
        'na_values':       'NaN',
        'quoting':         csv.QUOTE_MINIMAL,
    },
}

###############################################################################
### END INITS ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------

def load_raw_data(
        file_path: pathlib.Path, file_format: str, drop_non_numeric=True
        ) -> pd.DataFrame:

    # Initialise dispatcher and get formatter
    DATE_FORMATTERS = {
        'TOA5': _TOA5_date_formatter,
        'EddyPro': _EddyPro_date_formatter,
        }
    formatter = DATE_FORMATTERS.get(file_format)
    
    # Get data
    df = file_io.read_csv_data(
        file_path=file_path, 
        file_format=_FILE_FORMATS[file_format],
        on_bad_lines='skip'
        )
    
    # Apply formatting
    df = formatter(df)

    # Drop non-numeric and return
    return _drop_non_numeric(df=df, file_format=file_format)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------    

def _TOA5_date_formatter(df):
    
    dttm = pd.to_datetime(
        df['TIMESTAMP'],
        format=DATA_TIME_FORMAT,
        errors="coerce"
        )
    return df.set_index(keys=pd.Index(data=dttm, name=TIME_INDEX_NAME))
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _EddyPro_date_formatter(df):
    
    dttm = pd.to_datetime(
        df["date"] + " " + df["time"],
        format=DATA_TIME_FORMAT,
        errors="coerce"
        )
    return df.set_index(keys=pd.Index(data=dttm, name=TIME_INDEX_NAME))
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _drop_non_numeric(df, file_format):
    
    cols_to_drop = _FILE_FORMATS[file_format]['non_numeric_cols']
    return df.drop(columns=cols_to_drop, errors='ignore')
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_data_adapter(system_type):

    adapters = {
        "CSI": CSIDataAdapter(),
        "LICOR": LicorDataAdapter()
        }

    return adapters[system_type]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def load_raw_header(file_path, file_format: str) -> list[list]:
    
    fmt = _FILE_FORMATS[file_format]
    header_dict = fmt.get('header_lines')
    lines = list(header_dict.values())
    keys = list(header_dict.keys())
    return dict(zip(
        keys,
        file_io.read_lines(
            file_path=file_path, 
            begin=min(lines), 
            end=max(lines),
            sep=fmt['separator']
            )
        ))
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_header_adapter(system_type):

    adapters = {
        "CSI": CSIHeaderAdapter(),
        "LICOR": LicorHeaderAdapter()
        }

    return adapters[system_type]
# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------

class BaseAdapter:

    def load(self, file_path):
        raise NotImplementedError
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

class CSIDataAdapter(BaseAdapter):

    def load(self, file_path):

        df = load_raw_data(file_path=file_path, file_format="TOA5")
        df = df[~df.index.duplicated(keep="last")]

        return df        
# -----------------------------------------------------------------------------    

# -----------------------------------------------------------------------------
    
class LicorDataAdapter(BaseAdapter):

    def load(self, file_path):

        df = load_raw_data(file_path=file_path, file_format="EddyPro")
        df = df[~df.index.duplicated(keep="last")]

        return df
# -----------------------------------------------------------------------------  

# -----------------------------------------------------------------------------  

class CSIHeaderAdapter(BaseAdapter):

    def load(self, file_path):

        return load_raw_header(file_path=file_path, file_format="TOA5")
# ----------------------------------------------------------------------------- 

# -----------------------------------------------------------------------------  

class LicorHeaderAdapter(BaseAdapter):

    def load(self, file_path):

        return load_raw_header(file_path=file_path, file_format="EddyPro")
# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################

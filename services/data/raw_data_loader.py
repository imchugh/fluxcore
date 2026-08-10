#!/usr/bin/env python3
"""Load raw TOA5/EddyPro files into time-indexed DataFrames."""

import csv
import pathlib

import pandas as pd

from domain.constants import DATA_TIME_FORMAT, TIME_INDEX_NAME
from infrastructure import file_io

# TOA5 and EddyPro are fixed industry-standard formats; these constants
# capture only the fields consumed by this module (header line positions,
# separator, non-numeric columns, NA sentinel, and CSV quoting mode).
_FILE_FORMATS = {
    "TOA5": {
        "header_lines": {"info": 0, "variable": 1, "units": 2, "sampling": 3},
        "separator": ",",
        "non_numeric_cols": ["TIMESTAMP"],
        "na_values": "NAN",
        "quoting": csv.QUOTE_NONNUMERIC,
    },
    "EddyPro": {
        "header_lines": {"variable": 0, "units": 1},
        "separator": "\t",
        "non_numeric_cols": ["DATAH", "filename", "date", "time"],
        "na_values": "NaN",
        "quoting": csv.QUOTE_MINIMAL,
    },
}

# Maps the site config's logger system_type to the file format it produces.
_SYSTEM_TYPE_FORMAT_MAP = {"CSI": "TOA5", "LICOR": "EddyPro"}


def load_raw_data(
    file_path: pathlib.Path,
    file_format: str,
    drop_non_numeric: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a raw TOA5/EddyPro file into a time-indexed DataFrame.

    **read_csv_kwargs are forwarded to pd.read_csv (via file_io.read_csv_data)
    — e.g. usecols=[...] to restrict which columns are parsed out of a wide
    file. If usecols is given, it must include the file format's timestamp
    column(s) ('TIMESTAMP' for TOA5; 'date' and 'time' for EddyPro), since
    the date formatter below depends on them.
    """
    DATE_FORMATTERS = {
        "TOA5": _TOA5_date_formatter,
        "EddyPro": _EddyPro_date_formatter,
    }

    df = file_io.read_csv_data(
        file_path=file_path,
        file_format=_FILE_FORMATS[file_format],
        on_bad_lines="skip",
        **read_csv_kwargs,
    )
    df = DATE_FORMATTERS[file_format](df)

    if drop_non_numeric:
        df = _drop_non_numeric(df=df, file_format=file_format)

    return df


def _TOA5_date_formatter(df):

    dttm = pd.to_datetime(df["TIMESTAMP"], format=DATA_TIME_FORMAT, errors="coerce")
    return df.drop(columns=["TIMESTAMP"]).set_index(
        pd.Index(data=dttm, name=TIME_INDEX_NAME)
    )


def _EddyPro_date_formatter(df):

    dttm = pd.to_datetime(
        df["date"] + " " + df["time"], format=DATA_TIME_FORMAT, errors="coerce"
    )
    return df.set_index(keys=pd.Index(data=dttm, name=TIME_INDEX_NAME))


def _drop_non_numeric(df, file_format):

    cols_to_drop = _FILE_FORMATS[file_format]["non_numeric_cols"]
    return df.drop(columns=cols_to_drop, errors="ignore")


def get_data_adapter(system_type: str):
    """Return a `load(file_path)` closure bound to `system_type`'s file format."""
    file_format = _SYSTEM_TYPE_FORMAT_MAP[system_type]

    def load(file_path):
        return load_raw_data(file_path=file_path, file_format=file_format)

    return load


def load_raw_header(file_path, file_format: str) -> dict:
    """Read a raw file's header rows, keyed by line type ('variable', 'units', etc).

    Non-numeric columns (per `_FILE_FORMATS`) are dropped from every header
    row, matching the columns `load_raw_data` would drop if called with
    `drop_non_numeric=True`.
    """
    fmt = _FILE_FORMATS[file_format]
    header_dict = fmt.get("header_lines")
    lines = list(header_dict.values())
    keys = list(header_dict.keys())
    result = dict(
        zip(
            keys,
            file_io.read_lines(
                file_path=file_path,
                begin=min(lines),
                end=max(lines),
                sep=fmt["separator"],
            ),
        )
    )
    non_numeric = set(fmt.get("non_numeric_cols", []))
    if "variable" in result and non_numeric:
        drop_indices = {i for i, v in enumerate(result["variable"]) if v in non_numeric}
        for key in ("variable", "units", "sampling"):
            if key in result:
                result[key] = [
                    v for i, v in enumerate(result[key]) if i not in drop_indices
                ]
    return result


def get_header_adapter(system_type: str):
    """Return a `load(file_path)` closure bound to `system_type`'s file format."""
    file_format = _SYSTEM_TYPE_FORMAT_MAP[system_type]

    def load(file_path):
        return load_raw_header(file_path=file_path, file_format=file_format)

    return load

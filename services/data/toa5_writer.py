#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOA5 file writer.

Provides a single public function, `write_toa5`, for writing data and headers
to a Campbell Scientific TOA5-format file.  The function accepts either a
single pandas DataFrame or a list of DataFrames (concatenated in order before
writing), plus an explicit list of header rows.  The TOA5 info line (line 1)
is optional and falls back to a sensible set of dummy values.

Raw I/O mechanics (CSV quoting, atomic write) are handled by
``infrastructure.file_io.write_toa5_csv``.

The caller is responsible for ensuring that:
  - the DataFrame has a DatetimeIndex (used as the TIMESTAMP column).
  - the three header lists do NOT include TIMESTAMP — the writer prepends it
    to all three internally using the fixed TOA5 values ('TIMESTAMP', 'TS', '').
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

import pathlib

import pandas as pd

from collections.abc import Callable

from domain.constants import DATA_TIME_FORMAT
from infrastructure import data_conditioning, file_io
from infrastructure.datetime_utils import format_fast_timestamp

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CONSTANTS ###
###############################################################################

_TIMESTAMP_COL = 'TIMESTAMP'
_TIMESTAMP_UNITS = 'TS'
_TIMESTAMP_SAMPLING = ''

# Default TOA5 info line (line 1) used when no `info` argument is supplied.
DEFAULT_TOA5_INFO: list[str] = [
    'TOA5',
    'NoStation',
    'CR1000',
    '9999',
    'cr1000.std.99.99',
    'CPU:noprogram.cr1',
    '9999',
    'default_table',
]

###############################################################################
### END CONSTANTS ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------
def _coerce_integer_floats(data: pd.DataFrame) -> pd.DataFrame:
    """Convert float columns that contain only whole-number values to Int64.

    Pandas represents integer columns that contain any NaN as float64, which
    causes values like 50 to be written as 50.0.  This restores the original
    integer representation for any float column where every non-NaN value has
    a zero fractional part (e.g. RECORD, sample counts, diagnostic flags,
    totals).

    Args:
        data: DataFrame to process.

    Returns:
        DataFrame with qualifying float columns cast to Int64.
    """

    for col in data.select_dtypes(include='float').columns:
        non_null = data[col].dropna()
        if len(non_null) and (non_null % 1 == 0).all():
            data[col] = data[col].astype('Int64')
    return data
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def write_toa5(
    data: pd.DataFrame | list[pd.DataFrame],
    headers: list[list],
    file_path: str | pathlib.Path,
    info: list[str] | None = None,
    timestamp_formatter: Callable | None = None,
) -> None:
    """Write data and headers to a TOA5-format file.

    The TOA5 format has four header lines followed by data rows:
        1. Info      – station/logger metadata  (written from ``info``).
        2. Variables – column names             (``headers[0]``).
        3. Units     – engineering units        (``headers[1]``).
        4. Sampling  – statistical type         (``headers[2]``).

    Args:
        data:
            DataFrame or list of DataFrames to write.  If a list is supplied
            the frames are concatenated in order before writing.
        headers:
            List of three lists — [variables, units, sampling] — matching
            lines 2–4 of the TOA5 file, **excluding** the TIMESTAMP column.
            Each list must have exactly ``len(data.columns)`` entries.
            The writer prepends TIMESTAMP, 'TS', and '' to the three lists
            before writing.
        file_path:
            Destination file path.  Parent directories are created
            automatically if they do not already exist.
        info:
            Eight-element list for the TOA5 info line (line 1).  If None,
            ``DEFAULT_TOA5_INFO`` is used:
                ['TOA5', 'NoStation', 'CR1000', '9999', 'cr1000.std.99.99',
                 'CPU:noprogram.cr1', '9999', 'default_table']
        timestamp_formatter:
            Optional callable applied to each index value to produce the
            TIMESTAMP string.  When None (default), sub-second precision is
            detected automatically: if any index value has a non-zero
            microsecond component, ``format_fast_timestamp`` is used;
            otherwise ``strftime(DATA_TIME_FORMAT)`` (second precision).
            Pass an explicit callable only to override this behaviour.

    Returns:
        None.

    Raises:
        TypeError: if any DataFrame does not have a DatetimeIndex.
        ValueError: if a list of DataFrames is supplied and their inferred
            time intervals are not all equal; if ``headers`` does not contain
            exactly three lists; or if the variable-name list (``headers[0]``)
            length does not equal the number of DataFrame columns.
    """

    # --- normalise inputs ----------------------------------------------------

    if info is None:
        info = DEFAULT_TOA5_INFO

    if isinstance(data, list):
        for i, frame in enumerate(data):
            data_conditioning.check_datetime_index(frame)
        intervals = [data_conditioning.infer_interval(f) for f in data]
        if len(set(intervals)) > 1:
            raise ValueError(
                f'DataFrames have inconsistent intervals (minutes): {intervals}.'
            )
        data = (
            pd.concat(data)
            .sort_index(kind='stable')
            .loc[lambda df: ~df.index.duplicated(keep='first')]
        )
    else:
        data_conditioning.check_datetime_index(data)

    if len(headers) != 3:
        raise ValueError(
            f'`headers` must be a list of exactly 3 lists '
            f'(variables, units, sampling); got {len(headers)}.'
        )

    # --- validate header / column consistency --------------------------------

    n_data_cols = len(data.columns)
    n_header_cols = len(headers[0])
    if n_header_cols != n_data_cols:
        raise ValueError(
            f'Variable-name header has {n_header_cols} entries but data has '
            f'{n_data_cols} columns (TIMESTAMP excluded from both).'
        )

    # --- prepare data for output ---------------------------------------------

    data = data.copy()

    # Insert TIMESTAMP from the DatetimeIndex as the first column.
    # Auto-detect sub-second precision when no explicit formatter is given.
    if timestamp_formatter is not None:
        timestamps = data.index.map(timestamp_formatter)
    elif data.index.microsecond.any():
        timestamps = data.index.map(format_fast_timestamp)
    else:
        timestamps = data.index.strftime(DATA_TIME_FORMAT)
    data.insert(0, _TIMESTAMP_COL, timestamps)

    # Float columns whose non-NaN values are all whole numbers should be
    # written as integers (no decimal point), matching the original logger
    # output.  This handles RECORD, diagnostic flags, sample counts, totals,
    # etc. without needing to enumerate them explicitly.
    data = _coerce_integer_floats(data)

    # Prepend TIMESTAMP to all three header lists using fixed TOA5 format values.
    full_headers = [
        [_TIMESTAMP_COL] + list(headers[0]),
        [_TIMESTAMP_UNITS] + list(headers[1]),
        [_TIMESTAMP_SAMPLING] + list(headers[2]),
    ]

    # --- delegate write to infrastructure ------------------------------------

    file_io.write_toa5_csv(
        file_path=pathlib.Path(file_path),
        headers=[info] + full_headers,
        data=data,
    )

# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################

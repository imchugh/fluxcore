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
  - the DataFrame columns match the variable-name header list.
  - any timestamp column is already formatted as a string (the function writes
    data as-is and does not reformat datetime values).
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

import pathlib

import pandas as pd

from infrastructure import data_conditioning, file_io

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CONSTANTS ###
###############################################################################

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
def write_toa5(
    data: pd.DataFrame | list[pd.DataFrame],
    headers: list[list],
    file_path: str | pathlib.Path,
    info: list[str] | None = None,
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
            lines 2–4 of the TOA5 file.  The number of elements in the
            variable-name list must equal the number of columns in ``data``.
        file_path:
            Destination file path.  Parent directories are created
            automatically if they do not already exist.
        info:
            Eight-element list for the TOA5 info line (line 1).  If None,
            ``DEFAULT_TOA5_INFO`` is used:
                ['TOA5', 'NoStation', 'CR1000', '9999', 'cr1000.std.99.99',
                 'CPU:noprogram.cr1', '9999', 'default_table']

    Returns:
        None.

    Raises:
        TypeError: if any DataFrame does not have a DatetimeIndex.
        ValueError: if a list of DataFrames is supplied and their inferred
            time intervals are not all equal; if ``headers`` does not contain
            exactly three lists; or if the length of the variable-name list
            (``headers[0]``) does not match the number of columns in the
            (possibly concatenated) DataFrame.
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
            .sort_index()
            .loc[lambda df: ~df.index.duplicated(keep='first')]
        )
    else:
        data_conditioning.check_datetime_index(data)

    if len(headers) != 3:
        raise ValueError(
            f'`headers` must be a list of exactly 3 lists '
            f'(variables, units, sampling); got {len(headers)}.'
        )

    n_data_cols = len(data.columns)
    n_header_cols = len(headers[0])
    if n_header_cols != n_data_cols:
        raise ValueError(
            f'Variable-name header has {n_header_cols} entries but data has '
            f'{n_data_cols} columns.'
        )

    # --- delegate write to infrastructure ------------------------------------

    file_io.write_toa5_csv(
        file_path=pathlib.Path(file_path),
        headers=[info] + headers,
        data=data,
    )

# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EddyPro file writer.

Provides a single public function, `write_eddypro`, for writing data and
headers to a LI-COR EddyPro-format file (SmartFlux daily/master summary
files). The function accepts a DataFrame whose leading four columns are
the fixed EddyPro row-marker/provenance columns (DATAH, filename, date,
time) plus an explicit list of the remaining (numeric variable) header
rows.

Raw I/O mechanics (CSV quoting, atomic write) are handled by
``infrastructure.file_io.write_eddypro_csv``.

Unlike ``toa5_writer.write_toa5``, no column needs to be reconstructed
from the DataFrame's index: EddyPro's `date`/`time` columns survive
concatenation as ordinary data columns (only TOA5's single `TIMESTAMP`
column is collapsed into the DatetimeIndex on load), so they're written
straight through.

The caller is responsible for ensuring that:
  - the DataFrame has a DatetimeIndex.
  - the leading four columns are, in order, DATAH, filename, date, time.
  - the two header lists do NOT include those four columns — the writer
    prepends their fixed EddyPro header values internally.
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

_FIXED_COLS = ['DATAH', 'filename', 'date', 'time']
_FIXED_VARIABLE_HEADER = ['DATAH', 'filename', 'date', 'time']
_FIXED_UNITS_HEADER = ['DATAU', '', '[yyyy-mm-dd]', '[HH:MM]']

###############################################################################
### END CONSTANTS ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------
def write_eddypro(
    data: pd.DataFrame,
    headers: list[list],
    file_path: str | pathlib.Path,
) -> None:
    """Write data and headers to an EddyPro-format file.

    The EddyPro format has two header lines followed by data rows:
        1. Variables – column names (``headers[0]``), prefixed with the
           fixed DATAH/filename/date/time labels.
        2. Units     – engineering units (``headers[1]``), prefixed with
           the fixed DATAU/''/[yyyy-mm-dd]/[HH:MM] labels.

    Args:
        data:
            DataFrame to write, with a DatetimeIndex. Its leading four
            columns must be, in order, DATAH, filename, date, time,
            followed by the numeric variable columns.
        headers:
            List of two lists — [variables, units] — matching the numeric
            variable columns only (excluding the four fixed columns).
            Each list must have exactly ``len(data.columns) - 4`` entries.
        file_path:
            Destination file path. Parent directories are created
            automatically if they do not already exist.

    Returns:
        None.

    Raises:
        TypeError: if the DataFrame does not have a DatetimeIndex.
        ValueError: if ``headers`` does not contain exactly two lists; if
            the DataFrame's leading four columns are not
            DATAH/filename/date/time; or if the variable-name list
            (``headers[0]``) length does not match the number of remaining
            columns.
    """

    # --- validate inputs -------------------------------------------------

    data_conditioning.check_datetime_index(data)

    if len(headers) != 2:
        raise ValueError(
            f'`headers` must be a list of exactly 2 lists '
            f'(variables, units); got {len(headers)}.'
        )

    if list(data.columns[:4]) != _FIXED_COLS:
        raise ValueError(
            f'Expected leading columns {_FIXED_COLS}, got '
            f'{list(data.columns[:4])}.'
        )

    n_data_cols = len(data.columns) - 4
    n_header_cols = len(headers[0])
    if n_header_cols != n_data_cols:
        raise ValueError(
            f'Variable-name header has {n_header_cols} entries but data has '
            f'{n_data_cols} variable columns (DATAH/filename/date/time '
            f'excluded from both).'
        )

    # --- prepend fixed header cells ---------------------------------------

    full_headers = [
        _FIXED_VARIABLE_HEADER + list(headers[0]),
        _FIXED_UNITS_HEADER + list(headers[1]),
    ]

    # --- delegate write to infrastructure ---------------------------------

    file_io.write_eddypro_csv(
        file_path=pathlib.Path(file_path),
        headers=full_headers,
        data=data,
    )

# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 08:43:15 2026

@author: imchugh
"""

import pandas as pd


# -----------------------------------------------------------------------------
def check_datetime_index(df: pd.DataFrame) -> None:
    """Raise TypeError if ``df`` does not have a DatetimeIndex.

    Args:
        df: DataFrame to check.

    Raises:
        TypeError: if the index is not a DatetimeIndex.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('DataFrame must have a DatetimeIndex.')
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def infer_interval(df: pd.DataFrame) -> int:
    """Infer the time step of a DataFrame in minutes.

    Uses the median of successive index differences, which is robust to
    occasional gaps or duplicate timestamps.

    Args:
        df: DataFrame with a DatetimeIndex.

    Returns:
        Inferred interval in whole minutes.

    Raises:
        TypeError: if the index is not a DatetimeIndex.
    """

    check_datetime_index(df)
    return int(
        df.index.to_series()
        .diff()
        .median()
        .total_seconds() / 60
    )
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def condition_dataframe(
    df: pd.DataFrame,
    interval_in: int | None = None,
    interval_out: int | None = None,
    ) -> pd.DataFrame:
    """
    Condition flux dataframe for merge compatibility.

    Handles:
    - duplicate timestamps
    - padded time index (NaN rows inserted for any missing steps)
    - optional resampling to a different output interval

    Args:
        df: DataFrame with a DatetimeIndex.
        interval_in: input interval in minutes.  Inferred from the data if
            None.
        interval_out: output interval in minutes.  Defaults to
            ``interval_in`` if None.

    Returns:
        Conditioned DataFrame with a complete, uniform time index.

    Raises:
        TypeError: if the index is not a DatetimeIndex.
    """

    check_datetime_index(df)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    if interval_in is None:
        interval_in = infer_interval(df)

    interval = interval_out or interval_in
    return df.resample(f'{interval}min').asfreq()
# -----------------------------------------------------------------------------

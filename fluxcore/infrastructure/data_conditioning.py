#!/usr/bin/env python3
"""Dataframe conditioning utilities: index validation and gap-filling."""

import pandas as pd


def check_datetime_index(df: pd.DataFrame) -> None:
    """Raise TypeError if ``df`` does not have a DatetimeIndex.

    Args:
        df: DataFrame to check.

    Raises:
        TypeError: if the index is not a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")


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
        ValueError: if there are fewer than two distinct timestamps, so no
            interval can be inferred.
    """
    check_datetime_index(df)
    median = df.index.to_series().diff().median()
    if pd.isna(median):
        raise ValueError(
            "Cannot infer a time step from fewer than two distinct timestamps"
        )
    return int(median.total_seconds() / 60)


def condition_dataframe(
    df: pd.DataFrame,
    interval_in: int | None = None,
    interval_out: int | None = None,
) -> pd.DataFrame:
    """Condition flux dataframe for merge compatibility.

    Handles:
    - duplicate timestamps
    - padded time index (NaN rows inserted for any missing steps)
    - optional resampling to a different output interval

    Args:
        df: DataFrame with a DatetimeIndex.
        interval_in: input interval in minutes. Only used if
            ``interval_out`` is None; inferred from the data if also None.
        interval_out: output interval in minutes. Defaults to
            ``interval_in`` if None.

    Returns:
        Conditioned DataFrame with a complete, uniform time index.

    Raises:
        TypeError: if the index is not a DatetimeIndex.
        ValueError: if neither interval is given and the data has fewer than
            two distinct timestamps.
    """
    check_datetime_index(df)

    df = df.sort_index()
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]

    # Infer only when nothing was supplied: inference needs at least two
    # timestamps, and an incremental tail is routinely a single record (one
    # new 30-min row since the checkpoint), which would otherwise raise and
    # push the caller into a full rebuild.
    interval = interval_out or interval_in or infer_interval(df)
    return df.resample(f"{interval}min").asfreq()

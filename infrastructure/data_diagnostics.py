#!/usr/bin/env python3
"""Gap and duplicate-timestamp analysis for time-indexed dataframes."""

import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    """Assert that a DataFrame is fit for analysis.

    Raises:
        TypeError: If the index is not a DatetimeIndex.
        ValueError: If the DataFrame contains no rows.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")
    if df.empty:
        raise ValueError("DataFrame must contain data.")


def analyse_data_gaps(
    df: pd.DataFrame,
    interval_minutes: int,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict:
    """Analyse timestamp gaps in a dataframe.

    Args:
        df: Dataframe with DatetimeIndex.
        interval_minutes: Expected sampling interval in minutes.
        start: Start of the reference window. Defaults to ``df.index.min()``.
            Supply this (together with ``end``) when the window of interest
            extends beyond the available data — e.g. when the trailing
            portion of the window has no records at all and ``df`` would
            otherwise understate the missing percentage.
        end: End of the reference window. Defaults to ``df.index.max()``.

    Returns:
        Dict with keys ``n_missing`` (int), ``pct_missing`` (float),
        ``gap_distribution`` (pd.Series), ``gap_table`` (pd.DataFrame), and
        ``missing_timestamps`` (pd.DatetimeIndex).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")

    full_index = pd.date_range(
        start=start if start is not None else df.index.min() if not df.empty else None,
        end=end if end is not None else df.index.max() if not df.empty else None,
        freq=f"{interval_minutes}min",
    )

    if df.empty:
        if not full_index.size:
            raise ValueError("DataFrame is empty and no start/end window provided.")
        return {
            "n_missing": len(full_index),
            "pct_missing": 100.0,
            "gap_distribution": pd.Series(dtype=int),
            "gap_table": pd.DataFrame(
                columns=["gap_start", "gap_end", "missing_records"]
            ),
            "missing_timestamps": full_index,
        }

    # ensure clean index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    missing_timestamps = full_index.difference(df.index)

    expected = pd.Timedelta(minutes=interval_minutes)

    # timestamp differences
    diffs = df.index.to_series().diff()

    # gaps larger than expected interval
    gaps = diffs[diffs > expected]

    # number of missing records per gap
    gap_sizes = (gaps / expected).astype(int) - 1

    # gap bounds
    gap_starts = gaps.index - gaps
    gap_ends = gaps.index

    gap_table = pd.DataFrame(
        {
            "gap_start": gap_starts,
            "gap_end": gap_ends,
            "missing_records": gap_sizes.values,
        }
    ).reset_index(drop=True)

    # gap size distribution
    gap_distribution = gap_sizes.value_counts().sort_index()

    n_missing = len(full_index) - len(df)
    pct_missing = round(n_missing / len(full_index) * 100, 2)

    return {
        "n_missing": n_missing,
        "pct_missing": pct_missing,
        "gap_distribution": gap_distribution,
        "gap_table": gap_table,
        "missing_timestamps": missing_timestamps,
    }


def analyse_data_duplicates(df: pd.DataFrame) -> dict:
    """Analyse duplicate timestamps in a dataframe.

    Checks the index only, not row content: a duplicate is any timestamp
    that appears more than once, regardless of whether the associated data
    differs between occurrences.

    Args:
        df: Dataframe with DatetimeIndex.

    Returns:
        Dict with keys ``n_duplicates`` (int), ``pct_duplicates`` (float),
        ``duplicate_timestamps`` (pd.DatetimeIndex), and
        ``duplicate_counts`` (pd.Series).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")

    if df.empty:
        return {
            "n_duplicates": 0,
            "pct_duplicates": 0.0,
            "duplicate_timestamps": pd.DatetimeIndex([]),
            "duplicate_counts": pd.Series(dtype=int),
        }

    is_dupe = df.index.duplicated(keep="first")
    n_duplicates = int(is_dupe.sum())
    pct_duplicates = round(n_duplicates / len(df) * 100, 2)

    counts = df.index.value_counts()
    duplicate_counts = counts[counts > 1].sort_index()

    return {
        "n_duplicates": n_duplicates,
        "pct_duplicates": pct_duplicates,
        "duplicate_timestamps": duplicate_counts.index,
        "duplicate_counts": duplicate_counts,
    }

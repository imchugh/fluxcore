#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:13:20 2026

@author: imchugh
"""

import pandas as pd

def analyse_data_gaps(df: pd.DataFrame, interval_minutes: int) -> dict:
    """
    Analyse timestamp gaps in a dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe with DatetimeIndex
    interval_minutes : int
        Expected sampling interval

    Returns
    -------
    dict
        {
            'n_missing': int,
            'pct_missing': float,
            'gap_distribution': pd.Series,
            'gap_table': pd.DataFrame
        }
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have DatetimeIndex")

    if df.empty:
        raise ValueError('DataFrame must contain data!')

    # ensure clean index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

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

    gap_table = (
        pd.DataFrame(
            {
                "gap_start": gap_starts,
                "gap_end": gap_ends,
                "missing_records": gap_sizes.values
                }
            )
        .reset_index(drop=True)
        )

    # gap size distribution
    gap_distribution = gap_sizes.value_counts().sort_index()

    # total missing records
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{interval_minutes}min"
    )

    n_missing = len(full_index) - len(df)

    pct_missing = round(n_missing / len(full_index) * 100, 2)

    return {
        "n_missing": n_missing,
        "pct_missing": pct_missing,
        "gap_distribution": gap_distribution,
        "gap_table": gap_table
        }
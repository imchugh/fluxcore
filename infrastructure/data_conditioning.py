#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 08:43:15 2026

@author: imchugh
"""

import pandas as pd

def condition_dataframe(
    df: pd.DataFrame,
    interval_in: int | None = None,
    interval_out: int | None = None,
    ) -> pd.DataFrame:
    """
    Condition flux dataframe for merge compatibility.

    Handles:
    - duplicate timestamps
    - padded time index
    - optional resampling
    """

    # Check we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have DatetimeIndex")

    # Check data is sorted
    df = df.sort_index()

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # Infer interval if needed
    if interval_in is None:
        interval_in = int(
            df.index.to_series()
            .diff()
            .median()
            .total_seconds() / 60
            )

    # choose output interval
    interval = interval_out or interval_in

    # resample builds full timestamp grid automatically
    df = df.resample(f"{interval}min").asfreq()

    return df

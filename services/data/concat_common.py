#!/usr/bin/env python3
"""Shared validation helpers for the TOA5 and EddyPro file concatenators.

Public API
----------
validate_headers(master_headers, slave_path, slave_headers, labels) -> None
validate_interval(master_df, slave_df, slave_path) -> None
"""

import pathlib

import pandas as pd

from infrastructure.data_conditioning import infer_interval


def validate_headers(
    master_headers: list[list],
    slave_path: pathlib.Path,
    slave_headers: list[list],
    labels: tuple[str, ...],
) -> None:
    """Raise ValueError if slave headers are incompatible with master."""
    for label, master_row, slave_row in zip(labels, master_headers, slave_headers):
        if master_row != slave_row:
            raise ValueError(
                f"{slave_path.name}: {label} header does not match master.\n"
                f"  master : {master_row}\n"
                f"  slave  : {slave_row}"
            )


def validate_interval(
    master_df: pd.DataFrame,
    slave_df: pd.DataFrame,
    slave_path: pathlib.Path,
) -> None:
    """Raise ValueError if slave time step differs from master."""
    master_interval = infer_interval(master_df)
    slave_interval = infer_interval(slave_df)
    if master_interval != slave_interval:
        raise ValueError(
            f"{slave_path.name}: time step ({slave_interval} min) does not "
            f"match master ({master_interval} min)."
        )

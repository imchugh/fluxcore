#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 13:35:13 2026

@author: imchugh
"""

import csv
import json
import logging
import pandas as pd
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

def get_most_recent_file(
    *,
    root: Path,
    pattern: str = "*",
    recursive: bool = False,
    ) -> Path | None:
    """
    Return most recent file in directory matching pattern.

    Infrastructure-level utility.
    Returns None if no matching files.
    """

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    iterator = root.rglob(pattern) if recursive else root.glob(pattern)

    files = [p for p in iterator if p.is_file()]

    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def get_backup_files(file_path):
    
    file_path = Path(file_path)
    return sorted(file_path.parent.glob(f'{file_path.stem}*backup'))
# -----------------------------------------------------------------------------    

# -----------------------------------------------------------------------------    

def read_yml(file_path: Path) -> dict:
    
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def read_json(file_path: Path):
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def read_text(file_path: Path, encoding="utf-8") -> str:
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()
# -----------------------------------------------------------------------------    

#------------------------------------------------------------------------------
def read_lines(
        file_path: str | Path, begin: int=0, end: int=4, sep: str=','
        ) -> list:
    """Get a list of the header strings.

    Args:
        file: absolute path of file to parse.
        begin: line number of first header line.
        end: line number of last header line.
        sep: text separation character.

    Returns:
        List of sublists, each sublist containing the text elements of a header
            line.

    """

    line_list = []
    with open(file_path, 'r') as f:
        for i in range(end + 1):
            line = f.readline()
            if not i < begin:
                line_list.append(line)
    return [line for line in csv.reader(line_list, delimiter=sep)]
#------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def read_csv_data(file_path: str, file_format: dict, **kwargs) -> pd.DataFrame:
    """
    Reads a CSV/TSV file according to the provided file_format dictionary.

    Parameters
    ----------
    path : str
        Path to the data file.
    file_format : dict
        Dictionary containing parsing information. Expected keys:
        - info_line: number of initial info lines to skip
        - header_lines: dict with keys 'variable', 'units', 'sampling'
        - separator: str, column separator
        - non_numeric_cols: list of columns to treat as strings
        - time_variables: dict mapping time columns to indices
        - na_values: value(s) to treat as NaN
        - quoting: quoting level (0,1,2)
    **kwargs
        Any additional keyword arguments are passed directly to pd.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    
    # Extract skiprows and header line
    header_lines = file_format.get("header_lines", {})
    variable_row = header_lines.get("variable", 0)
    skiprows = set(header_lines.values()) - {variable_row}

    # Handle quoting
    quoting = file_format.get("quoting", 0)

    # Columns to treat as strings
    dtype = {col: str for col in file_format.get("non_numeric_cols", [])}

    # Separator
    sep = file_format.get("separator", ",")

    # NA values
    na_values = file_format.get("na_values", None)

    # Read the file
    return pd.read_csv(
        file_path,
        sep=sep,
        skiprows=skiprows,
        header=0,
        dtype=dtype,
        na_values=na_values,
        quoting=quoting,
        **kwargs  # pass any extra kwargs
        )
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def read_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def write_json_file(
    *,
    file_path: Path,
    data: list[dict],
    run_id: str,
) -> None:

    logger.info(
        "json_write_start",
        extra={
            "run_id": run_id,
            "path": str(file_path),
        },
    )

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(
        "json_write_complete",
        extra={
            "run_id": run_id,
            "path": str(file_path),
            "records": len(data),
        },
    )
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def write_yml_file(
    file_path: str | Path,
    data: dict,
    *,
    sort_keys: bool = False,
    default_flow_style: bool = False
    ) -> None:
    """
    Write a dictionary to a YAML file.

    Args:
        data: dictionary to write.
        file: output file path.
        sort_keys (optional): sort dictionary keys alphabetically.
        default_flow_style (optional): write in inline YAML format.

    Returns:
        None
    """

    file = Path(file_path)

    with file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=sort_keys,
            default_flow_style=default_flow_style,
            allow_unicode=True
        )
# -----------------------------------------------------------------------------
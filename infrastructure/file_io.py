#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 13:35:13 2026

@author: imchugh
"""

import csv
import json
import logging
import os
import pandas as pd
import yaml
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class UniqueKeyLoader(yaml.SafeLoader):
    pass

def construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key detected in YAML: {key}")
        value = loader.construct_object(value_node, deep=deep)
        mapping[key] = value
    return mapping

UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_mapping
)


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
def get_backup_files(file_path, abs_path=True):
    
    file_path = Path(file_path)
    paths = sorted(file_path.parent.glob(f'{file_path.stem}*backup'))
    if abs_path:
        return paths
    return [path.name for path in paths]
# -----------------------------------------------------------------------------    

# -----------------------------------------------------------------------------    
def list_available_files(
        dir_path: Path | str, pattern: str | list[str]
        ) -> list[Path]:
    
    if isinstance(pattern, str):
        pattern_list = [pattern]
    elif isinstance(pattern, list):
        pattern_list = pattern
    files = set()
    for this_pattern in pattern_list:
        files.update(Path(dir_path).glob(this_pattern))
    return sorted(files)
# -----------------------------------------------------------------------------    

# -----------------------------------------------------------------------------    

def read_yml(file_path: Path, enforce_unique_keys=False) -> dict:
    
    with open(file_path, "r", encoding="utf-8") as f:
        
        if enforce_unique_keys:
            return yaml.load(f, Loader=UniqueKeyLoader)
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
def write_json(
        *,
        file_path: Path,
        data: Any,
        indent: int = 2,
        sort_keys: bool = False,
        atomic: bool = True,
        ) -> None:
    """
    Write JSON data to disk.

    Args:
        file_path:
            Destination path.

        data:
            JSON-serializable object.

        indent:
            JSON indentation level.

        sort_keys:
            Whether to sort dictionary keys.

        atomic:
            If True, write via temporary file replacement to avoid
            partially-written/corrupted files.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if atomic:

        tmp_path = file_path.with_suffix(
            f"{file_path.suffix}.tmp"
            )

        with open(tmp_path, "w", encoding="utf-8") as f:

            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=sort_keys,
                )

            f.flush()
            os.fsync(f.fileno())

        tmp_path.replace(file_path)

    else:

        with open(file_path, "w", encoding="utf-8") as f:

            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=sort_keys,
                )
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def write_toa5_csv(
        *,
        file_path: Path,
        headers: list[list],
        data: pd.DataFrame,
        atomic: bool = True,
        ) -> None:
    """
    Write headers and data to a TOA5-format CSV file.

    Handles raw I/O mechanics only: TOA5 CSV quoting conventions, atomic
    write via a temporary file, and fsync.  All domain concerns (building
    the info line, concatenating multiple DataFrames, validating column
    counts) are the caller's responsibility.

    Args:
        file_path:
            Destination path.  Parent directories are created automatically.
        headers:
            Ordered list of rows to write before the data.  For a standard
            TOA5 file this is four lists: info, variable-names, units, and
            sampling-type.
        data:
            DataFrame to write.  Written without its index; the caller must
            include any timestamp column explicitly as a data column.
        atomic:
            If True (default), write via a temporary file that is atomically
            renamed to ``file_path`` on success.  The temporary file is
            removed on failure.  Set False only when atomicity is guaranteed
            by the caller.
    """

    _SEP = ','
    _NA  = 'NAN'

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(f) -> None:
        writer = csv.writer(f, delimiter=_SEP, quoting=csv.QUOTE_ALL)
        for row in headers:
            writer.writerow(row)
        data.to_csv(
            f,
            header=False,
            index=False,
            na_rep=_NA,
            sep=_SEP,
            quoting=csv.QUOTE_NONNUMERIC,
        )

    if atomic:
        tmp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        try:
            with open(tmp_path, 'w', newline='\n') as f:
                _write(f)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(file_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        with open(file_path, 'w', newline='\n') as f:
            _write(f)

    logger.info('Wrote TOA5 file: %s', file_path.name)
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
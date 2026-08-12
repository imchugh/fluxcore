#!/usr/bin/env python3
"""Load raw TOA5/EddyPro files into time-indexed DataFrames."""

import csv
import logging
import pathlib

import pandas as pd

from domain.constants import DATA_TIME_FORMAT, TIME_INDEX_NAME
from infrastructure import file_io

logger = logging.getLogger(__name__)

# TOA5 and EddyPro are fixed industry-standard formats; these constants
# capture only the fields consumed by this module (header line positions,
# separator, non-numeric columns, NA sentinel, and CSV quoting mode).
_FILE_FORMATS = {
    "TOA5": {
        "header_lines": {"info": 0, "variable": 1, "units": 2, "sampling": 3},
        "separator": ",",
        "non_numeric_cols": ["TIMESTAMP"],
        "na_values": "NAN",
        "quoting": csv.QUOTE_NONNUMERIC,
    },
    "EddyPro": {
        "header_lines": {"variable": 0, "units": 1},
        "separator": "\t",
        "non_numeric_cols": ["DATAH", "filename", "date", "time"],
        "na_values": "NaN",
        "quoting": csv.QUOTE_MINIMAL,
    },
}

# Maps the site config's logger system_type to the file format it produces.
_SYSTEM_TYPE_FORMAT_MAP = {"CSI": "TOA5", "LICOR": "EddyPro"}


def load_raw_data(
    file_path: pathlib.Path,
    file_format: str,
    drop_non_numeric: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a raw TOA5/EddyPro file into a time-indexed DataFrame.

    **read_csv_kwargs are forwarded to pd.read_csv (via file_io.read_csv_data)
    — e.g. usecols=[...] to restrict which columns are parsed out of a wide
    file. If usecols is given, it must include the file format's timestamp
    column(s) ('TIMESTAMP' for TOA5; 'date' and 'time' for EddyPro), since
    the date formatter below depends on them.
    """
    df = file_io.read_csv_data(
        file_path=file_path,
        file_format=_FILE_FORMATS[file_format],
        on_bad_lines="skip",
        **read_csv_kwargs,
    )
    df = DATE_FORMATTERS[file_format](df)

    if drop_non_numeric:
        df = _drop_non_numeric(df=df, file_format=file_format)

    return df


def _TOA5_date_formatter(df):

    dttm = pd.to_datetime(df["TIMESTAMP"], format=DATA_TIME_FORMAT, errors="coerce")
    return df.drop(columns=["TIMESTAMP"]).set_index(
        pd.Index(data=dttm, name=TIME_INDEX_NAME)
    )


def _EddyPro_date_formatter(df):

    dttm = pd.to_datetime(
        df["date"] + " " + df["time"], format=DATA_TIME_FORMAT, errors="coerce"
    )
    return df.set_index(keys=pd.Index(data=dttm, name=TIME_INDEX_NAME))


DATE_FORMATTERS = {
    "TOA5": _TOA5_date_formatter,
    "EddyPro": _EddyPro_date_formatter,
}


def _drop_non_numeric(df, file_format):

    cols_to_drop = _FILE_FORMATS[file_format]["non_numeric_cols"]
    return df.drop(columns=cols_to_drop, errors="ignore")


def _toa5_line_timestamp(line: str) -> pd.Timestamp | None:
    """Parse a raw TOA5 data line's quoted TIMESTAMP field (column 0).

    None on any parse failure (malformed line, wrong field count, garbage
    text) — callers treat None as "can't compare this line".
    """
    try:
        raw = next(csv.reader([line], delimiter=","))[0]
        return pd.to_datetime(raw, format=DATA_TIME_FORMAT)
    except (StopIteration, IndexError, ValueError):
        return None


def _eddypro_line_timestamp(line: str) -> pd.Timestamp | None:
    """Parse a raw EddyPro data line's date+time fields (tab-separated, fields 2, 3)."""
    fields = line.split("\t")
    try:
        return pd.to_datetime(f"{fields[2]} {fields[3]}", format=DATA_TIME_FORMAT)
    except (IndexError, ValueError):
        return None


LINE_TIMESTAMP_PARSERS = {
    "TOA5": _toa5_line_timestamp,
    "EddyPro": _eddypro_line_timestamp,
}

# Bytes read from the end of a master file to recover its last record's
# timestamp without loading the whole file. TOA5 flux-slow tables can run
# to 100+ float columns (several KB/line); EddyPro summary lines are much
# narrower. Generous on both sides — the read itself is cheap either way.
_TAIL_PEEK_BYTES = {"TOA5": 65536, "EddyPro": 16384}


def peek_last_timestamp(file_path, file_format: str) -> pd.Timestamp | None:
    """Read the last data line's timestamp without parsing the whole file.

    Shared by load_raw_data_since's tail-peek short-circuit and
    eddypro_concatenator.select_new_slaves' new-candidate pre-filter — both
    need "what's the newest record in this file" without a full parse.

    Returns None if no line in the peek window parses (e.g. a header-only
    file, or — defensively — a truncated in-progress write). Callers
    should treat None as "nothing to compare against": load_raw_data_since
    treats it the same as "last record predates start_date" (skip);
    select_new_slaves treats it as "nothing to filter against" (keep
    every candidate).
    """
    path = pathlib.Path(file_path)
    window = _TAIL_PEEK_BYTES[file_format]
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - window))
        tail = f.read()

    lines = [line for line in tail.decode(errors="ignore").splitlines() if line.strip()]
    parser = LINE_TIMESTAMP_PARSERS[file_format]
    for line in reversed(lines):
        ts = parser(line)
        if ts is not None:
            return ts
    return None


def load_raw_data_since(
    file_path: pathlib.Path,
    file_format: str,
    start_date: pd.Timestamp,
    drop_non_numeric: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a raw TOA5/EddyPro file, skipping rows before start_date where possible.

    Performance-only variant of load_raw_data: a cheap tail-peek plus a
    byte-offset binary search (infrastructure.file_io.find_seek_offset)
    avoid parsing years of history that a start_date filter would just
    discard. Always produces the same result as calling load_raw_data and
    trimming to `df.index >= start_date` — on any error in the fast path,
    or if `read_csv_kwargs` are given (the fast path can't resolve column
    names for a headerless seeked read under e.g. usecols), this falls
    back to exactly that: a full load_raw_data call, trimmed.

    Args:
        file_path: path to the data file.
        file_format: 'TOA5' or 'EddyPro'.
        start_date: only rows at/after this timestamp are returned.
        drop_non_numeric: as load_raw_data.
        **read_csv_kwargs: forwarded to load_raw_data on the fallback path
            only; any kwarg given always takes the fallback path.

    Returns:
        Time-indexed DataFrame containing only rows with index >= start_date.
    """
    if not read_csv_kwargs:
        try:
            return _load_raw_data_since_fast(
                file_path=file_path,
                file_format=file_format,
                start_date=start_date,
                drop_non_numeric=drop_non_numeric,
            )
        except Exception:
            logger.debug(
                "load_raw_data_since: fast path failed for %s, falling back to "
                "full read",
                file_path,
                exc_info=True,
            )

    df = load_raw_data(
        file_path=file_path,
        file_format=file_format,
        drop_non_numeric=drop_non_numeric,
        **read_csv_kwargs,
    )
    return df[df.index >= start_date]


def _load_raw_data_since_fast(
    file_path: pathlib.Path,
    file_format: str,
    start_date: pd.Timestamp,
    drop_non_numeric: bool,
) -> pd.DataFrame:
    """Fast path for load_raw_data_since: tail-peek + binary search + seeked read."""
    fmt = _FILE_FORMATS[file_format]
    parser = LINE_TIMESTAMP_PARSERS[file_format]

    last_ts = peek_last_timestamp(file_path, file_format)
    if last_ts is None or last_ts < start_date:
        return pd.DataFrame()

    n_header_lines = max(fmt["header_lines"].values()) + 1
    low = file_io.header_end_offset(file_path, n_header_lines)
    high = pathlib.Path(file_path).stat().st_size

    def line_qualifies(line: str) -> bool | None:
        ts = parser(line)
        return None if ts is None else ts < start_date

    seek_offset = file_io.find_seek_offset(
        file_path=file_path, low=low, high=high, line_qualifies=line_qualifies
    )

    variable_row = fmt["header_lines"]["variable"]
    names = file_io.read_lines(
        file_path=file_path,
        begin=variable_row,
        end=variable_row,
        sep=fmt["separator"],
    )[0]

    df = file_io.read_csv_tail(
        file_path=file_path,
        file_format=fmt,
        names=names,
        seek_offset=seek_offset,
        on_bad_lines="skip",
    )
    df = DATE_FORMATTERS[file_format](df)
    df = df[df.index >= start_date]

    if drop_non_numeric:
        df = _drop_non_numeric(df=df, file_format=file_format)

    return df


def get_data_adapter(system_type: str, start_date: pd.Timestamp | None = None):
    """Return a `load(file_path)` closure bound to `system_type`'s file format.

    If start_date is given, the closure uses load_raw_data_since (skips
    parsing rows before start_date where possible) instead of the default
    full-file load_raw_data.
    """
    file_format = _SYSTEM_TYPE_FORMAT_MAP[system_type]

    def load(file_path):
        if start_date is not None:
            return load_raw_data_since(
                file_path=file_path, file_format=file_format, start_date=start_date
            )
        return load_raw_data(file_path=file_path, file_format=file_format)

    return load


def load_raw_header(file_path, file_format: str) -> dict:
    """Read a raw file's header rows, keyed by line type ('variable', 'units', etc).

    Non-numeric columns (per `_FILE_FORMATS`) are dropped from every header
    row, matching the columns `load_raw_data` would drop if called with
    `drop_non_numeric=True`.
    """
    fmt = _FILE_FORMATS[file_format]
    header_dict = fmt.get("header_lines")
    lines = list(header_dict.values())
    keys = list(header_dict.keys())
    result = dict(
        zip(
            keys,
            file_io.read_lines(
                file_path=file_path,
                begin=min(lines),
                end=max(lines),
                sep=fmt["separator"],
            ),
        )
    )
    non_numeric = set(fmt.get("non_numeric_cols", []))
    if "variable" in result and non_numeric:
        drop_indices = {i for i, v in enumerate(result["variable"]) if v in non_numeric}
        for key in ("variable", "units", "sampling"):
            if key in result:
                result[key] = [
                    v for i, v in enumerate(result[key]) if i not in drop_indices
                ]
    return result


def get_header_adapter(system_type: str):
    """Return a `load(file_path)` closure bound to `system_type`'s file format."""
    file_format = _SYSTEM_TYPE_FORMAT_MAP[system_type]

    def load(file_path):
        return load_raw_header(file_path=file_path, file_format=file_format)

    return load

#!/usr/bin/env python3
"""File I/O: YAML/JSON/text/CSV reading, and atomic writers for TOA5/EddyPro/NetCDF."""

import csv
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

logger = logging.getLogger(__name__)


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys instead of overwriting."""

    pass


def construct_mapping(loader, node, deep=False):
    """Build a mapping node, raising ValueError on a duplicate key."""
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key detected in YAML: {key}")
        value = loader.construct_object(value_node, deep=deep)
        mapping[key] = value
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
)


def _atomic_text_write(file_path: Path, write_fn, atomic: bool) -> None:
    """Write text content to `file_path` via `write_fn(file_handle)`.

    If `atomic`, writes to a temporary file (fsync'd) and atomically renames
    it into place; the temporary file is removed if `write_fn` raises.
    """
    if atomic:
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", newline="\n") as f:
                write_fn(f)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(file_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        with open(file_path, "w", newline="\n") as f:
            write_fn(f)


def get_most_recent_file(
    *,
    root: Path,
    pattern: str = "*",
    recursive: bool = False,
) -> Path | None:
    """Return the most recently modified file in `root` matching `pattern`.

    Returns:
        Path to the most recent match, or None if no file matches.
    """
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    iterator = root.rglob(pattern) if recursive else root.glob(pattern)

    files = [p for p in iterator if p.is_file()]

    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def get_backup_files(
    file_path: str | Path, abs_path: bool = True
) -> list[Path] | list[str]:
    """Return backup files alongside `file_path`, matching `<stem>*backup`.

    Args:
        file_path: reference file; its parent directory and stem are used
            to build the glob pattern.
        abs_path: if True (default), return full paths; if False, return
            just the filenames.

    Returns:
        Sorted list of matching backup files (as Paths or filenames).
    """
    file_path = Path(file_path)
    paths = sorted(file_path.parent.glob(f"{file_path.stem}*backup"))
    if abs_path:
        return paths
    return [path.name for path in paths]


def list_available_files(dir_path: Path | str, pattern: str | list[str]) -> list[Path]:
    """Return sorted files under `dir_path` matching one or more glob patterns.

    Args:
        dir_path: directory to search (non-recursive).
        pattern: a single glob pattern, or a list of patterns whose matches
            are unioned.

    Returns:
        Sorted list of matching paths.
    """
    if isinstance(pattern, str):
        pattern_list = [pattern]
    elif isinstance(pattern, list):
        pattern_list = pattern
    files = set()
    for this_pattern in pattern_list:
        files.update(Path(dir_path).glob(this_pattern))
    return sorted(files)


def read_yml(file_path: Path, enforce_unique_keys: bool = False) -> dict:
    """Load a YAML file.

    Args:
        file_path: path to the YAML file.
        enforce_unique_keys: if True, raise ValueError on a duplicate
            mapping key instead of silently keeping the last one.

    Returns:
        Parsed YAML content.
    """
    with open(file_path, encoding="utf-8") as f:
        if enforce_unique_keys:
            return yaml.load(f, Loader=UniqueKeyLoader)
        return yaml.safe_load(f)


def read_json(file_path: Path) -> dict | list:
    """Load a JSON file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def read_text(file_path: Path, encoding="utf-8") -> str:
    """Read a file's entire contents as text."""
    with open(file_path, encoding=encoding) as f:
        return f.read()


def read_lines(
    file_path: str | Path, begin: int = 0, end: int = 4, sep: str = ","
) -> list:
    """Get a list of the header strings.

    Args:
        file_path: absolute path of file to parse.
        begin: line number of first header line.
        end: line number of last header line.
        sep: text separation character.

    Returns:
        List of sublists, each sublist containing the text elements of a header
            line.
    """
    line_list = []
    with open(file_path) as f:
        for i in range(end + 1):
            line = f.readline()
            if not i < begin:
                line_list.append(line)
    return [line for line in csv.reader(line_list, delimiter=sep)]


def _csv_format_kwargs(file_format: dict) -> dict:
    """Assemble the pandas.read_csv kwargs shared by full-file and tail reads.

    Derives sep/dtype/na_values/quoting from a file_format dict (see
    read_csv_data). Excludes header/skiprows/names, which differ between a
    full read (starts at the file's header) and a tail read (starts
    mid-file, past a header already consumed elsewhere) — see
    read_csv_data vs read_csv_tail.
    """
    return {
        "sep": file_format.get("separator", ","),
        "dtype": {col: str for col in file_format.get("non_numeric_cols", [])},
        "na_values": file_format.get("na_values", None),
        "quoting": file_format.get("quoting", 0),
    }


def read_csv_data(file_path: str, file_format: dict, **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV file according to the provided file_format dictionary.

    Args:
        file_path: path to the data file.
        file_format: dictionary containing parsing information. Expected
            keys: ``header_lines`` (dict with keys 'variable', 'units',
            'sampling'), ``separator`` (column separator), ``non_numeric_cols``
            (columns to treat as strings), ``na_values`` (value(s) to treat
            as NaN), and ``quoting`` (quoting level, 0-2).
        **kwargs: additional keyword arguments passed directly to
            pd.read_csv.

    Returns:
        Parsed DataFrame.
    """
    # Extract skiprows and header line
    header_lines = file_format.get("header_lines", {})
    variable_row = header_lines.get("variable", 0)
    skiprows = set(header_lines.values()) - {variable_row}

    # Read the file
    return pd.read_csv(
        file_path,
        skiprows=skiprows,
        header=0,
        **_csv_format_kwargs(file_format),
        **kwargs,  # pass any extra kwargs
    )


def read_csv_tail(
    file_path: str | Path,
    file_format: dict,
    names: list[str],
    seek_offset: int,
    **kwargs,
) -> pd.DataFrame:
    """Read file_path from `seek_offset` to EOF as headerless CSV data rows.

    Sibling to read_csv_data for a file whose header has already been
    consumed elsewhere (e.g. a binary-search tail read via
    find_seek_offset) — `names` supplies the column names read_csv_data
    would otherwise take from the header row.

    Args:
        file_path: path to the data file.
        file_format: same shape as read_csv_data's file_format.
        names: column names, in file order, including the timestamp
            column(s) (e.g. TOA5's 'TIMESTAMP').
        seek_offset: byte offset to start reading from; must be a valid
            line-start offset (e.g. as returned by find_seek_offset or
            header_end_offset).
        **kwargs: additional keyword arguments passed directly to
            pd.read_csv.

    Returns:
        Parsed DataFrame with `names` as columns.
    """
    with open(file_path, "rb") as f:
        f.seek(seek_offset)
        return pd.read_csv(
            f,
            header=None,
            names=names,
            **_csv_format_kwargs(file_format),
            **kwargs,
        )


def header_end_offset(file_path: str | Path, n_header_lines: int) -> int:
    """Return the byte offset right after a file's first `n_header_lines` lines."""
    with open(file_path, "rb") as f:
        for _ in range(n_header_lines):
            f.readline()
        return f.tell()


def _next_line_after(
    f, offset: int, ceiling: int, chunk_size: int = 4096, max_scan: int = 1 << 20
) -> tuple[int | None, str]:
    """Return (byte offset, text) of the first full line starting at/after `offset`.

    Scans forward in `chunk_size` steps (handling lines wider than one
    chunk) for the next newline, then reads the line that begins
    immediately after it. `f` must be open in binary mode. Returns
    (None, "") if no such line starts before `ceiling`, or if no newline
    is found within `max_scan` bytes (a pathologically long line — treated
    as unparseable rather than scanned indefinitely).
    """
    pos = offset
    scanned = 0
    while pos < ceiling and scanned < max_scan:
        f.seek(pos)
        chunk = f.read(min(chunk_size, ceiling - pos))
        if not chunk:
            return None, ""
        nl = chunk.find(b"\n")
        if nl != -1:
            line_start = pos + nl + 1
            if line_start >= ceiling:
                return None, ""
            f.seek(line_start)
            line = f.readline()
            if not line:
                return None, ""
            return line_start, line.decode(errors="replace").rstrip("\n")
        pos += len(chunk)
        scanned += len(chunk)
    return None, ""


def find_seek_offset(
    file_path: str | Path,
    low: int,
    high: int,
    line_qualifies: Callable[[str], bool | None],
    margin: int = 8192,
) -> int:
    """Binary-search a text file for a byte offset safely at-or-before a boundary.

    Bisects between `low` and `high` (byte offsets) using `line_qualifies`
    to test probe lines. `low` is only ever advanced to an offset that a
    line has *proven* itself (`line_qualifies` returned True for that
    exact line's text), so the returned offset is always safe to read
    from without skipping wanted data — either `low` unchanged (nothing
    proven yet) or a line-start offset at or before the target boundary,
    never past it. A line `line_qualifies` can't parse/compare (returns
    None) narrows `high` only, never advances `low` past unproven ground.

    Does not attempt to land exactly on the boundary line — stops once
    `high - low <= margin`, leaving a small number of extra lines for the
    caller to trim afterward. This keeps the search robust to corrupt or
    ambiguous lines near the boundary without needing byte-exact
    precision.

    Args:
        file_path: file to search.
        low: starting (minimum) byte offset — must already be a valid
            line-start offset (e.g. the offset right after a file's header
            rows, from header_end_offset).
        high: ending (maximum) byte offset, typically the file size.
        line_qualifies: given one line of text (no trailing newline),
            return True if it belongs strictly before the target boundary,
            False if at/after it, or None if the line can't be
            parsed/compared.
        margin: stop bisecting once the search window is this small, in
            bytes.

    Returns:
        A byte offset at or before the boundary; either the input `low`
        or the start of a line `line_qualifies` proved to be before it.
    """
    with open(file_path, "rb") as f:
        while high - low > margin:
            mid = (low + high) // 2
            line_start, line_text = _next_line_after(f, mid, ceiling=high)
            if line_start is None:
                high = mid
                continue
            try:
                qualifies = line_qualifies(line_text)
            except Exception:
                qualifies = None
            if qualifies is None:
                high = line_start
            elif qualifies:
                low = line_start
            else:
                high = line_start
    return low


def read_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file with pandas' default parsing (no custom format handling)."""
    return pd.read_csv(file_path)


def write_json(
    *,
    file_path: Path,
    data: Any,
    indent: int = 2,
    sort_keys: bool = False,
    atomic: bool = True,
) -> None:
    """Write JSON data to disk.

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
        tmp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")

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


def write_toa5_csv(
    *,
    file_path: Path,
    headers: list[list],
    data: pd.DataFrame,
    atomic: bool = True,
) -> None:
    """Write headers and data to a TOA5-format CSV file.

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
    _SEP = ","
    _NA = "NAN"
    _FLOAT_FMT = "%.7g"  # matches Campbell logger's 7-significant-figure output

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(f) -> None:
        # Headers: every cell quoted (QUOTE_ALL), standard TOA5 convention.
        writer = csv.writer(f, delimiter=_SEP, quoting=csv.QUOTE_ALL)
        for row in headers:
            writer.writerow(row)

        # Data rows: per-column-type formatting written directly, bypassing
        # to_csv so that float_format and quoting can be controlled
        # independently.
        #   - string  columns → quoted         e.g. "2025-01-01 00:00:00"
        #   - integer columns → bare integer   e.g. 190
        #   - float   columns → %.7g, unquoted e.g. 0, 16.31809, 6.704941e-05
        #   - NaN in any column → "NAN" (quoted, matching logger output)
        col_fmt = []
        for col in data.columns:
            s = data[col]
            if pd.api.types.is_string_dtype(s):
                col_fmt.append([f'"{v}"' if pd.notna(v) else f'"{_NA}"' for v in s])
            elif pd.api.types.is_integer_dtype(s):
                col_fmt.append([_NA if pd.isna(v) else str(v) for v in s])
            else:  # float
                arr = s.to_numpy(dtype=float, na_value=np.nan)
                nan_mask = np.isnan(arr)
                formatted = np.empty(len(arr), dtype=object)
                formatted[nan_mask] = f'"{_NA}"'
                formatted[~nan_mask] = [
                    (_FLOAT_FMT % v).replace("e", "E") for v in arr[~nan_mask]
                ]
                col_fmt.append(formatted.tolist())

        for row in zip(*col_fmt):
            f.write(_SEP.join(row) + "\n")

    _atomic_text_write(file_path, _write, atomic)

    logger.info("Wrote TOA5 file: %s", file_path.name)


def write_eddypro_csv(
    *,
    file_path: Path,
    headers: list[list],
    data: pd.DataFrame,
    atomic: bool = True,
) -> None:
    """Write headers and data to an EddyPro-format (tab-separated) CSV file.

    Handles raw I/O mechanics only: EddyPro's tab-delimited, minimally-quoted
    convention, atomic write via a temporary file, and fsync. All domain
    concerns (building the header rows, validating column counts) are the
    caller's responsibility. Unlike ``write_toa5_csv``, there is no legacy
    logger format to replicate byte-for-bit, so data rows are written with
    plain ``DataFrame.to_csv`` rather than manual per-column formatting.

    Args:
        file_path:
            Destination path. Parent directories are created automatically.
        headers:
            Ordered list of rows to write before the data. For a standard
            EddyPro file this is two lists: variable-names and units.
        data:
            DataFrame to write. Written without its index; the caller must
            include the DATAH/filename/date/time columns explicitly as data
            columns.
        atomic:
            If True (default), write via a temporary file that is
            atomically renamed to ``file_path`` on success. The temporary
            file is removed on failure. Set False only when atomicity is
            guaranteed by the caller.
    """
    _SEP = "\t"
    _NA = "NaN"

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(f) -> None:
        writer = csv.writer(f, delimiter=_SEP, quoting=csv.QUOTE_MINIMAL)
        for row in headers:
            writer.writerow(row)

        data.to_csv(
            f,
            header=False,
            index=False,
            na_rep=_NA,
            sep=_SEP,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )

    _atomic_text_write(file_path, _write, atomic)

    logger.info("Wrote EddyPro file: %s", file_path.name)


def _serialize_attrs(attrs: dict) -> dict:
    """Convert attr values to NetCDF-compatible types.

    Enums are replaced with their .value; tuples are converted to lists;
    None entries are dropped. All other values pass through unchanged.
    """
    result = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, Enum):
            v = v.value
        elif isinstance(v, tuple):
            v = list(v)
        result[k] = v
    return result


def serialize_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Coerce attrs to NetCDF-writable types (Enum, tuple, None).

    Generic type coercion only — no domain-specific attr rewriting (e.g.
    URI or instrument-history formatting), which is the caller's
    responsibility. Callers must run this before `write_netcdf`, which
    does not serialize attrs itself.

    Modifies ds directly — safe to call on a year slice since xarray copies
    attrs on sel(), so the source dataset is not affected.
    """
    ds.attrs = _serialize_attrs(ds.attrs)
    for name in list(ds.variables):
        if ds[name].attrs:
            ds[name].attrs = _serialize_attrs(ds[name].attrs)
    return ds


def peek_netcdf_variables(file_path: Path) -> list[str]:
    """List data variable names in a NetCDF file without loading any data.

    xarray's netCDF4 backend reads only variable/coordinate metadata on
    open; array data is read lazily on first access. This opens and
    immediately closes the file, so no data is read from disk.

    Args:
        file_path: Path to the .nc file.

    Returns:
        Data variable names (excludes coordinates).
    """
    with xr.open_dataset(file_path) as ds:
        return list(ds.data_vars)


def read_netcdf(file_path: Path, variables: list[str] | None = None) -> xr.Dataset:
    """Load a NetCDF file into an in-memory xarray Dataset.

    xarray opens NetCDF files lazily, so restricting `variables` before
    loading means only those variables' data is read from disk — the rest
    of the file is never touched. Use this to pull a small subset out of
    a large file cheaply. Use `peek_netcdf_variables` first if the
    available variable names aren't already known.

    Args:
        file_path: Path to the .nc file.
        variables: Data variable names to load. If None, the whole file
            is loaded.

    Returns:
        Dataset fully loaded into memory; the underlying file handle is
        closed before this function returns.
    """
    file_path = Path(file_path)

    with xr.open_dataset(file_path) as ds:
        if variables is not None:
            missing = set(variables) - set(ds.data_vars)
            if missing:
                raise KeyError(
                    f"Variable(s) not found in {file_path.name}: {sorted(missing)}"
                )

            ds = ds[variables]

        return ds.load()


def write_netcdf(
    *,
    ds: xr.Dataset,
    file_path: Path,
    time_units: str = None,
    atomic: bool = True,
) -> None:
    """Write an xarray Dataset to NetCDF.

    Does not serialize attrs — callers must ensure `ds` is already
    NetCDF-writable (see `serialize_dataset_attrs`) before calling this.

    Args:
        ds: Dataset to write.
        file_path: Destination path. Parent directories created automatically.
        time_units: CF units string for time encoding (e.g. 'seconds since
            1800-01-01'). If None, xarray chooses its own default.
        atomic: If True (default), write via a temporary file that is
            atomically renamed on success.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    encoding = {"time": {"units": time_units}} if time_units else {}

    if atomic:
        # Write to a local temp file then move to the destination.  This avoids
        # HDF5 file-locking hangs when the destination is on a network filesystem.
        fd, tmp_str = tempfile.mkstemp(suffix=".nc.tmp", dir=file_path.parent)
        os.close(fd)
        tmp_path = Path(tmp_str)
        try:
            ds.to_netcdf(tmp_path, encoding=encoding)
            shutil.move(tmp_path, file_path)
            file_path.chmod(0o640)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        ds.to_netcdf(file_path, encoding=encoding)

    logger.info("Wrote NetCDF file: %s", file_path.name)


def _chmod_tree(root: Path, *, file_mode: int, dir_mode: int) -> None:
    """Apply file_mode to every file and dir_mode to every directory under root."""
    root.chmod(dir_mode)
    for path in root.rglob("*"):
        path.chmod(dir_mode if path.is_dir() else file_mode)


def write_zarr(*, ds: xr.Dataset, store_path: Path, atomic: bool = True) -> None:
    """Write an xarray Dataset to a Zarr store (a directory).

    Does not serialize attrs — callers must ensure `ds` is already
    Zarr-writable (see `serialize_dataset_attrs`) before calling this, for
    the same reason as `write_netcdf`.

    Unlike `write_netcdf`, a Zarr store is a directory tree, not a single
    file, which changes two things: the atomic-write temp object is a
    directory (moved into place with `shutil.move`, which nests rather than
    replaces if the destination already exists as a directory, so an
    existing store is removed immediately before the move — a small
    non-atomic window, acceptable for this manually-invoked tool), and
    permissions must be applied recursively rather than via one chmod call.

    Args:
        ds: Dataset to write.
        store_path: Destination directory path (the Zarr store root).
            Parent directories created automatically.
        atomic: If True (default), write to a temporary sibling directory
            then move it into place.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if atomic:
        tmp_dir = Path(tempfile.mkdtemp(suffix=".zarr.tmp", dir=store_path.parent))
        try:
            ds.to_zarr(tmp_dir, mode="w")
            if store_path.exists():
                shutil.rmtree(store_path)
            shutil.move(str(tmp_dir), str(store_path))
            _chmod_tree(store_path, file_mode=0o640, dir_mode=0o750)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    else:
        ds.to_zarr(store_path, mode="w")

    logger.info("Wrote Zarr store: %s", store_path.name)


def append_zarr(*, ds: xr.Dataset, store_path: Path) -> None:
    """Append a tail slice to an existing Zarr store and refresh its global attrs.

    Does not serialize attrs — see `write_zarr`. Unlike `write_zarr`, this
    assumes `store_path` already exists with a schema (variables/dims)
    matching `ds`; a mismatch (e.g. a site-config change adding/removing a
    variable) raises from `to_zarr` itself — callers should catch that and
    fall back to a full rebuild via `write_zarr` rather than handling it
    here.

    `ds.attrs` are not applied to the store by `to_zarr(mode="a")`, so they
    are written explicitly onto the root group afterward — this is how
    callers refresh whole-history attrs (record count, time coverage,
    instrument history) that must reflect the combined store, not just the
    newly appended slice.

    Args:
        ds: Tail-slice dataset to append along the existing `time` dim.
        store_path: Path to the existing Zarr store.
    """
    store_path = Path(store_path)

    ds.to_zarr(store_path, mode="a", append_dim="time")

    group = zarr.open_group(str(store_path), mode="a")
    group.attrs.update(ds.attrs)

    _chmod_tree(store_path, file_mode=0o640, dir_mode=0o750)

    logger.info(
        "Appended %d record(s) to Zarr store: %s", ds.sizes["time"], store_path.name
    )


def write_yml_file(
    file_path: str | Path,
    data: dict,
    *,
    sort_keys: bool = False,
    default_flow_style: bool = False,
) -> None:
    """Write a dictionary to a YAML file.

    Args:
        file_path: output file path.
        data: dictionary to write.
        sort_keys: sort dictionary keys alphabetically. Defaults to False.
        default_flow_style: write in inline YAML format. Defaults to False.
    """
    file = Path(file_path)

    with file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=sort_keys,
            default_flow_style=default_flow_style,
            allow_unicode=True,
        )

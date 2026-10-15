#!/usr/bin/env python3
"""File I/O: YAML and raw-file reading, and atomic NetCDF/Zarr writers."""

import csv
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path

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
    matching `ds`; a variable-set mismatch (e.g. a site-config change adding
    or removing a variable) is checked explicitly before to_zarr runs and
    raises ValueError here, rather than being left to to_zarr's own
    behavior — which is inconsistent across mismatch kinds: some raise
    immediately, but adding a variable that doesn't yet exist in the store
    does not raise at all. to_zarr silently creates the new array at
    whatever length the tail happens to be, leaving the store internally
    inconsistent (existing variables at the full combined length, the new
    one only at the tail's length) until something next tries to open the
    store as a whole and fails on the dimension-size conflict. Callers
    should catch this ValueError (or the pre-existing to_zarr failure modes
    this doesn't change) and fall back to a full rebuild via `write_zarr`.

    `to_zarr(mode="a", append_dim=...)` replaces the store's root attrs
    wholesale with `ds.attrs` as part of the append (it does not merge),
    so any attr the caller didn't put on `ds` — anything not recomputed on
    every tail build, e.g. a `config_hash` stamped only at full-rebuild
    time — would otherwise be silently dropped on the very next incremental
    update. The store's pre-append attrs are read first and merged with
    `ds.attrs` (which wins per-key) when writing the root group's attrs
    back afterward, so callers only need to supply the keys they want to
    refresh (record count, time coverage, instrument history); anything
    else already on the store persists untouched.

    Args:
        ds: Tail-slice dataset to append along the existing `time` dim.
        store_path: Path to the existing Zarr store.

    Raises:
        ValueError: if ds's data variables don't exactly match the existing
            store's (added and/or removed variables are named in the message).
    """
    store_path = Path(store_path)

    existing_ds = xr.open_zarr(store_path)
    existing_vars = set(existing_ds.data_vars)
    existing_attrs = dict(existing_ds.attrs)
    new_vars = set(ds.data_vars)
    if new_vars != existing_vars:
        raise ValueError(
            f"Cannot append to {store_path.name}: variable set mismatch "
            f"(added: {sorted(new_vars - existing_vars)}, "
            f"removed: {sorted(existing_vars - new_vars)})"
        )

    ds.to_zarr(store_path, mode="a", append_dim="time")

    group = zarr.open_group(str(store_path), mode="a")
    group.attrs.update({**existing_attrs, **ds.attrs})
    # to_zarr writes a consolidated .zmetadata cache that xr.open_zarr reads
    # from by default; the attrs.update() above only touches the plain
    # .zattrs file, so without re-consolidating here, every reader would
    # keep seeing to_zarr's wholesale-replaced (pre-merge) attrs.
    zarr.consolidate_metadata(str(store_path))

    _chmod_tree(store_path, file_mode=0o640, dir_mode=0o750)

    logger.info(
        "Appended %d record(s) to Zarr store: %s", ds.sizes["time"], store_path.name
    )

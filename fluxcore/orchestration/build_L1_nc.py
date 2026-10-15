#!/usr/bin/env python3
"""Export step: write a site's L1 Zarr store out as annual NetCDF files.

NetCDF is export-only: build_from_zarr() reads the already-processed L1 Zarr
store (orchestration/build_L1_zarr.py), never raw data, splits it by calendar
year and writes one file per year. To export a site that has no store yet,
build the store first with build_L1_zarr.build().

The dataset-finishing helpers defined here (spatial dims, QC flags, attr
filtering and serialization) are also used by build_L1_zarr when it writes
the store.

slice_year_from_store, decode_instrument_history,
assign_global_attrs, and get_ds_years are also reused by
orchestration/build_L2_nc.py -- their content is generic to any
Zarr-sourced, already-processed dataset, not specific to L1, despite the
naming.

Generic core: takes a SiteContext and explicit directories, never a bare
site name to resolve. Every file gets the network-agnostic global attributes
in configs/nc_global_attrs.yml; network-specific ones (e.g. TERN's
configs/nc_metadata.yml) are passed in as `global_attrs` by the caller
and override them per key. The TERN site-name wrappers live in
orchestration/build_runner.py.
"""

import datetime
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from fluxcore.domain.constants import DATA_TIME_FORMAT, NC_ENCODING
from fluxcore.infrastructure import file_io
from fluxcore.orchestration import incremental_zarr
from fluxcore.orchestration.dataset_finishing import (
    assign_coverage_attrs,
    assign_global_attrs,
    decode_instrument_history,
    non_dim_variables,
)


def get_ds_years(ds):
    """Return the sorted list of distinct calendar years present in ds.time."""
    return np.unique(ds.time.dt.year).tolist()


def build_from_zarr(
    site_name: str,
    zarr_dir: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    year: int | None = None,
    global_attrs: dict | None = None,
) -> list[pathlib.Path]:
    """Build L1 NetCDF files for a site by reading from its Zarr store.

    The store is already unit-converted, merged, QC-flagged and attributed,
    so this only slices by year, re-derives the year-scoped attrs, and
    writes NetCDF. Raises if the site's store doesn't exist yet.

    Args:
        site_name: the site's name, used only for store/file naming
            ({site_name}_L1.zarr in, {site_name}_{year}_L1.nc out).
        zarr_dir: directory containing the site's Zarr store.
        output_dir: directory to write NetCDF files into.
        year: if provided, only build that calendar year's file.
        global_attrs: extra global attributes to (re)stamp on every file
            (see assign_global_attrs).

    Returns:
        List of paths to files written.
    """
    return export_store(
        site_name,
        zarr_dir,
        output_dir,
        level="L1",
        year=year,
        global_attrs=global_attrs,
    )


def export_store(
    site_name: str,
    zarr_dir: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    level: str,
    year: int | None = None,
    global_attrs: dict | None = None,
) -> list[pathlib.Path]:
    """Write a site's `level` Zarr store out as annual NetCDF files.

    Shared by the L1 and L2 exporters: {site_name}_{level}.zarr in,
    {site_name}_{year}_{level}.nc out, one file per calendar year (or only
    `year`).
    """
    store_path = incremental_zarr.store_path_for(site_name, zarr_dir, level)

    ds = xr.open_zarr(store_path).load()
    ds = decode_instrument_history(ds)
    # Refresh date_created/history to this NetCDF build's own time, rather
    # than whenever the Zarr store was last written/appended.
    ds = assign_global_attrs(ds, global_attrs=global_attrs, processing_level=level)

    output_dir = pathlib.Path(output_dir)
    years = [year] if year is not None else get_ds_years(ds)

    written = []
    for ds_year in years:
        year_ds = slice_year_from_store(ds=ds, year=ds_year)
        file_path = output_dir / f"{site_name}_{ds_year}_{level}.nc"
        file_io.write_netcdf(ds=year_ds, file_path=file_path, time_units=NC_ENCODING)
        written.append(file_path)

    return written


def _year_window(year: int, time_step) -> tuple[datetime.datetime, datetime.datetime]:
    """First and last period-end timestamps of a calendar year's records."""
    return (
        datetime.datetime(year, 1, 1) + datetime.timedelta(minutes=time_step),
        datetime.datetime(year + 1, 1, 1),
    )


def _year_time_bounds(ds, year) -> list[str]:
    """Calendar-year time_bounds for slicing ds, as DATA_TIME_FORMAT strings."""
    return [
        bound.strftime(DATA_TIME_FORMAT)
        for bound in _year_window(year, ds.attrs["time_step"])
    ]


def slice_year_from_store(ds, year):
    """Slice a Zarr-sourced (already-processed) dataset to one year.

    Skips the steps already baked into the store (QC flags, attr filtering,
    uri/units serialization) and only re-derives what's genuinely
    year-scoped: title/record-count/time-coverage attrs and
    instrument-history clipping.

    Args:
        ds: xarray dataset loaded from a site's Zarr store (already passed
            through decode_instrument_history).
        year: calendar year to slice out.

    Returns:
        Sliced, fully attributed and serialized single-year dataset.
    """
    year_ds = ds.sel(time=slice(*_year_time_bounds(ds, year)))
    year_ds = assign_coverage_attrs(
        year_ds, title=f"Flux tower data set from the {ds.site_name} site {year}"
    )
    year_ds = serialize_inst_history(ds=year_ds, year=year)
    year_ds = file_io.serialize_dataset_attrs(ds=year_ds)

    return year_ds


def serialize_inst_history(ds, year):
    """Serialize per-year instrument-changeover history into compact attr strings.

    Collapses the instrument_history dict (simple or compound) built during
    dataset assembly into '|'- and ';'-joined strings clipped to this year's
    date range, and sets 'instrument' to the last instrument used in the year.
    The open-ended-date resolution itself lives in resolve_instrument_history;
    this function only adds the calendar-year windowing and NetCDF attr
    reassembly on top of it.
    """
    year_start, year_end = _year_window(year, ds.attrs["time_step"])
    year_start = max(year_start, pd.Timestamp(ds.time.values[0]).to_pydatetime())
    year_end = min(year_end, pd.Timestamp(ds.time.values[-1]).to_pydatetime())

    for var in non_dim_variables(ds):
        attrs = ds[var].attrs

        if "instrument_history" not in attrs:
            if isinstance(attrs.get("instrument"), dict):
                attrs["instrument"] = ",".join(attrs["instrument"].values())
            continue

        history = attrs["instrument_history"]
        first_val = next(iter(history.values()))

        if "start_date" in first_val:
            # Simple: {inst_name: {start_date, end_date}}
            if isinstance(attrs.get("instrument"), dict):
                attrs["instrument"] = ",".join(attrs["instrument"].values())
            serialised, last_inst = _serialise_simple_history(
                history, year_start, year_end
            )
        else:
            # Compound: {alias: {inst_name: {start_date, end_date}}}
            serialised, last_inst = _serialise_compound_history(
                history, year_start, year_end
            )

        if not serialised:
            del attrs["instrument_history"]
        elif "start_date" in first_val:
            # Simple history
            if len(serialised) == 1:
                attrs["instrument"] = last_inst
                del attrs["instrument_history"]
            else:
                attrs["instrument_history"] = "|".join(serialised)
                if last_inst is not None:
                    attrs["instrument"] = last_inst
        else:
            # Compound history: last_inst is a dict {alias: name}
            attrs["instrument_history"] = ";".join(serialised)
            inst = attrs.get("instrument")
            current = dict(inst) if isinstance(inst, dict) else {}
            if last_inst:
                current.update(last_inst)
            attrs["instrument"] = ",".join(current.values())

    return ds


def resolve_instrument_history(
    history: dict,
    window_start: datetime.datetime,
    window_end: datetime.datetime,
) -> list[tuple[str, datetime.datetime, datetime.datetime]]:
    """Resolve a simple instrument-history dict against a time window.

    This is the piece that actually needs data, not just config: an
    instrument era with no recorded start_date/end_date in the site config
    — either because the instrument never changed, or because a changeover
    boundary was simply never dated — is open-ended, and the only sensible
    way to resolve "open-ended" into a concrete date is to clamp it to
    whatever window of actual data is being asked about; there's no
    config-only answer. Callers are responsible for that clamping (e.g.
    serialize_inst_history clamps its calendar-year window to the
    dataset's actual time extent) before calling this.

    Args:
        history: {instrument_name: {"start_date": Timestamp|None,
            "end_date": Timestamp|None}}.
        window_start: window lower bound.
        window_end: window upper bound.

    Returns:
        (instrument_name, resolved_start, resolved_end) tuples, in
        `history`'s order, restricted to eras overlapping the window and
        clipped to it.
    """
    resolved = []
    for inst, dates in history.items():
        start = dates["start_date"] if dates["start_date"] is not None else window_start
        end = dates["end_date"] if dates["end_date"] is not None else window_end
        if end < window_start or start > window_end:
            continue
        resolved.append((inst, max(window_start, start), min(window_end, end)))
    return resolved


def _format_resolved_history(
    resolved: list[tuple[str, datetime.datetime, datetime.datetime]],
) -> tuple[list[str], str | None]:
    """Format resolved (instrument, start, end) tuples into NetCDF attr segments.

    Returns the formatted '(inst,start,end)' segment strings and the name
    of whichever instrument's resolved era ends latest (the "current"
    instrument for this window).
    """
    serialised = [
        f"({inst},{start.isoformat()},{end.isoformat()})"
        for inst, start, end in resolved
    ]
    last_inst = None
    last_end = None
    for inst, _start, end in resolved:
        if last_end is None or end > last_end:
            last_end = end
            last_inst = inst
    return serialised, last_inst


def _serialise_simple_history(
    history: dict,
    year_start,
    year_end,
) -> tuple[list[str], str | None]:
    """Serialise a simple instrument history to a list of formatted strings."""
    resolved = resolve_instrument_history(history, year_start, year_end)
    return _format_resolved_history(resolved)


def _serialise_compound_history(
    history: dict,
    year_start,
    year_end,
) -> tuple[list[str], dict[str, str]]:
    """Serialise a compound instrument history keyed by alias.

    Produces one segment per alias: alias>(inst,start,end)|(inst,start,end)
    Segments joined with ';' by the caller. Returns last_by_alias as a
    {alias: instrument_name} dict for the most recently used instruments.
    """
    alias_segments = []
    last_by_alias: dict[str, str] = {}
    for alias, inst_history in history.items():
        parts, last_inst = _serialise_simple_history(inst_history, year_start, year_end)
        if parts:
            alias_segments.append(f"{alias}>{'|'.join(parts)}")
            if last_inst is not None:
                last_by_alias[alias] = last_inst

    return alias_segments, last_by_alias

#!/usr/bin/env python3
"""Export step: convert an L1 xarray Dataset into annual NetCDF files.

Adds spatial dims, QC flags, global/variable metadata, then splits by
calendar year and writes one NetCDF file per year.

build_from_zarr() is the operational path (construct_L1_nc): reads an
already-processed site from its Zarr store (orchestration/build_L1_zarr.py)
rather than rebuilding from raw data. build() (raw-data, from scratch) is
kept for manual/legacy rebuilds, bootstrapping a site before its Zarr store
exists, and diagnosing zarr/raw discrepancies.
"""

import datetime
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from domain.constants import DATA_TIME_FORMAT, NC_ENCODING, SITE_PLACEHOLDER
from infrastructure import file_io, paths
from orchestration import dataset_builder
from services import config_loader

STD_METADATA = config_loader.load_config_file_from_name(name="nc_metadata")
NC_DIM_ATTRS = config_loader.load_config_file_from_name(name="nc_dim_attrs")

VARIABLE_NC_ATTRS = {
    "units",
    "long_name",
    "standard_name",
    "height",
    "height_range",
    "instrument",
    "instrument_history",
    "instrument_uri",
    "statistic_type",
    "valid_range",
}


def get_ds_years(ds):
    """Return the sorted list of distinct calendar years present in ds.time."""
    return np.unique(ds.time.dt.year).tolist()


def build(
    site_name: str,
    output_dir: pathlib.Path | str | None = None,
    year: int | None = None,
    legacy: bool = False,
) -> list[pathlib.Path]:
    """Build L1 NetCDF files for all data years at a site.

    Constructs the base L1 dataset via L1_constructor, augments it with
    NetCDF global attributes and spatial dimensions, then writes one file
    per data year to output_dir.

    The dataset is truncated to the temporal extent of the flux file group,
    so ancillary data (e.g. soil loggers) predating the flux system does not
    produce empty output years.

    Args:
        site_name: registered site name.
        output_dir: directory to write files into. Defaults to the
            homogenised_data/nc stream path for this site.
        year: if provided, only build that calendar year and discard all
            earlier records during data loading.  Use for the 30-min
            operational update cycle to avoid reading full site history.
        legacy: if True, build from the site's legacy config snapshot
            (site_configs/legacy) instead of its operational config. Use
            for one-off rebuilds of a site under an earlier generation of
            instruments/variables/files. Not wired into scheduled tasks —
            callers should pass an explicit output_dir to avoid overwriting
            the operational L1 output for the site.

    Returns:
        List of paths to files written.
    """
    if output_dir is None:
        output_dir = paths.get_local_stream_path("homogenised_data", "nc") / site_name
    output_dir = pathlib.Path(output_dir)

    start_date = pd.Timestamp(year, 1, 1) if year is not None else None
    ds = dataset_builder.build_dataset_from_site_name(
        site_name, start_date=start_date, legacy=legacy
    )
    ds = build_L1_ds_complete(ds)

    written = []
    for ds_year in get_ds_years(ds):
        year_ds = build_L1_ds_by_year(ds=ds, year=ds_year)
        file_path = output_dir / f"{site_name}_{ds_year}_L1.nc"
        file_io.write_netcdf(ds=year_ds, file_path=file_path, time_units=NC_ENCODING)
        written.append(file_path)

    return written


def build_from_zarr(
    site_name: str,
    zarr_dir: pathlib.Path | str | None = None,
    output_dir: pathlib.Path | str | None = None,
    year: int | None = None,
) -> list[pathlib.Path]:
    """Build L1 NetCDF files for a site by reading from its Zarr store.

    Replaces build()'s raw-data rebuild for the operational cron path: the
    store (kept current by construct_L1_zarr/rebuild_L1_zarr — see
    orchestration/build_L1_zarr.py) is already unit-converted, merged,
    QC-flagged, and attributed, so this only slices by year, re-derives the
    year-scoped attrs, and writes NetCDF. See build() for the from-scratch
    raw-data path (manual/legacy/bootstrap use, or diagnosing zarr/raw
    discrepancies) — also the fallback if a site's Zarr store doesn't exist
    yet; this function raises rather than silently falling back to it,
    matching the fail-closed style already used for tasks.csv.

    Args:
        site_name: registered site name.
        zarr_dir: directory containing the site's Zarr store. Defaults to
            the homogenised_data/zarr/L1 stream path.
        output_dir: directory to write NetCDF files into. Defaults to the
            homogenised_data/nc stream path for this site.
        year: if provided, only build that calendar year's file.

    Returns:
        List of paths to files written.
    """
    if zarr_dir is None:
        zarr_dir = paths.get_local_stream_path("homogenised_data", "zarr") / "L1"
    store_path = pathlib.Path(zarr_dir) / f"{site_name}_L1.zarr"

    ds = xr.open_zarr(store_path).load()
    ds = _rehydrate_instrument_history(ds)
    ds = _reorder_compound_instrument(ds)
    # Refresh date_created/history to this NetCDF build's own time, rather
    # than whenever the Zarr store was last written/appended.
    ds = assign_L1_global_generic_attrs(ds)

    if output_dir is None:
        output_dir = paths.get_local_stream_path("homogenised_data", "nc") / site_name
    output_dir = pathlib.Path(output_dir)

    years = [year] if year is not None else get_ds_years(ds)

    written = []
    for ds_year in years:
        year_ds = build_L1_year_from_zarr(ds=ds, year=ds_year)
        file_path = output_dir / f"{site_name}_{ds_year}_L1.nc"
        file_io.write_netcdf(ds=year_ds, file_path=file_path, time_units=NC_ENCODING)
        written.append(file_path)

    return written


def build_L1_ds_complete(ds):
    """Apply spatial dims, CRS variable, and generic global attrs to the dataset."""
    ds = do_dim_ops(ds=ds)
    ds = assign_crs_variable(ds=ds)
    ds = assign_L1_global_generic_attrs(ds=ds)

    return ds


def _year_time_bounds(ds, year) -> list[str]:
    """Calendar-year time_bounds for slicing ds, as DATA_TIME_FORMAT strings."""
    time_step = ds.attrs["time_step"]
    return [
        bound.strftime(DATA_TIME_FORMAT)
        for bound in (
            datetime.datetime(year, 1, 1) + datetime.timedelta(minutes=time_step),
            datetime.datetime(year + 1, 1, 1),
        )
    ]


def build_L1_ds_by_year(ds, year):
    """Slice ds to one calendar year and apply the per-year attrs/serialization steps.

    Args:
        ds: xarray dataset already assembled for the full site history.
        year: calendar year to slice out.

    Returns:
        Sliced, fully attributed and serialized single-year dataset.
    """
    year_ds = ds.sel(time=slice(*_year_time_bounds(ds, year)))
    year_ds = assign_variable_flags(year_ds)
    year_ds = assign_valid_range(year_ds)
    year_ds = assign_L1_data_year_attrs(ds=year_ds, year=year)
    year_ds = filter_variable_attrs(ds=year_ds)
    year_ds = serialize_uri(ds=year_ds)
    year_ds = serialize_inst_history(ds=year_ds, year=year)
    year_ds = serialize_units(ds=year_ds)
    year_ds = file_io.serialize_dataset_attrs(ds=year_ds)

    return year_ds


def build_L1_year_from_zarr(ds, year):
    """Slice a Zarr-sourced (already-processed) dataset to one year.

    Counterpart to build_L1_ds_by_year for data sourced from
    build_from_zarr rather than freshly built from raw: skips the steps
    already baked into the Zarr store (QC flags, valid_range, attr
    filtering, uri/units serialization) and only re-derives what's
    genuinely year-scoped — title/record-count/time-coverage attrs and
    instrument-history clipping — reusing the existing
    assign_L1_data_year_attrs/serialize_inst_history unchanged.

    Args:
        ds: xarray dataset loaded from a site's Zarr store (already passed
            through _rehydrate_instrument_history).
        year: calendar year to slice out.

    Returns:
        Sliced, fully attributed and serialized single-year dataset.
    """
    year_ds = ds.sel(time=slice(*_year_time_bounds(ds, year)))
    year_ds = assign_L1_data_year_attrs(ds=year_ds, year=year)
    year_ds = serialize_inst_history(ds=year_ds, year=year)
    year_ds = file_io.serialize_dataset_attrs(ds=year_ds)

    return year_ds


def do_dim_ops(ds):
    """Add latitude/longitude dims and attach CF attrs to the dimension coordinates."""
    # Add spatial coordinates
    ds = ds.assign_coords(
        latitude=ds.attrs["latitude"],
        longitude=ds.attrs["longitude"],
    ).expand_dims(["latitude", "longitude"])
    ds = ds.transpose("time", "latitude", "longitude")

    # Attach CF attrs to dimension coordinates
    for dim in ("time", "latitude", "longitude"):
        ds[dim].attrs.update(NC_DIM_ATTRS[dim])

    return ds


def assign_crs_variable(ds):
    """Add a scalar 'crs' coordinate-reference-system variable to ds.

    Args:
        ds: xarray dataset.

    Returns:
        ds, with the 'crs' variable added.
    """
    ds["crs"] = ([], np.int32(0), NC_DIM_ATTRS["coordinate_reference_system"])
    return ds


def assign_L1_data_year_attrs(ds, year):
    """Add per-year global attrs (title, record count, time coverage) to ds.

    Args:
        ds: xarray dataset, already sliced to one calendar year.
        year: calendar year, used to build the title string.

    Returns:
        ds, with the year-specific global attrs added.
    """

    def _format_time(x):
        return pd.to_datetime(x).strftime(DATA_TIME_FORMAT)

    begin = _format_time(ds.time.values[0])
    end = _format_time(ds.time.values[-1])

    # Make title string
    title_str = f"Flux tower data set from the {ds.site_name} site {year}"

    # Assign attrs
    ds.attrs.update(
        {
            "title": title_str,
            "nc_nrecs": len(ds.time),
            "time_coverage_start": begin,
            "time_coverage_end": end,
        }
    )

    return ds


def assign_L1_global_generic_attrs(ds):
    """Add the site-agnostic global attrs from nc_metadata config, plus date/history."""
    # Get and edit the generic global attribute fields
    site_metadata = STD_METADATA.copy()
    site_metadata["metadata_link"] = site_metadata["metadata_link"].replace(
        SITE_PLACEHOLDER, ds.site_name
    )

    # Add time-sensitive fields
    date = datetime.datetime.now()
    this_year = date.strftime("%Y")
    this_month = date.strftime("%b")
    site_metadata.update(
        {
            "date_created": date.strftime(DATA_TIME_FORMAT),
            "history": f"{this_month} {this_year} processing",
        }
    )

    # Update dataset metadata
    ds.attrs.update(site_metadata)

    return ds


def assign_variable_flags(ds):
    """Assign the variable QC flags to the existing dataset.

    Args:
        ds: xarray dataset.

    Returns:
        ds, with a '{var}_QCFlag' variable added for every data variable.
    """
    var_list = [var for var in ds.variables if var not in ds.dims and var != "crs"]
    for var in var_list:
        ds[f"{var}_QCFlag"] = (
            ["time", "latitude", "longitude"],
            pd.isnull(ds[var]).astype(int),
            {"long_name": f"{var} QC flag", "units": "1"},
        )
    return ds


def assign_valid_range(ds):
    """Assign the CF `valid_range` attribute to variables carrying valid bounds.

    Cast to the variable's own dtype so it matches what is written to disk.

    Args:
        ds: xarray dataset.

    Returns:
        ds.

    """
    for var in ds.variables:
        attrs = ds[var].attrs
        vmin = attrs.get("valid_min")
        vmax = attrs.get("valid_max")
        if vmin is None or vmax is None:
            continue
        ds[var].attrs["valid_range"] = np.array([vmin, vmax], dtype=ds[var].dtype)
    return ds


def filter_variable_attrs(ds):
    """Drop every variable attr not in VARIABLE_NC_ATTRS (crs is left untouched)."""
    for var in ds.variables:
        if var == "crs":
            continue
        ds[var].attrs = {
            k: v for k, v in ds[var].attrs.items() if k in VARIABLE_NC_ATTRS
        }

    return ds


def serialize_units(ds):
    """Rewrite 'dimensionless' units to '1', the CF-compliant form."""
    for var in ds.variables:
        if ds[var].attrs.get("units") == "dimensionless":
            ds[var].attrs["units"] = "1"

    return ds


def serialize_uri(ds):
    """Flatten a compound-instrument instrument_uri dict to a comma-joined string."""
    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
        attrs = ds[var].attrs

        if isinstance(attrs.get("instrument_uri"), dict):
            attrs["instrument_uri"] = ",".join(
                f"{uri}" for uri in attrs["instrument_uri"].values()
            )

    return ds


def _rehydrate_instrument_history(ds):
    """Reverse build_L1_zarr._json_safe_instrument_history's ISO-string encoding.

    Parses each variable's instrument_history start_date/end_date back to
    pd.Timestamp (or None), so serialize_inst_history can run against
    Zarr-sourced data exactly as it does against freshly built data.
    """
    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
        history = ds[var].attrs.get("instrument_history")
        if history is None:
            continue

        first_val = next(iter(history.values()))
        if "start_date" in first_val:
            ds[var].attrs["instrument_history"] = _parse_history(history)
        else:
            # Compound: alias-level keys need the same reordering as
            # `instrument` itself — see _reorder_compound_instrument.
            ds[var].attrs["instrument_history"] = {
                alias: _parse_history(history[alias])
                for alias in _COMPOUND_INSTRUMENT_KEY_ORDER
                if alias in history
            }

    return ds


def _parse_history(history: dict) -> dict:
    """Parse ISO-string start_date/end_date within a simple instrument-history dict."""
    return {
        inst: {
            "start_date": pd.Timestamp(dates["start_date"])
            if dates["start_date"] is not None
            else None,
            "end_date": pd.Timestamp(dates["end_date"])
            if dates["end_date"] is not None
            else None,
        }
        for inst, dates in history.items()
    }


_COMPOUND_INSTRUMENT_KEY_ORDER = ("sonic_anemometer", "irga")


def _reorder_compound_instrument(ds):
    """Restore canonical sonic_anemometer/irga key order for compound `instrument`.

    Zarr's attrs round-trip does not preserve dict key insertion order (its
    JSON attrs encoding is not order-stable — confirmed empirically: a
    {"sonic_anemometer": ..., "irga": ...} dict comes back key-alphabetised
    after a to_zarr/open_zarr round trip). Left uncorrected, this would
    scramble the ",".join(attrs["instrument"].values()) order
    serialize_inst_history uses for compound-instrument (flux/covariance)
    variables. `instrument`'s only valid compound keys are these two (see
    CLAUDE.md's site-config instrument notation), so the canonical order is
    fixed rather than derived.
    """
    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
        inst = ds[var].attrs.get("instrument")
        if isinstance(inst, dict):
            ds[var].attrs["instrument"] = {
                k: inst[k] for k in _COMPOUND_INSTRUMENT_KEY_ORDER if k in inst
            }
    return ds


def serialize_inst_history(ds, year):
    """Serialize per-year instrument-changeover history into compact attr strings.

    Collapses the instrument_history dict (simple or compound) built during
    dataset assembly into '|'- and ';'-joined strings clipped to this year's
    date range, and sets 'instrument' to the last instrument used in the year.
    The open-ended-date resolution itself lives in resolve_instrument_history;
    this function only adds the calendar-year windowing and NetCDF attr
    reassembly on top of it.
    """
    time_step = ds.attrs["time_step"]
    year_start = datetime.datetime(year, 1, 1) + datetime.timedelta(minutes=time_step)
    year_end = datetime.datetime(year + 1, 1, 1)
    year_start = max(year_start, pd.Timestamp(ds.time.values[0]).to_pydatetime())
    year_end = min(year_end, pd.Timestamp(ds.time.values[-1]).to_pydatetime())

    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
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

#!/usr/bin/env python3
"""Export step: convert an L1 xarray Dataset into annual NetCDF files.

Adds spatial dims, QC flags, global/variable metadata, then splits by
calendar year and writes one NetCDF file per year.
"""

import datetime
import pathlib

import numpy as np
import pandas as pd

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


def build_L1_ds_complete(ds):
    """Apply spatial dims, CRS variable, and generic global attrs to the dataset."""
    ds = do_dim_ops(ds=ds)
    ds = assign_crs_variable(ds=ds)
    ds = assign_L1_global_generic_attrs(ds=ds)

    return ds


def build_L1_ds_by_year(ds, year):
    """Slice ds to one calendar year and apply the per-year attrs/serialization steps.

    Args:
        ds: xarray dataset already assembled for the full site history.
        year: calendar year to slice out.

    Returns:
        Sliced, fully attributed and serialized single-year dataset.
    """
    # Get network-specific valid year data bounds
    time_step = ds.attrs["time_step"]
    time_bounds = [
        bound.strftime(DATA_TIME_FORMAT)
        for bound in (
            datetime.datetime(year, 1, 1) + datetime.timedelta(minutes=time_step),
            datetime.datetime(year + 1, 1, 1),
        )
    ]

    year_ds = ds.sel(time=slice(*time_bounds))
    year_ds = assign_variable_flags(year_ds)
    year_ds = assign_valid_range(year_ds)
    year_ds = assign_L1_data_year_attrs(ds=year_ds, year=year)
    year_ds = filter_variable_attrs(ds=year_ds)
    year_ds = serialize_uri(ds=year_ds)
    year_ds = serialize_inst_history(ds=year_ds, year=year)
    year_ds = serialize_units(ds=year_ds)
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


def serialize_inst_history(ds, year):
    """Serialize per-year instrument-changeover history into compact attr strings.

    Collapses the instrument_history dict (simple or compound) built during
    dataset assembly into '|'- and ';'-joined strings clipped to this year's
    date range, and sets 'instrument' to the last instrument used in the year.
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


def _serialise_simple_history(
    history: dict,
    year_start,
    year_end,
) -> tuple[list[str], str | None]:
    """Serialise a simple instrument history to a list of formatted strings."""
    serialised = []
    last_inst = None
    last_end = None
    for inst, dates in history.items():
        dates = dict(dates)
        start = dates["start_date"] if dates["start_date"] is not None else year_start
        end = dates["end_date"] if dates["end_date"] is not None else year_end
        if end < year_start or start > year_end:
            continue
        use_start = max(year_start, start)
        use_end = min(year_end, end)
        serialised.append(f"({inst},{use_start.isoformat()},{use_end.isoformat()})")
        if last_end is None or use_end > last_end:
            last_end = use_end
            last_inst = inst
    return serialised, last_inst


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

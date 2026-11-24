#!/usr/bin/env python3
"""Finish an assembled dataset for output: structure, flags, attrs, encoding.

Shared by the Zarr builders (build_L1_zarr, build_L2_zarr) and the NetCDF
exporters (build_L1_nc, build_L2_nc): spatial dims and CRS variable, global
and coverage attrs, missing-value flags, CF-style attr filtering and
serialization, and the Zarr-safe encoding of instrument_history (with the
inverse decode the exporters run on a store's data).
"""

import datetime

import numpy as np
import pandas as pd

from fluxcore.domain.constants import CONFIG_PATH, DATA_TIME_FORMAT, SITE_PLACEHOLDER
from fluxcore.infrastructure.file_io import read_yml

NC_DIM_ATTRS = read_yml(file_path=CONFIG_PATH / "nc_dim_attrs.yml")
NC_GLOBAL_ATTRS = read_yml(file_path=CONFIG_PATH / "nc_global_attrs.yml")
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
    "flag_values",
    "flag_meanings",
}


def non_dim_variables(ds) -> list[str]:
    """Names of ds's variables that aren't dimension coordinates."""
    return [var for var in ds.variables if var not in ds.dims]


def add_spatial_dims(ds):
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


def assign_global_attrs(
    ds, global_attrs: dict | None = None, processing_level: str = "L1"
):
    """Add the core and caller-supplied global attrs, plus generated fields.

    The core's network-agnostic attrs (NC_GLOBAL_ATTRS, from
    configs/nc_global_attrs.yml) are always applied. global_attrs is a
    network's own global attribute template, overriding the core attrs per
    key. Any
    string value containing SITE_PLACEHOLDER has it replaced with the
    dataset's site_name. processing_level is set by the level's own builder
    and always overrides any value from either source.
    """
    site_metadata = {
        key: (
            value.replace(SITE_PLACEHOLDER, ds.site_name)
            if isinstance(value, str)
            else value
        )
        for key, value in {**NC_GLOBAL_ATTRS, **(global_attrs or {})}.items()
    }

    # Add time-sensitive fields
    date = datetime.datetime.now()
    this_year = date.strftime("%Y")
    this_month = date.strftime("%b")
    site_metadata.update(
        {
            "processing_level": processing_level,
            "date_created": date.strftime(DATA_TIME_FORMAT),
            "history": f"{this_month} {this_year} processing",
        }
    )

    # Update dataset metadata
    ds.attrs.update(site_metadata)

    return ds


def add_missing_value_flags(ds):
    """Assign the variable QC flags to the existing dataset.

    Args:
        ds: xarray dataset.

    Returns:
        ds, with a '{var}_QCFlag' variable added for every data variable.
    """
    var_list = [
        var
        for var in ds.variables
        if var not in ds.dims and var not in ("crs", "day_night")
    ]
    for var in var_list:
        ds[f"{var}_QCFlag"] = (
            ["time", "latitude", "longitude"],
            pd.isnull(ds[var]).astype(int),
            {"long_name": f"{var} QC flag", "units": "1"},
        )
    return ds


def assign_coverage_attrs(
    ds, title: str | None = None, range_start=None, nrecs_base: int = 0
):
    """Set the record-count and time-coverage global attrs (and title, if given).

    Args:
        ds: dataset whose time axis the attrs describe.
        title: new title attr; None leaves any existing title alone.
        range_start: use as time_coverage_start instead of ds's first time —
            for a tail slice, the existing store's recorded start.
        nrecs_base: records already in the store before ds (0 unless ds is a
            tail slice being appended).

    Returns:
        ds, with the attrs set.
    """

    def _format_time(x):
        return pd.to_datetime(x).strftime(DATA_TIME_FORMAT)

    attrs = {"title": title} if title is not None else {}
    attrs.update(
        {
            "nc_nrecs": nrecs_base + ds.sizes["time"],
            "time_coverage_start": _format_time(
                range_start if range_start is not None else ds.time.values[0]
            ),
            "time_coverage_end": _format_time(ds.time.values[-1]),
        }
    )
    ds.attrs.update(attrs)
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


def flatten_instrument_uri(ds):
    """Flatten a compound-instrument instrument_uri dict to a comma-joined string.

    Unresolved (None) entries are dropped; a dict with no resolved URIs at
    all serializes to None, consistent with a scalar unresolved
    instrument_uri, so the later None-attr filter drops the attribute
    rather than writing a literal "None" string.
    """
    for var in non_dim_variables(ds):
        attrs = ds[var].attrs

        if isinstance(attrs.get("instrument_uri"), dict):
            uris = [uri for uri in attrs["instrument_uri"].values() if uri is not None]
            attrs["instrument_uri"] = ",".join(uris) if uris else None

    return ds


def cf_normalise_units(ds):
    """Rewrite 'dimensionless' units to '1', the CF-compliant form."""
    for var in ds.variables:
        if ds[var].attrs.get("units") == "dimensionless":
            ds[var].attrs["units"] = "1"

    return ds

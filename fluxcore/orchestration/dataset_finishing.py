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

from fluxcore.domain.constants import DATA_TIME_FORMAT, SITE_PLACEHOLDER
from fluxcore.services import config_loader

NC_DIM_ATTRS = config_loader.load_config("nc_dim_attrs")
NC_GLOBAL_ATTRS = config_loader.load_config("nc_global_attrs")
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


def encode_instrument_history(ds):
    """Make each variable's instrument/instrument_history attrs Zarr-safe.

    Converts nested start_date/end_date timestamps to ISO strings without
    collapsing or year-clipping the structure — unlike
    build_L1_nc.serialize_inst_history, which flattens *and* clips to one
    calendar year. Preserving the structure here (rather than flattening
    at Zarr-write time) is what lets NetCDF export reproduce that per-year
    clipping later, from data that's otherwise already fully processed by
    the time it reaches this store — see build_L1_nc.build_from_zarr and
    its decode_instrument_history counterpart.

    Also converts any alias-keyed dict — the compound-instrument
    `instrument` attr itself ({"sonic_anemometer": name, "irga": name}),
    and instrument_history's alias-level structure — to an ordered list of
    [key, value] pairs. Zarr's attrs JSON writer alphabetizes dict keys but
    preserves list order (confirmed empirically: a {"sonic_anemometer":
    ..., "irga": ...} dict comes back key-alphabetised after a
    to_zarr/open_zarr round trip, silently swapping which value is which
    instrument once downstream code joins .values() positionally). A list
    of pairs round-trips exactly as written, so this is a real fix rather
    than a guessed fixed key order applied after the fact.

    Args:
        ds: dataset — the full history for `build()`, or a tail slice for
            `update()`. instrument_history is config-derived and always
            reflects the site's full history regardless of which slice of
            data `ds` covers, so no date-range argument is needed here.
    """
    for var in non_dim_variables(ds):
        attrs = ds[var].attrs

        inst = attrs.get("instrument")
        if isinstance(inst, dict):
            attrs["instrument"] = list(inst.items())

        history = attrs.get("instrument_history")
        if history is None:
            continue

        first_val = next(iter(history.values()))
        if "start_date" in first_val:
            # Simple: {inst_name: {start_date, end_date}}
            attrs["instrument_history"] = _isoformat_history(history)
        else:
            # Compound: {alias: {inst_name: {start_date, end_date}}}
            attrs["instrument_history"] = [
                [alias, _isoformat_history(inst_history)]
                for alias, inst_history in history.items()
            ]

    return ds


def _isoformat_history(history: dict) -> dict:
    """ISO-format start_date/end_date within a simple instrument-history dict."""
    return {
        inst: {
            "start_date": dates["start_date"].isoformat()
            if dates["start_date"] is not None
            else None,
            "end_date": dates["end_date"].isoformat()
            if dates["end_date"] is not None
            else None,
        }
        for inst, dates in history.items()
    }


def decode_instrument_history(ds):
    """Reverse dataset_finishing.encode_instrument_history's Zarr-safe encoding.

    Parses each variable's instrument_history start_date/end_date back to
    pd.Timestamp (or None), so serialize_inst_history can run against
    Zarr-sourced data exactly as it does against freshly built data. Also
    reconstructs `instrument`/instrument_history's alias-level structure
    from the [key, value]-pair lists written at Zarr-build time back into
    plain dicts, in their original order — a compound instrument_history is
    a list at this point (never a dict), which is what tells this apart
    from a simple one, so no key-order guessing is needed the way
    _reorder_compound_instrument used to do it.
    """
    for var in non_dim_variables(ds):
        attrs = ds[var].attrs

        inst = attrs.get("instrument")
        if isinstance(inst, list):
            attrs["instrument"] = dict(inst)

        history = attrs.get("instrument_history")
        if history is None:
            continue

        if isinstance(history, list):
            # Compound: [[alias, {inst_name: {start_date, end_date}}], ...]
            attrs["instrument_history"] = {
                alias: _parse_history(inst_history) for alias, inst_history in history
            }
        else:
            attrs["instrument_history"] = _parse_history(history)

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

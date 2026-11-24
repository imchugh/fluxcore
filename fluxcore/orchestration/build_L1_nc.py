#!/usr/bin/env python3
"""Export step: write a site's L1 Zarr store out as annual NetCDF files.

NetCDF is export-only: build_from_zarr() reads the already-processed L1 Zarr
store (build_L1_zarr), never raw data, splits it by calendar year and writes
one file per year. To export a site that has no store yet, build the store
first with build_L1_zarr.build(). export_store() is the level-agnostic
exporter that build_L2_nc uses too.

Every file gets the network-agnostic global attributes in
configs/nc_global_attrs.yml, overlaid per key by a caller's `global_attrs`
(see dataset_finishing.assign_global_attrs). Directories are arguments;
site_name is used only for file naming.
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
    non_dim_variables,
)
from fluxcore.services.metadata.instrument_history import InstrumentHistory


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
    # The store's config_hash is incremental-update bookkeeping, not metadata.
    ds.attrs.pop("config_hash", None)
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
        ds: xarray dataset loaded from a site's Zarr store.
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
    """Write each variable's NetCDF instrument attrs for one calendar year.

    Replaces the full instrument_history attr with its year-clipped NetCDF
    string (or drops it), and sets `instrument` to what was in use that
    year; see InstrumentHistory.netcdf_attrs. The year is clamped to the
    dataset's actual time extent, which resolves open-ended periods.
    """
    year_start, year_end = _year_window(year, ds.attrs["time_step"])
    year_start = max(year_start, pd.Timestamp(ds.time.values[0]).to_pydatetime())
    year_end = min(year_end, pd.Timestamp(ds.time.values[-1]).to_pydatetime())

    for var in non_dim_variables(ds):
        attrs = ds[var].attrs
        if "instrument_history" not in attrs:
            continue
        history = InstrumentHistory.from_attr(attrs["instrument_history"])
        instrument, history_str = history.netcdf_attrs(year_start, year_end)
        attrs["instrument"] = instrument
        if history_str is None:
            del attrs["instrument_history"]
        else:
            attrs["instrument_history"] = history_str

    return ds

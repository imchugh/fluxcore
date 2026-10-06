#!/usr/bin/env python3
"""Export step: convert an L2 xarray Dataset into annual NetCDF files.

L2's Zarr store (orchestration/build_L2_zarr.py) is already QC-flagged/
masked and fully attributed (inherited unchanged from L1, whose own build
already ran filter_variable_attrs), so this only slices by year, re-derives
the year-scoped title/record-count/time-coverage attrs, and writes NetCDF —
reusing build_L1_nc.py's already-generic Zarr-sourced helpers directly rather
than duplicating them. Unlike L1, there is no from-scratch raw-data build
path here: L2 is always QC-derived from L1's Zarr store, never rebuilt
directly from raw data, so this module has one entry point, not a pair.

Not wired into tasks/build_tasks.py -- built for manual benchmarking against
real PyFluxPro L2 NetCDF output (the L2 Zarr store itself is rebuilt
nightly by the rebuild_L2_zarr task). The TERN site-name wrapper and CLI
live in orchestration/tern/build_runner.py.
"""

import pathlib

import xarray as xr

from domain.constants import NC_ENCODING
from infrastructure import file_io
from orchestration.build_L1_nc import (
    _rehydrate_instrument_history,
    assign_L1_global_generic_attrs,
    build_L1_year_from_zarr,
    get_ds_years,
)
from orchestration.build_L2_zarr import PROCESSING_LEVEL


def build_from_zarr(
    site_name: str,
    zarr_dir: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    year: int | None = None,
    global_attrs: dict | None = None,
) -> list[pathlib.Path]:
    """Build L2 NetCDF files for a site by reading from its L2 Zarr store.

    Args:
        site_name: the site's name, used only for store/file naming
            ({site_name}_L2.zarr in, {site_name}_{year}_L2.nc out).
        zarr_dir: directory containing the site's L2 Zarr store.
        output_dir: directory to write NetCDF files into.
        year: if provided, only build that calendar year's file.
        global_attrs: extra global attributes to (re)stamp on every file
            (see build_L1_nc.assign_L1_global_generic_attrs).

    Returns:
        List of paths to files written.
    """
    store_path = pathlib.Path(zarr_dir) / f"{site_name}_L2.zarr"

    ds = xr.open_zarr(store_path).load()
    ds = _rehydrate_instrument_history(ds)
    # Refresh date_created/history to this NetCDF build's own time, rather
    # than whenever the L2 Zarr store was last written/appended.
    ds = assign_L1_global_generic_attrs(
        ds, global_attrs=global_attrs, processing_level=PROCESSING_LEVEL
    )

    output_dir = pathlib.Path(output_dir)

    years = [year] if year is not None else get_ds_years(ds)

    written = []
    for ds_year in years:
        year_ds = build_L1_year_from_zarr(ds=ds, year=ds_year)
        file_path = output_dir / f"{site_name}_{ds_year}_L2.nc"
        file_io.write_netcdf(ds=year_ds, file_path=file_path, time_units=NC_ENCODING)
        written.append(file_path)

    return written

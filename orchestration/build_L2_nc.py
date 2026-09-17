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
real PyFluxPro L2 NetCDF output, same deferred-until-needed status as
build_L2_zarr.py's own task wiring.

Usage (from project root, with ep_cntl activated):
    python -m orchestration.build_L2_nc SITE_NAME [--zarr-dir DIR] \
        [--output-dir DIR] [--year YEAR]
"""

import argparse
import pathlib

import xarray as xr

from domain.constants import NC_ENCODING
from infrastructure import file_io, paths
from orchestration.build_L1_nc import (
    _rehydrate_instrument_history,
    assign_L1_global_generic_attrs,
    build_L1_year_from_zarr,
    get_ds_years,
)


def build_from_zarr(
    site_name: str,
    zarr_dir: pathlib.Path | str | None = None,
    output_dir: pathlib.Path | str | None = None,
    year: int | None = None,
) -> list[pathlib.Path]:
    """Build L2 NetCDF files for a site by reading from its L2 Zarr store.

    Args:
        site_name: registered site name.
        zarr_dir: directory containing the site's L2 Zarr store. Defaults to
            the homogenised_data/zarr/L2 stream path.
        output_dir: directory to write NetCDF files into. Defaults to the
            homogenised_data/nc stream path for this site.
        year: if provided, only build that calendar year's file.

    Returns:
        List of paths to files written.
    """
    if zarr_dir is None:
        zarr_dir = paths.get_local_stream_path("homogenised_data", "zarr") / "L2"
    store_path = pathlib.Path(zarr_dir) / f"{site_name}_L2.zarr"

    ds = xr.open_zarr(store_path).load()
    ds = _rehydrate_instrument_history(ds)
    # Refresh date_created/history to this NetCDF build's own time, rather
    # than whenever the L2 Zarr store was last written/appended.
    ds = assign_L1_global_generic_attrs(ds)

    if output_dir is None:
        output_dir = paths.get_local_stream_path("homogenised_data", "nc") / site_name
    output_dir = pathlib.Path(output_dir)

    years = [year] if year is not None else get_ds_years(ds)

    written = []
    for ds_year in years:
        year_ds = build_L1_year_from_zarr(ds=ds, year=ds_year)
        file_path = output_dir / f"{site_name}_{ds_year}_L2.nc"
        file_io.write_netcdf(ds=year_ds, file_path=file_path, time_units=NC_ENCODING)
        written.append(file_path)

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a site's L2 NetCDF files from its L2 Zarr store."
    )
    parser.add_argument("site_name", help="Registered site name, e.g. Dookie2.")
    parser.add_argument(
        "--zarr-dir",
        type=pathlib.Path,
        default=None,
        help="Override source L2 Zarr directory (default: homogenised_data/zarr/L2 stream).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Override NetCDF output directory (default: homogenised_data/nc stream).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Only build this calendar year's file (default: every year present).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    written = build_from_zarr(
        site_name=args.site_name,
        zarr_dir=args.zarr_dir,
        output_dir=args.output_dir,
        year=args.year,
    )
    for file_path in written:
        print(f"Wrote NetCDF file: {file_path}")


if __name__ == "__main__":
    main()

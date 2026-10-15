#!/usr/bin/env python3
"""Export step: convert an L2 xarray Dataset into annual NetCDF files.

L2's Zarr store (orchestration/build_L2_zarr.py) is already QC-flagged/
masked and fully attributed (inherited unchanged from L1, whose own build
already ran filter_variable_attrs), so this only slices by year, re-derives
the year-scoped title/record-count/time-coverage attrs, and writes NetCDF —
reusing build_L1_nc.py's already-generic Zarr-sourced helpers directly rather
than duplicating them.

Not wired into tasks/build_tasks.py -- built for manual benchmarking against
real PyFluxPro L2 NetCDF output (the L2 Zarr store itself is rebuilt
nightly by the rebuild_L2_zarr task). The TERN site-name wrapper and CLI
live in orchestration/build_runner.py.
"""

import pathlib

from fluxcore.orchestration.build_L1_nc import export_store
from fluxcore.orchestration.build_L2_zarr import PROCESSING_LEVEL


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
    return export_store(
        site_name,
        zarr_dir,
        output_dir,
        level=PROCESSING_LEVEL,
        year=year,
        global_attrs=global_attrs,
    )

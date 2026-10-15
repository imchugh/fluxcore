#!/usr/bin/env python3
"""Export step: write a site's L2 Zarr store out as annual NetCDF files.

The L2 store is already QC-flagged, masked and attributed, so this is the
same per-year export as L1's (build_L1_nc.export_store).
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
            (see dataset_finishing.assign_global_attrs).

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

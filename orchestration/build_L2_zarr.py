#!/usr/bin/env python3
"""Apply L2 QC to a site's L1 Zarr store, writing a whole-history L2 Zarr store.

Structural mirror of orchestration/build_L1_zarr.py's build()/update()
duality, but reads from the site's existing L1 store (orchestration/
build_L1_zarr.py) instead of rebuilding from raw data — L1 is already fully
built, so this is a store-to-store transform via qc_pipeline.apply_qc, not
another dataframe_builder/dataset_builder pass.

`build()` does a full rebuild: read the entire L1 store, run QC, overwrite
the L2 store. `update()` is the cheap incremental path: checkpoint off the
L2 store's own last timestamp, read only the new tail from L1 (plus a
lookback window sized to the largest configured mad_filter.window_days, so
the despike test gets a genuine neighbourhood rather than being starved by a
handful of new records — see update()'s docstring), and append. Falls back
to a full build() on any exception, same pattern as L1's update().

Usage (from project root, with ep_cntl activated):
    python -m orchestration.build_L2_zarr SITE_NAME [--update] [--output-dir DIR]
"""

import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from domain.constants import DATA_TIME_FORMAT
from infrastructure import file_io, paths
from orchestration import qc_pipeline
from services.metadata import qc_config_schema

logger = logging.getLogger(__name__)


def _resolve_store_path(
    site_name: str, output_dir: pathlib.Path | str | None
) -> pathlib.Path:
    """Resolve the target L2 Zarr store path for a site."""
    if output_dir is None:
        output_dir = paths.get_local_stream_path("homogenised_data", "zarr") / "L2"
    return pathlib.Path(output_dir) / f"{site_name}_L2.zarr"


def _resolve_l1_store_path(
    site_name: str, l1_dir: pathlib.Path | str | None
) -> pathlib.Path:
    """Resolve the source L1 Zarr store path for a site."""
    if l1_dir is None:
        l1_dir = paths.get_local_stream_path("homogenised_data", "zarr") / "L1"
    return pathlib.Path(l1_dir) / f"{site_name}_L1.zarr"


def _last_store_timestamp(store_path: pathlib.Path) -> pd.Timestamp | None:
    """Return the store's last timestamp, or None if the store doesn't exist yet."""
    if not store_path.exists():
        return None
    return pd.Timestamp(xr.open_zarr(store_path)["time"].values[-1])


def _checkable_variables(ds: xr.Dataset) -> set[str]:
    """Data variables eligible to be QC-checked: excludes flags and crs."""
    return {var for var in ds.data_vars if not var.endswith("_QCFlag") and var != "crs"}


def build(
    site_name: str,
    output_dir: pathlib.Path | str | None = None,
    l1_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Build a single, whole-history L2 Zarr store for a site (full rebuild).

    Reads the entire L1 store, runs the site's QC config over it, and
    overwrites the L2 store in full. Used to seed a new store and for a
    periodic full-rebuild reconciliation pass — see update() for the cheap
    incremental path.

    Args:
        site_name: registered site name.
        output_dir: directory to write the L2 store into. Defaults to the
            homogenised_data/zarr/L2 stream path.
        l1_dir: directory containing the source L1 store. Defaults to the
            homogenised_data/zarr/L1 stream path.

    Returns:
        Path to the L2 Zarr store written.
    """
    store_path = _resolve_store_path(site_name, output_dir)
    l1_path = _resolve_l1_store_path(site_name, l1_dir)

    ds = xr.open_zarr(l1_path)
    qc_config = qc_config_schema.load_qc_config(site_name)
    qc_config_schema.validate_qc_config_variables(qc_config, _checkable_variables(ds))

    ds = qc_pipeline.apply_qc(ds, qc_config)
    file_io.write_zarr(ds=ds, store_path=store_path)

    return store_path


def update(
    site_name: str,
    output_dir: pathlib.Path | str | None = None,
    l1_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Incrementally update a site's L2 Zarr store with new L1 records only.

    Reads the L2 store's last timestamp as a checkpoint. Rather than reading
    just the new L1 tail, also reads back a lookback window sized to the
    largest configured mad_filter.window_days across the site's QC config: a
    windowed despike test run on only a handful of newly-arrived records
    would have no real neighbourhood to test against (the algorithm's core
    windowing assumption would be violated, not just shifted slightly), so
    QC runs over lookback+tail and only the genuinely-new records are then
    appended. This isn't bit-identical to a full rebuild — window boundaries
    still won't line up exactly — but it's bounded drift of the same kind the
    periodic full-rebuild reconciliation pass already exists to correct, not
    a correctness bug. Sites/variables with no mad_filter configured pay no
    cost: lookback_days is 0 and this degenerates to a tail-only read.

    If the store doesn't exist yet, seeds it with a full build(). If the
    incremental path fails for any reason, falls back to a full rebuild for
    this cycle, mirroring build_L1_zarr.update()'s fallback pattern.

    Args:
        site_name: registered site name.
        output_dir: directory containing the L2 store. Defaults to the
            homogenised_data/zarr/L2 stream path.
        l1_dir: directory containing the source L1 store. Defaults to the
            homogenised_data/zarr/L1 stream path.

    Returns:
        Path to the L2 Zarr store (updated in place, or freshly built).
    """
    store_path = _resolve_store_path(site_name, output_dir)
    l1_path = _resolve_l1_store_path(site_name, l1_dir)

    if not store_path.exists():
        return build(site_name, output_dir=output_dir, l1_dir=l1_dir)

    try:
        checkpoint_ts = _last_store_timestamp(store_path)
        qc_config = qc_config_schema.load_qc_config(site_name)

        lookback_days = max(
            (
                spec.mad_filter.window_days
                for spec in qc_config.variables.values()
                if spec.mad_filter is not None
            ),
            default=0,
        )

        l1_ds = xr.open_zarr(l1_path)
        time_step = int(l1_ds.attrs["time_step"])
        read_start = checkpoint_ts - pd.Timedelta(days=lookback_days)
        tail_start = checkpoint_ts + pd.Timedelta(minutes=time_step)

        ds = l1_ds.sel(time=slice(read_start, None))
        if ds.sizes["time"] == 0 or ds.time.values[-1] <= np.datetime64(checkpoint_ts):
            return store_path

        qc_config_schema.validate_qc_config_variables(
            qc_config, _checkable_variables(ds)
        )
        ds = qc_pipeline.apply_qc(ds, qc_config)

        ds = ds.sel(time=slice(tail_start, None))
        if ds.sizes["time"] == 0:
            return store_path

        ds = _assign_L2_tail_attrs(ds, store_path=store_path)
        file_io.append_zarr(ds=ds, store_path=store_path)
    except Exception:
        logger.exception(
            "Incremental L2 Zarr update failed for %s, falling back to full rebuild",
            site_name,
        )
        return build(site_name, output_dir=output_dir, l1_dir=l1_dir)

    return store_path


def _assign_L2_tail_attrs(ds: xr.Dataset, store_path: pathlib.Path) -> xr.Dataset:
    """Refresh whole-history attrs on a tail slice before it's appended.

    apply_qc doesn't touch ds.attrs, so a tail slice still carries L1's
    whole-history time_coverage_start/nc_nrecs — appending it unmodified
    would let append_zarr overwrite the store's own combined attrs with
    L1's, jumping time_coverage_start forward and understating nc_nrecs.
    Mirrors build_L1_zarr.assign_L1_data_full_attrs's tail-attrs handling.
    """
    existing_attrs = dict(xr.open_zarr(store_path).attrs)
    range_start = existing_attrs["time_coverage_start"]
    nrecs_base = int(existing_attrs.get("nc_nrecs", 0))

    ds.attrs.update(
        {
            "nc_nrecs": nrecs_base + ds.sizes["time"],
            "time_coverage_start": range_start,
            "time_coverage_end": pd.to_datetime(ds.time.values[-1]).strftime(
                DATA_TIME_FORMAT
            ),
        }
    )
    return ds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply L2 QC to a site's L1 Zarr store, building/updating an L2 store."
    )
    parser.add_argument("site_name", help="Registered site name, e.g. Dookie2.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incrementally append new records instead of a full rebuild.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Override L2 output directory (default: homogenised_data/zarr/L2 stream).",
    )
    parser.add_argument(
        "--l1-dir",
        type=pathlib.Path,
        default=None,
        help="Override source L1 directory (default: homogenised_data/zarr/L1 stream).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    if args.update:
        store_path = update(
            site_name=args.site_name,
            output_dir=args.output_dir,
            l1_dir=args.l1_dir,
        )
    else:
        store_path = build(
            site_name=args.site_name,
            output_dir=args.output_dir,
            l1_dir=args.l1_dir,
        )
    print(f"Wrote Zarr store: {store_path}")


if __name__ == "__main__":
    main()

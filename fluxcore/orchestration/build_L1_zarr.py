#!/usr/bin/env python3
"""Build and incrementally update a site's whole-history L1 Zarr store.

One store per site, covering every data year. `build()` is a full rebuild
from all raw data (to seed a store, or reconcile it periodically);
`update()` is the cheap incremental path: it reads the store's last
timestamp as a checkpoint, then loads and appends only newer records.

`instrument_history` is stored in full structured form (JSON-safe, see
dataset_finishing.encode_instrument_history), not flattened and clipped to a
year as in NetCDF; the NetCDF exporter decodes it and does the clipping.

Takes a SiteContext and an explicit output directory, never a site name to
resolve.
"""

import logging
import pathlib

import pandas as pd
import xarray as xr

from fluxcore.infrastructure import file_io
from fluxcore.orchestration import incremental_zarr
from fluxcore.orchestration.dataset_builder import build_dataset_from_context
from fluxcore.orchestration.dataset_finishing import (
    add_missing_value_flags,
    add_spatial_dims,
    assign_coverage_attrs,
    assign_crs_variable,
    assign_global_attrs,
    cf_normalise_units,
    encode_instrument_history,
    filter_variable_attrs,
    flatten_instrument_uri,
)
from fluxcore.services.metadata.core.site_metadata_builder import SiteContext

logger = logging.getLogger(__name__)


def build(
    ctx: SiteContext,
    output_dir: pathlib.Path | str,
    year: int | None = None,
    global_attrs: dict | None = None,
) -> pathlib.Path:
    """Build a single, whole-history L1 Zarr store for a site (full rebuild).

    Reads all raw data (or all data from `year` onward), rebuilds the
    dataset from scratch, and overwrites the store in full. Used to seed a
    new store and for the periodic full-rebuild reconciliation pass — see
    `update()` for the cheap incremental path used on the 30-min cadence.
    Stamps the store with a hash of the site's assembled runtime config
    (see `incremental_zarr.hash_config`), which `update()` checks on every
    cycle to detect config changes that would otherwise leave already-
    stored history silently stale.

    Args:
        ctx: the site's assembled runtime config and metadata.
        output_dir: directory to write the store into.
        year: if provided, discard all records before this calendar year
            during data loading.
        global_attrs: extra global attributes to stamp on the store (see
            dataset_finishing.assign_global_attrs).

    Returns:
        Path to the Zarr store written.
    """
    store_path = incremental_zarr.store_path_for(
        ctx.runtime_config.site_name, output_dir, "L1"
    )

    start_date = pd.Timestamp(year, 1, 1) if year is not None else None
    ds = build_dataset_from_context(ctx=ctx, start_date=start_date)
    ds = _add_dims_crs_and_global_attrs(ds, global_attrs=global_attrs)
    ds = _finish_for_store(ds)
    ds.attrs["config_hash"] = incremental_zarr.hash_config(ctx.runtime_config)

    file_io.write_zarr(ds=ds, store_path=store_path)

    return store_path


def update(
    ctx: SiteContext,
    output_dir: pathlib.Path | str,
    global_attrs: dict | None = None,
) -> pathlib.Path:
    """Incrementally update a site's L1 Zarr store with new records only.

    Reads the store's last timestamp as a checkpoint and loads only records
    after it — `dataset_builder`/`raw_data_loader` push that filter down to
    a tail-peek + seeked read at the raw-file level, so this is cheap
    regardless of total site history length, unlike `build()`. Appends just
    that tail slice rather than rewriting the whole store.

    If the store doesn't exist yet, or the site's assembled runtime config
    no longer matches the hash stamped on the store (a site-config edit
    changed how history should be processed — unit conversion, calibration,
    instrument-period boundary, rename, etc.), seeds/reseeds it with a full
    `build()`. If the incremental path fails for any reason — most notably
    a site-config change that added or removed a variable, so the tail's
    schema no longer matches the store's, or a store that's already
    internally inconsistent (e.g. a prior cycle appended some variables but
    not others) — falls back to a full rebuild for this cycle rather than
    leaving the store stale or raising. This mirrors
    `raw_data_loader.load_raw_data_since`'s fallback-on-exception pattern.
    The checkpoint read itself is inside this fallback: a store an earlier
    bug left inconsistent would otherwise fail there on every cycle,
    forever, instead of self-healing. See `fluxcore.orchestration.incremental_zarr`
    for the shared envelope this delegates to (shared with
    `build_L2_zarr.update()`).

    Args:
        ctx: the site's assembled runtime config and metadata.
        output_dir: directory containing the store.
        global_attrs: extra global attributes to stamp on the store (see
            dataset_finishing.assign_global_attrs).

    Returns:
        Path to the Zarr store (updated in place, or freshly built).
    """
    site_name = ctx.runtime_config.site_name
    store_path = incremental_zarr.store_path_for(site_name, output_dir, "L1")

    def produce_tail(checkpoint: pd.Timestamp) -> xr.Dataset | None:
        time_step = int(xr.open_zarr(store_path).attrs["time_step"])
        start_date = checkpoint + pd.Timedelta(minutes=time_step)

        ds = build_dataset_from_context(ctx=ctx, start_date=start_date)
        if ds.sizes["time"] == 0:
            return None

        ds = _add_dims_crs_and_global_attrs(ds, global_attrs=global_attrs)
        return _finish_for_store(ds, append_to=store_path)

    def full_rebuild() -> pathlib.Path:
        return build(ctx, output_dir=output_dir, global_attrs=global_attrs)

    return incremental_zarr.incremental_update(
        store_path=store_path,
        config_hash=incremental_zarr.hash_config(ctx.runtime_config),
        produce_tail=produce_tail,
        full_rebuild=full_rebuild,
        logger=logger,
        label=f"L1 {site_name}",
    )


def _add_dims_crs_and_global_attrs(ds, global_attrs: dict | None = None):
    """Apply spatial dims, CRS variable, and generic global attrs to the dataset."""
    ds = add_spatial_dims(ds=ds)
    ds = assign_crs_variable(ds=ds)
    ds = assign_global_attrs(ds=ds, global_attrs=global_attrs)

    return ds


def _finish_for_store(ds, append_to: pathlib.Path | None = None):
    """Apply the flags/attrs/serialization pipeline for writing ds to a store.

    Covers the whole store rather than one calendar year: no time slicing,
    and title/coverage attrs span the store's full time range. With
    `append_to`, ds is a tail slice bound for that existing store, so the
    coverage attrs combine the store's recorded start and record count with
    the tail's; otherwise appending would make time_coverage_start jump
    forward to the tail's start. instrument_history needs no such combining:
    it's config-derived and stored in full structured form (see
    `encode_instrument_history`).
    """
    range_start, nrecs_base = (
        incremental_zarr.store_coverage(append_to) if append_to else (None, 0)
    )

    ds = add_missing_value_flags(ds)
    ds = assign_coverage_attrs(
        ds,
        title=f"Flux tower data set from the {ds.site_name} site",
        range_start=range_start,
        nrecs_base=nrecs_base,
    )
    ds = filter_variable_attrs(ds=ds)
    ds = flatten_instrument_uri(ds=ds)
    ds = encode_instrument_history(ds=ds)
    ds = cf_normalise_units(ds=ds)
    ds = file_io.serialize_dataset_attrs(ds=ds)

    return ds

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

Generic core: takes an already-loaded SiteQCConfig and explicit store
directories. The TERN site-name wrappers and CLI live in
orchestration/build_runner.py.
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from fluxcore.domain.constants import DATA_TIME_FORMAT
from fluxcore.infrastructure import file_io
from fluxcore.orchestration import incremental_zarr, qc_pipeline
from fluxcore.services.metadata.core import qc_config_schema
from fluxcore.services.metadata.core.qc_config_schema import SiteQCConfig

logger = logging.getLogger(__name__)

PROCESSING_LEVEL = "L2"


def store_path_for(site_name: str, output_dir: pathlib.Path | str) -> pathlib.Path:
    """Path of a site's L2 Zarr store within output_dir."""
    return pathlib.Path(output_dir) / f"{site_name}_L2.zarr"


def _l1_store_path(site_name: str, l1_dir: pathlib.Path | str) -> pathlib.Path:
    """Path of a site's source L1 Zarr store within l1_dir."""
    return pathlib.Path(l1_dir) / f"{site_name}_L1.zarr"


def build(
    qc_config: SiteQCConfig,
    output_dir: pathlib.Path | str,
    l1_dir: pathlib.Path | str,
) -> pathlib.Path:
    """Build a single, whole-history L2 Zarr store for a site (full rebuild).

    Reads the entire L1 store, runs the site's QC config over it, and
    overwrites the L2 store in full. Used to seed a new store and for a
    periodic full-rebuild reconciliation pass — see update() for the cheap
    incremental path. Stamps the store with a hash of the site's assembled
    QC config (site overrides + shared defaults — see
    `incremental_zarr.hash_config`), which `update()` checks on every cycle
    to detect config changes that would otherwise leave already-stored
    history silently stale.

    Args:
        qc_config: the site's own QC config (overrides only; the shared
            defaults are loaded and merged here). Its site_name names the
            stores ({site_name}_L1.zarr in, {site_name}_L2.zarr out).
        output_dir: directory to write the L2 store into.
        l1_dir: directory containing the source L1 store.

    Returns:
        Path to the L2 Zarr store written.
    """
    store_path = store_path_for(qc_config.site_name, output_dir)
    l1_path = _l1_store_path(qc_config.site_name, l1_dir)

    ds = xr.open_zarr(l1_path)
    qc_config_schema.validate_qc_config_variables(
        qc_config, qc_pipeline.checkable_variables(ds)
    )

    dependency_defaults = qc_config_schema.load_dependency_defaults()
    range_defaults = qc_config_schema.load_range_defaults()
    resolved_qc_config = qc_config_schema.resolve_qc_config(
        qc_config, dependency_defaults, range_defaults, ds
    )
    ds = qc_pipeline.apply_qc(ds, resolved_qc_config)
    ds.attrs["processing_level"] = PROCESSING_LEVEL
    ds.attrs["config_hash"] = incremental_zarr.hash_config(
        qc_config, dependency_defaults, range_defaults
    )
    file_io.write_zarr(ds=ds, store_path=store_path)

    return store_path


def update(
    qc_config: SiteQCConfig,
    output_dir: pathlib.Path | str,
    l1_dir: pathlib.Path | str,
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
    cost: lookback_days is 0 and this degenerates to a tail-only read. When a
    mad_filter is configured the newest L1 record is held back (not appended)
    until its successor arrives, since the filter can't test a record with no
    later neighbour -- the L2 store then lags L1 by one record.

    If the store doesn't exist yet, or the site's assembled QC config no
    longer matches the hash stamped on the store (a QC-config edit changed
    a range bound, dependency, flag rule, etc.), seeds/reseeds it with a
    full build(). If the incremental path fails for any reason, falls back
    to a full rebuild for this cycle, mirroring build_L1_zarr.update()'s
    fallback pattern. See `fluxcore.orchestration.incremental_zarr` for the shared
    envelope this delegates to (shared with `build_L1_zarr.update()`).

    Args:
        qc_config: the site's own QC config (see build()).
        output_dir: directory containing the L2 store.
        l1_dir: directory containing the source L1 store.

    Returns:
        Path to the L2 Zarr store (updated in place, or freshly built).
    """
    store_path = store_path_for(qc_config.site_name, output_dir)
    l1_path = _l1_store_path(qc_config.site_name, l1_dir)

    dependency_defaults = qc_config_schema.load_dependency_defaults()
    range_defaults = qc_config_schema.load_range_defaults()

    def produce_tail(checkpoint: pd.Timestamp) -> xr.Dataset | None:
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
        read_start = checkpoint - pd.Timedelta(days=lookback_days)
        tail_start = checkpoint + pd.Timedelta(minutes=time_step)

        ds = l1_ds.sel(time=slice(read_start, None))
        if ds.sizes["time"] == 0 or ds.time.values[-1] <= np.datetime64(checkpoint):
            return None

        qc_config_schema.validate_qc_config_variables(
            qc_config, qc_pipeline.checkable_variables(ds)
        )
        resolved_qc_config = qc_config_schema.resolve_qc_config(
            qc_config, dependency_defaults, range_defaults, ds
        )
        ds = qc_pipeline.apply_qc(ds, resolved_qc_config)

        ds = ds.sel(time=slice(tail_start, None))
        if lookback_days > 0:
            # The mad_filter can't test the newest record (its second
            # difference needs a successor), and an appended record is never
            # re-evaluated, so hold it back until the next cycle when it is
            # no longer the last record of the QC input.
            ds = ds.isel(time=slice(None, -1))
        if ds.sizes["time"] == 0:
            return None

        return _assign_L2_tail_attrs(ds, store_path=store_path)

    def full_rebuild() -> pathlib.Path:
        return build(qc_config, output_dir=output_dir, l1_dir=l1_dir)

    return incremental_zarr.incremental_update(
        store_path=store_path,
        config_hash=incremental_zarr.hash_config(
            qc_config, dependency_defaults, range_defaults
        ),
        produce_tail=produce_tail,
        full_rebuild=full_rebuild,
        logger=logger,
        label=f"L2 {qc_config.site_name}",
    )


def _assign_L2_tail_attrs(ds: xr.Dataset, store_path: pathlib.Path) -> xr.Dataset:
    """Refresh whole-history attrs on a tail slice before it's appended.

    apply_qc doesn't touch ds.attrs, so a tail slice still carries L1's
    processing_level and whole-history time_coverage_start/nc_nrecs —
    appending it unmodified would let append_zarr overwrite the store's own
    combined attrs with L1's, relabelling the store L1, jumping
    time_coverage_start forward and understating nc_nrecs.
    Mirrors build_L1_zarr.assign_L1_data_full_attrs's tail-attrs handling.
    """
    existing_attrs = dict(xr.open_zarr(store_path).attrs)
    range_start = existing_attrs["time_coverage_start"]
    nrecs_base = int(existing_attrs.get("nc_nrecs", 0))

    ds.attrs.update(
        {
            "processing_level": PROCESSING_LEVEL,
            "nc_nrecs": nrecs_base + ds.sizes["time"],
            "time_coverage_start": range_start,
            "time_coverage_end": pd.to_datetime(ds.time.values[-1]).strftime(
                DATA_TIME_FORMAT
            ),
        }
    )
    return ds

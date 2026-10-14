#!/usr/bin/env python3
"""Export an L1 xarray Dataset to a single whole-history Zarr store per site.

Wired into the production task registry as construct_L1_zarr (30-min
incremental) and rebuild_L1_zarr (nightly full-rebuild reconciliation) —
see tasks/build_tasks.py. construct_L1_nc (build_L1_nc.build_from_zarr)
now reads its NetCDF output from this store instead of rebuilding from raw
data. Mirrors build_L1_nc.py's structure and reuses its dataset-preparation
functions where they aren't year-scoped, writing one Zarr store per site
(covering all data years) instead of one NetCDF file per calendar year.

`build()` does a full rebuild from all raw data (used to seed a new store,
and for the periodic full-rebuild reconciliation pass). `update()` is the
cheap 30-minute-cadence path: reads the store's last timestamp as a
checkpoint, loads and appends only newer records.

Unlike build_L1_nc.py's NetCDF path, `instrument_history` is stored here
in full structured form (JSON-safe: timestamps as ISO strings), not
flattened to a year-clipped string — Zarr attrs are JSON-native, so this
survives the round trip natively. NetCDF export re-hydrates it and
reproduces the year clipping itself; see `_json_safe_instrument_history`.

Generic core: takes a SiteContext and an explicit output directory. The TERN
site-name wrappers and CLI live in orchestration/build_runner.py.
"""

import logging
import pathlib

import pandas as pd
import xarray as xr

from fluxcore.domain.constants import DATA_TIME_FORMAT
from fluxcore.infrastructure import file_io
from fluxcore.orchestration import incremental_zarr
from fluxcore.orchestration.build_L1_nc import (
    assign_crs_variable,
    assign_L1_global_generic_attrs,
    assign_variable_flags,
    do_dim_ops,
    filter_variable_attrs,
    serialize_units,
    serialize_uri,
)
from fluxcore.orchestration.dataset_builder import build_dataset_from_context
from fluxcore.services.metadata.core.site_metadata_builder import SiteContext

logger = logging.getLogger(__name__)


def store_path_for(site_name: str, output_dir: pathlib.Path | str) -> pathlib.Path:
    """Path of a site's L1 Zarr store within output_dir."""
    return pathlib.Path(output_dir) / f"{site_name}_L1.zarr"


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
            build_L1_nc.assign_L1_global_generic_attrs).

    Returns:
        Path to the Zarr store written.
    """
    store_path = store_path_for(ctx.runtime_config.site_name, output_dir)

    start_date = pd.Timestamp(year, 1, 1) if year is not None else None
    ds = build_dataset_from_context(ctx=ctx, start_date=start_date)
    ds = build_L1_ds_complete(ds, global_attrs=global_attrs)
    ds = build_L1_ds_full(ds)
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
            build_L1_nc.assign_L1_global_generic_attrs).

    Returns:
        Path to the Zarr store (updated in place, or freshly built).
    """
    site_name = ctx.runtime_config.site_name
    store_path = store_path_for(site_name, output_dir)

    def produce_tail(checkpoint: pd.Timestamp) -> xr.Dataset | None:
        time_step = int(xr.open_zarr(store_path).attrs["time_step"])
        start_date = checkpoint + pd.Timedelta(minutes=time_step)

        ds = build_dataset_from_context(ctx=ctx, start_date=start_date)
        if ds.sizes["time"] == 0:
            return None

        ds = build_L1_ds_complete(ds, global_attrs=global_attrs)
        return build_L1_ds_tail(ds, store_path=store_path)

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


def build_L1_ds_complete(ds, global_attrs: dict | None = None):
    """Apply spatial dims, CRS variable, and generic global attrs to the dataset.

    Reused unchanged from build_L1_nc — none of these steps are year-scoped.
    """
    ds = do_dim_ops(ds=ds)
    ds = assign_crs_variable(ds=ds)
    ds = assign_L1_global_generic_attrs(ds=ds, global_attrs=global_attrs)

    return ds


def build_L1_ds_full(ds):
    """Apply the flags/attrs/serialization pipeline over a whole-history dataset.

    Adaptation of build_L1_nc.build_L1_ds_by_year for a single store spanning
    every data year instead of one file per year: no time slicing, and the
    two steps that clip to a calendar year (title/coverage attrs, instrument
    history serialization) clip to the dataset's actual full time range
    instead. Used by `build()`. See `build_L1_ds_tail` for the counterpart
    used by `update()`, where the whole-history range must come from the
    existing store rather than from `ds` itself.
    """
    ds = assign_variable_flags(ds)
    ds = assign_L1_data_full_attrs(ds=ds)
    ds = filter_variable_attrs(ds=ds)
    ds = serialize_uri(ds=ds)
    ds = _json_safe_instrument_history(ds=ds)
    ds = serialize_units(ds=ds)
    ds = file_io.serialize_dataset_attrs(ds=ds)

    return ds


def build_L1_ds_tail(ds, store_path: pathlib.Path):
    """Apply the flags/attrs/serialization pipeline to a tail slice being appended.

    Same steps as `build_L1_ds_full`, except the whole-history attrs
    (title, `nc_nrecs`, time coverage) are computed against the *combined*
    range — the existing store's recorded start and record count, plus
    this tail slice — rather than the tail's own local bounds. Without
    this, appending would make `time_coverage_start` jump forward to the
    tail's start. `instrument_history` needs no such combining: it's
    config-derived (independent of the queried date range) and stored in
    full structured form via `_json_safe_instrument_history`, not clipped
    at write time — see that function's docstring.

    Args:
        ds: tail-slice dataset, already passed through `build_L1_ds_complete`.
        store_path: path to the existing store being appended to, read here
            only for its current global attrs (cheap — no data load).
    """
    existing_attrs = dict(xr.open_zarr(store_path).attrs)
    range_start = existing_attrs["time_coverage_start"]
    nrecs_base = int(existing_attrs["nc_nrecs"])

    ds = assign_variable_flags(ds)
    ds = assign_L1_data_full_attrs(
        ds=ds, range_start=range_start, nrecs_base=nrecs_base
    )
    ds = filter_variable_attrs(ds=ds)
    ds = serialize_uri(ds=ds)
    ds = _json_safe_instrument_history(ds=ds)
    ds = serialize_units(ds=ds)
    ds = file_io.serialize_dataset_attrs(ds=ds)

    return ds


def assign_L1_data_full_attrs(ds, range_start=None, nrecs_base: int = 0):
    """Add whole-history global attrs (title, record count, time coverage) to ds.

    Whole-history counterpart of build_L1_nc.assign_L1_data_year_attrs.

    Args:
        ds: dataset — the full history for `build()`, or a tail slice for
            `update()`.
        range_start: if given, used as the store's `time_coverage_start`
            instead of `ds.time.values[0]` — pass the *existing* store's
            recorded start when `ds` is only a tail slice.
        nrecs_base: record count already present in the store before this
            call (0 for a full build); added to `len(ds.time)` for
            `nc_nrecs`.
    """

    def _format_time(x):
        return pd.to_datetime(x).strftime(DATA_TIME_FORMAT)

    begin = _format_time(range_start if range_start is not None else ds.time.values[0])
    end = _format_time(ds.time.values[-1])

    ds.attrs.update(
        {
            "title": f"Flux tower data set from the {ds.site_name} site",
            "nc_nrecs": nrecs_base + len(ds.time),
            "time_coverage_start": begin,
            "time_coverage_end": end,
        }
    )

    return ds


def _json_safe_instrument_history(ds):
    """Make each variable's instrument/instrument_history attrs Zarr-safe.

    Converts nested start_date/end_date timestamps to ISO strings without
    collapsing or year-clipping the structure — unlike
    build_L1_nc.serialize_inst_history, which flattens *and* clips to one
    calendar year. Preserving the structure here (rather than flattening
    at Zarr-write time) is what lets NetCDF export reproduce that per-year
    clipping later, from data that's otherwise already fully processed by
    the time it reaches this store — see build_L1_nc.build_from_zarr and
    its _rehydrate_instrument_history counterpart.

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
    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
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

#!/usr/bin/env python3
"""Export an L1 xarray Dataset to a single whole-history Zarr store per site.

Wired into the production task registry as construct_L1_zarr (30-min
incremental) and rebuild_L1_zarr (nightly full-rebuild reconciliation) —
see tasks/build_tasks.py. Runs in parallel with the existing NetCDF export
(construct_L1_nc); NetCDF still reads raw data directly for now, not this
store — that cutover is a deferred follow-up once this has been proven
against real production data. Mirrors build_L1_nc.py's structure and
reuses its dataset-preparation functions where they aren't year-scoped,
writing one Zarr store per site (covering all data years) instead of one
NetCDF file per calendar year.

`build()` does a full rebuild from all raw data (used to seed a new store,
and for the periodic full-rebuild reconciliation pass). `update()` is the
cheap 30-minute-cadence path: reads the store's last timestamp as a
checkpoint, loads and appends only newer records.

Usage (from project root, with ep_cntl activated):
    python -m orchestration.build_L1_zarr SITE_NAME [--update] [--year YEAR] \
        [--output-dir DIR]
"""

import argparse
import logging
import pathlib

import pandas as pd
import xarray as xr

from domain.constants import DATA_TIME_FORMAT
from infrastructure import file_io, paths
from orchestration import dataset_builder
from orchestration.build_L1_nc import (
    _serialise_compound_history,
    _serialise_simple_history,
    assign_crs_variable,
    assign_L1_global_generic_attrs,
    assign_valid_range,
    assign_variable_flags,
    do_dim_ops,
    filter_variable_attrs,
    serialize_units,
    serialize_uri,
)

logger = logging.getLogger(__name__)


def _resolve_store_path(
    site_name: str, output_dir: pathlib.Path | str | None
) -> pathlib.Path:
    """Resolve the target Zarr store path for a site, applying the default stream."""
    if output_dir is None:
        output_dir = paths.get_local_stream_path("homogenised_data", "zarr") / "L1"
    return pathlib.Path(output_dir) / f"{site_name}_L1.zarr"


def _last_store_timestamp(store_path: pathlib.Path) -> pd.Timestamp | None:
    """Return the store's last timestamp, or None if the store doesn't exist yet."""
    if not store_path.exists():
        return None
    return pd.Timestamp(xr.open_zarr(store_path)["time"].values[-1])


def build(
    site_name: str,
    output_dir: pathlib.Path | str | None = None,
    year: int | None = None,
    legacy: bool = False,
) -> pathlib.Path:
    """Build a single, whole-history L1 Zarr store for a site (full rebuild).

    Reads all raw data (or all data from `year` onward), rebuilds the
    dataset from scratch, and overwrites the store in full. Used to seed a
    new store and for the periodic full-rebuild reconciliation pass — see
    `update()` for the cheap incremental path used on the 30-min cadence.

    Args:
        site_name: registered site name.
        output_dir: directory to write the store into. Defaults to the
            homogenised_data/zarr/L1 stream path.
        year: if provided, discard all records before this calendar year
            during data loading.
        legacy: if True, build from the site's legacy config snapshot
            instead of its operational config.

    Returns:
        Path to the Zarr store written.
    """
    store_path = _resolve_store_path(site_name, output_dir)

    start_date = pd.Timestamp(year, 1, 1) if year is not None else None
    ds = dataset_builder.build_dataset_from_site_name(
        site_name, start_date=start_date, legacy=legacy
    )
    ds = build_L1_ds_complete(ds)
    ds = build_L1_ds_full(ds)

    file_io.write_zarr(ds=ds, store_path=store_path)

    return store_path


def update(
    site_name: str,
    output_dir: pathlib.Path | str | None = None,
    legacy: bool = False,
) -> pathlib.Path:
    """Incrementally update a site's L1 Zarr store with new records only.

    Reads the store's last timestamp as a checkpoint and loads only records
    after it — `dataset_builder`/`raw_data_loader` push that filter down to
    a tail-peek + seeked read at the raw-file level, so this is cheap
    regardless of total site history length, unlike `build()`. Appends just
    that tail slice rather than rewriting the whole store.

    If the store doesn't exist yet, seeds it with a full `build()`. If the
    incremental path fails for any reason — most notably a site-config
    change that added or removed a variable, so the tail's schema no longer
    matches the store's — falls back to a full rebuild for this cycle
    rather than leaving the store stale or raising. This mirrors
    `raw_data_loader.load_raw_data_since`'s fallback-on-exception pattern.

    Args:
        site_name: registered site name.
        output_dir: directory containing the store. Defaults to the
            homogenised_data/zarr/L1 stream path.
        legacy: if True, build from the site's legacy config snapshot
            instead of its operational config.

    Returns:
        Path to the Zarr store (updated in place, or freshly built).
    """
    store_path = _resolve_store_path(site_name, output_dir)
    last_ts = _last_store_timestamp(store_path)

    if last_ts is None:
        return build(site_name, output_dir=output_dir, legacy=legacy)

    try:
        time_step = int(xr.open_zarr(store_path).attrs["time_step"])
        start_date = last_ts + pd.Timedelta(minutes=time_step)

        ds = dataset_builder.build_dataset_from_site_name(
            site_name, start_date=start_date, legacy=legacy
        )
        if ds.sizes["time"] == 0:
            return store_path

        ds = build_L1_ds_complete(ds)
        ds = build_L1_ds_tail(ds, store_path=store_path)
        file_io.append_zarr(ds=ds, store_path=store_path)
    except Exception:
        logger.exception(
            "Incremental Zarr update failed for %s, falling back to full rebuild",
            site_name,
        )
        return build(site_name, output_dir=output_dir, legacy=legacy)

    return store_path


def build_L1_ds_complete(ds):
    """Apply spatial dims, CRS variable, and generic global attrs to the dataset.

    Reused unchanged from build_L1_nc — none of these steps are year-scoped.
    """
    ds = do_dim_ops(ds=ds)
    ds = assign_crs_variable(ds=ds)
    ds = assign_L1_global_generic_attrs(ds=ds)

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
    ds = assign_valid_range(ds)
    ds = assign_L1_data_full_attrs(ds=ds)
    ds = filter_variable_attrs(ds=ds)
    ds = serialize_uri(ds=ds)
    ds = serialize_inst_history_full(ds=ds)
    ds = serialize_units(ds=ds)
    ds = file_io.serialize_dataset_attrs(ds=ds)

    return ds


def build_L1_ds_tail(ds, store_path: pathlib.Path):
    """Apply the flags/attrs/serialization pipeline to a tail slice being appended.

    Same steps as `build_L1_ds_full`, except the whole-history attrs
    (title, `nc_nrecs`, time coverage, instrument history) are computed
    against the *combined* range — the existing store's recorded start and
    record count, plus this tail slice — rather than the tail's own local
    bounds. Without this, appending would make `time_coverage_start` jump
    forward to the tail's start and `serialize_inst_history_full` would
    drop instrument eras that predate the tail entirely.

    Args:
        ds: tail-slice dataset, already passed through `build_L1_ds_complete`.
        store_path: path to the existing store being appended to, read here
            only for its current global attrs (cheap — no data load).
    """
    existing_attrs = dict(xr.open_zarr(store_path).attrs)
    range_start = existing_attrs["time_coverage_start"]
    nrecs_base = int(existing_attrs["nc_nrecs"])

    ds = assign_variable_flags(ds)
    ds = assign_valid_range(ds)
    ds = assign_L1_data_full_attrs(
        ds=ds, range_start=range_start, nrecs_base=nrecs_base
    )
    ds = filter_variable_attrs(ds=ds)
    ds = serialize_uri(ds=ds)
    ds = serialize_inst_history_full(ds=ds, range_start=range_start)
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


def serialize_inst_history_full(ds, range_start=None):
    """Serialize instrument-changeover history clipped to the dataset's full time range.

    Whole-history counterpart of build_L1_nc.serialize_inst_history — same
    logic, but the clip window is the dataset's actual first/last timestamp
    instead of one calendar year. Reuses the same private history-formatting
    helpers as the NetCDF path.

    Args:
        ds: dataset — the full history for `build()`, or a tail slice for
            `update()`.
        range_start: if given, used as the clip window's start instead of
            `ds.time.values[0]` — pass the *existing* store's recorded
            start when `ds` is only a tail slice, so instrument eras that
            predate the tail aren't dropped from the serialized history.
    """
    range_start = (
        pd.Timestamp(range_start).to_pydatetime()
        if range_start is not None
        else pd.Timestamp(ds.time.values[0]).to_pydatetime()
    )
    range_end = pd.Timestamp(ds.time.values[-1]).to_pydatetime()

    var_list = [var for var in ds.variables if var not in ds.dims]
    for var in var_list:
        attrs = ds[var].attrs

        if "instrument_history" not in attrs:
            if isinstance(attrs.get("instrument"), dict):
                attrs["instrument"] = ",".join(attrs["instrument"].values())
            continue

        history = attrs["instrument_history"]
        first_val = next(iter(history.values()))

        if "start_date" in first_val:
            # Simple: {inst_name: {start_date, end_date}}
            if isinstance(attrs.get("instrument"), dict):
                attrs["instrument"] = ",".join(attrs["instrument"].values())
            serialised, last_inst = _serialise_simple_history(
                history, range_start, range_end
            )
        else:
            # Compound: {alias: {inst_name: {start_date, end_date}}}
            serialised, last_inst = _serialise_compound_history(
                history, range_start, range_end
            )

        if not serialised:
            del attrs["instrument_history"]
        elif "start_date" in first_val:
            # Simple history
            if len(serialised) == 1:
                attrs["instrument"] = last_inst
                del attrs["instrument_history"]
            else:
                attrs["instrument_history"] = "|".join(serialised)
                if last_inst is not None:
                    attrs["instrument"] = last_inst
        else:
            # Compound history: last_inst is a dict {alias: name}
            attrs["instrument_history"] = ";".join(serialised)
            inst = attrs.get("instrument")
            current = dict(inst) if isinstance(inst, dict) else {}
            if last_inst:
                current.update(last_inst)
            attrs["instrument"] = ",".join(current.values())

    return ds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or incrementally update a site's L1 Zarr store."
    )
    parser.add_argument("site_name", help="Registered site name, e.g. Dookie2.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incrementally append new records instead of a full rebuild.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Discard records before this calendar year during data loading "
        "(--update only checkpoints off the store's own last timestamp; "
        "this is ignored with --update).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Override output directory (default: homogenised_data/zarr/L1 stream).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Build from the site's legacy config snapshot instead of operational.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    if args.update:
        store_path = update(
            site_name=args.site_name,
            output_dir=args.output_dir,
            legacy=args.legacy,
        )
    else:
        store_path = build(
            site_name=args.site_name,
            output_dir=args.output_dir,
            year=args.year,
            legacy=args.legacy,
        )
    print(f"Wrote Zarr store: {store_path}")


if __name__ == "__main__":
    main()

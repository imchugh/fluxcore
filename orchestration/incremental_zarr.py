#!/usr/bin/env python3
"""Shared incremental-update envelope for a per-site whole-history Zarr store.

Generic core (tier 1): no TERN dependency, no knowledge of L1 vs L2.
`build_L1_zarr.update()` and `build_L2_zarr.update()` each supply the
level-specific pieces (how to produce a processed tail dataset, how to do a
full rebuild) as callbacks; this module owns only the shared envelope:

    store doesn't exist?  -> full_rebuild()
    stored config hash != current config hash?  -> full_rebuild()
    try:
        checkpoint = last_store_timestamp(store)
        tail = produce_tail(checkpoint)
        if tail is empty: no-op, return store unchanged
        append tail to store
    except:
        full_rebuild()

`hash_config()` hashes the assembled config object(s) that drive processing
(not raw YAML text, so a comment or key-reorder edit doesn't force a
rebuild) -- `full_rebuild()` implementations are expected to stamp the
resulting hash onto the store's `config_hash` attr so the next cycle's
comparison is meaningful.
"""

import dataclasses
import datetime
import enum
import hashlib
import json
import logging
import pathlib
from collections.abc import Callable

import pandas as pd
import xarray as xr
from pydantic import BaseModel

from infrastructure import file_io


def last_store_timestamp(store_path: pathlib.Path) -> pd.Timestamp | None:
    """Return the store's last timestamp, or None if the store doesn't exist yet."""
    if not store_path.exists():
        return None
    return pd.Timestamp(xr.open_zarr(store_path)["time"].values[-1])


def hash_config(*configs: object) -> str:
    """Deterministic hash of one or more assembled config objects.

    Hashes the assembled object tree (dataclasses -- stdlib or pydantic --,
    pydantic models, dicts/lists/enums/paths/datetimes), not raw file text,
    so a comment or key-reorder in a YAML source doesn't change the hash.
    Pass every piece that feeds processing (e.g. a site's own QC overrides
    plus the shared defaults tables) as separate positional args; argument
    order is part of the hash, so a caller must pass them in the same order
    every time.
    """
    blob = json.dumps([_to_jsonable(c) for c in configs], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _to_jsonable(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_to_jsonable(v) for v in obj]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, default=str))
    return obj


def incremental_update(
    *,
    store_path: pathlib.Path,
    config_hash: str,
    produce_tail: Callable[[pd.Timestamp | None], xr.Dataset | None],
    full_rebuild: Callable[[], pathlib.Path],
    logger: logging.Logger,
    label: str,
) -> pathlib.Path:
    """Shared checkpoint/tail-append/fallback envelope for a Zarr `update()`.

    Args:
        store_path: path to the store being updated.
        config_hash: hash of the config that would drive a rebuild right
            now (see `hash_config`). Compared against the store's own
            stamped `config_hash` attr; a mismatch means something that
            affects historical processing changed since the store was last
            (re)built, so a full rebuild is forced rather than appending
            under stale rules. `full_rebuild` must stamp the store with
            this same hash so the next cycle's comparison is meaningful.
        produce_tail: given the store's checkpoint timestamp (None if the
            store is empty), returns the fully-processed dataset to append,
            or None (or an empty dataset) if there's nothing new yet. All
            level-specific logic -- raw data loading, QC, lookback windows,
            tail-attrs refresh -- lives in this callback.
        full_rebuild: performs a complete rebuild of the store. Used to seed
            a store that doesn't exist yet, and as the fallback on a
            config-hash mismatch or any failure below.
        logger: caller's logger, used for the fallback warning.
        label: short human-readable label for log messages (e.g.
            "L1 TestSite").

    Returns:
        Path to the store (updated in place, or freshly (re)built).
    """
    if not store_path.exists():
        return full_rebuild()

    try:
        stored_hash = xr.open_zarr(store_path).attrs.get("config_hash")
        if stored_hash != config_hash:
            logger.info(
                "Config hash changed for %s store, forcing full rebuild", label
            )
            return full_rebuild()

        checkpoint = last_store_timestamp(store_path)
        tail_ds = produce_tail(checkpoint)
        if tail_ds is None or tail_ds.sizes["time"] == 0:
            return store_path

        file_io.append_zarr(ds=tail_ds, store_path=store_path)
    except Exception:
        logger.exception(
            "Incremental %s Zarr update failed, falling back to full rebuild", label
        )
        return full_rebuild()

    return store_path

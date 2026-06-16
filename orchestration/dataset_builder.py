#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build L1 xarray Datasets from a site context.

Public API
----------
build_dataset_from_site_name(site_name, pad_humidity, pad_co2, start_date) -> xr.Dataset
build_dataset_from_context(ctx, pad_humidity, pad_co2, start_date)         -> xr.Dataset

DatasetBuildIntermediate  — exported for use by derived_quantities
"""

import datetime
import pandas as pd
import xarray as xr
from dataclasses import dataclass

from domain.enums import StatisticType
from services.metadata.site_registry import SiteRegistry, SiteContext
from services.metadata import file_group_builder
from services.metadata.variable_registry import VariableSpec, build_variable_registry
from orchestration.dataframe_builder import build_dataframe


###############################################################################
### BEGIN TYPES ###
###############################################################################

@dataclass
class DatasetBuildIntermediate:
    """Staging object carrying the canonical DataFrame and xarray variable metadata."""

    df: pd.DataFrame
    var_attrs: dict[str, dict]

###############################################################################
### END TYPES ###
###############################################################################


###############################################################################
### BEGIN INITS ###
###############################################################################

ATTRS_SUBSET = [
    'site_name', 'fluxnet_id', 'latitude', 'longitude', 'elevation',
    'time_step', 'time_zone', 'canopy_height', 'tower_height', 'soil',
    'vegetation', 'date_commissioned',
    ]

SITE_REGISTRY = SiteRegistry()

###############################################################################
### END INITS ###
###############################################################################


###############################################################################
### BEGIN PUBLIC FUNCTIONS ###
###############################################################################

def build_dataset_from_site_name(
        site_name: str,
        pad_humidity: bool = True,
        pad_co2: bool = True,
        start_date: pd.Timestamp | None = None,
        ) -> xr.Dataset:
    """Convenience wrapper — resolves site name to context via registry."""

    ctx = SITE_REGISTRY.get_context(site=site_name)
    return build_dataset_from_context(
        ctx=ctx, pad_humidity=pad_humidity, pad_co2=pad_co2, start_date=start_date
        )


def build_dataset_from_context(
        ctx: SiteContext,
        pad_humidity: bool = True,
        pad_co2: bool = True,
        start_date: pd.Timestamp | None = None,
        ) -> xr.Dataset:
    """Build an L1 xarray dataset from a fully-assembled site context."""

    result = _build_result(ctx, start_date=start_date)
    result = _apply_padding(result, pad_humidity=pad_humidity, pad_co2=pad_co2)

    ds = result.df.to_xarray()
    ds = _apply_variable_metadata(ds, result.var_attrs)
    ds = _apply_global_metadata(ds, ctx)

    return ds

###############################################################################
### END PUBLIC FUNCTIONS ###
###############################################################################


###############################################################################
### BEGIN PRIVATE FUNCTIONS — metadata application ###
###############################################################################

def _apply_padding(
        result: DatasetBuildIntermediate,
        pad_humidity: bool,
        pad_co2: bool,
        ) -> DatasetBuildIntermediate:
    # Lazy import to avoid circular dependency (derived_quantities imports
    # DatasetBuildIntermediate from this module)
    from orchestration.derived_quantities import (
        pad_humidity as _pad_humidity,
        pad_co2 as _pad_co2,
        )
    if pad_humidity:
        result = _pad_humidity(result)
    if pad_co2:
        result = _pad_co2(result)
    return result

def _apply_variable_metadata(
        ds: xr.Dataset,
        var_attrs: dict[str, dict],
        ) -> xr.Dataset:

    for variable in [v for v in ds.variables if v not in ds.dims]:
        ds[variable].attrs = {
            k: v for k, v in var_attrs[variable].items() if v is not None
            }

    return ds

def _apply_global_metadata(
        ds: xr.Dataset,
        ctx: SiteContext,
        ) -> xr.Dataset:

    for attr in ATTRS_SUBSET:
        value = ctx.metadata.get(attr)
        if value is None:
            continue
        if isinstance(value, (datetime.date, datetime.datetime)):
            value = value.isoformat()
        ds.attrs[attr] = value

    flux_system = ctx.runtime_config.flux_system
    if flux_system is not None:
        ds.attrs['flux_system'] = flux_system

    return ds

###############################################################################
### END PRIVATE FUNCTIONS — metadata application ###
###############################################################################


###############################################################################
### BEGIN PRIVATE FUNCTIONS — dataset build ###
###############################################################################

def _build_result(
        ctx: SiteContext,
        start_date: pd.Timestamp | None = None,
        ) -> DatasetBuildIntermediate:
    """
    Build a canonical dataframe and per-variable xarray attrs from a site
    context.

    Args:
        ctx: fully-assembled site context (runtime config + site metadata).
             Site metadata provides n_samples (time_step * freq_hz * 60),
             needed when COUNTER variables are stored as invalid_count.
        start_date: if provided, records before this timestamp are discarded
             after loading each file.  Use to limit processing to a single
             year without reading full site history.

    Returns:
        DatasetBuildIntermediate containing the canonical dataframe and a dict
        of per-variable xarray attribute dicts keyed by canonical name.
    """

    runtime_cfg = ctx.runtime_config

    file_groups = file_group_builder.build_file_groups(runtime_cfg)
    for group in file_groups.values():
        group.validate()

    registry = build_variable_registry(runtime_cfg=runtime_cfg, file_groups=file_groups)

    df = build_dataframe(
        file_groups=file_groups,
        registry=registry,
        n_samples=ctx.metadata.n_samples,
        start_date=start_date,
        flux_file=runtime_cfg.flux_file,
        time_step=ctx.metadata.time_step,
        )

    var_attrs = _build_var_attrs(registry=registry)

    return DatasetBuildIntermediate(df=df, var_attrs=var_attrs)


# -----------------------------------------------------------------------------
# Variable attribute construction
# -----------------------------------------------------------------------------

def _build_var_attrs(registry: dict[str, VariableSpec]) -> dict[str, dict]:
    """Build per-variable xarray attribute dicts keyed by canonical output name."""

    rslt = {}

    canonical_groups = _canonical_from_registry(registry)

    for _, var_specs in canonical_groups.items():

        main_spec = var_specs[-1]

        attrs = {
            'height':             main_spec.height,
            'height_range':       main_spec.height_range,
            'instrument':         main_spec.instrument,
            'instrument_history': _history_from_instruments(var_specs),
            'long_name':          main_spec.long_name,
            'quantity':           main_spec.quantity,
            'standard_name':      main_spec.standard_name,
            'statistic_type':     _output_statistic(main_spec),
            'units':              main_spec.canonical_units,
            }

        rslt[_canonical_output_name(main_spec)] = attrs

    return rslt


def _canonical_from_registry(
        registry: dict[str, VariableSpec],
        ) -> dict[str, list[VariableSpec]]:
    """Group registry entries by canonical variable name."""

    groups: dict[str, list[VariableSpec]] = {}
    for entry in registry.values():
        groups.setdefault(entry.canonical_name, []).append(entry)
    return groups


def _history_from_instruments(
        var_specs: list[VariableSpec],
        ) -> dict[str, dict] | None:
    """
    Return instrument changeover history, or None if no instrument changed.

    For simple (string) instruments: keyed by instrument name.
    For compound (dict) instruments: keyed by alias, containing per-instrument
    histories only for aliases where the instrument actually changed. Aliases
    that were constant across all periods are omitted.

    Dates are always explicit for multi-period variables (required for the
    merge step), so no defaults are needed.
    """

    if isinstance(var_specs[0].instrument, dict):
        return _history_from_compound_instruments(var_specs)

    instruments = {spec.instrument for spec in var_specs}
    if len(instruments) == 1:
        return None
    return {
        spec.instrument: {'start_date': spec.begin, 'end_date': spec.end}
        for spec in var_specs
        }


def _history_from_compound_instruments(
        var_specs: list[VariableSpec],
        ) -> dict[str, dict] | None:
    """
    Per-alias history for compound quantities. Only aliases where the
    instrument changed across periods are included.
    """

    aliases = list(var_specs[0].instrument.keys())
    result = {}
    for alias in aliases:
        alias_instruments = {spec.instrument[alias] for spec in var_specs}
        if len(alias_instruments) > 1:
            result[alias] = {
                spec.instrument[alias]: {
                    'start_date': spec.begin,
                    'end_date': spec.end,
                    }
                for spec in var_specs
                }
    return result if result else None


def _canonical_output_name(var_spec: VariableSpec) -> str:
    """
    Return the output variable name.

    Variance variables are output as standard deviation, so the _Vr suffix
    is replaced with _Sd.
    """

    name = var_spec.canonical_name
    if var_spec.statistic_type == StatisticType.VAR and name.endswith('_Vr'):
        return f"{name[:-2]}Sd"
    return name


def _output_statistic(var_spec: VariableSpec) -> StatisticType | None:
    """
    Return the output statistic type.

    VAR is replaced with STDEV because the pipeline always converts variance
    to standard deviation before output.
    """

    if var_spec.statistic_type == StatisticType.VAR:
        return StatisticType.STDEV
    return var_spec.statistic_type


###############################################################################
### END PRIVATE FUNCTIONS — dataset build ###
###############################################################################

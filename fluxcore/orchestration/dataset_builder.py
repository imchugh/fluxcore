#!/usr/bin/env python3
"""Build L1 xarray Datasets from a site context.

Public API
----------
build_dataset_from_context(ctx, derive_humidity, derive_co2, start_date)
    -> xr.Dataset
"""

import datetime

import pandas as pd
import xarray as xr

from fluxcore.domain.enums import StatisticType
from fluxcore.orchestration import derived_quantities
from fluxcore.orchestration.dataframe_builder import build_dataframe_and_specs
from fluxcore.orchestration.derived_quantities import DatasetBuildIntermediate
from fluxcore.services.metadata.instrument_history import InstrumentHistory
from fluxcore.services.metadata.site_metadata_builder import SiteContext
from fluxcore.services.metadata.variable_registry import (
    VariableSpec,
    canonical_output_name,
    group_by_canonical_name,
)

ATTRS_SUBSET = [
    "site_name",
    "fluxnet_id",
    "latitude",
    "longitude",
    "elevation",
    "time_step",
    "time_zone",
    "canopy_height",
    "tower_height",
    "soil",
    "vegetation",
    "date_commissioned",
]


def build_dataset_from_context(
    ctx: SiteContext,
    derive_humidity: bool = True,
    derive_co2: bool = True,
    start_date: pd.Timestamp | None = None,
) -> xr.Dataset:
    """Build an L1 xarray dataset from a fully-assembled site context."""
    result = _build_result(ctx, start_date=start_date)
    result = _derive_quantities(
        result, derive_humidity=derive_humidity, derive_co2=derive_co2
    )

    ds = result.df.to_xarray()
    ds = _apply_variable_metadata(ds, result.var_attrs)
    ds = _apply_global_metadata(ds, ctx)
    ds = derived_quantities.add_day_night_indicator(ds)

    return ds


def _derive_quantities(
    result: DatasetBuildIntermediate,
    derive_humidity: bool,
    derive_co2: bool,
) -> DatasetBuildIntermediate:
    if derive_humidity:
        result = derived_quantities.derive_missing_humidity(result)
    if derive_co2:
        result = derived_quantities.derive_co2_mole_fraction(result)
    # Always calculated, never gated -- see add_net_radiation's docstring.
    result = derived_quantities.add_net_radiation(result)
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
        ds.attrs["flux_system"] = flux_system

    return ds


def _build_result(
    ctx: SiteContext,
    start_date: pd.Timestamp | None = None,
) -> DatasetBuildIntermediate:
    """Build a canonical dataframe and per-variable xarray attrs from a site context.

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
    df, specs_by_alias = build_dataframe_and_specs(ctx, start_date=start_date)
    var_attrs = _build_var_attrs(specs_by_alias=specs_by_alias)

    return DatasetBuildIntermediate(df=df, var_attrs=var_attrs)


# Variable attribute construction


def _build_var_attrs(specs_by_alias: dict[str, VariableSpec]) -> dict[str, dict]:
    """Build per-variable xarray attribute dicts keyed by canonical output name."""
    rslt = {}

    canonical_groups = group_by_canonical_name(specs_by_alias)

    for _, var_specs in canonical_groups.items():
        main_spec = var_specs[-1]
        history = InstrumentHistory.from_specs(var_specs)

        attrs = {
            "height": main_spec.height,
            "height_range": main_spec.height_range,
            "instrument": history.current(),
            "instrument_uri": main_spec.instrument_uri,
            "instrument_history": history.to_attr(),
            "long_name": main_spec.long_name,
            "quantity": main_spec.quantity,
            "standard_name": main_spec.standard_name,
            "statistic_type": _output_statistic(main_spec),
            "units": main_spec.canonical_units,
        }

        rslt[canonical_output_name(main_spec)] = attrs

    return rslt


def _output_statistic(var_spec: VariableSpec) -> StatisticType | None:
    """Return the output statistic type.

    VAR is replaced with STDEV because the pipeline always converts variance
    to standard deviation before output.
    """
    if var_spec.statistic_type == StatisticType.VAR:
        return StatisticType.STDEV
    return var_spec.statistic_type

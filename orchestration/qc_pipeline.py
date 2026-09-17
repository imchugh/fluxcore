#!/usr/bin/env python3
"""Apply a site's L2 QC config to an xr.Dataset.

The one place that knows about xr.Dataset in the QC path: squeezes the
singleton lat/lon dims L1 carries on every data variable, runs the registered
checks (services/data/qc_service.py) in dependency order, combines their
results into a bitmask, masks flagged values to NaN, and overwrites each
configured variable's {var}_QCFlag.
"""

import numpy as np
import pandas as pd
import xarray as xr

from domain.enums import StatisticType
from services.data.qc_service import get_check
from services.metadata.core.variable_name_parser import (
    NameParser,
    VariableNameParseError,
)
from services.metadata.qc_config_schema import SiteQCConfig

# One bit per check, summed into {var}_QCFlag. 0 = OK. CF flag_masks/
# flag_meanings attrs are attached to each flag variable for standards
# compliance. This is a deliberate improvement over PyFluxPro's own scheme,
# which uses enumerated single-cause codes that a later check silently
# overwrites (see pfp_ck.do_rangecheck's unconditional Flag[idx] = code) —
# a bitmask preserves every cause a record failed.
QC_FLAG_BITS = {
    "missing": 1,
    "range_check": 2,
    "exclude_dates": 4,
    "mad_filter": 8,
    "dependency_check": 16,
}


def checkable_variables(ds: xr.Dataset) -> set[str]:
    """Data variables eligible to be QC-checked: excludes flags and crs."""
    return {var for var in ds.data_vars if not var.endswith("_QCFlag") and var != "crs"}


def apply_qc(
    ds: xr.Dataset,
    qc_config: SiteQCConfig,
    range_defaults: dict | None = None,
) -> xr.Dataset:
    """Run qc_config's checks over ds, masking flagged values and writing flags.

    Processes configured variables in qc_config.dependency_graph_order() —
    every dependency is fully resolved before anything that depends on it, so
    chained dependency_check references (A depends on B depends on C)
    propagate correctly in a single ordered pass, unlike PyFluxPro's own flat
    two-pass local/dependency split. A dependency that isn't itself a
    configured variable is resolved directly from isnull() rather than an
    ordered flag — this holds even where that same variable gets a default
    range_check below; defaults are not woven into dependency resolution, to
    keep that one documented, predictable rule rather than two.

    If range_defaults is given (see qc_config_schema.load_range_defaults),
    every checkable variable NOT in qc_config.variables is also looked up
    there (by quantity, then qualifier or statistic — see
    configs/qc/_range_defaults.yml's header) and, if a bound is found, gets
    an implicit range_check + isnull() (nothing else — no dependency_check/
    mad_filter/exclude_dates for a default-only variable). A variable with
    no explicit config AND no resolvable default passes through unchanged,
    same as when range_defaults is omitted entirely.
    """
    ds = ds.copy()
    resolved_bad: dict[str, pd.Series] = {}

    for var_name in qc_config.dependency_graph_order():
        spec = qc_config.variables[var_name]
        series = _extract_series(ds, var_name)

        flag_bits = pd.Series(np.zeros(len(series), dtype=int), index=series.index)
        flag_bits += series.isnull().astype(int) * QC_FLAG_BITS["missing"]

        if spec.range_check is not None:
            bad = get_check("range_check")(
                series, spec.range_check.lower, spec.range_check.upper
            )
            flag_bits += bad.astype(int) * QC_FLAG_BITS["range_check"]

        if spec.exclude_dates is not None:
            bad = get_check("exclude_dates")(series.index, spec.exclude_dates)
            flag_bits += bad.astype(int) * QC_FLAG_BITS["exclude_dates"]

        if spec.mad_filter is not None:
            reference = _extract_series(ds, spec.mad_filter.reference_var)
            bad = get_check("mad_filter")(
                series,
                reference,
                time_step_minutes=int(ds.attrs["time_step"]),
                fsd_threshold=spec.mad_filter.fsd_threshold,
                window_days=spec.mad_filter.window_days,
                zfc=spec.mad_filter.zfc,
                edge_threshold=spec.mad_filter.edge_threshold,
            )
            flag_bits += bad.astype(int) * QC_FLAG_BITS["mad_filter"]

        if spec.dependency_check is not None:
            dep_flags = [
                resolved_bad[dep]
                if dep in resolved_bad
                else _extract_series(ds, dep).isnull()
                for dep in spec.dependency_check
            ]
            bad = get_check("dependency_check")(dep_flags)
            flag_bits += bad.astype(int) * QC_FLAG_BITS["dependency_check"]

        resolved_bad[var_name] = flag_bits != 0
        ds = _write_flag_and_mask(ds, var_name, flag_bits)

    if range_defaults:
        name_parser = NameParser()
        unconfigured = checkable_variables(ds) - set(qc_config.variables)
        for var_name in sorted(unconfigured):
            bounds = lookup_default_range(ds, var_name, range_defaults, name_parser)
            if bounds is None:
                continue

            series = _extract_series(ds, var_name)
            flag_bits = pd.Series(np.zeros(len(series), dtype=int), index=series.index)
            flag_bits += series.isnull().astype(int) * QC_FLAG_BITS["missing"]

            bad = get_check("range_check")(series, bounds[0], bounds[1])
            flag_bits += bad.astype(int) * QC_FLAG_BITS["range_check"]

            ds = _write_flag_and_mask(ds, var_name, flag_bits)

    return ds


def lookup_default_range(
    ds: xr.Dataset,
    var_name: str,
    range_defaults: dict,
    name_parser: NameParser,
) -> tuple[float, float] | None:
    """Resolve var_name's default [min, max] from range_defaults, or None.

    quantity and qualifier come from parsing var_name itself; statistic
    comes from ds[var_name].attrs["statistic_type"] (the authoritative
    source — required for every continuous variable by site_config_schema,
    not something to re-derive from the name, which fails on several real
    canonical names). Returns None (no default, pass through) wherever the
    name doesn't parse, the quantity has no entry, or the entry is a dict
    with no matching qualifier or statistic key.

    Public (used by both apply_qc's default-fallback and
    orchestration.legacy_rtmc_export's range-limiting step).
    """
    try:
        parsed = name_parser.parse_variable_name(var_name)
    except VariableNameParseError:
        return None

    entry = range_defaults.get(parsed.quantity)
    if entry is None:
        return None
    if isinstance(entry, list):
        return tuple(entry)

    if parsed.qualifier is not None and parsed.qualifier in entry:
        return tuple(entry[parsed.qualifier])

    statistic_attr = ds[var_name].attrs.get("statistic_type")
    if statistic_attr is not None:
        statistic_suffix = StatisticType(statistic_attr).suffix
        if statistic_suffix in entry:
            return tuple(entry[statistic_suffix])

    return None


def _extract_series(ds: xr.Dataset, var_name: str) -> pd.Series:
    """Return var_name's data as a flat, time-indexed pd.Series."""
    da = ds[var_name]
    non_time_dims = [d for d in da.dims if d != "time"]
    if non_time_dims:
        da = da.squeeze(non_time_dims, drop=True)
    series = da.to_pandas()
    series.index = pd.DatetimeIndex(series.index)
    return series


def _write_flag_and_mask(ds: xr.Dataset, var_name: str, flag_bits: pd.Series) -> xr.Dataset:
    """Overwrite {var_name}_QCFlag and mask var_name to NaN where flagged."""
    da = ds[var_name]
    non_time_dims = [d for d in da.dims if d != "time"]
    reshape = (-1,) + tuple(1 for _ in non_time_dims)

    flag_values = flag_bits.to_numpy().reshape(reshape)
    ds[f"{var_name}_QCFlag"] = (da.dims, flag_values.astype(int))
    ds[f"{var_name}_QCFlag"].attrs.update(
        {
            "long_name": f"{var_name} QC flag",
            "units": "1",
            "flag_masks": list(QC_FLAG_BITS.values()),
            "flag_meanings": " ".join(QC_FLAG_BITS.keys()),
        }
    )

    mask = (flag_bits.to_numpy() != 0).reshape(reshape)
    ds[var_name] = da.where(~xr.DataArray(mask, dims=da.dims, coords=da.coords))

    return ds

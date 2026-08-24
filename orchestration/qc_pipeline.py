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

from services.data.qc_service import get_check
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


def apply_qc(ds: xr.Dataset, qc_config: SiteQCConfig) -> xr.Dataset:
    """Run qc_config's checks over ds, masking flagged values and writing flags.

    Processes configured variables in qc_config.dependency_graph_order() —
    every dependency is fully resolved before anything that depends on it, so
    chained dependency_check references (A depends on B depends on C)
    propagate correctly in a single ordered pass, unlike PyFluxPro's own flat
    two-pass local/dependency split. A dependency that isn't itself a
    configured variable is resolved directly from isnull() rather than an
    ordered flag. Variables not present in qc_config.variables pass through
    unchanged.
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

    return ds


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

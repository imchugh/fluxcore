#!/usr/bin/env python3
"""Apply a site's L2 QC config to an xr.Dataset.

The one place that knows about xr.Dataset in the QC path: squeezes the
singleton lat/lon dims L1 carries on every data variable, runs the registered
checks (services/data/qc_service.py) in dependency order, combines their
results into a single-cause code, masks flagged values to NaN, and overwrites
each configured variable's {var}_QCFlag.
"""

import numpy as np
import pandas as pd
import xarray as xr

from services.data.qc_service import get_check
from services.metadata.qc_config_schema import SiteQCConfig

# Matches PyFluxPro's own codes exactly (grepped from pfp_ck.py/pfp_io.py, not
# from memory), so {var}_QCFlag is directly comparable against real PyFluxPro
# output for benchmarking. 0 = OK. Combination is PyFluxPro's own scheme too:
# each check unconditionally overwrites the code wherever it fires, applied in
# PyFluxPro's own execution order (range_check -> exclude_dates -> mad_filter,
# dependency_check as a separate later pass) -- the *last* check to fire wins,
# not a sum. A missing value's comparisons never evaluate true (NaN < lower is
# False, same as PyFluxPro's masked-array behaviour), so "missing" survives
# being run through the other checks with no special-casing needed.
#
# An OR'd bitmask (missing=1, range_check=2, exclude_dates=4, mad_filter=8,
# dependency_check=16) was built and considered here instead -- a genuine
# improvement over PyFluxPro's own lossy overwrite (see pfp_ck.do_rangecheck's
# unconditional Flag[idx] = code, which silently drops an earlier cause) since
# it preserves every cause a record failed rather than just the last. Shelved
# in favour of benchmark comparability, not lost: see git commits dafe90a and
# 95606ee if multi-cause tracking becomes more valuable than that comparability.
QC_FLAG_CODES = {
    "missing": 1,
    "range_check": 2,
    "exclude_dates": 6,
    "dependency_check": 23,
    "mad_filter": 24,
}


def checkable_variables(ds: xr.Dataset) -> set[str]:
    """Data variables eligible to be QC-checked: excludes flags and crs."""
    return {var for var in ds.data_vars if not var.endswith("_QCFlag") and var != "crs"}


def apply_qc(
    ds: xr.Dataset,
    qc_config: SiteQCConfig,
) -> xr.Dataset:
    """Run qc_config's checks over ds, masking flagged values and writing flags.

    Processes configured variables in qc_config.dependency_graph_order() —
    every dependency is fully resolved before anything that depends on it, so
    chained dependency_check references (A depends on B depends on C)
    propagate correctly in a single ordered pass, unlike PyFluxPro's own flat
    two-pass local/dependency split. A dependency that isn't itself a
    configured variable is resolved directly from isnull() rather than an
    ordered flag.

    qc_config is expected to already be fully resolved (site config merged
    with configs/qc/_range_defaults.yml and _dependency_defaults.yml via
    qc_config_schema.resolve_qc_config) before it reaches this function —
    apply_qc itself no longer knows about defaults or does any fallback
    lookup. This used to be a separate two-pass design (a loop over
    explicitly-configured variables, then a second fallback pass applying
    range_defaults to anything left unconfigured) — collapsed into a single
    pass once defaults needed to participate in dependency_check resolution
    too (a Diag_SONIC default range_check has to be visible to any
    dependency_check that names it, which only happens via
    dependency_graph_order()/resolved_bad below, not a separate later pass).
    """
    ds = ds.copy()
    resolved_bad: dict[str, pd.Series] = {}

    for var_name in qc_config.dependency_graph_order():
        spec = qc_config.variables[var_name]
        series = _extract_series(ds, var_name)

        code = pd.Series(np.zeros(len(series), dtype=int), index=series.index)
        code[series.isnull()] = QC_FLAG_CODES["missing"]

        if spec.range_check is not None:
            bad = get_check("range_check")(
                series, spec.range_check.lower, spec.range_check.upper
            )
            code[bad] = QC_FLAG_CODES["range_check"]

        if spec.exclude_dates is not None:
            bad = get_check("exclude_dates")(series.index, spec.exclude_dates)
            code[bad] = QC_FLAG_CODES["exclude_dates"]

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
            code[bad] = QC_FLAG_CODES["mad_filter"]

        if spec.dependency_check is not None:
            dep_flags = [
                resolved_bad[dep]
                if dep in resolved_bad
                else _extract_series(ds, dep).isnull()
                for dep in spec.dependency_check
            ]
            bad = get_check("dependency_check")(dep_flags)
            code[bad] = QC_FLAG_CODES["dependency_check"]

        resolved_bad[var_name] = code != 0
        ds = _write_flag_and_mask(ds, var_name, code)

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


_FLAG_VALUES = [0] + sorted(QC_FLAG_CODES.values())
_FLAG_MEANINGS = " ".join(
    ["good"]
    + [
        name
        for _, name in sorted((code, name) for name, code in QC_FLAG_CODES.items())
    ]
)


def _write_flag_and_mask(ds: xr.Dataset, var_name: str, code: pd.Series) -> xr.Dataset:
    """Overwrite {var_name}_QCFlag and mask var_name to NaN where flagged."""
    da = ds[var_name]
    non_time_dims = [d for d in da.dims if d != "time"]
    reshape = (-1,) + tuple(1 for _ in non_time_dims)

    code_values = code.to_numpy().reshape(reshape)
    ds[f"{var_name}_QCFlag"] = (da.dims, code_values.astype(int))
    ds[f"{var_name}_QCFlag"].attrs.update(
        {
            "long_name": f"{var_name} QC flag",
            "units": "1",
            "flag_values": _FLAG_VALUES,
            "flag_meanings": _FLAG_MEANINGS,
        }
    )

    mask = (code.to_numpy() != 0).reshape(reshape)
    ds[var_name] = da.where(~xr.DataArray(mask, dims=da.dims, coords=da.coords))

    return ds

#!/usr/bin/env python3
"""Apply a site's L2 QC config to an xr.Dataset.

The one place that knows about xr.Dataset in the QC path: squeezes the
singleton lat/lon dims L1 carries on every data variable, applies the
configured corrections, runs the checks (services/data/qc_service.py) in
dependency order, combines their results into a single-cause code, masks
flagged values to NaN, and overwrites each configured variable's {var}_QCFlag.
"""

import numpy as np
import pandas as pd
import xarray as xr

from fluxcore.services.data import qc_service
from fluxcore.services.metadata.qc_config_schema import SiteQCConfig

# PyFluxPro's own codes (from pfp_ck.py/pfp_io.py), so {var}_QCFlag is directly
# comparable with real PyFluxPro output for benchmarking. 0 = OK. Combination
# is PyFluxPro's scheme too: the corrections (linear, wd_offset) run first and
# mark what they changed 10, then each check overwrites the code wherever it
# fires, in PyFluxPro's execution order (range_check -> flag_check ->
# exclude_dates -> mad_filter, then dependency_check), so the *last* one wins. A
# missing value's comparisons are never true (NaN < lower is False, as with
# PyFluxPro's masked arrays), so "missing" survives the other checks with no
# special-casing. An OR'd bitmask would keep every cause rather than the last,
# but would lose that comparability.
QC_FLAG_CODES = {
    "missing": 1,
    "range_check": 2,
    "exclude_dates": 6,
    "flag_check": 9,
    "corrected": 10,
    "dependency_check": 23,
    "mad_filter": 24,
}

# Codes that mark usable data: a corrected value is good, not masked.
GOOD_CODES = [0, QC_FLAG_CODES["corrected"]]


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

    qc_config must already be fully resolved (site config merged with
    configs/qc/_range_defaults.yml and _dependency_defaults.yml via
    qc_config_schema.resolve_qc_config); apply_qc knows nothing about
    defaults. Defaults therefore take part in dependency resolution like any
    other entry: a default range_check on Diag_SONIC is visible to every
    dependency_check that names it.
    """
    ds = ds.copy()
    corrected = _apply_corrections(ds, qc_config)
    resolved_bad: dict[str, pd.Series] = {}

    for var_name in qc_config.dependency_graph_order():
        spec = qc_config.variables[var_name]
        series = _extract_series(ds, var_name)

        code = pd.Series(np.zeros(len(series), dtype=int), index=series.index)
        if var_name in corrected:
            code[corrected[var_name]] = QC_FLAG_CODES["corrected"]
        code[series.isnull()] = QC_FLAG_CODES["missing"]

        if spec.range_check is not None:
            bad = qc_service.range_check(
                series, spec.range_check.lower, spec.range_check.upper
            )
            code[bad] = QC_FLAG_CODES["range_check"]

        if spec.flag_check is not None:
            flag_series = [_extract_series(ds, src) for src in spec.flag_check.source]
            bad = qc_service.flag_check(flag_series, spec.flag_check.reject)
            code[bad] = QC_FLAG_CODES["flag_check"]

        if spec.exclude_dates is not None:
            bad = qc_service.exclude_dates_check(series.index, spec.exclude_dates)
            code[bad] = QC_FLAG_CODES["exclude_dates"]

        if spec.mad_filter is not None:
            reference = _extract_series(ds, spec.mad_filter.reference_var)
            bad = qc_service.mad_filter(
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
            bad = qc_service.dependency_check(dep_flags)
            code[bad] = QC_FLAG_CODES["dependency_check"]

        resolved_bad[var_name] = ~code.isin(GOOD_CODES)
        ds = _write_flag_and_mask(ds, var_name, code)

    # Logger quality-flag sources are gate-only: left unmasked, and L1's own
    # placeholder {src}_QCFlag is dropped rather than left looking like a real
    # L2 flag -- unless the site explicitly configured the flag variable itself.
    for src in sorted(qc_config.flag_sources - set(qc_config.variables)):
        if f"{src}_QCFlag" in ds:
            ds = ds.drop_vars(f"{src}_QCFlag")

    return ds


def _apply_corrections(ds: xr.Dataset, qc_config: SiteQCConfig) -> dict[str, pd.Series]:
    """Apply every configured linear/wd_offset correction to ds, in place.

    Runs before any check, as PyFluxPro does, so every check (including
    another variable's dependency_check or mad_filter reference) sees
    corrected values. Each corrected variable records its windows in a
    linear_correction / wd_offset attribute.

    Returns:
        {variable: mask of the records a correction changed}.
    """
    corrected: dict[str, pd.Series] = {}
    for var_name, spec in qc_config.variables.items():
        series = _extract_series(ds, var_name)
        changed = pd.Series(False, index=series.index)
        attrs = {}
        if spec.linear is not None:
            series, mask = qc_service.linear_correction(series, spec.linear)
            changed |= mask
            attrs["linear_correction"] = _describe(spec.linear, "slope", "offset")
        if spec.wd_offset is not None:
            series, mask = qc_service.wd_offset(series, spec.wd_offset)
            changed |= mask
            attrs["wd_offset"] = _describe(spec.wd_offset, "offset")
        if attrs:
            da = ds[var_name]
            ds[var_name] = da.copy(data=series.to_numpy().reshape(da.shape))
            ds[var_name].attrs.update(attrs)
            corrected[var_name] = changed
    return corrected


def _describe(windows, *fields: str) -> str:
    """Windows as 'start,end,field...;...' (blank start/end: open-ended)."""

    def fmt(dt):
        return "" if dt is None else dt.strftime("%Y-%m-%d %H:%M")

    return ";".join(
        ",".join([fmt(w.start), fmt(w.end), *(str(getattr(w, f)) for f in fields)])
        for w in windows
    )


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
    + [name for _, name in sorted((code, name) for name, code in QC_FLAG_CODES.items())]
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

    mask = (~code.isin(GOOD_CODES)).to_numpy().reshape(reshape)
    ds[var_name] = da.where(~xr.DataArray(mask, dims=da.dims, coords=da.coords))

    return ds

#!/usr/bin/env python3
"""L2 QC check registry: RangeCheck, ExcludeDates, DependencyCheck, MADFilter.

Ported from PyFluxPro's scripts/pfp_ck.py (do_rangecheck, do_excludedates,
do_dependencycheck, do_madfilter/_1/_2), reimplemented as small pure functions
against this pipeline's own pd.Series/numpy rather than PyFluxPro's
configobj/DataStructure/masked-array machinery.

`@register_check` populates CHECK_REGISTRY, keyed by check name; `get_check`
looks it back up. Every check function returns a boolean "bad" mask (True =
fails this check), same length/index as its input — never a masked-and-filled
series — so DependencyCheck can consume other checks' outputs uniformly, and
all flag-combination policy stays in orchestration/qc_pipeline.py.
"""

import numpy as np
import pandas as pd

CHECK_REGISTRY = {}


def register_check(name):
    """Register the decorated function in CHECK_REGISTRY under `name`."""

    def decorator(func):
        CHECK_REGISTRY[name] = func
        return func

    return decorator


def get_check(name):
    """Look up a registered check function by name."""
    return CHECK_REGISTRY[name]


@register_check("range_check")
def range_check(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Boolean mask, True where series < lower or series > upper.

    NaN comparisons are False by construction — already-missing values are
    not additionally flagged here; qc_pipeline combines isnull() separately.
    """
    return (series < lower) | (series > upper)


@register_check("exclude_dates")
def exclude_dates_check(
    index: pd.DatetimeIndex, ranges: list[tuple]
) -> pd.Series:
    """Boolean mask, True where index falls within any [start, end] range (inclusive)."""
    bad = np.zeros(len(index), dtype=bool)
    for start, end in ranges:
        bad |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return pd.Series(bad, index=index)


@register_check("dependency_check")
def dependency_check(dependency_flags: list[pd.Series]) -> pd.Series:
    """OR-combine already-computed boolean bad-flags for a variable's dependencies."""
    if not dependency_flags:
        raise ValueError("dependency_check requires at least one dependency flag series")
    combined = dependency_flags[0]
    for flags in dependency_flags[1:]:
        combined = combined | flags
    return combined


@register_check("mad_filter")
def mad_filter(
    series: pd.Series,
    reference: pd.Series,
    *,
    time_step_minutes: int,
    fsd_threshold: float = 12.0,
    window_days: int = 13,
    zfc: float = 5.5,
    edge_threshold: float | tuple[float, float] = (20.0, 80.0),
) -> pd.Series:
    """Boolean mask, True where series fails the two-stage MAD despike test.

    Direct reimplementation of pfp_ck.do_madfilter_1/_2: day/night split off
    `reference` (e.g. Fsd) at `fsd_threshold`, with a one-step bleed either
    side of a day/night transition; per-window (default 13 days) test of the
    series' second differences against a median +/- zfc*MAD/0.6745 envelope;
    then a second-stage check on points untouched by either window (gap
    edges) against `edge_threshold`. Uses numpy.nanmedian/nanpercentile in
    place of PyFluxPro's masked-array operations, since this pipeline
    represents missing data as NaN rather than a masked array.
    """
    values = series.to_numpy(dtype=float)
    ref = reference.reindex(series.index).to_numpy(dtype=float)
    nrecs = len(values)

    bad = np.zeros(nrecs, dtype=bool)
    if nrecs < 3:
        return pd.Series(bad, index=series.index)

    if isinstance(edge_threshold, tuple):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            edge_thr = 0.0
        else:
            lo, hi = np.nanpercentile(finite, edge_threshold[0]), np.nanpercentile(
                finite, edge_threshold[1]
            )
            edge_thr = abs(hi - lo)
    else:
        edge_thr = float(edge_threshold)

    nperday = int(24 * 60 / time_step_minutes)
    window_nrecs = window_days * nperday
    if window_nrecs <= 0 or window_nrecs >= nrecs:
        n_windows = 1
        window_nrecs = nrecs
        last_window_nrecs = nrecs
    else:
        n_windows = nrecs // window_nrecs
        leftover_nrecs = nrecs - window_nrecs * n_windows
        addons = leftover_nrecs // n_windows
        window_nrecs = window_nrecs + addons
        last_window_nrecs = window_nrecs + (nrecs - n_windows * window_nrecs)

    ones = np.ones(nrecs, dtype=int)
    zeros = np.zeros(nrecs, dtype=int)
    day = np.where(ref > fsd_threshold, ones, zeros)
    night = np.where(ref <= fsd_threshold, ones, zeros)
    day_ext = (day + np.roll(day, 1) + np.roll(day, -1)) > 0
    night_ext = (night + np.roll(night, 1) + np.roll(night, -1)) > 0

    # cidx: 0 = untouched (candidate gap edge), 2 = fails MAD test, 3 = passes
    cidx = np.zeros(nrecs, dtype=int)

    for ext_mask in (night_ext, day_ext):
        masked = np.where(ext_mask, values, np.nan)
        diffs = np.full(nrecs, np.nan)
        dm1 = masked[1 : nrecs - 1] - masked[0 : nrecs - 2]
        dp1 = masked[2:nrecs] - masked[1 : nrecs - 1]
        diffs[1 : nrecs - 1] = dm1 - dp1

        for i in range(n_windows):
            si = i * window_nrecs
            ei = si + window_nrecs
            if i == n_windows - 1:
                ei = si + last_window_nrecs
            ei = min(ei, nrecs)
            window = diffs[si:ei]
            if np.all(np.isnan(window)):
                continue
            median = np.nanmedian(window)
            median_abs = np.nanmedian(np.abs(window - median))
            upr = median + zfc * median_abs / 0.6745
            lwr = median - zfc * median_abs / 0.6745
            upr, lwr = max(upr, lwr), min(upr, lwr)

            local = cidx[si:ei]
            fail = (window > upr) | (window < lwr)
            passed = (window >= lwr) & (window <= upr) & ~np.isnan(window)
            local[fail] = 2
            local[passed] = 3
            cidx[si:ei] = local

    edge_idx = np.where(cidx == 0)[0]
    edge_idx = edge_idx[(edge_idx > 0) & (edge_idx < nrecs - 1)]
    if edge_idx.size:
        before = edge_idx - 1
        after = edge_idx + 1
        diff_before = np.abs(values[edge_idx] - values[before])
        diff_after = np.abs(values[after] - values[edge_idx])
        fail_edge = (diff_before > edge_thr) | (diff_after > edge_thr)
        cidx[edge_idx[fail_edge]] = 2
        cidx[edge_idx[~fail_edge]] = 3

    bad = cidx != 3
    bad[~np.isfinite(values)] = False  # missing handled separately by qc_pipeline
    return pd.Series(bad, index=series.index)

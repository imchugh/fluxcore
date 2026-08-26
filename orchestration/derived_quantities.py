#!/usr/bin/env python3
"""Derive and pad missing quantities in a DatasetBuildIntermediate.

pad_humidity: for each (instrument, height) group with Ta and exactly one of
{RH, AH}, derives the missing variable from the other plus Ta and ps.

pad_co2: for each CO2c column (Av and Sd only), derives the corresponding CO2
dry mole fraction if not already present. Ta_Av and ps_Av are used as
period-representative inputs for both Av and Sd derivations.

Public API
----------
pad_humidity(result) -> DatasetBuildIntermediate
pad_co2(result)      -> DatasetBuildIntermediate
"""

import pandas as pd

from domain.enums import StatisticType
from orchestration.dataset_builder import DatasetBuildIntermediate
from services.data.transform_service import get_calculation
from services.metadata.core.canonical_quantity_registry import (
    build_canonical_quantity_registry,
)

_HUMIDITY_QUANTITIES = frozenset({"Ta", "RH", "AH"})


def pad_humidity(result: DatasetBuildIntermediate) -> DatasetBuildIntermediate:
    """Derive and add the missing humidity variable (RH or AH) for each group.

    A group is an instrument+height combination that has Ta and exactly one
    of {RH, AH}. Returns the result unchanged if no pressure variable is
    found or there is nothing to derive.
    """
    df = result.df.copy()
    var_attrs = dict(result.var_attrs)

    ps_col = _find_quantity_av(df, var_attrs, "ps")
    if ps_col is None:
        return result

    groups = _group_by_instrument_height(var_attrs)

    for group in groups.values():
        ta_col = group.get("Ta")
        rh_col = group.get("RH")
        ah_col = group.get("AH")

        if ta_col is None:
            continue
        if (rh_col is None) == (ah_col is None):  # both present or both absent
            continue

        if rh_col is not None:
            new_col = _derived_name(rh_col, "RH", "AH")
            df[new_col] = get_calculation("AH")(
                Ta=df[ta_col], RH=df[rh_col], ps=df[ps_col]
            )
            var_attrs[new_col] = _build_attrs(
                source_attrs=var_attrs[rh_col], quantity="AH"
            )
        else:
            new_col = _derived_name(ah_col, "AH", "RH")
            df[new_col] = get_calculation("RH")(
                AH=df[ah_col], Ta=df[ta_col], ps=df[ps_col]
            )
            var_attrs[new_col] = _build_attrs(
                source_attrs=var_attrs[ah_col], quantity="RH"
            )

    return DatasetBuildIntermediate(df=df, var_attrs=var_attrs)


def pad_co2(result: DatasetBuildIntermediate) -> DatasetBuildIntermediate:
    """Derive CO2 dry mole fraction (umol/mol) from CO2c for all sites.

    Processes Av and Sd columns; skips counters and any column where the
    corresponding CO2 column already exists (e.g. CumberlandPlain).
    For each CO2c column, the Ta_Av at the closest height is used; ps_Av is
    taken globally (pressure variation over tower height is negligible).

    Returns the result unchanged if ps is absent or no Ta column can be matched.
    """
    df = result.df.copy()
    var_attrs = dict(result.var_attrs)

    ps_col = _find_quantity_av(df, var_attrs, "ps")
    if ps_col is None:
        return result

    for col, attrs in result.var_attrs.items():
        if attrs.get("quantity") != "CO2c":
            continue
        if attrs.get("statistic_type") not in (StatisticType.AVG, StatisticType.STDEV):
            continue
        new_col = _derived_name(col, "CO2c", "CO2")
        if new_col in df.columns:
            continue
        ta_col = _find_closest_quantity_av(df, var_attrs, "Ta", attrs.get("height"))
        if ta_col is None:
            continue
        df[new_col] = get_calculation("CO2")(CO2c=df[col], Ta=df[ta_col], ps=df[ps_col])
        var_attrs[new_col] = _build_attrs(source_attrs=attrs, quantity="CO2")

    return DatasetBuildIntermediate(df=df, var_attrs=var_attrs)


def _find_quantity_av(
    df: pd.DataFrame,
    var_attrs: dict[str, dict],
    quantity: str,
) -> str | None:
    """Return the first Av column for the given quantity present in df, or None."""
    for var_name, attrs in var_attrs.items():
        if (
            attrs.get("quantity") == quantity
            and attrs.get("statistic_type") == StatisticType.AVG
            and var_name in df.columns
        ):
            return var_name
    return None


def _find_closest_quantity_av(
    df: pd.DataFrame,
    var_attrs: dict[str, dict],
    quantity: str,
    target_height: float | None,
) -> str | None:
    """Return the Av column for quantity whose height is closest to target_height."""
    best_col = None
    best_dist = float("inf")
    for var_name, attrs in var_attrs.items():
        if (
            attrs.get("quantity") != quantity
            or attrs.get("statistic_type") != StatisticType.AVG
            or var_name not in df.columns
        ):
            continue
        h = attrs.get("height")
        if h is None or target_height is None:
            if best_col is None:  # fall back to first available
                best_col = var_name
            continue
        dist = abs(h - target_height)
        if dist < best_dist:
            best_dist = dist
            best_col = var_name
    return best_col


def _group_by_instrument_height(
    var_attrs: dict[str, dict],
) -> dict[tuple, dict[str, str]]:
    """Group humidity-relevant column names by (instrument, height).

    Returns dict keyed by (instrument, height); values are quantity → column name
    for quantities in {Ta, RH, AH}.
    """
    groups: dict[tuple, dict[str, str]] = {}
    for var_name, attrs in var_attrs.items():
        qty = attrs.get("quantity")
        if qty not in _HUMIDITY_QUANTITIES:
            continue
        instrument = attrs["instrument"]
        if isinstance(instrument, dict):
            instrument = frozenset(instrument.items())
        key = (instrument, attrs["height"])
        groups.setdefault(key, {})[qty] = var_name
    return groups


def _derived_name(source_name: str, source_qty: str, new_qty: str) -> str:
    """Replace the quantity prefix of source_name with new_qty."""
    suffix = source_name[len(source_qty) :]  # preserves leading '_'
    return f"{new_qty}{suffix}"


def _build_attrs(source_attrs: dict, quantity: str) -> dict:
    """Build attrs for a derived variable, inheriting spatial/instrument context."""
    canonical = build_canonical_quantity_registry().get_base_metadata(quantity)
    return {
        "height": source_attrs["height"],
        "height_range": source_attrs["height_range"],
        "instrument": source_attrs["instrument"],
        "instrument_history": source_attrs["instrument_history"],
        "long_name": canonical.long_name,
        "quantity": quantity,
        "standard_name": canonical.standard_name,
        "statistic_type": source_attrs["statistic_type"],
        "units": canonical.standard_units,
        "valid_min": canonical.valid_min,
        "valid_max": canonical.valid_max,
    }

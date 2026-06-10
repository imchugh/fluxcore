#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build canonical DataFrames from raw instrument file groups.

Public API
----------
build_dataframe_from_site_name(site_name, quantities, n_samples, start_date) -> pd.DataFrame
build_dataframe_from_context(ctx, quantities, n_samples, start_date)         -> pd.DataFrame
build_dataframe(file_groups, registry, quantities, n_samples, start_date,
                flux_file, time_step)                                         -> pd.DataFrame
"""

from typing import Callable

import pandas as pd

from domain.enums import DiagnosticType, StatisticType, VariableType
from services.metadata.file_group_builder import FileGroup, build_file_groups
from services.metadata.canonical_quantity_registry import resolve_variance_units
from services.metadata.variable_registry import VariableSpec, build_variable_registry
from services.metadata.site_registry import SiteRegistry, SiteContext
from services.data import raw_data_loader, transform_service
from infrastructure.data_conditioning import condition_dataframe


SITE_REGISTRY = SiteRegistry()


def build_dataframe_from_site_name(
        site_name: str,
        quantities: set[str] | None = None,
        start_date: pd.Timestamp | None = None,
        ) -> pd.DataFrame:
    """Convenience wrapper — resolves site name to context via registry."""

    ctx = SITE_REGISTRY.get_context(site=site_name)
    return build_dataframe_from_context(
        ctx=ctx,
        quantities=quantities,
        start_date=start_date,
        )


def build_dataframe_from_context(
        ctx: SiteContext,
        quantities: set[str] | None = None,
        start_date: pd.Timestamp | None = None,
        ) -> pd.DataFrame:
    """
    Build a canonical DataFrame from a fully-assembled site context.

    Assembles file_groups and registry from the context, then delegates to
    build_dataframe. All site-specific parameters (n_samples, flux_file,
    time_step) are sourced from the context.

    Args:
        ctx: Site runtime config and metadata.
        quantities: Optional set of base quantity names to load (e.g.
            ``{'Fco2', 'Fh'}``). When None, all configured variables are built.
        start_date: If provided, records before this timestamp are discarded.

    Returns:
        Canonical DataFrame with DatetimeIndex.
    """

    runtime_cfg = ctx.runtime_config
    file_groups = build_file_groups(runtime_cfg)
    registry = build_variable_registry(runtime_cfg=runtime_cfg, file_groups=file_groups)

    return build_dataframe(
        file_groups=file_groups,
        registry=registry,
        quantities=quantities,
        n_samples=ctx.metadata.n_samples,
        start_date=start_date,
        flux_file=runtime_cfg.flux_file,
        time_step=ctx.metadata.time_step,
        )


def build_dataframe(
        file_groups: dict[str, FileGroup],
        registry: dict[str, VariableSpec],
        n_samples: int | None = None,
        start_date: pd.Timestamp | None = None,
        flux_file: str | None = None,
        time_step: int | None = None,
        quantities: set[str] | None = None,
        ) -> pd.DataFrame:
    """
    Build the canonical dataframe.

    When ``quantities`` is supplied, the registry is filtered to only entries
    where ``spec.quantity in quantities``, and ``file_groups`` is further
    reduced to only those groups referenced by the filtered registry.  When
    ``None``, the full registry and all file groups are used.

    Step 1 — per file group: load all files, apply aliasing, conversions,
              condition, and concatenate vertically.
    Step 2 — concatenate all file groups horizontally.
    Step 3 — merge overlapping instrument periods into single columns.
    Step 4 — rename aliased columns to canonical output names.
    Step 5 — truncate to flux file group's temporal extent, so that ancillary
              data predating the flux system does not produce empty output years.
    """

    if quantities is not None:
        registry = {
            alias: spec
            for alias, spec in registry.items()
            if spec.quantity in quantities
            }
        active_groups = {spec.file_group for spec in registry.values()}
        file_groups = {
            name: fg
            for name, fg in file_groups.items()
            if name in active_groups
            }

    # Step 1
    dfs = []
    flux_index = None
    for group_name, mapper in file_groups.items():
        loader = raw_data_loader.get_data_adapter(system_type=mapper.file_format)
        group_df = _build_file_group_dataframe(
            mapper=mapper,
            loader=loader,
            registry=registry,
            n_samples=n_samples,
            start_date=start_date,
            time_step=time_step,
            )
        if not group_df.empty:
            if group_name == flux_file:
                flux_index = group_df.index
            dfs.append(group_df)

    # Step 2
    df = pd.concat(dfs, axis=1)

    # Step 3
    df = _merge_overlapping_variables(df=df, registry=registry)

    # Step 4
    df = _rename_to_canonical(df=df, registry=registry)

    # Step 5
    if flux_index is not None:
        df = df.loc[flux_index.min():flux_index.max()]

    return df


def _build_file_group_dataframe(
        mapper: FileGroup,
        loader: Callable,
        registry: dict[str, VariableSpec],
        n_samples: int | None = None,
        start_date: pd.Timestamp | None = None,
        time_step: int | None = None,
        ) -> pd.DataFrame:
    """
    Load and process all files in a file group.

    For each file: load, filter to expected variables, apply aliasing,
    apply statistical transforms and unit conversions.  Then concatenate
    vertically across master + backup files and condition the time index.
    Files whose latest record predates start_date are skipped entirely.
    """

    rename_map = {
        entry.raw_name: entry.alias
        for entry in registry.values()
        if entry.file_group == mapper.group
        }

    dfs = []
    for file in mapper.variables_by_file:

        df = loader(file)

        if start_date is not None:
            if df.index.max() < start_date:
                continue
            df = df[df.index >= start_date]

        df = _filter_variables(df=df, variables=mapper.expected_variables)

        df = df.rename(
            columns={col: rename_map[col] for col in df.columns if col in rename_map}
            )

        df = _apply_conversions(df=df, registry=registry, n_samples=n_samples)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs).sort_index()

    return condition_dataframe(df, interval_out=time_step)


def _filter_variables(df: pd.DataFrame, variables: set) -> pd.DataFrame:
    """Keep only columns present in the expected variable set."""

    return df[[c for c in df.columns if c in variables]].copy()


def _apply_conversions(
        df: pd.DataFrame,
        registry: dict[str, VariableSpec],
        n_samples: int | None = None,
        ) -> pd.DataFrame:
    """
    Apply statistical transforms and unit conversions in a single pass.

    For variance variables: take the square root (data → stdev), then derive
    the post-transform units and convert further if site stdev-form units
    differ from canonical stdev-form units.

    For counter variables: output is always invalid_count (0 = no error).
    Sites storing valid_count are converted via transform_service; sites
    already storing invalid_count pass through unchanged. diag_type on the
    VariableSpec determines which path is taken.

    For all other variables: convert units if site units differ from canonical.

    QC variables (dimensionless on both sides) pass through unchanged.
    """

    for variable in df.columns:

        if variable not in registry:
            continue

        spec = registry[variable]
        from_units = spec.site_units

        if spec.statistic_type == StatisticType.VAR:
            df[variable] = df[variable] ** 0.5
            from_units = resolve_variance_units(spec.site_units)

        if spec.variable_type == VariableType.COUNTER:
            if spec.diag_type == DiagnosticType.VALID_COUNT:
                if n_samples is None:
                    raise ValueError(
                        f"Counter variable {variable!r} has diag_type="
                        f"'valid_count' but n_samples is not available. "
                        f"Ensure site metadata includes time_step and freq_hz."
                        )
                converter = transform_service.get_unit_conversion('Diag')
                try:
                    result = converter(
                        data=df[variable],
                        from_units='valid_count',
                        n_samples=n_samples,
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Diagnostic conversion failed for {variable!r}"
                        ) from e
                if result is None:
                    raise RuntimeError(
                        f"Diagnostic converter returned None for {variable!r}"
                        )
                df[variable] = result
            # INVALID_COUNT: pass through unchanged
            continue

        if from_units != spec.canonical_units:
            converter = transform_service.get_unit_conversion(spec.quantity)
            try:
                result = converter(data=df[variable], from_units=from_units)
            except Exception as e:
                raise RuntimeError(
                    f"Unit conversion failed for {variable!r} "
                    f"({from_units!r} → {spec.canonical_units!r})"
                    ) from e
            if result is None:
                raise RuntimeError(
                    f"Unit conversion returned None for {variable!r}: "
                    f"converter for quantity {spec.quantity!r} does not "
                    f"handle from_units={from_units!r}"
                    )
            df[variable] = result

    return df


def _get_merge_blocks(registry: dict[str, VariableSpec]) -> dict:
    """
    Identify canonical variables constructed from multiple raw inputs
    (instrument changeovers).

    Returns:
        dict keyed by canonical name; values are alias → {begin, end} dicts.
    """

    rslt = {}

    for canonical_name, var_specs in _canonical_from_registry(registry).items():

        if len(var_specs) <= 1:
            continue

        rslt[canonical_name] = {
            spec.alias: {'begin': spec.begin, 'end': spec.end}
            for spec in var_specs
            }

    return rslt


def _merge_overlapping_variables(
        df: pd.DataFrame,
        registry: dict[str, VariableSpec],
        ) -> pd.DataFrame:
    """Merge overlapping instrument periods into single canonical columns."""

    merge_blocks = _get_merge_blocks(registry)

    if not merge_blocks:
        return df

    merged = {}
    drop_cols = []

    for canonical_name, block in merge_blocks.items():
        merged[canonical_name] = _merge_block(df=df, block=block)
        drop_cols.extend(block.keys())

    return (
        df.drop(columns=drop_cols, errors='ignore')
        .join(pd.DataFrame(merged))
        )


def _merge_block(
        df: pd.DataFrame,
        block: dict[str, dict],
        ) -> pd.Series:
    """Assemble one canonical series from consecutive instrument segments."""

    result = pd.Series(index=df.index, dtype=float)

    for alias, info in block.items():
        segment = df.loc[info['begin']: info['end'], alias]
        result.loc[segment.index] = segment

    return result


def _rename_to_canonical(
        df: pd.DataFrame,
        registry: dict[str, VariableSpec],
        ) -> pd.DataFrame:
    """Rename aliased columns to their canonical output names."""

    rename_map = {
        alias: _canonical_output_name(spec)
        for alias, spec in registry.items()
        if alias in df.columns
        }

    return df.rename(columns=rename_map)


def _canonical_from_registry(
        registry: dict[str, VariableSpec],
        ) -> dict[str, list[VariableSpec]]:
    """Group registry entries by canonical variable name."""

    groups: dict[str, list[VariableSpec]] = {}
    for entry in registry.values():
        groups.setdefault(entry.canonical_name, []).append(entry)
    return groups


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

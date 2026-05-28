#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a canonical, unit-corrected pandas DataFrame from a site runtime
configuration, together with per-variable xarray attribute dicts.

Public API
----------
build(runtime_cfg)      -> DataframeBuildResult
DataframeBuildResult    dataclass(df, var_attrs)
VariableSpec            dataclass (exported for downstream type-hinting)
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

import pandas as pd
from dataclasses import dataclass
from typing import Callable

# -----------------------------------------------------------------------------

from domain.enums import StatisticType, VariableType
from services.metadata.variable_metadata_service import SiteRuntimeConfig
from services.metadata.site_registry import SiteContext
from services.metadata import file_mapping_service
from services.metadata.file_mapping_service import FileGroup
from services.metadata.metadata_conversion_service import resolve_variance_units
from services.data import raw_data_loader, conversion_service
from infrastructure.data_conditioning import condition_dataframe

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------

@dataclass
class VariableSpec:
    """Flat specification for a single raw-to-canonical variable mapping."""

    # Identity
    raw_name: str
    canonical_name: str
    quantity: str

    # Units — canonical_units reflects pipeline output:
    #   VAR variables → stdev-form (pipeline always outputs stdev, not variance)
    #   QC variables  → 'dimensionless'
    #   all others    → base canonical units
    site_units: str
    canonical_units: str

    # Statistical context
    statistic_type: StatisticType | None
    variable_type: VariableType

    # xarray variable attrs
    height: float
    height_range: tuple[float, float] | None
    instrument: str
    long_name: str
    standard_name: str | None

    # Grouping / aliasing
    alias: str
    file_group: str

    # Temporal validity (instrument changeover merge)
    begin: pd.Timestamp | None
    end: pd.Timestamp | None

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass
class DataframeBuildResult:
    """Output of the dataframe build pipeline."""

    df: pd.DataFrame
    var_attrs: dict[str, dict]

# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################


###############################################################################
### BEGIN PUBLIC FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------

def build(ctx: SiteContext) -> DataframeBuildResult:
    """
    Build a canonical dataframe and per-variable xarray attrs from a site
    context.

    Validates file groups, loads and unit-converts raw data, merges
    overlapping instrument periods, and renames to canonical variable names.

    Args:
        ctx: fully-assembled site context (runtime config + site metadata).
             Site metadata provides n_samples (time_step * freq_hz * 60),
             needed when COUNTER variables are stored as invalid_count.

    Returns:
        DataframeBuildResult containing the canonical dataframe and a dict
        of per-variable xarray attribute dicts keyed by canonical name.
    """

    runtime_cfg = ctx.runtime_config

    # File groups contain mapping and validation of variables to files
    file_groups = file_mapping_service.build_file_groups(runtime_cfg)
    for group in file_groups.values():
        group.validate()

    # Registry entries generated for all input variables specified in config
    registry = _build_registry(runtime_cfg=runtime_cfg, file_groups=file_groups)

    # Build the dataframe
    df = _build_dataframe(
        file_groups=file_groups,
        registry=registry,
        n_samples=ctx.metadata.n_samples,
        )

    # Create the requisite variable attrs for the L1 NetCDF file
    var_attrs = _build_var_attrs(registry=registry)

    # Return as a class
    return DataframeBuildResult(df=df, var_attrs=var_attrs)

# -----------------------------------------------------------------------------

###############################################################################
### END PUBLIC FUNCTIONS ###
###############################################################################


###############################################################################
### BEGIN PRIVATE FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------
# Registry construction
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _build_registry(
        runtime_cfg: SiteRuntimeConfig,
        file_groups: dict[str, FileGroup],
        ) -> dict[str, VariableSpec]:
    """Build alias-keyed registry of VariableSpec objects."""

    registry = {}

    for i, (group_name, _) in enumerate(file_groups.items()):

        group_id = f'gp{i}'

        for canonical_name, var_cfg in runtime_cfg.variables.items():

            for raw_input in var_cfg.raw_inputs:

                if raw_input.file != group_name:
                    continue

                alias = f"{raw_input.raw_name}_{group_id}"

                # canonical_units reflects pipeline output:
                #   VAR → stdev-form (resolve_metadata gave variance-form, but
                #         the pipeline always converts to stdev before output)
                #   others → standard_units as resolved by canonical registry
                if var_cfg.statistic_type == StatisticType.VAR:
                    canonical_units = resolve_variance_units(
                        var_cfg.canonical.standard_units
                        )
                else:
                    canonical_units = var_cfg.canonical.standard_units

                # Normalise site units.
                # QC: no meaningful unit regardless of what the file says.
                # COUNTER: preserve raw_units so conversion fires when needed,
                #   but treat 'dimensionless' as 'valid_count' (the count is
                #   already in canonical form, just unlabelled in the source).
                # Others: use raw_units if provided, else fall back to canonical.
                if var_cfg.variable_type == VariableType.QUALITY_FLAG:
                    site_units = 'dimensionless'
                elif var_cfg.variable_type == VariableType.COUNTER:
                    site_units = raw_input.raw_units or 'valid_count'
                    if site_units == 'dimensionless':
                        site_units = 'valid_count'
                else:
                    site_units = (
                        raw_input.raw_units or var_cfg.canonical.standard_units
                        )

                registry[alias] = VariableSpec(
                    raw_name=raw_input.raw_name,
                    canonical_name=canonical_name,
                    alias=alias,
                    quantity=var_cfg.quantity,
                    long_name=var_cfg.canonical.long_name,
                    standard_name=var_cfg.canonical.standard_name,
                    canonical_units=canonical_units,
                    site_units=site_units,
                    statistic_type=var_cfg.statistic_type,
                    variable_type=var_cfg.variable_type,
                    height=var_cfg.height,
                    height_range=var_cfg.height_range,
                    instrument=raw_input.instrument,
                    file_group=group_name,
                    begin=raw_input.begin,
                    end=raw_input.end,
                    )

    return registry

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Variable attribute construction
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _build_var_attrs(registry: dict[str, VariableSpec]) -> dict[str, dict]:
    """
    Build per-variable xarray attribute dicts keyed by canonical output name.
    """

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
            'standard_name':      main_spec.standard_name,
            'statistic_type':     _output_statistic(main_spec),
            'units':              main_spec.canonical_units,
            }

        rslt[_canonical_output_name(main_spec)] = attrs

    return rslt

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _canonical_from_registry(
        registry: dict[str, VariableSpec],
        ) -> dict[str, list[VariableSpec]]:
    """Group registry entries by canonical variable name."""

    groups: dict[str, list[VariableSpec]] = {}
    for entry in registry.values():
        groups.setdefault(entry.canonical_name, []).append(entry)
    return groups

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def _history_from_instruments(
        var_specs: list[VariableSpec],
        ) -> dict[str, dict] | None:
    """
    Return instrument changeover history, or None for single-instrument variables.

    Dates are always explicit for multi-instrument variables (required for the
    merge step), so no defaults are needed.
    """

    instruments = {spec.instrument for spec in var_specs}
    if len(instruments) == 1:
        return None
    return {
        spec.instrument: {'start_date': spec.begin, 'end_date': spec.end}
        for spec in var_specs
        }

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _output_statistic(var_spec: VariableSpec) -> StatisticType | None:
    """
    Return the output statistic type.

    VAR is replaced with STDEV because the pipeline always converts variance
    to standard deviation before output.
    """

    if var_spec.statistic_type == StatisticType.VAR:
        return StatisticType.STDEV
    return var_spec.statistic_type

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Dataframe construction
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _build_dataframe(
        file_groups: dict[str, FileGroup],
        registry: dict[str, VariableSpec],
        n_samples: int | None = None,
        ) -> pd.DataFrame:
    """
    Build the canonical dataframe.

    Step 1 — per file group: load all files, apply aliasing, conversions,
              condition, and concatenate vertically.
    Step 2 — concatenate all file groups horizontally.
    Step 3 — merge overlapping instrument periods into single columns.
    Step 4 — rename aliased columns to canonical output names.
    """

    # Step 1
    dfs = []
    for _, mapper in file_groups.items():
        loader = raw_data_loader.get_data_adapter(system_type=mapper.file_format)
        dfs.append(
            _build_file_group_dataframe(
                mapper=mapper,
                loader=loader,
                registry=registry,
                n_samples=n_samples,
                )
            )

    # Step 2
    df = pd.concat(dfs, axis=1)

    # Step 3
    df = _merge_overlapping_variables(df=df, registry=registry)

    # Step 4
    df = _rename_to_canonical(df=df, registry=registry)

    return df

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _build_file_group_dataframe(
        mapper: FileGroup,
        loader: Callable,
        registry: dict[str, VariableSpec],
        n_samples: int | None = None,
        ) -> pd.DataFrame:
    """
    Load and process all files in a file group.

    For each file: load, filter to expected variables, apply aliasing,
    apply statistical transforms and unit conversions.  Then concatenate
    vertically across master + backup files and condition the time index.
    """

    rename_map = {
        entry.raw_name: entry.alias
        for entry in registry.values()
        if entry.file_group == mapper.group
        }

    dfs = []
    for file in mapper.variables_by_file:

        df = loader.load(file)

        df = _filter_variables(df=df, variables=mapper.expected_variables)

        df = df.rename(
            columns={col: rename_map[col] for col in df.columns if col in rename_map}
            )

        df = _apply_conversions(df=df, registry=registry, n_samples=n_samples)

        dfs.append(df)

    df = pd.concat(dfs).sort_index()

    return condition_dataframe(df, interval_out=30)

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _filter_variables(df: pd.DataFrame, variables: set) -> pd.DataFrame:
    """Keep only columns present in the expected variable set."""

    return df[[c for c in df.columns if c in variables]].copy()

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

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

    For counter variables stored as invalid_count: subtract from n_samples
    to yield valid_count.

    For all other variables: convert units if site units differ from canonical.

    QC variables (dimensionless on both sides) pass through unchanged.
    """

    for variable in df.columns:

        if variable not in registry:
            continue

        spec = registry[variable]
        from_units = spec.site_units

        # Statistical transform: variance → standard deviation
        if spec.statistic_type == StatisticType.VAR:
            df[variable] = df[variable] ** 0.5
            from_units = resolve_variance_units(spec.site_units)

        # Unit conversion — no-op when from_units already matches canonical
        if from_units != spec.canonical_units: 

            # Counter: invalid_count → valid_count = n_samples - invalid_count
            if spec.variable_type == VariableType.COUNTER:
                if n_samples is None:
                    raise ValueError(
                        f"Counter variable {variable!r} has from_units="
                        f"{from_units!r} but n_samples is not available. "
                        f"Ensure site metadata includes time_step and freq_hz."
                        )
                df[variable] = n_samples - df[variable]
                continue

            converter = conversion_service.get_unit_conversion(spec.quantity)
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

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Merge and rename
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

###############################################################################
### END PRIVATE FUNCTIONS ###
###############################################################################

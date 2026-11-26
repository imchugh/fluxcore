#!/usr/bin/env python3
"""Build a registry of VariableSpec objects from a runtime config and file groups.

Public API
----------
VariableSpec          — flat specification for a single raw-to-canonical mapping
build_variable_registry(runtime_cfg, file_groups) -> dict[str, VariableSpec]
input_alias(canonical_name, position) -> str
group_by_canonical_name(registry) -> dict[str, list[VariableSpec]]
canonical_output_name(var_spec) -> str
"""

from datetime import datetime

from pydantic.dataclasses import dataclass

from fluxcore.domain.enums import DiagnosticType, StatisticType, VariableType
from fluxcore.services.metadata.canonical_quantity_registry import (
    resolve_variance_units,
)
from fluxcore.services.metadata.file_group_builder import FileGroup
from fluxcore.services.metadata.runtime_config_builder import SiteRuntimeConfig


@dataclass(frozen=True)
class VariableSpec:
    """Flat specification for a single raw-to-canonical variable mapping."""

    # Identity (raw_name is None for a calculated input)
    raw_name: str | None
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
    instrument: str | dict[str, str]
    instrument_uri: str | dict[str, str] | None
    long_name: str
    standard_name: str | None

    # Grouping / aliasing (file_group is None for a calculated input)
    alias: str
    file_group: str | None

    # Temporal validity (instrument changeover merge)
    begin: datetime | str | None
    end: datetime | str | None

    # Diagnostic counter direction (Diag variables only)
    diag_type: DiagnosticType | None = None

    # The raw data's statistic when it differs from statistic_type (a
    # standard-deviation input on a variance variable, or the reverse).
    # None means the same as statistic_type; see raw_statistic_type.
    input_statistic_type: StatisticType | None = None

    # A calculated input: the canonical variables its period is calculated
    # from (dataframe_builder calculates it after the measured inputs are
    # assembled). None for an input read from a raw column.
    calculate_from: tuple[str, ...] | None = None

    @property
    def raw_statistic_type(self) -> StatisticType | None:
        """The statistic the raw data is stored as, which drives conversion."""
        return self.input_statistic_type or self.statistic_type


def build_variable_registry(
    runtime_cfg: SiteRuntimeConfig,
    file_groups: dict[str, FileGroup],
) -> dict[str, "VariableSpec"]:
    """Build alias-keyed registry of VariableSpec objects.

    One entry per input, keyed by input_alias(canonical name, position in
    the variable's input list), in config order: per variable, its inputs in
    time order. The alias names the input's column in the wide frame
    dataframe_builder assembles before stitching inputs into their variable,
    so one raw column can feed several inputs (eras of one column whose
    units or instrument changed, or two variables). A calculated input has
    no file group and no raw column, but is still registered: it carries
    its period's instrument, and marks it calculated in instrument history.
    """
    registry = {}

    for canonical_name, var_cfg in runtime_cfg.variables.items():
        for position, raw_input in enumerate(var_cfg.raw_inputs):
            calculated = raw_input.calculate_from is not None
            if not calculated and raw_input.file not in file_groups:
                continue
            group_name = raw_input.file

            alias = input_alias(canonical_name, position)
            if var_cfg.statistic_type == StatisticType.VAR:
                canonical_units = resolve_variance_units(
                    var_cfg.canonical.standard_units
                )
            else:
                canonical_units = var_cfg.canonical.standard_units

            if var_cfg.variable_type in (
                VariableType.QUALITY_FLAG,
                VariableType.COUNTER,
            ):
                site_units = "dimensionless"
            elif calculated:
                # Calculated in canonical units from canonical variables.
                site_units = var_cfg.canonical.standard_units
            else:
                site_units = raw_input.raw_units or var_cfg.canonical.standard_units

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
                instrument_uri=raw_input.instrument_uri,
                file_group=group_name,
                begin=raw_input.begin,
                end=raw_input.end,
                diag_type=raw_input.diag_type,
                input_statistic_type=(
                    raw_input.statistic_type
                    if not calculated
                    and raw_input.statistic_type != var_cfg.statistic_type
                    else None
                ),
                calculate_from=raw_input.calculate_from,
            )

    return registry


def input_alias(canonical_name: str, position: int) -> str:
    """Registry key for a variable's input at ``position`` in its input list.

    '#' cannot occur in a canonical name, so an alias never collides with a
    canonical name (some end in _<digit>, e.g. Precip_1) or another alias.
    """
    return f"{canonical_name}#{position}"


def group_by_canonical_name(
    registry: dict[str, VariableSpec],
) -> dict[str, list[VariableSpec]]:
    """Group registry entries by canonical variable name."""
    groups: dict[str, list[VariableSpec]] = {}
    for entry in registry.values():
        groups.setdefault(entry.canonical_name, []).append(entry)
    return groups


def canonical_output_name(var_spec: VariableSpec) -> str:
    """Return the output variable name.

    Variance variables are output as standard deviation, so the _Vr suffix
    is replaced with _Sd.
    """
    name = var_spec.canonical_name
    if var_spec.statistic_type == StatisticType.VAR and name.endswith("_Vr"):
        return f"{name[:-2]}Sd"
    return name

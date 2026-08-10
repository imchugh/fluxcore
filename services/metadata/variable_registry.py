#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a registry of VariableSpec objects from a runtime config and file groups.

Public API
----------
VariableSpec          — flat specification for a single raw-to-canonical mapping
build_variable_registry(runtime_cfg, file_groups) -> dict[str, VariableSpec]
group_by_canonical_name(registry) -> dict[str, list[VariableSpec]]
canonical_output_name(var_spec) -> str
"""

from datetime import datetime
from pydantic.dataclasses import dataclass

from domain.enums import DiagnosticType, StatisticType, VariableType
from services.metadata.runtime_config_loader import SiteRuntimeConfig
from services.metadata.file_group_builder import FileGroup
from services.metadata.canonical_quantity_registry import resolve_variance_units


@dataclass(frozen=True)
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
    instrument: str | dict[str, str]
    long_name: str
    standard_name: str | None
    valid_min: float | None
    valid_max: float | None

    # Grouping / aliasing
    alias: str
    file_group: str

    # Temporal validity (instrument changeover merge)
    begin: datetime | str | None
    end: datetime | str | None

    # Diagnostic counter direction (Diag variables only)
    diag_type: DiagnosticType | None = None


def build_variable_registry(
        runtime_cfg: SiteRuntimeConfig,
        file_groups: dict[str, FileGroup],
        ) -> dict[str, 'VariableSpec']:
    """Build alias-keyed registry of VariableSpec objects."""

    registry = {}

    for i, (group_name, _) in enumerate(file_groups.items()):

        group_id = f'gp{i}'

        for canonical_name, var_cfg in runtime_cfg.variables.items():

            for raw_input in var_cfg.raw_inputs:

                if raw_input.file != group_name:
                    continue

                alias = f"{raw_input.raw_name}_{group_id}"

                if var_cfg.statistic_type == StatisticType.VAR:
                    canonical_units = resolve_variance_units(
                        var_cfg.canonical.standard_units
                        )
                else:
                    canonical_units = var_cfg.canonical.standard_units

                if var_cfg.variable_type in (VariableType.QUALITY_FLAG, VariableType.COUNTER):
                    site_units = 'dimensionless'
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
                    valid_min=var_cfg.canonical.valid_min,
                    valid_max=var_cfg.canonical.valid_max,
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
                    diag_type=var_cfg.diag_type,
                    )

    return registry


def group_by_canonical_name(
        registry: dict[str, VariableSpec],
        ) -> dict[str, list[VariableSpec]]:
    """Group registry entries by canonical variable name."""

    groups: dict[str, list[VariableSpec]] = {}
    for entry in registry.values():
        groups.setdefault(entry.canonical_name, []).append(entry)
    return groups


def canonical_output_name(var_spec: VariableSpec) -> str:
    """
    Return the output variable name.

    Variance variables are output as standard deviation, so the _Vr suffix
    is replaced with _Sd.
    """

    name = var_spec.canonical_name
    if var_spec.statistic_type == StatisticType.VAR and name.endswith('_Vr'):
        return f"{name[:-2]}Sd"
    return name

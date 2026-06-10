#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a registry of VariableSpec objects from a runtime config and file groups.

Public API
----------
VariableSpec          — flat specification for a single raw-to-canonical mapping
build_variable_registry(runtime_cfg, file_groups) -> dict[str, VariableSpec]
"""

import pandas as pd
from dataclasses import dataclass

from domain.enums import DiagnosticType, StatisticType, VariableType
from services.metadata.variable_metadata_service import SiteRuntimeConfig
from services.metadata.file_mapping_service import FileGroup
from services.metadata.metadata_conversion_service import resolve_variance_units


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

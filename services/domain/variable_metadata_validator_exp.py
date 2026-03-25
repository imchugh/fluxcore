#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:52:01 2026

@author: imchugh
"""

from __future__ import annotations
from typing import Dict, Optional, ClassVar
from pydantic import BaseModel, field_validator, model_validator
from infrastructure.file_io import read_yml


# --------------------------------------------------------------------------
# Per-variable configuration
# --------------------------------------------------------------------------
class VariableConfig(BaseModel):
    instrument: str
    statistic_type: str
    units: str
    height: str
    name: str

    logger: Optional[str] = None
    table: Optional[str] = None
    file: Optional[str] = None

    diag_type: Optional[str] = None
    standard_name: Optional[str] = None

    class Config:
        extra = "allow"

    @field_validator("diag_type")
    def validate_diag_type_value(cls, v):
        if v is None:
            return v
        if v not in {"valid_count", "invalid_count"}:
            raise ValueError("diag_type must be one of: valid_count, invalid_count")
        return v

    @model_validator(mode="after")
    def validate_schema_choice(self):
        if self.file is not None:
            if self.logger is not None or self.table is not None:
                raise ValueError("Use either file OR logger+table, not both.")
        else:
            if self.logger is None or self.table is None:
                raise ValueError("Must define either file OR (logger AND table).")
        return self


# --------------------------------------------------------------------------
# Site-level configuration
# --------------------------------------------------------------------------
class Config(BaseModel):
    site: str
    system_type: str
    variables: Dict[str, VariableConfig]

    # class-level constants for validation
    diag_prefixes: ClassVar[list[str]] = ["Diag_"]
    sonic_suffix: ClassVar[str] = "_SONIC"
    irga_suffix: ClassVar[str] = "_IRGA"
    flux_prefixes: ClassVar[list[str]] = ["Fco2", "Fe", "Fh", "Fm", "ustar"]


    # ----------------------------------------------------------------------
    # Top-level field validators
    # ----------------------------------------------------------------------
    @field_validator("system_type")
    def validate_system_type(cls, v):
        allowed = {"CSI", "LICOR"}
        if v not in allowed:
            raise ValueError(f"system_type must be one of {allowed}, got '{v}'")
        return v

    # ----------------------------------------------------------------------
    # Deterministic validation rules only
    # ----------------------------------------------------------------------
    @model_validator(mode="after")
    def enforce_diag_rules(self):
        diag_types = {
            cfg.diag_type
            for name, cfg in self.variables.items()
            if any(name.startswith(prefix) for prefix in self.diag_prefixes)
            }
        if diag_types and len(diag_types) > 1:
            raise ValueError(
                f"Diagnostic variables have inconsistent diag_type values: {diag_types}. Must all be same."
                )
        return self

    @model_validator(mode="after")
    def enforce_instrument_consistency(self):
        sonic_instruments = {
            cfg.instrument
            for name, cfg in self.variables.items()
            if name.endswith(self.sonic_suffix)
        }
        irga_instruments = {
            cfg.instrument
            for name, cfg in self.variables.items()
            if name.endswith(self.irga_suffix)
        }
        if len(sonic_instruments) > 1:
            raise ValueError(f"SONIC variables must use the same instrument; found {sonic_instruments}")
        if len(irga_instruments) > 1:
            raise ValueError(f"IRGA variables must use the same instrument; found {irga_instruments}")
        return self

    @model_validator(mode="after")
    def enforce_flux_suffix(self):
        suffixes_found = set()
        for var_name in self.variables:
            for prefix in self.flux_prefixes:
                if var_name.startswith(prefix):
                    parts = var_name.split("_", 1)
                    if len(parts) != 2 or parts[1] not in {"EP", "EF", "DL"}:
                        raise ValueError(
                            f"Flux variable '{var_name}' must have suffix EP, EF, or DL."
                        )
                    suffixes_found.add(parts[1])
        if len(suffixes_found) > 1:
            raise ValueError(f"Flux variables must share the same suffix. Found: {suffixes_found}")
        return self


# --------------------------------------------------------------------------
# Helper function
# --------------------------------------------------------------------------
def validate_L1_config_structure(file: str) -> Config:
    """Validate YAML structure and return Config object."""
    return Config(**read_yml(file_path=file))

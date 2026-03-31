#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:52:01 2026

@author: imchugh
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, ClassVar, Union
from pydantic import BaseModel, field_validator, model_validator

from infrastructure.file_io import read_yml

# --------------------------------------------------------------------------
# Input-level configuration (raw variables)
# --------------------------------------------------------------------------
class InputVariableConfig(BaseModel):
    
    instrument: str
    file: str
    units: str

    diag_type: Optional[str] = None
    begin: Optional[Union[datetime, str]] = None
    end: Optional[Union[datetime, str]] = None

    class Config:
        extra = "allow"

    @field_validator("diag_type")
    def validate_diag_type_value(cls, v):
        if v is None:
            return v
        if v not in {"valid_count", "invalid_count"}:
            raise ValueError(
                "diag_type must be one of: valid_count, invalid_count"
            )
        return v

    @field_validator("begin", "end", mode="before")
    def parse_datetime(cls, v):
        
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            if v.lower() in {"none", "null", ""}:
                return None
            return datetime.fromisoformat(v)
        raise TypeError(f"Invalid type for datetime field: {type(v)}")

# --------------------------------------------------------------------------
# Output variable configuration
# --------------------------------------------------------------------------
class VariableConfig(BaseModel):
    statistic_type: str
    height: str

    input_variables: Dict[str, InputVariableConfig]

    standard_name: Optional[str] = None

    class Config:
        extra = "allow"


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
    # Diagnostic consistency (now based on input variables)
    # ----------------------------------------------------------------------
    @model_validator(mode="after")
    def enforce_diag_rules(self):
        diag_types = set()

        for var_name, var_cfg in self.variables.items():
            if any(var_name.startswith(prefix) for prefix in self.diag_prefixes):
                for input_cfg in var_cfg.input_variables.values():
                    if input_cfg.diag_type is not None:
                        diag_types.add(input_cfg.diag_type)

        if diag_types and len(diag_types) > 1:
            raise ValueError(
                f"Diagnostic variables have inconsistent diag_type values: {diag_types}. Must all be same."
            )

        return self

    # ----------------------------------------------------------------------
    # Instrument consistency (updated for nested structure)
    # ----------------------------------------------------------------------
    @model_validator(mode="after")
    def enforce_instrument_consistency(self):
        sonic_instruments = set()
        irga_instruments = set()

        for var_name, var_cfg in self.variables.items():
            for input_cfg in var_cfg.input_variables.values():
                if var_name.endswith(self.sonic_suffix):
                    sonic_instruments.add(input_cfg.instrument)
                if var_name.endswith(self.irga_suffix):
                    irga_instruments.add(input_cfg.instrument)

        if len(sonic_instruments) > 1:
            raise ValueError(
                f"SONIC variables must use the same instrument; found {sonic_instruments}"
            )
        if len(irga_instruments) > 1:
            raise ValueError(
                f"IRGA variables must use the same instrument; found {irga_instruments}"
            )

        return self

    # ----------------------------------------------------------------------
    # Flux suffix consistency (unchanged)
    # ----------------------------------------------------------------------
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
            raise ValueError(
                f"Flux variables must share the same suffix. Found: {suffixes_found}"
            )

        return self


# --------------------------------------------------------------------------
# Helper function
# --------------------------------------------------------------------------
def validate_L1_config_structure(file: str) -> Config:
    """Validate YAML structure and return Config object."""
    return Config(**read_yml(file_path=file))

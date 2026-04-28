#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:52:01 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from __future__ import annotations

import re

from datetime import datetime
from typing import Dict, Optional, ClassVar, Union
from pydantic import BaseModel, field_validator, model_validator, Field, ValidationError

from infrastructure.file_io import read_yml
from domain.enums import StatisticType, ALLOWED_FILE_TYPES

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN PYDANTIC VALIDATION CLASSES ###
###############################################################################


# -----------------------------------------------------------------------------
# Input-level configuration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class InputVariableConfig(BaseModel):
    """Define the architecture / rules validation rules for input variables"""
    
    # -------------------------------------------------------------------------
    # Define attributes
    
    # Mandatory
    instrument: str
    file: str
    units: str

    # Optional
    diag_type: Optional[str] = None
    begin: Optional[Union[datetime, str]] = None
    end: Optional[Union[datetime, str]] = None

    # Allow future fields
    class Config:
        extra = "allow"
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    @field_validator("diag_type")
    def validate_diag_type_value(cls, v):
        """Check that diag_type (where present) takes one of two values only"""
        
        if v is None:
            return v
        if v not in {"valid_count", "invalid_count"}:
            raise ValueError(
                "diag_type must be one of: valid_count, invalid_count"
            )
        return v
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    @field_validator("begin", "end", mode="before")
    def parse_datetime(cls, v):
        """Ensure that date fields are either datetimes or None"""
        
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            if v.lower() in {"none", "null", ""}:
                return None
            return datetime.fromisoformat(v)
        raise TypeError(f"Invalid type for datetime field: {type(v)}")
    # -------------------------------------------------------------------------

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class CustomMetadataConfig(BaseModel):
    """Define architecture of custom metadata"""
    
    # -------------------------------------------------------------------------
    # Define attributes
    
    # Mandatory
    horizontal_location: HorizontalLocationConfig

    # Allow future fields
    class Config:
        extra = "allow"  
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class HorizontalLocationConfig(BaseModel):
    """Define the architecture / rules for validation of horizontal location"""

    # -------------------------------------------------------------------------
    # Define attributes

    # All optional    
    belowground: Optional[Dict[str, str]] = None
    aboveground: Optional[Dict[str, str]] = None

    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    @field_validator("belowground", "aboveground")
    def validate_keys(cls, v):
        """Ensure that horizontal location keys are alphabetic"""
        
        if v is None:
            return v

        for key in v:
            if not re.fullmatch(r"[a-zA-Z]", key):
                raise ValueError(
                    f"Horizontal location keys must be single alphabetic characters, got '{key}'"
                )
        return v
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------    
    @model_validator(mode="after")
    def ensure_at_least_one_category(self):
        """Ensure that at least one expected field is in structure"""
        
        if not self.belowground and not self.aboveground:
            raise ValueError(
                "horizontal_location must contain at least one of 'belowground' or 'aboveground'"
            )
        return self
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class FileTypesConfig(BaseModel):
    """Define the architecture / rules for validation of file_types"""

    # -------------------------------------------------------------------------
    # Define attributes
    
    default: str
    overrides: Dict[str, str] = Field(default_factory=dict)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    @field_validator("default")
    def validate_default(cls, v):
        """Ensure that passed file type is allowed"""
        
        if v not in ALLOWED_FILE_TYPES:
            raise ValueError(
                f"default file type must be one of {ALLOWED_FILE_TYPES}")
        return v
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    @field_validator("overrides")
    def validate_overrides(cls, v):
        """Ensure that passed file type is allowed"""
        
        invalid = {
            k: val for k, val in v.items() if val not in ALLOWED_FILE_TYPES
            }
        if invalid:
            raise ValueError(
                f"override file type must be one of {ALLOWED_FILE_TYPES}"
                )
        return v
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Output-level variable configuration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class VariableConfig(BaseModel):
    """Define the complete architecture for canonical variable"""
    
    # -------------------------------------------------------------------------
    # Define attributes
    
    statistic_type: StatisticType
    height: str

    input_variables: Dict[str, InputVariableConfig]

    standard_name: Optional[str] = None
    
    # Allow future fields
    class Config:
        extra = "allow"
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Complete site configuration
# -----------------------------------------------------------------------------
class SiteConfig(BaseModel):

    # -------------------------------------------------------------------------
    # Define attributes

    site: str
    file_formats: FileTypesConfig
    variables: Dict[str, VariableConfig]

    custom_metadata: Optional[CustomMetadataConfig] = None

    # class-level constants for validation
    diag_prefixes: ClassVar[list[str]] = ["Diag_"]
    sonic_suffix: ClassVar[str] = "_SONIC"
    irga_suffix: ClassVar[str] = "_IRGA"
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def enforce_file_type_consistency(self):
        """
        Accumulate file groups and ensure any files referenced in format 
        overrides occur in input variable declarations
        """
        
        used_files = {
            input_cfg.file
            for var_cfg in self.variables.values()
            for input_cfg in var_cfg.input_variables.values()
            }
    
        overrides = set(self.file_formats.overrides.keys())
    
        # overrides must refer to real files
        invalid = overrides - used_files
        if invalid:
            raise ValueError(
                f"Overrides specified for unused files: {invalid}"
            )
    
        return self
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------    
    @model_validator(mode="after")
    def enforce_diag_rules(self):
        """"""
        
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
    # -------------------------------------------------------------------------    

    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

###############################################################################
### END PYDANTIC VALIDATION CLASSES ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------
def validate_variable(config_data, key):
    """
    External variable validator used to test e.g. UI validity after user 
    changes
    """
    
    try:
        VariableConfig(**config_data["variables"][key])
        return True, []
    except ValidationError as e:
        return False, e.errors()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def validate_all(config_data):
    """
    External config validator used to test e.g. UI validity after user changes
    """

    try:
        SiteConfig(**config_data)
        return True, []
    except ValidationError as e:
        return False, e.errors()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def validate_L1_config_structure(file: str) -> SiteConfig:
    """Validate YAML structure and return Config object."""
    return SiteConfig(**read_yml(file_path=file, enforce_unique_keys=True))
# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################


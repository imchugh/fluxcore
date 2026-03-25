#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:12:51 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from dataclasses import dataclass
from typing import Optional

###############################################################################
### BEGIN IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RawVariableMetadata:
    raw_name: str
    raw_units: str
    logger: str
    table: str
    instrument: str
    height: str
    statistic_type: str
    quantity: str
    file: Optional[str] = None
    diag_type: Optional[str] = None
    instrument_type: Optional[str] = None
    vertical_location: Optional[str] = None
    horizontal_location: Optional[str] = None
    replicate: Optional[str] = None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalVariableMetadata:
    long_name: str
    standard_name: str
    standard_units: str
    plausible_min: Optional[float] = None
    plausible_max: Optional[float] = None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class VariableDefinition:
    site_variable_name: str
    raw: RawVariableMetadata
    canonical: CanonicalVariableMetadata
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build_variable_definition(
    site_var_name: str,
    raw_config: dict,
    parsed_name: dict,
    canonical: dict
    ) -> VariableDefinition:

    # 1. Merge YAML + parsed name into RawVariableMetadata
    raw_meta = RawVariableMetadata(
        raw_name = raw_config["name"],
        raw_units = raw_config["units"],
        logger = raw_config["logger"],
        table = raw_config["table"],
        instrument = raw_config.get("instrument"),
        height = raw_config.get('height'),
        statistic_type = raw_config.get('statistic_type'),
        file = raw_config.get("file"),
        diag_type = raw_config.get('diag_type'),
        quantity = parsed_name.get("quantity"),
        instrument_type = parsed_name.get("instrument_type"),
        vertical_location = parsed_name.get("vertical_location"),
        horizontal_location = parsed_name.get("horizontal_location"),
        replicate = parsed_name.get("replicate"),
    )

    # 2 Build canonical metadata object
    canonical_meta = CanonicalVariableMetadata(**canonical)

    # 3 Return immutable VariableDefinition
    return VariableDefinition(
        site_variable_name = site_var_name,
        raw = raw_meta,
        canonical = canonical_meta
    )
# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################

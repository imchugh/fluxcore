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
    file: str
    instrument: str
    begin: Optional[str] = None
    end: Optional[str] = None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalVariableMetadata:
    long_name: str
    standard_name: str
    standard_units: str
    valid_input_units: str | None
    plausible_min: Optional[float] = None
    plausible_max: Optional[float] = None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class VariableDefinition:
    
    # Mandatory attributes
    variable_name: str
    instrument: str
    height: str
    statistic_type: str
    quantity: str

    # Mandatory subclasses    
    raw_inputs: RawVariableMetadata
    canonical: CanonicalVariableMetadata
    
    # Optional attributes
    instrument_type: Optional[str] = None
    vertical_location: Optional[str] = None
    horizontal_location: Optional[str] = None
    replicate: Optional[str] = None
    diag_type: Optional[str] = None
# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################


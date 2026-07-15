#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:12:51 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from datetime import datetime
from pydantic.dataclasses import dataclass, ConfigDict
from typing import Optional, Union

# -----------------------------------------------------------------------------

from domain import enums

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
    instrument: str | dict[str, str]
    begin: Optional[Union[datetime, str]] = None
    end: Optional[Union[datetime, str]] = None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedVariableName:

    quantity: str
    qualifier: Optional[str] = None
    vertical_location: Optional[str] = None
    horizontal_location: Optional[str] = None
    replicate: Optional[str] = None
    statistic_id: Optional[str] = None      # set when suffix is a StatisticType (Av, Sd, Vr ...)
    variable_type_id: Optional[str] = None  # set when suffix is a VariableType  (QC, Ct)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CanonicalQuantityMetadata:

    long_name: str
    standard_name: Optional[str]
    standard_units: str
    valid_input_units: list[str]
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class VariableDefinition:

    # Core identity
    variable_name: str
    quantity: str
    variable_type: enums.VariableType

    # Physical metadata
    instrument: str | dict[str, str]
    height: float

    # Structural metadata
    raw_inputs: tuple[RawVariableMetadata, ...]
    canonical: CanonicalQuantityMetadata
    parsed_name_elems: ParsedVariableName
    
    # Optionals    
    height_range: Optional[tuple[float, float]] = None
    statistic_type: Optional[enums.StatisticType] = None
    diag_type: Optional[enums.DiagnosticType] = None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

class SiteMetadata(dict):
    """
    Lightweight metadata container.

    Provides:
    - automatic type formatting
    - attribute-style access
    - dictionary behaviour
    """

    DATA_DTYPES = {
        'id': str,
        'fluxnet_id': str,
        'date_commissioned': datetime,
        'date_decommissioned': datetime,
        'latitude': float,
        'longitude': float,
        'elevation': float,
        'time_step': int,
        'freq_hz': int,
        'soil': str,
        'tower_height': float,
        'vegetation': str,
        'canopy_height': float,
        'time_zone': str,
        'UTC_offset': float,
        }

    # -------------------------------------------------------------------------

    def __init__(self, data: dict):

        formatted = {}

        for key, value in data.items():

            if value in (None, "", "nan"):
                formatted[key] = None
                continue

            dtype = self.DATA_DTYPES.get(key)

            try:

                if dtype is float:
                    formatted[key] = float(value)

                elif dtype is int:
                    formatted[key] = int(float(value))

                elif dtype is str:
                    formatted[key] = str(value)

                elif dtype is datetime:
                    formatted[key] = datetime.fromisoformat(value)

                else:
                    formatted[key] = value

            except Exception:
                formatted[key] = value

        super().__init__(formatted)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------    
    
    def __getattr__(self, key):
        
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    
    def __setattr__(self, key, value):
        
        raise AttributeError("SiteMetadata is immutable")
    # -------------------------------------------------------------------------        

    # -------------------------------------------------------------------------
    def __dir__(self):
        
        return sorted(set(super().__dir__()) | set(self.keys()))
    # -------------------------------------------------------------------------    

    # -------------------------------------------------------------------------

    @property
    def n_samples(self) -> int | None:
        """
        Expected number of high-frequency samples in one averaging period.

        Calculated from time_step (minutes) and freq_hz (Hz).  Returns None
        if either field is absent or not yet populated.
        """

        time_step = self.get('time_step')
        freq_hz = self.get('freq_hz')
        if time_step is None or freq_hz is None:
            return None
        return time_step * 60 * freq_hz

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def __repr__(self):

        label = self.get("site_name", None)
        if label:
            return f"<SiteMetadata {label}>"
        return f"<SiteMetadata {self.get('site_name','unknown')}>"
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def to_yaml_dict(self):

        return {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in self.items()
            }
    # -------------------------------------------------------------------------    

# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################


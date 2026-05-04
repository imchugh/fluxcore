#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:28:57 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from dataclasses import dataclass
from typing import List, Tuple

from services import config_loader

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN INITS ###
###############################################################################

CANONICAL_VARS = config_loader.load_config_file_from_name('canonical_variables')

###############################################################################
### END INITS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalVariableDefinition:
    """Container for site metadata"""
    
    quantity: str
    instrument_type: str | None
    process: str | None
    vertical_location: str | None
    horizontal_location: str | None
    replicate: str | None
    long_name: str
    plausible_min: float | None
    plausible_max: float | None
    standard_name: str | None
    standard_units: str
    valid_input_units: list[str]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedVariableName:

    quantity: str
    instrument_type: str | None
    process: str | None
    vertical_location: str | None
    horizontal_location: str | None
    replicate: str | None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

class CanonicalResolver:

    def __init__(self, canonical_vars: dict):
        self._canonical = canonical_vars

    def validate(self, parsed: ParsedVariableName) -> None:
        if parsed.quantity not in self._canonical:
            raise VariableNameParseError(
                f"Unknown quantity '{parsed.quantity}'"
                )

    def to_model(
        self, parsed: ParsedVariableName
        ) -> CanonicalVariableDefinition:

        self.validate(parsed)
        canonical = self._canonical[parsed.quantity]
        return CanonicalVariableDefinition(**(parsed.__dict__ | canonical))
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

class VariableNameParseError(Exception):
    """Raised when a variable name fails parsing/validation."""
    pass
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class NameParser:
    """Service for parsing and validating standardised variable names."""

    # Private module-level constants
    _VALID_INSTRUMENTS = ['SONIC', 'IRGA', 'RAD']
    _VALID_LOC_UNITS = ['m']
    _VALID_SUFFIXES = {
        'Av': 'average', 'Sd': 'standard_deviation', 'Vr': 'variance',
        'Sum': 'sum', 'Ct': 'sum', 'QC': 'quality_control_flag'
        }

    # -------------------------------------------------------------------------
    
    def __init__(self):
        """Load the standard variable names from the fixed internal config."""
        
        pass
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def parse_variable_name(self, variable_name: str) -> dict:
        """Parse a variable name and return structured components.

        Raises:
            VariableNameParseError: If parsing or validation fails.
        """
        
        # Split the variable name on underscore
        elems = variable_name.split("_")

        # Quantity / Instrument quantity
        quantity, instrument_type, elems = self._parse_quantity(elems)

        # Exhaust the components of the name string
        process, elems = self._parse_process(elems)
               
        # Check there is only one element remaining 
        # (there should be only one remaining component)
        if len(elems) > 1:
            raise VariableNameParseError(
                f"Unrecognized elements remain in '{variable_name}': {elems}"
                )
        
        # Split elements that aren't separated by underscores
        vertical_location, elems = self._parse_vertical_location(elems)
        horizontal_location, elems = self._parse_horizontal_location(elems)
        replicate, elems = self._parse_replicate(elems)
        
        # Raise if unprocessed elements remain
        if elems:
            raise VariableNameParseError(
                f"Unrecognized elements remain in '{variable_name}': {elems}"
                )
       
        return {
            'quantity': quantity,
            'instrument_type': instrument_type,
            'process': process,
            'vertical_location': vertical_location,
            'horizontal_location': horizontal_location,
            'replicate': replicate           
            }
    
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### Component parsers ###
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    
    def _parse_quantity(
            self, elems: List[str]
            ) -> Tuple[str, str | None, List[str]]:
        """
        Extract the fundamental quantity

        Args:
            elems: list of name elements (substrings).

        Raises:
            VariableNameParseError: raised if quantity not listed in config.

        Returns:
            quantity and remaining elements.

        """
        
        # Raise if nothing
        if not elems:
            raise VariableNameParseError("Variable name is empty")

        # Initialise
        quantity = elems[0]
        instrument = None
        remainder = elems[1:]

        # Check if an instrument quantity e.g. AH versus AH_IRGA
        if remainder and remainder[0] in self._VALID_INSTRUMENTS:
            instrument = remainder[0]
            quantity = f"{quantity}_{instrument}"
            remainder = remainder[1:]

        return quantity, instrument, remainder
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def _parse_process(self, elems: List[str]) -> Tuple[str | None, List[str]]:
        """
        Extract statistical_process (is ALWAYS the final element).

        Args:
            elems: list of name elements (substrings).

        Returns:
            process and remaining elements.

        """
        
        process = None
        if len(elems) > 0:
            candidate = elems[-1]
            if candidate in self._VALID_SUFFIXES:
                process = candidate
                elems = elems[:-1]
        return process, elems       
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def _parse_vertical_location(
            self, elems: List[str]
            ) -> Tuple[str | None, List[str]]:
        """
        Extract vertical location.

        Args:
            elems: list of name elements (substrings).

        Raises:
            VariableNameParseError: raised if first element not numeric.

        Returns:
            vertical location and remaining elements.

        """
        
        vertical_location = None
        if len(elems) > 0:
            candidate = elems[0]
            for unit in self._VALID_LOC_UNITS:
                if unit in candidate:
                    elems = candidate.split(unit)                
                    if elems[-1] == '':
                        elems = elems[:-1]
                    try:
                        float(elems[0])
                    except ValueError:
                        raise VariableNameParseError(
                            'Characters preceding height / depth units must be '
                            'numeric!'
                            )
                    vertical_location = f'{elems[0]}{unit}' 
                    elems = elems[1:]
                    break
        return vertical_location, elems
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def _parse_horizontal_location(
            self, elems: List[str]
            ) -> Tuple[str | None, List[str]]:
        """
        Extract horizontal location.

        Args:
            elems: list of name elements (substrings).

        Returns:
            horizontal location and remaining elements.

        """

        horizontal_location = None
        if len(elems) > 0:
            candidate = elems[0]
            if candidate[0].isalpha():
                horizontal_location = candidate[0]
                elems = candidate[1:]
        return horizontal_location, elems
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    
    def _parse_replicate(self, elems: List[str]) -> Tuple[str | None, List[str]]:
        """
        Extract replicates.

        Args:
            elems: list of name elements (substrings).

        Returns:
            replicate and remaining elements.

        """
        
        replicate = None
        if len(elems) > 0:
            candidate = elems[0]
            if candidate.isdigit():
                replicate = candidate
                elems = elems[1:]
        return replicate, elems
# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################

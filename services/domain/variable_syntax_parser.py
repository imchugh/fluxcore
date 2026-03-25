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

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class VariableDefinition:
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
    _VALID_FLUX_SYSTEMS = {'EF': 'EasyFlux', 'EP': 'EddyPro', 'DL': 'TERNflux'}
    _VALID_LOC_UNITS = ['cm', 'm']
    _VALID_SUFFIXES = {
        'Av': 'average', 'Sd': 'standard_deviation', 'Vr': 'variance',
        'Sum': 'sum', 'Ct': 'sum', 'QC': 'quality_control_flag'
        }

    # -------------------------------------------------------------------------
    
    def __init__(self):
        """Load the standard variable names from the fixed internal config."""
        
        pass
        # # Load YAML via config_loader
        # self.canonical_vars = (
        #     config_loader.load_config_file_from_name("pfp_std_names")
        #     )
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
        
        # Get system type
        sys_type, elems = self._parse_system_type(elems)
        
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

        # # Step 4: return structured definition object
        # return VariableDefinition(
            
        #     # Structural components
        #     quantity=quantity,
        #     instrument_type=instrument_type,
        #     process=process,
        #     vertical_location=vertical_location,
        #     horizontal_location=horizontal_location,
        #     replicate=replicate,
        
        #     # Canonical metadata
        #     long_name=canonical["long_name"],
        #     standard_name=canonical.get("standard_name"),
        #     standard_units=canonical["standard_units"],
        #     plausible_min=canonical.get("plausible_min"),
        #     plausible_max=canonical.get("plausible_max"),
        #     )
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
    
    def _parse_system_type(
            self, elems: List[str]
            ) -> Tuple[str | None, List[str]]:
        """
        Extract system type.

        Args:
            elems: list of name elements (substrings).

        Returns:
            system type and remaining elements.

        """
        
        system_type = None
        if len(elems) > 0:
            candidate = elems[0]
            if candidate in self._VALID_FLUX_SYSTEMS:
                system_type = self._VALID_FLUX_SYSTEMS[candidate]
                elems = elems[1:]
        return system_type, elems
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

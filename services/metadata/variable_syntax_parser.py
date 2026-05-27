#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:28:57 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from domain.data_models.metadata_classes import ParsedVariableName
from domain.enums import StatisticType, VariableType

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

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

    # -------------------------------------------------------------------------
    
    def __init__(self):
        pass
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def parse_variable_name(self, variable_name: str) -> ParsedVariableName:
        """Parse a variable name and return structured components.

        Raises:
            VariableNameParseError: If parsing or validation fails.
        """
        
        # Split the variable name on underscore
        elems = variable_name.split("_")

        # Quantity / Instrument quantity
        quantity, qualifier, elems = self._parse_quantity(elems)
        
        # Exhaust the components of the name string
        statistic_id, variable_type_id, elems = self._parse_process(elems)

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

        # Return parsed variable class
        return ParsedVariableName(

            quantity = quantity,
            qualifier = qualifier,
            statistic_id = statistic_id,
            variable_type_id = variable_type_id,
            vertical_location = vertical_location,
            horizontal_location = horizontal_location,
            replicate = replicate

            )
    
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ### Component parsers ###
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    
    def _parse_quantity(
            self, elems: list[str]
            ) -> tuple[str, str | None, list[str]]:
        """
        Extract the fundamental quantity and optional qualifier.

        For ``Diag`` variables the element immediately following ``Diag``
        (if present) is consumed as the qualifier (e.g. ``IRGA``, ``SONIC``,
        ``Fe``) without being merged into the quantity name.  This means
        ``Diag_IRGA_Ct``, ``Diag_Fe_Ct``, and ``Diag_SONIC_Ct`` all resolve
        to quantity ``'Diag'``, distinguished only by their qualifier.

        For all other quantities, the element immediately following the
        quantity is consumed as the qualifier and merged into the quantity name
        when it is one of the recognised instrument types (e.g. ``AH_IRGA``).

        Args:
            elems: list of name elements (substrings).

        Returns:
            quantity, qualifier (or None), and remaining elements.
        """

        # Raise if nothing
        if not elems:
            raise VariableNameParseError("Variable name is empty")

        quantity = elems[0]
        qualifier = None
        remainder = elems[1:]

        if quantity == 'Diag':
            # Dedicated diagnostic branch: next element is a free-form
            # qualifier and is NOT merged into the quantity name.
            if remainder:
                qualifier = remainder[0]
                remainder = remainder[1:]

        elif remainder and remainder[0] in self._VALID_INSTRUMENTS:
            # Instrument qualifier: stored in qualifier, NOT merged into the
            # quantity name.  The canonical quantity lookup always uses the
            # base name (e.g. 'AH' for 'AH_IRGA_Av').
            qualifier = remainder[0]
            remainder = remainder[1:]

        return quantity, qualifier, remainder
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def _parse_process(
            self, elems: list[str]
            ) -> tuple[str | None, str | None, list[str]]:
        """
        Extract the role-identifying suffix, which MUST be the final element.

        The candidate is tried first against StatisticType (Av, Sd, Vr ...),
        then against VariableType (QC, Ct).  If neither matches (e.g. '2m'
        in 'Ta_2m') it is left in place for the location parsers to consume.

        Args:
            elems: list of name elements (substrings).

        Returns:
            statistic_id (or None), variable_type_id (or None), and remaining
            elements.  Exactly one of the two id fields is non-None when a
            suffix is recognised.
        """

        statistic_id = None
        variable_type_id = None

        if elems:
            candidate = elems[-1]

            try:
                StatisticType.from_suffix(candidate)
                statistic_id = candidate
            except ValueError:
                pass

            if statistic_id is None:
                try:
                    VariableType.from_suffix(candidate)
                    variable_type_id = candidate
                except ValueError:
                    pass

            if statistic_id is not None or variable_type_id is not None:
                elems = elems[:-1]

        return statistic_id, variable_type_id, elems
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def _parse_vertical_location(
            self, elems: list[str]
            ) -> tuple[str | None, list[str]]:
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
        if elems:
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
            self, elems: list[str]
            ) -> tuple[str | None, list[str]]:
        """
        Extract horizontal location.

        Args:
            elems: list of name elements (substrings).

        Returns:
            horizontal location and remaining elements.

        """

        horizontal_location = None
        if elems:
            candidate = elems[0]
            if candidate[0].islower():
                horizontal_location = candidate[0]
                remainder = candidate[1:]
                elems = [remainder] if remainder else []
        return horizontal_location, elems
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    
    def _parse_replicate(self, elems: list[str]) -> tuple[str | None, list[str]]:
        """
        Extract replicates.

        Args:
            elems: list of name elements (substrings).

        Returns:
            replicate and remaining elements.

        """
        
        replicate = None
        if elems:
            candidate = elems[0]
            if candidate.isdigit():
                replicate = candidate
                elems = elems[1:]
        return replicate, elems
# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################

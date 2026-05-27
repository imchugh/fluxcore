#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:42:59 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from dataclasses import asdict

from domain.data_models.metadata_classes import CanonicalQuantityMetadata
from domain.enums import StatisticType, VariableType
from services import config_loader
from services.metadata.metadata_conversion_service import resolve_variance_units

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------
class CanonicalQuantityRegistry:
    """
    Registry containing invariant canonical quantity metadata plus
    logic for deriving realized metadata representations (QC, variance).
    """

    # -------------------------------------------------------------------------
    def __init__(
            self, 
            quantity_definitions: dict[str, CanonicalQuantityMetadata]
            ):
        """
        Parameters
        ----------
        quantity_definitions : dict
            Mapping of quantity name to materialised CanonicalQuantityMetadata.
        """

        self._registry = quantity_definitions

    # -------------------------------------------------------------------------
    def has_quantity(self, quantity: str) -> bool:
        """Check whether quantity exists in registry."""

        return quantity in self._registry

    # -------------------------------------------------------------------------
    def get_base_metadata(
        self,
        quantity: str
    ) -> CanonicalQuantityMetadata:
        """
        Return invariant canonical metadata.
        """

        if quantity not in self._registry:
            raise KeyError(f"Unknown canonical quantity: {quantity}")

        return self._registry[quantity]

    # -------------------------------------------------------------------------
    def resolve_metadata(
        self,
        quantity: str,
        variable_type: VariableType = VariableType.CONTINUOUS,
        statistic_type: StatisticType | None = None,
        ) -> CanonicalQuantityMetadata:
        """
        Return realized metadata for a variable context.

        Examples
        --------
        CO2 + variance
            -> squared units

        CO2 + quality_flag
            -> dimensionless units
        """

        base = self.get_base_metadata(quantity)

        # Deep copy prevents accidental mutation
        attrs = asdict(base)

        # ---------------------------------------------------------------------
        # QUALITY FLAGS
        # ---------------------------------------------------------------------

        if variable_type == VariableType.QUALITY_FLAG:

            attrs["standard_units"] = "dimensionless"
            attrs["valid_input_units"] = ["dimensionless"]

            attrs["plausible_max"] = None
            attrs["plausible_min"] = None

            if attrs["long_name"]:
                attrs["long_name"] = (
                    f"{attrs['long_name']} quality flag"
                )

            if attrs["standard_name"] is not None:
                attrs["standard_name"] = (
                    f"{attrs['standard_name']}_quality_flag"
                )

        # ---------------------------------------------------------------------
        # VARIANCE
        # ---------------------------------------------------------------------

        elif statistic_type == StatisticType.VAR:

            attrs["standard_units"] = resolve_variance_units(
                units=base.standard_units, from_variance=False
                )

            attrs["valid_input_units"] = [
                resolve_variance_units(
                    units=units, 
                    from_variance=False
                    )
                for units in attrs["valid_input_units"]
                ]

            if base.plausible_min is not None:
                attrs["plausible_min"] = 0
            if base.plausible_max is not None:
                attrs["plausible_max"] = base.plausible_max ** 2

        # ---------------------------------------------------------------------

        return CanonicalQuantityMetadata(**attrs)

    # -------------------------------------------------------------------------

    @property
    def quantities(self) -> list[str]:
        """Return sorted quantity names."""

        return sorted(self._registry.keys())
    # -------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------
def build_canonical_quantity_registry() -> CanonicalQuantityRegistry:
    """
    Build the canonical quantity registry from the standard config file.

    This is the canonical factory for production use. The registry class
    itself accepts already-materialised quantity definitions, so tests or
    other callers can construct an instance directly without touching the
    config file.
    """

    raw = config_loader.load_config_file_from_name('canonical_quantities')
    return CanonicalQuantityRegistry(
        quantity_definitions={
            quantity: CanonicalQuantityMetadata(**quantity_dict)
            for quantity, quantity_dict in raw.items()
            }
        )

# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################

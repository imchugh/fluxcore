#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:59:42 2026
@author: imchugh

Assembles SiteRuntimeConfig — the runtime variable metadata object containing
all information required to build a generic dataset from site data.

Responsibilities:
    - structural validation of the site config YML
      (via variable_structural_validator)
    - variable name parsing and syntax validation
      (via variable_syntax_parser)
    - canonical quantity resolution for each variable
      (via canonical_quantity_registry)
    - assembly of VariableDefinition and SiteRuntimeConfig objects

"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

# -----------------------------------------------------------------------------

from services.metadata.variable_structural_validator import validate_L1_config_structure
from services.metadata.variable_syntax_parser import NameParser
from services.metadata.canonical_quantity_registry import build_canonical_quantity_registry
from domain.data_models.metadata_classes import RawVariableMetadata, VariableDefinition
from domain.enums import FileType, FluxSystemType

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SiteRuntimeConfig:

    site_name: str

    # Store FILE FORMATS, not extensions
    file_format_default: str
    file_format_overrides: dict[str, str]

    flux_system: FluxSystemType
    flux_file: str

    custom_metadata: dict[str, dict[str, dict[str, str]]] | None

    variables: dict[str, VariableDefinition]

    # -------------------------------------------------------------------------
    # File format helpers
    # -------------------------------------------------------------------------

    def get_file_format(self, file_group: str) -> str:
        """
        Resolve file format name for file group.

        Returns:
            e.g. 'CSI'
        """

        return self.file_format_overrides.get(
            file_group,
            self.file_format_default
            )

    # -------------------------------------------------------------------------

    def get_file_type(self, file_group: str) -> FileType:
        """
        Resolve FileType enum for file group.
        """

        format_name = self.get_file_format(file_group)

        return FileType[format_name]

    # -------------------------------------------------------------------------

    def get_file_extension(self, file_group: str) -> str:
        """
        Resolve extension for file group.

        Returns:
            e.g. 'dat'
        """

        return self.get_file_type(file_group).extension

    # -------------------------------------------------------------------------

    def get_filename(self, file_group: str) -> str:
        """
        Construct canonical filename.
        """

        ext = self.get_file_extension(file_group)

        return f"{file_group}.{ext}"

    # -------------------------------------------------------------------------

    @property
    def flux_filename(self) -> str:
        """
        Canonical flux filename.
        """

        return self.get_filename(self.flux_file)

    # -------------------------------------------------------------------------

    @cached_property
    def input_file_groups(self) -> tuple[str, ...]:
    
        rslt = {
            raw_var.file
            for attrs in self.variables.values()
            for raw_var in attrs.raw_inputs
            if raw_var.file
            }
    
        return tuple(sorted(rslt))

    # -------------------------------------------------------------------------
    
    @cached_property
    def sonic_instrument(self) -> str | None:
        return compute_sonic_instrument(self.variables)

    @cached_property
    def irga_instrument(self) -> str | None:
        return compute_irga_instrument(self.variables)
    
# -----------------------------------------------------------------------------



###############################################################################
### END CLASSES ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------

def compute_sonic_instrument(
        variables: dict[str, VariableDefinition]
        ) -> str | None:
    """
    Get the sonic instrument.
                                                                    
    Args:
        variables: dict collection of variable definitions.

    Returns:
        name of instrument.

    """
    
    return _compute_instrument(
        variables=variables, 
        instrument_substring='_SONIC'
        )
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def compute_irga_instrument(
        variables: dict[str, VariableDefinition],
        ) -> str | None:

    """
    Get the irga instrument.
                                                                    
    Args:
        variables: dict collection of variable definitions.

    Returns:
        name of instrument.

    """
    
    return _compute_instrument(
        variables=variables, 
        instrument_substring='_IRGA'
        )
# -----------------------------------------------------------------------------
    
# -----------------------------------------------------------------------------

def _compute_instrument(
        variables: dict[str, VariableDefinition],
        instrument_substring: str,
        ) -> str | None:
    """
    Get the instrument from the dict of variable definitions (note that
    validation function has already enforced only a single instrument can be
    defined for SONIC or IRGA variables - therefore no error raising).

    Args:
        variables: dict collection of variable definitions.

    Returns:
        name of instrument.

    """
    
    instruments = set()
    for name, var in variables.items():
        if instrument_substring in name:
            for raw_input in var.raw_inputs:
                instruments.add(raw_input.instrument)
    return next(iter(instruments), None)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def load_runtime_config(file_path: Path) -> SiteRuntimeConfig:  
    """
    Assemble the runtime_config class.

    Args:
        file_path: absolute path to file.

    Raises:
        ValueError: raised if any variable names in the config file are not 
        found in the canonical variable definitions.

    Returns:
        the runtime_config class

    """

    # Initialise parser
    name_parser = NameParser()
    
    # Load canonical metadata
    canonical_metadata = build_canonical_quantity_registry()
    
    # Validate site-level configuration file
    validated_config = validate_L1_config_structure(file=file_path)

    # Load custom metadata
    custom_metadata = (
        validated_config.custom_metadata.model_dump()
        if validated_config.custom_metadata
        else None
        )

    # Build VariableDefinitions
    site_variables = {}
    for variable, raw_cfg in validated_config.variables.items():
        
        # Parse the variable name (syntax validation)
        parsed_name_elems = name_parser.parse_variable_name(variable_name=variable)

        # Ensure the quantity exists in canonical metadata
        quantity = parsed_name_elems.quantity
        if quantity not in canonical_metadata.quantities:
            raise ValueError(
                f"Variable '{variable}': canonical quantity '{quantity}' not found"
                )

        # Resolve its output (ensure canonical fields are correct for
        # variances, QC vals etc)
        quantity_canonical_metadata = canonical_metadata.resolve_metadata(
            quantity=quantity,
            variable_type=raw_cfg.variable_type,
            statistic_type=raw_cfg.statistic_type
            )

        # Assemble per-input metadata
        raw_inputs = tuple(
            RawVariableMetadata(
                raw_name=raw_name,
                raw_units=cfg.units,
                instrument=cfg.instrument,
                file=cfg.file,
                begin=cfg.begin,
                end=cfg.end,
            )
            for raw_name, cfg in raw_cfg.input_variables.items()
        )

        # instrument and diag_type are validated to be consistent across
        # all inputs (enforced by validate_L1_config_structure), so take
        # from the first input explicitly rather than relying on loop-scope.
        first_input = next(iter(raw_cfg.input_variables.values()))

        site_variables[variable] = VariableDefinition(
            variable_name=variable,
            quantity=parsed_name_elems.quantity,
            variable_type=raw_cfg.variable_type,
            instrument=first_input.instrument,
            height=raw_cfg.height,
            statistic_type=raw_cfg.statistic_type,
            raw_inputs=raw_inputs,
            canonical=quantity_canonical_metadata,
            parsed_name_elems=parsed_name_elems,
            diag_type=first_input.diag_type,
            )

    # Return immutable SiteRuntimeConfig object
    return SiteRuntimeConfig(
        site_name=validated_config.site,
        file_format_default=validated_config.file_formats.default,
        file_format_overrides=validated_config.file_formats.overrides,
        flux_system=validated_config.flux_system,
        flux_file=validated_config.flux_file,
        custom_metadata=custom_metadata,
        variables=site_variables
        )
# -----------------------------------------------------------------------------    
    
###############################################################################
### END FUNCTIONS ###
###############################################################################
    
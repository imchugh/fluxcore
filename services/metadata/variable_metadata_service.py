#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:59:42 2026
@author: imchugh

This module creates a runtime variable metadata configuration class containing
information required to build a generic dataset from site data.

It combines validation functionality from the following modules:
    - variable_metadata_validator: structural validation of the input yml
    - variable_syntax_parser: syntax validation of the top-level yml variable 
    names
It creates (static) class-based variable object based on definitions in 
variable_definition_builder.
It combines variable metadata with canonical metadata accessed from 
configs/pfp_std_names.yml

"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional

# -----------------------------------------------------------------------------

from services.metadata.variable_definition_builder import build_variable_definition
from services.metadata.variable_structural_validator import validate_L1_config_structure
from services.metadata.variable_syntax_parser import NameParser
from services.metadata.canonical_quantity_registry import build_canonical_quantity_registry
from domain.enums import FileType

if TYPE_CHECKING:
    from domain.metadata.variable_definition import VariableDefinition

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# # -----------------------------------------------------------------------------
# @dataclass(frozen=True)
# class FileFormatResolver:
#     """Simple class to store file format defaults and overrides, """
    
#     default: str
#     overrides: Dict[str, str]

#     def resolve(self, file_group: str) -> str:
#         """
#         Return override format if file group name is in overrides, 
#         otherwise default
#         """
        
#         return self.overrides.get(file_group, self.default)
# # -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

# @dataclass(frozen=True)
# class SiteRuntimeConfig:
#     """Simple container for: 
#         1) site name, and;
#         2) the loaded and validated variable definition  

#     """
    
#     site_name: str
#     file_formats: FileFormatResolver
#     flux_system: str
#     flux_file: str
#     custom_metadata: Dict[str, Dict[str, Dict[str, str]]]
#     variables: Dict[str, 'VariableDefinition']

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SiteRuntimeConfig:

    site_name: str

    # Store FILE FORMATS, not extensions
    file_format_default: str
    file_format_overrides: dict[str, str]

    flux_system: str
    flux_file: str

    custom_metadata: Optional[dict[str, dict[str, dict[str, str]]]]

    variables: dict[str, 'VariableDefinition']

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
    def sonic_instrument(self) -> Optional[str]:
        return compute_sonic_instrument(self.variables)
    
    @cached_property
    def irga_instrument(self) -> Optional[str]:
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
        ) -> Optional[str]:
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
        ) -> Optional[str]:

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
        ) -> Optional[str]:
    """
    Get the instrument from the dict of variable definitions (note that
    validation function has already enforced only a single instrument can be
    defined for SONIC or IRGA variables - therefore no error raising).

    Args:
        variables: dict collection of variable definitions.

    Returns:
        name of instrument.

    """
    
    instruments = []
    for name, var in variables.items():
        for raw_input in var.raw_inputs:
            if instrument_substring in name:
                instruments.append(raw_input.instrument)
    instruments = set(instruments)
    return next(iter(instruments), None) if instruments else None
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
    # load_config_file_from_name('canonical_variables')
    
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

        # Resolve it's output (ensure canonical fields are correct for 
        # variances, QC vals etc) 
        quantity_canonical_metadata = canonical_metadata.resolve_metadata(
            quantity=quantity,
            variable_type=raw_cfg.variable_type,
            statistic_type=raw_cfg.statistic_type
            )

        # Merge raw config + parsed name + canonical metadata into a simple 
        # metadata structure
        var_def = build_variable_definition(
            var_name=variable,
            raw_cfg=raw_cfg,
            parsed_name_elems=parsed_name_elems,
            canonical_metadata=quantity_canonical_metadata
            )
        site_variables[variable] = var_def

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
    
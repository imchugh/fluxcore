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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# -----------------------------------------------------------------------------

from infrastructure.paths import get_local_stream_path

from services.metadata.variable_definition_builder import build_variable_definition, VariableDefinition
from services.metadata.variable_metadata_validator import validate_L1_config_structure
from services.metadata.variable_syntax_parser import NameParser
from services.config_loader import load_config_file_from_name

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FileFormatResolver:
    """Simple class to store file format defaults and overrides, """
    
    default: str
    overrides: Dict[str, str]

    def resolve(self, file_group: str) -> str:
        """
        Return override format if file group name is in overrides, 
        otherwise default
        """
        
        return self.overrides.get(file_group, self.default)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteRuntimeConfig:
    """Simple container for: 
        1) site name, and;
        2) the loaded and validated variable definition  

    """
    
    site_name: str
    file_formats: FileFormatResolver
    custom_metadata: Dict[str, Dict[str, Dict[str, str]]]
    variables: Dict[str, 'VariableDefinition']
    
    @property
    def sonic_instrument(self) -> Optional[str]:
        return compute_sonic_instrument(self.variables)
    
    @property
    def irga_instrument(self) -> Optional[str]:
        return compute_irga_instrument(self.variables)
    
    @property
    def diag_type(self) -> Optional[str]:
        return compute_diag_type(self.variables)
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

def compute_diag_type(variables: dict[str, VariableDefinition]) -> Optional[str]:
    """
    Get the diagnostic type from the dict of variable definitions (note that
    validation function has already enforced only a single diagnostic type can be
    diagnosed for any diagnostic variable - therefore no error raising).

    Args:
        variables: dict collection of variable definitions.

    Returns:
        diag type.

    """
    
    
    diag_prefixes = ["Diag_"]
    diag_types = {
        var.raw.diag_type
        for name, var in variables.items()
        if any(name.startswith(prefix) for prefix in diag_prefixes)
        }
    return next(iter(diag_types), None) if diag_types else None
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
    canonical_metadata = load_config_file_from_name('pfp_std_names')
    
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
        quantity = parsed_name_elems["quantity"]
        if quantity not in canonical_metadata:
            raise ValueError(
                f"Variable '{variable}': canonical quantity '{quantity}' not found"
            )
        
        # Merge raw config + parsed name + canonical metadata into a simple 
        # metadata structure
        var_def = build_variable_definition(
            var_name=variable,
            raw_cfg=raw_cfg,
            parsed_name_elems=parsed_name_elems,
            canonical_metadata=canonical_metadata[quantity]
            )
        site_variables[variable] = var_def

    # Return immutable SiteRuntimeConfig object
    return SiteRuntimeConfig(
        site_name=validated_config.site,
        file_formats=FileFormatResolver(
            default=validated_config.file_formats.default,
            overrides=validated_config.file_formats.overrides
            ),
        custom_metadata=custom_metadata,
        variables=site_variables
        )
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def dictify_custom_metadata(metadata):
    
    if metadata is None: 
        return
    rslt = {}
    for name, field in dict(metadata).items():
        sub_rslt = {}
        for sub_name, sub_field in dict(field).items():
            sub_rslt[sub_name] = sub_field
        rslt[name] = sub_rslt
    return rslt
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def load_runtime_config_by_site(site: str) -> SiteRuntimeConfig:
    """
    Convenience function for load_runtime_config to allow site-based calling.

    Args:
        site: name of site.

    Returns:
        the SiteRunTimeConfig object that contains the validator plus mapping.

    """
    
    # # Get base directory
    # file_path = get_local_stream_path(
    #     resource='configs', 
    #     stream='site_config_files',
    #     )
    
    
    
    # Temporary hack
    # file_path = file_path.parent / 'Test' / f'{site}.yml'
    
    file_path = Path(f'/opt/TERN_EP/site_configs/new_exp/{site}.yml')
    
    # Do static validation
    return load_runtime_config(file_path=file_path)
# -----------------------------------------------------------------------------        
    
###############################################################################
### END FUNCTIONS ###
###############################################################################
    
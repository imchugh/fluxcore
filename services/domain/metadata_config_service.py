#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:59:42 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# -----------------------------------------------------------------------------

from infrastructure.paths import get_local_stream_path

from services.domain.variable_definition_builder import build_variable_definition, VariableDefinition
from services.domain.variable_metadata_validator import validate_L1_config_structure
from services.domain.variable_syntax_parser import NameParser
from services.domain.config_loader import load_config_file_from_name

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteRuntimeConfig:
    """Simple container for: 
        1) site name, and;
        2) the loaded and validated variable definition  

    """
    
    site_name: str
    system_type: str
    variables: Dict[str, 'VariableDefinition']
    
    @property
    def sonic_instrument(self) -> Optional[str]:
        return compute_sonic_instrument(self.variables)
    
    @property
    def irga_instrument(self) -> Optional[str]:
        return compute_irga_instrument(self.variables)
    
    @property
    def flux_suffix(self) -> Optional[str]:
        return compute_flux_suffix(self.variables)
    
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
    Get the sonic instrument from the dict of variable definitions (note that
    validation function has already enforced only a single instrument can be
    defined for any _SONIC variable - therefore no error raising).
                                                                    
    Args:
        variables: dict collection of variable definitions.

    Returns:
        name of instrument.

    """
    
    instruments = {
        var.raw.instrument
        for name, var in variables.items()
        if name.endswith("_SONIC")
        }
    return next(iter(instruments), None) if instruments else None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def compute_irga_instrument(
        variables: dict[str, VariableDefinition]
        ) -> Optional[str]:
    """
    Get the irga instrument from the dict of variable definitions (note that
    validation function has already enforced only a single instrument can be
    defined for any _IRGA variable - therefore no error raising).

    Args:
        variables: dict collection of variable definitions.

    Returns:
        name of instrument.

    """
    
    instruments = {
        var.raw.instrument
        for name, var in variables.items()
        if name.endswith("_IRGA")
        }
    return next(iter(instruments), None) if instruments else None
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def compute_flux_suffix(
        variables: dict[str, VariableDefinition]
        ) -> Optional[str]:
    """
    Get the flux suffix from the dict of variable definitions (note that
    validation function has already enforced only a single suffix can be
    defined for any flux variable - therefore no error raising).

    Args:
        variables: dict collection of variable definitions.

    Returns:
        suffix.

    """
    
    flux_prefixes = ["Fco2", "Fe", "Fh", "Fm", "ustar"]
    suffixes = set()
    for name in variables:
        for prefix in flux_prefixes:
            if name.startswith(prefix):
                parts = name.split("_", 1)
                if len(parts) == 2:
                    suffixes.add(parts[1])
    return next(iter(suffixes), None) if suffixes else None
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

    # Initialise parser and load canonical metadata
    name_parser = NameParser()
    canonical_metadata = load_config_file_from_name('pfp_std_names')
    
    # Validate site-level configuration
    validated_config = validate_L1_config_structure(file=file_path)

    # Build VariableDefinitions
    site_variables = {}
    for variable, raw_cfg in validated_config.variables.items():
        
        # Parse the variable name (syntax validation)
        parsed_name = name_parser.parse_variable_name(variable_name=variable)

        # Ensure the quantity exists in canonical metadata
        quantity = parsed_name["quantity"]
        if quantity not in canonical_metadata:
            raise ValueError(
                f"Variable '{variable}': canonical quantity '{quantity}' not found"
            )
            
        # Merge raw config + parsed name + canonical metadata
        var_def = build_variable_definition(
            site_var_name=variable,
            raw_config=dict(raw_cfg),
            parsed_name=parsed_name,
            canonical=canonical_metadata[quantity]
            )
        site_variables[variable] = var_def

    # Return immutable SiteRuntimeConfig object
    return SiteRuntimeConfig(
        site_name=validated_config.site,
        system_type=validated_config.system_type,
        variables=site_variables
        )
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
    
    # Get base directory
    file_path = get_local_stream_path(
        resource='configs', 
        stream='site_config_files',
        )
    
    # Temporary hack
    file_path = file_path.parent / 'Test' / f'{site}.yml'
    
    # Do static validation
    return load_runtime_config(file_path=file_path)
# -----------------------------------------------------------------------------        
    
###############################################################################
### END FUNCTIONS ###
###############################################################################
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:09:44 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from infrastructure import paths
from pathlib import Path
from services.domain.metadata_config_service import SiteRuntimeConfig
from services.domain.variable_definition_builder import RawVariableMetadata

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN INITS ###
###############################################################################

FLUX_VAR = 'UzT'

###############################################################################
### END INITS ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------

def resolve_filename(site_name: str, variable: RawVariableMetadata) -> str:
    """
    Either return the existing file name or build and return if doesn't exist.

    Args:
        site_name: name of site.
        variable: raw variable metadata class.

    Returns:
        file name.

    """

    if variable.file:
        return variable.file

    return f"{site_name}_{variable.logger}_{variable.table}.dat"
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def resolve_variable_path(
    site_name: str,
    variable: RawVariableMetadata
    ) -> Path:
    """
    Get the absolute path to a specific variable.

    Args:
        site_name: name of site.
        variable: raw variable metadata class.

    Returns:
        absolute path to file.

    """

    base_path = _get_site_raw_data_path(site_name)
    filename = resolve_filename(site_name, variable)

    return base_path / filename
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build_file_map(runtime_cfg: SiteRuntimeConfig) -> dict[Path, list[str]]:
    """
    Map absolute file path to list of raw variable names expected in that file.

    Args:
        runtime_cfg: site-specific runtime configuration class.

    Returns:
        mapping dictionary.

    """

    files: dict[Path, list[str]] = {}

    for var_def in runtime_cfg.variables.values():

        path = resolve_variable_path(
            runtime_cfg.site_name,
            var_def.raw
        )

        files.setdefault(path, []).append(var_def.raw.raw_name)

    return files
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_flux_file_path(runtime_cfg: SiteRuntimeConfig) -> Path:
    """
    Get the raw data path for the flux file.

    Args:
        runtime_cfg: site-specific runtime configuration class.

    Returns:
        Path: DESCRIPTION.

    """
    
    for var_name, var_def in runtime_cfg.variables.items():
        if var_def.raw.quantity == FLUX_VAR:
            return resolve_variable_path(
                site_name=runtime_cfg.site_name, variable=var_def.raw)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def _get_site_raw_data_path(site_name: str) -> Path:
    """
    Get the raw data path for the site.

    Args:
        site_name: name of site.

    Returns:
        absolute path to parent directory.

    """
    
    return paths.get_local_stream_path(
        resource="raw_data",
        stream="flux_slow",
        site=site_name
    )
# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################

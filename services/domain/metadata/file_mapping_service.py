#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:09:44 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from infrastructure import paths, file_io
from pathlib import Path
from services.domain.metadata_config_service import SiteRuntimeConfig

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

def build_file_map(runtime_cfg: SiteRuntimeConfig) -> dict[Path, list[str]]:
    """
    Map absolute file path to list of raw variable names expected in that file.

    Args:
        runtime_cfg: site-specific runtime configuration class.

    Returns:
        mapping dictionary.

    """

    files: dict[Path, list[str]] = {}

    base_path = get_base_path(site_name=runtime_cfg.site_name)

    for var_def in runtime_cfg.variables.values():

        for raw_var in var_def.raw:
            path = base_path / raw_var.file

            files.setdefault(path, []).append(raw_var.raw_name)

    return files
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build_file_list(runtime_cfg: SiteRuntimeConfig) -> dict[Path, list[str]]:
    """
    Map absolute file path to list of raw variable names expected in that file.

    Args:
        runtime_cfg: site-specific runtime configuration class.

    Returns:
        mapping dictionary.

    """

    files = []

    base_path = get_base_path(site_name=runtime_cfg.site_name)

    for var_def in runtime_cfg.variables.values():

        for raw_var in var_def.raw:
            files.append(base_path / raw_var.file)

    return list(set(files))
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build_file_groups(runtime_cfg: SiteRuntimeConfig):
    
    # Get the list of files
    file_list = build_file_list(runtime_cfg=runtime_cfg)
    
    # Get their file groups (i.e. include all backups)
    file_groups = {}
    for file in file_list:
        
        group_name = file.name
        group_dict = {}
        
        master = file.parent / f'{file.name}.dat'
        slaves = file_io.get_backup_files(file_path=master)
        
        group_dict['master'] = master
        group_dict['slaves'] = slaves
        
        file_groups[group_name] = group_dict
        
    return file_groups
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def get_base_path(site_name) -> Path:
    
    return paths.get_local_stream_path(
        resource="raw_data",
        stream="flux_slow",
        site=site_name
        )
# -----------------------------------------------------------------------------

###############################################################################
### END FUNCTIONS ###
###############################################################################

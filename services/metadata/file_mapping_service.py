#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:09:44 2026

@author: imchugh
"""

###############################################################################
### BEGIN IMPORTS ###
###############################################################################

from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, Dict, List

from domain.enums import FileType
from services.metadata.variable_metadata_service import SiteRuntimeConfig
from services.data import raw_data_loader
from infrastructure import paths, file_io

###############################################################################
### END IMPORTS ###
###############################################################################


###############################################################################
### BEGIN CLASSES ###
###############################################################################

# -----------------------------------------------------------------------------

@dataclass
class FileGroup:
    """
    Represents a group of files (master + backups) and the variables 
    expected and actually found in them. Header discovery is lazy.
    """
    group: str
    master: Path
    backups: List[Path]
    file_format: str
    expected_variables: Set[str] = field(default_factory=set)
    _variables_by_file_cache: Dict[Path, Set[str]] = field(default_factory=dict, init=False, repr=False)

    # -------------------------------------------------------------------------
    
    @property
    def all_files(self) -> List[Path]:
        return [self.master, *self.backups]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    @property
    def variables_by_file(self) -> Dict[Path, Set[str]]:
        """
        Lazy evaluation: read headers only on first access and cache results.
        """
        if not self._variables_by_file_cache:
            for file in self.all_files:
                header_vars = get_variables_from_file(
                    file_path=file, system_type=self.file_format
                    )
                self._variables_by_file_cache[file] = set(header_vars)
        return self._variables_by_file_cache
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    
    def validate(self) -> Dict[str, Set[str]]:
        """
        Compare expected variables vs discovered variables.
        Returns a dict with 'found' and 'missing' sets.
        """
        found = set().union(*self.variables_by_file.values())
        missing = self.expected_variables - found
        return {"found": found & self.expected_variables, "missing": missing}
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------

    def files_for_variable(self, variable: str) -> List[Path]:
        """
        Return the list of files in which the variable was found.
        """
        return [f for f, vars in self.variables_by_file.items() if variable in vars]
    # -------------------------------------------------------------------------

# -----------------------------------------------------------------------------

###############################################################################
### END CLASSES ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------

def get_variables_from_file(
        file_path: Path | str, system_type: str, incl_backups: bool=False
        ):
    """Get the variable names from the file header"""
    
    # Set FileType
    ftype = FileType[system_type]
    
    # Sort out paths
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'File {file_path} does not exist!')
    
    # Build iterable
    file_list = [file_path]
    if incl_backups:
        file_list.extend(file_io.get_backup_files(file_path=file_path))

    # Load adapter
    header_adapter = raw_data_loader.get_header_adapter(
        system_type=ftype.name
        )

    # Iterate over files
    rslt = set()
    for file in file_list:
        rslt.update(header_adapter.load(file_path=file)['variable'])
    
    # Return a list
    return sorted(rslt)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

def build_file_groups(runtime_cfg: SiteRuntimeConfig) -> Dict[str, FileGroup]:
    """
    Build FileGroup objects for all variable groups in the runtime config.
    """
    base_path = paths.get_local_stream_path(
        resource="raw_data",
        stream="flux_slow",
        site=runtime_cfg.site_name
        )
    groups: Dict[str, FileGroup] = {}

    for var_def in runtime_cfg.variables.values():
        for raw_var in var_def.raw_inputs:
            group_name = raw_var.file
            file_format = runtime_cfg.get_file_format(group_name)
            file_ext = FileType[file_format].extension
            group = groups.get(group_name)
            if group is None:
                master = base_path / f"{group_name}.{file_ext}"
                backups = file_io.get_backup_files(
                    file_path=master,
                    abs_path=True,
                    )
                group = FileGroup(
                    group=group_name,
                    master=master,
                    backups=backups,
                    file_format=file_format,
                    )
                groups[group_name] = group

            group.expected_variables.add(raw_var.raw_name)

    return groups
# -----------------------------------------------------------------------------

###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

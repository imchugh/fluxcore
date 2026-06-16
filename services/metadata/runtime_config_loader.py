#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:59:42 2026
@author: imchugh

Assembles SiteRuntimeConfig — the runtime variable metadata object containing
all information required to build a generic dataset from site data.

Responsibilities:
    - structural validation of the site config YML
      (via site_config_schema)
    - variable name parsing and syntax validation
      (via variable_name_parser)
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

from services.metadata.site_config_schema import validate_L1_config_structure
from services.metadata.variable_name_parser import NameParser
from services.metadata.canonical_quantity_registry import build_canonical_quantity_registry
from domain.data_models.metadata_classes import RawVariableMetadata, VariableDefinition
from domain.enums import FileType, FluxSystemType, StatisticType, VariableType

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
    

# -----------------------------------------------------------------------------



###############################################################################
### END CLASSES ###
###############################################################################


###############################################################################
### BEGIN FUNCTIONS ###
###############################################################################

# -----------------------------------------------------------------------------

def _check_name_config_consistency(
        variable: str,
        parsed_name,
        statistic_type,
        variable_type,
        ) -> None:
    """
    Cross-check parsed name fields against config declarations.

    The parser resolves the terminal suffix into either statistic_id (a
    StatisticType suffix: Av, Sd, Vr ...) or variable_type_id (a
    VariableType suffix: QC, Ct).  The config independently declares
    statistic_type and variable_type.  This function verifies they agree.

    Args:
        variable:      canonical variable name (for error messages).
        parsed_name:   ParsedVariableName produced by the parser.
        statistic_type: StatisticType declared in the config, or None.
        variable_type:  VariableType declared in the config.
    """

    if parsed_name.statistic_id is not None:
        stat_from_name = StatisticType.from_suffix(parsed_name.statistic_id)
        if statistic_type != stat_from_name:
            raise ValueError(
                f"Variable '{variable}': name suffix '{parsed_name.statistic_id}'"
                f" implies statistic_type={stat_from_name.value!r}, but config"
                f" declares statistic_type={statistic_type!r}"
                )

    if parsed_name.variable_type_id is not None:
        vtype_from_name = VariableType.from_suffix(parsed_name.variable_type_id)
        if variable_type != vtype_from_name:
            raise ValueError(
                f"Variable '{variable}': name suffix '{parsed_name.variable_type_id}'"
                f" implies variable_type={vtype_from_name.value!r}, but config"
                f" declares variable_type={variable_type!r}"
                )

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

        # Cross-check name suffix against config declarations
        _check_name_config_consistency(
            variable=variable,
            parsed_name=parsed_name_elems,
            statistic_type=raw_cfg.statistic_type,
            variable_type=raw_cfg.variable_type,
            )

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

        # Validate that each input's declared units are acceptable for this
        # quantity. COUNTER variables are skipped — their units field is a
        # placeholder ("dimensionless"); the meaningful unit (valid_count vs
        # invalid_count) is carried by diag_type and validated separately.
        if raw_cfg.variable_type != VariableType.COUNTER:
            for raw_name, cfg in raw_cfg.input_variables.items():
                if cfg.units not in quantity_canonical_metadata.valid_input_units:
                    raise ValueError(
                        f"Variable '{variable}' input '{raw_name}': "
                        f"units '{cfg.units}' is not valid for quantity "
                        f"'{quantity}'. Expected one of: "
                        f"{quantity_canonical_metadata.valid_input_units}"
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

        first_input = next(iter(raw_cfg.input_variables.values()))

        site_variables[variable] = VariableDefinition(
            variable_name=variable,
            quantity=parsed_name_elems.quantity,
            variable_type=raw_cfg.variable_type,
            instrument=first_input.instrument,
            height=raw_cfg.height,
            height_range=raw_cfg.height_range,
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
    
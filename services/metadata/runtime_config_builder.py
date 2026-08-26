#!/usr/bin/env python3
"""Assemble SiteRuntimeConfig from a structurally-validated SiteConfig.

Generic EC core: name syntax, quantity/units checks, and object assembly
only. No TERN-specific resolution (instrument vocab/RDF lookups) happens
here — `instrument_uris` is an optional adapter-supplied enrichment map;
when omitted, every variable's `instrument_uri` resolves to None.

Two entry points:
    - build_runtime_config_from_file(file_path): raw YAML in, no TERN
      dependency at all.
    - build_runtime_config(validated_config, instrument_uris=None): an
      already-validated SiteConfig in (e.g. produced/enriched by a TERN
      adapter such as services/metadata/runtime_config_loader.py), plus an
      optional name -> URI enrichment map.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from domain.data_models.metadata_classes import RawVariableMetadata, VariableDefinition
from domain.enums import FileType, FluxSystemType, StatisticType, VariableType
from services.metadata.canonical_quantity_registry import (
    build_canonical_quantity_registry,
)
from services.metadata.site_config_schema import (
    SiteConfig,
    validate_L1_config_structure,
)
from services.metadata.variable_name_parser import NameParser


@dataclass(frozen=True)
class SiteRuntimeConfig:
    """Assembled runtime metadata for one site: variables, file formats, flux system."""

    site_name: str

    # Store FILE FORMATS, not extensions; keyed by file stem
    file_formats: dict[str, str]

    flux_system: FluxSystemType
    flux_file: str

    custom_metadata: dict[str, dict[str, dict[str, str]]] | None

    variables: dict[str, VariableDefinition]

    # Adapter-populated: where this site's raw input files live. None on the
    # pure-core path (build_runtime_config_from_file) — same pattern as
    # instrument_uri. Required by file_group_builder.build_file_groups.
    input_data_path: Path | None = None

    # File format helpers

    def get_file_format(self, file_group: str) -> str:
        """Resolve file format name for file group.

        Returns:
            e.g. 'CSI'
        """
        return self.file_formats[file_group]

    def get_file_type(self, file_group: str) -> FileType:
        """Resolve FileType enum for file group."""
        format_name = self.get_file_format(file_group)

        return FileType[format_name]

    def get_file_extension(self, file_group: str) -> str:
        """Resolve extension for file group.

        Returns:
            e.g. 'dat'
        """
        return self.get_file_type(file_group).extension

    def get_filename(self, file_group: str) -> str:
        """Construct canonical filename."""
        ext = self.get_file_extension(file_group)

        return f"{file_group}.{ext}"

    @property
    def flux_filename(self) -> str:
        """Canonical flux filename."""
        return self.get_filename(self.flux_file)

    @cached_property
    def input_file_groups(self) -> tuple[str, ...]:
        """Sorted, deduplicated file-group names referenced by any raw input."""
        rslt = {
            raw_var.file
            for attrs in self.variables.values()
            for raw_var in attrs.raw_inputs
            if raw_var.file
        }

        return tuple(sorted(rslt))


def _check_name_config_consistency(
    variable: str,
    parsed_name,
    statistic_type,
    variable_type,
) -> None:
    """Cross-check parsed name fields against config declarations.

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


def _resolve_instrument_uri(
    instrument: str | dict[str, str],
    instrument_uris: dict[str, str | None],
) -> str | dict[str, str] | None:
    """Resolve instrument name(s) to enrichment URI(s); None if unresolved.

    Pure lookup against an already-resolved map — this function makes no
    external calls itself. Producing the map (e.g. via TERN's RDF vocab) is
    the caller's responsibility; an empty/omitted map simply leaves every
    instrument_uri as None.

    Args:
        instrument: raw instrument name(s) from the config.
        instrument_uris: name -> URI map, or empty if none available.
    """
    if isinstance(instrument, dict):
        return {alias: instrument_uris.get(name) for alias, name in instrument.items()}
    return instrument_uris.get(instrument)


def build_runtime_config(
    validated_config: SiteConfig,
    instrument_uris: dict[str, str | None] | None = None,
    input_data_path: Path | None = None,
) -> SiteRuntimeConfig:
    """Build a SiteRuntimeConfig from a structurally-validated SiteConfig.

    Performs name-syntax, quantity-existence, and units checks as part of
    assembly, raising on the first failure found. No TERN/instrument-vocab
    calls are made here — `instrument_uris` is optional, adapter-supplied
    enrichment; instruments referenced in `validated_config` but absent
    from the map simply resolve to instrument_uri=None. Likewise,
    `input_data_path` is optional adapter-supplied data (where this site's
    raw files live); left None, callers requiring it (e.g.
    file_group_builder.build_file_groups) will raise.

    Args:
        validated_config: structurally-validated SiteConfig.
        instrument_uris: optional name -> URI map for enrichment.
        input_data_path: optional path to this site's raw input files.

    Returns:
        Immutable SiteRuntimeConfig.
    """
    instrument_uris = instrument_uris or {}

    name_parser = NameParser()
    canonical_metadata = build_canonical_quantity_registry()

    custom_metadata = (
        validated_config.custom_metadata.model_dump()
        if validated_config.custom_metadata
        else None
    )

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
            statistic_type=raw_cfg.statistic_type,
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
                instrument_uri=_resolve_instrument_uri(cfg.instrument, instrument_uris),
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
            instrument_uri=raw_inputs[0].instrument_uri,
        )

    return SiteRuntimeConfig(
        site_name=validated_config.site,
        file_formats=validated_config.file_formats,
        flux_system=validated_config.flux_system,
        flux_file=validated_config.flux_file,
        custom_metadata=custom_metadata,
        variables=site_variables,
        input_data_path=input_data_path,
    )


def build_runtime_config_from_file(
    file_path: Path,
    input_data_path: Path | None = None,
) -> SiteRuntimeConfig:
    """Assemble a SiteRuntimeConfig from a site YAML file.

    Generic-core entry point: structural validation plus assembly only, no
    TERN vocab/registry calls of any kind — every variable's instrument_uri
    resolves to None. For TERN's instrument-validated/URI-enriched
    equivalent, see services.metadata.runtime_config_loader.load_runtime_config.

    `input_data_path` is optional, caller-supplied data (where this site's
    raw files live) — left None, callers requiring it (e.g.
    file_group_builder.build_file_groups) will raise. A non-TERN caller who
    wants a fully runnable config from a single call passes their own path
    here directly.

    Args:
        file_path: absolute path to the site config YAML.
        input_data_path: optional path to this site's raw input files.

    Returns:
        Immutable SiteRuntimeConfig.
    """
    validated_config = validate_L1_config_structure(file=file_path)
    return build_runtime_config(validated_config, input_data_path=input_data_path)

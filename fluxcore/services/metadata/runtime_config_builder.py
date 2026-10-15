#!/usr/bin/env python3
"""Assemble SiteRuntimeConfig from a structurally-validated SiteConfig.

Name syntax, quantity/units checks, and object assembly only; no external
lookups. Two inputs are optional, adapter-supplied data: `instrument_uris`
(instrument name -> URI; omitted, every `instrument_uri` is None) and
`input_data_path` (where the site's raw files live; required later by
file_group_builder.build_file_groups).

Two entry points:
    - build_runtime_config_from_file(file_path, input_data_path=None): raw
      site-config YAML in.
    - build_runtime_config(validated_config, instrument_uris=None,
      input_data_path=None): an already-validated (possibly adapter-enriched)
      SiteConfig in.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic.dataclasses import dataclass as pydantic_dataclass

from fluxcore.domain.enums import (
    DEFAULT_FILE_FORMAT,
    DiagnosticType,
    FileFormat,
    FluxSystemType,
    LoggerType,
    StatisticType,
    VariableType,
)
from fluxcore.services.metadata.canonical_quantity_registry import (
    CanonicalQuantityMetadata,
    build_canonical_quantity_registry,
)
from fluxcore.services.metadata.site_config_schema import (
    SiteConfig,
    validate_L1_config_structure,
)
from fluxcore.services.metadata.variable_name_parser import (
    ParsedVariableName,
    parse_variable_name,
)


@pydantic_dataclass(frozen=True)
class RawVariableMetadata:
    """Metadata for one raw instrument-file input that feeds a canonical variable."""

    raw_name: str
    raw_units: str
    file: str
    instrument: str | dict[str, str]
    begin: datetime | str | None = None
    end: datetime | str | None = None
    instrument_uri: str | dict[str, str | None] | None = None


@pydantic_dataclass(frozen=True)
class VariableDefinition:
    """Complete definition of one canonical variable.

    Assembled by runtime_config_builder from its raw inputs, canonical
    quantity metadata, and parsed name. variable_registry then flattens each
    of its raw inputs into a VariableSpec.
    """

    # Core identity
    variable_name: str
    quantity: str
    variable_type: VariableType

    # Physical metadata
    instrument: str | dict[str, str]
    height: float

    # Structural metadata
    raw_inputs: tuple[RawVariableMetadata, ...]
    canonical: CanonicalQuantityMetadata
    parsed_name_elems: ParsedVariableName

    # Optionals
    height_range: tuple[float, float] | None = None
    statistic_type: StatisticType | None = None
    diag_type: DiagnosticType | None = None
    instrument_uri: str | dict[str, str | None] | None = None


@dataclass(frozen=True)
class SiteRuntimeConfig:
    """Assembled runtime metadata for one site: variables, file formats, flux system."""

    site_name: str

    # Logger family per file group (the site config's file_formats values)
    logger_types: dict[str, LoggerType]

    flux_system: FluxSystemType
    flux_file: str

    custom_metadata: dict[str, dict[str, dict[str, str]]] | None

    variables: dict[str, VariableDefinition]

    # Adapter-populated: where this site's raw input files live. None on the
    # pure-core path (build_runtime_config_from_file) — same pattern as
    # instrument_uri. Required by file_group_builder.build_file_groups.
    input_data_path: Path | None = None

    def get_logger_type(self, file_group: str) -> LoggerType:
        """Logger family that wrote file_group."""
        return self.logger_types[file_group]

    def get_file_format(self, file_group: str) -> FileFormat:
        """File format fluxcore reads file_group as."""
        return DEFAULT_FILE_FORMAT[self.get_logger_type(file_group)]

    def get_filename(self, file_group: str) -> str:
        """File group's master file name, e.g. 'Site_EC_flux.dat'."""
        return f"{file_group}.{self.get_file_format(file_group).extension}"

    @property
    def flux_filename(self) -> str:
        """Canonical flux filename."""
        return self.get_filename(self.flux_file)


def _check_statistic_matches_name(
    variable: str, parsed_name, statistic_type: StatisticType | None
) -> None:
    """Raise if the name's statistic suffix (Av, Sd, Vr ...) disagrees with config.

    The matching variable_type check (QC, Ct suffixes) is SiteConfig's
    enforce_name_type_consistency, which has already run on validated_config.
    """
    if parsed_name.statistic_id is None:
        return
    stat_from_name = StatisticType.from_suffix(parsed_name.statistic_id)
    if statistic_type != stat_from_name:
        raise ValueError(
            f"Variable '{variable}': name suffix '{parsed_name.statistic_id}'"
            f" implies statistic_type={stat_from_name.value!r}, but config"
            f" declares statistic_type={statistic_type!r}"
        )


def _check_no_repeated_instrument_era(
    variable: str, instruments: list[str | dict[str, str]]
) -> None:
    """Raise if an instrument returns after a changeover (e.g. A -> B -> A).

    Instrument history is keyed by instrument name, so a repeated era would
    silently overwrite the earlier one. Inputs that all share one instrument
    carry no history and are fine.
    """
    if isinstance(instruments[0], dict):
        sequences = {
            alias: [inst[alias] for inst in instruments] for alias in instruments[0]
        }
    else:
        sequences = {"instrument": instruments}

    for role, sequence in sequences.items():
        if 1 < len(set(sequence)) < len(sequence):
            raise ValueError(
                f"Variable '{variable}': {role} returns to an earlier instrument "
                f"({' -> '.join(sequence)}); instrument history cannot represent "
                "a repeated era"
            )


def _resolve_instrument_uri(
    instrument: str | dict[str, str],
    instrument_uris: dict[str, str | None],
) -> str | dict[str, str] | None:
    """Resolve instrument name(s) to enrichment URI(s); None if unresolved.

    Pure lookup against an already-resolved map — this function makes no
    external calls itself. Producing the map (e.g. from a network's
    instrument vocabulary) is
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
    assembly, raising on the first failure found. No external lookups are
    made here — `instrument_uris` is optional, adapter-supplied enrichment;
    instruments referenced in `validated_config` but absent from the map
    simply resolve to instrument_uri=None. Likewise,
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

    canonical_metadata = build_canonical_quantity_registry()

    custom_metadata = (
        validated_config.custom_metadata.model_dump()
        if validated_config.custom_metadata
        else None
    )

    site_variables = {}
    for variable, raw_cfg in validated_config.variables.items():
        # Parse the variable name (syntax validation)
        parsed_name_elems = parse_variable_name(variable)

        _check_statistic_matches_name(
            variable, parsed_name_elems, raw_cfg.statistic_type
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

        _check_no_repeated_instrument_era(
            variable, [cfg.instrument for cfg in raw_cfg.input_variables.values()]
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
        logger_types=validated_config.logger_types,
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

    Structural validation plus assembly only, with no enrichment: every
    variable's instrument_uri is None. An adapter that validates instruments
    or resolves URIs builds its own SiteConfig and calls build_runtime_config.

    `input_data_path` is where this site's raw files live; pass it to get a
    runnable config in one call. Left None, callers requiring it (e.g.
    file_group_builder.build_file_groups) will raise.

    Args:
        file_path: absolute path to the site config YAML.
        input_data_path: optional path to this site's raw input files.

    Returns:
        Immutable SiteRuntimeConfig.
    """
    validated_config = validate_L1_config_structure(file=file_path)
    return build_runtime_config(validated_config, input_data_path=input_data_path)

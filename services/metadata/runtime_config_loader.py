#!/usr/bin/env python3
"""Assemble SiteRuntimeConfig from a site YAML config.

SiteRuntimeConfig is the runtime variable metadata object containing all
information required to build a generic dataset from site data.

Responsibilities:
    - structural validation of the site config YML
      (via site_config_schema)
    - instrument name validation against the TERN instrument registry
      (via instrument_registry)
    - variable name parsing and syntax validation
      (via variable_name_parser)
    - canonical quantity resolution for each variable
      (via canonical_quantity_registry)
    - assembly of VariableDefinition and SiteRuntimeConfig objects
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from domain.data_models.metadata_classes import RawVariableMetadata, VariableDefinition
from domain.enums import FileType, FluxSystemType, StatisticType, VariableType
from services.metadata import instrument_validation_cache
from services.metadata.canonical_quantity_registry import (
    build_canonical_quantity_registry,
)
from services.metadata.site_config_schema import (
    SiteConfig,
    validate_L1_config_structure,
)
from services.metadata.variable_name_parser import NameParser

# Canonical EC flux quantities: the sole source for a site's flux-system
# sonic/irga instrument identity (see SiteRuntimeConfig._flux_instrument_pair).
FLUX_QUANTITIES = ("Fco2", "Fh", "Fe")


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

    @cached_property
    def sonic_instrument(self) -> str | None:
        """Site-wide flux-system sonic anemometer name, or None if undeclared."""
        return self._flux_instrument_pair()[0]

    @cached_property
    def irga_instrument(self) -> str | None:
        """Site-wide flux-system IRGA instrument name, or None if undeclared."""
        return self._flux_instrument_pair()[1]

    def _flux_instrument_pair(self) -> tuple[str | None, str | None]:
        """Get the site's (sonic_anemometer, irga) flux-system instrument names.

        Sourced only from the canonical EC flux quantities (Fco2/Fh/Fe),
        never from unrelated compound instrument entries elsewhere in the
        config (e.g. an integrated sensor reused for an ancillary met
        variable). Split-sensor sites declare Fco2/Fe as a
        {sonic_anemometer, irga} dict; integrated-sensor sites (single
        physical instrument does both roles) declare all three as the same
        plain string, which is then reported for both slots. Schema
        validation (`enforce_compound_instrument_consistency`) has already
        enforced consistency within each group, so no error raising is
        needed here.
        """
        dict_pairs = set()
        string_instruments = set()

        for var in self.variables.values():
            if var.quantity not in FLUX_QUANTITIES:
                continue
            for raw_input in var.raw_inputs:
                inst = raw_input.instrument
                if isinstance(inst, dict):
                    if "sonic_anemometer" in inst and "irga" in inst:
                        dict_pairs.add((inst["sonic_anemometer"], inst["irga"]))
                else:
                    string_instruments.add(inst)

        if dict_pairs:
            return next(iter(dict_pairs))
        if string_instruments:
            name = next(iter(string_instruments))
            return name, name
        return None, None


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


def _validate_instrument_names(
    validated_config: SiteConfig, content_hash: str
) -> dict[str, str | None]:
    """Check every instrument name in the config against the TERN instrument registry.

    A prior clean pass for this exact file content is trusted via
    `instrument_validation_cache` instead of re-querying the registry —
    the RDF-backed vocab lookup is the expensive/network-dependent part of
    config loading, and site configs change rarely. Any edit changes the
    content hash, so the next load for changed content naturally falls
    through to a real re-validation; nothing needs to watch for drift.

    Collects all failures before raising so the caller sees the complete
    list of unknown instruments in one pass.

    Args:
        validated_config: structurally-validated SiteConfig.
        content_hash: SHA-256 of the source file's exact bytes.

    Returns:
        Mapping of instrument name -> TERN vocab URI (or None if
        unresolvable), covering every distinct instrument referenced in
        the config.

    Raises:
        ValueError: if any instrument name is not recognised by the registry.
    """
    cached = instrument_validation_cache.lookup(content_hash)
    if cached is not None:
        return cached["instrument_uris"]

    # Deferred to avoid circular import:
    # instrument_registry → site_metadata_repository → site_registry → here
    from services.metadata import instrument_registry

    errors = []
    instrument_uris: dict[str, str | None] = {}
    for variable, var_cfg in validated_config.variables.items():
        for raw_name, input_cfg in var_cfg.input_variables.items():
            inst = input_cfg.instrument
            names = list(inst.values()) if isinstance(inst, dict) else [inst]
            for name in names:
                if not instrument_registry.is_valid_instrument(name):
                    errors.append(
                        f"Variable '{variable}' input '{raw_name}': "
                        f"unknown instrument '{name}'"
                    )
                elif name not in instrument_uris:
                    try:
                        instrument_uris[name] = instrument_registry.get_instrument_uri(
                            name
                        )
                    except KeyError:
                        instrument_uris[name] = None

    if errors:
        raise ValueError(
            f"Instrument validation failed for '{validated_config.site}':\n"
            + "\n".join(f"  {e}" for e in errors)
        )

    instrument_validation_cache.record(
        content_hash, instrument_uris=instrument_uris, label=validated_config.site
    )
    return instrument_uris


def _resolve_instrument_uri(
    instrument: str | dict[str, str],
    instrument_uris: dict[str, str | None],
) -> str | dict[str, str] | None:
    """Resolve instrument name(s) to TERN vocab URI(s); None if unresolvable.

    Some instrument names pass `_validate_instrument_names` (e.g. "pending"
    instruments declared via instrument_name_corrections.yml) but have no
    vocab entry yet, so URI resolution can legitimately fail even for a
    validated name — that case resolves to None rather than raising.

    Args:
        instrument: raw instrument name(s) from the config.
        instrument_uris: name -> URI map already resolved by
            `_validate_instrument_names`.
    """
    if isinstance(instrument, dict):
        return {alias: instrument_uris.get(name) for alias, name in instrument.items()}
    return instrument_uris.get(instrument)


def _build_runtime_config(
    validated_config: SiteConfig, instrument_uris: dict[str, str | None]
) -> SiteRuntimeConfig:
    """Build a SiteRuntimeConfig from a structurally/semantically-validated SiteConfig.

    Performs name-syntax, quantity-existence, and units checks as part of
    assembly, raising on the first failure found.

    Args:
        validated_config: structurally-validated SiteConfig (instrument names
            already verified by _validate_instrument_names).
        instrument_uris: name -> URI map resolved by _validate_instrument_names.

    Returns:
        Immutable SiteRuntimeConfig.
    """
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
    )


def load_runtime_config(file_path: Path) -> SiteRuntimeConfig:
    """Assemble a SiteRuntimeConfig from a site YAML config file.

    Three phases:
        1. Structural validation  — YAML schema (site_config_schema)
        2. Instrument validation  — all instrument names against TERN registry
           (skipped, per content hash, when a prior clean pass is cached —
           see instrument_validation_cache)
        3. Build                  — name syntax, quantity/units checks, assembly

    Args:
        file_path: absolute path to the site config YAML.

    Returns:
        Immutable SiteRuntimeConfig.
    """
    file_path = Path(file_path)
    content_hash = instrument_validation_cache.hash_file(file_path)
    validated_config = validate_L1_config_structure(file=file_path)
    instrument_uris = _validate_instrument_names(validated_config, content_hash)
    return _build_runtime_config(validated_config, instrument_uris)

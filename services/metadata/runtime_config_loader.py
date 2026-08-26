#!/usr/bin/env python3
"""TERN-adapter orchestration for loading a site's SiteRuntimeConfig.

Responsibilities (TERN-specific only — generic assembly lives in
services.metadata.runtime_config_builder):
    - instrument name validation against the TERN instrument registry
      (via instrument_registry)
    - instrument URI resolution against the TERN RDF vocab
    - delegating structural validation and object assembly to
      runtime_config_builder

load_runtime_config(file_path) is the TERN-integrated equivalent of
runtime_config_builder.build_runtime_config_from_file(file_path): same
structural validation, plus TERN instrument-name/URI resolution threaded
through as enrichment.
"""

from pathlib import Path

from services.metadata import instrument_validation_cache
from services.metadata.runtime_config_builder import (
    SiteRuntimeConfig,
    build_runtime_config,
)
from services.metadata.site_config_schema import (
    SiteConfig,
    validate_L1_config_structure,
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


def load_runtime_config(file_path: Path) -> SiteRuntimeConfig:
    """Assemble a SiteRuntimeConfig from a site YAML config file.

    Three phases:
        1. Structural validation  — YAML schema (site_config_schema)
        2. Instrument validation  — all instrument names against TERN registry
           (skipped, per content hash, when a prior clean pass is cached —
           see instrument_validation_cache)
        3. Build                  — delegated to runtime_config_builder

    Args:
        file_path: absolute path to the site config YAML.

    Returns:
        Immutable SiteRuntimeConfig.
    """
    file_path = Path(file_path)
    content_hash = instrument_validation_cache.hash_file(file_path)
    validated_config = validate_L1_config_structure(file=file_path)
    instrument_uris = _validate_instrument_names(validated_config, content_hash)
    return build_runtime_config(validated_config, instrument_uris)

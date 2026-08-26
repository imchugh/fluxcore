#!/usr/bin/env python3
"""Generic-core, strict, file-based loader for a single site's SiteMetadata.

Companion to runtime_config_builder.py, for the other half of a SiteContext.
TERN's own production pipeline is unaffected by this module — it keeps
sourcing SiteMetadata as an object via SiteRegistry.get_metadata()
(services/metadata/site_registry.py), reading configs/site_metadata.yml or
the RDF endpoint. This module exists purely as a standalone, file-based
entry point for a non-TERN caller who has no adapter of their own.

Deliberately a single file (schema + builder together) rather than mirroring
SiteConfig's schema/builder split — this piece is small enough not to earn
the extra file.

Deliberately strict (raises clearly on malformed input), unlike
SiteMetadata's own tolerant coercion (domain/data_models/metadata_classes.py,
SiteMetadata.__init__), which logs a warning and keeps bad values raw. That
tolerant behaviour is a TERN accommodation for messy upstream RDF/yml data;
it stays untouched, and is never exercised by this module, since validation
here happens strictly before SiteMetadata's constructor ever sees the data.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, field_validator

from domain.data_models.metadata_classes import SiteMetadata
from infrastructure.file_io import read_yml


class SiteMetadataSchema(BaseModel):
    """Schema for one site's standalone metadata YAML file.

    Required fields are the minimum a site needs to actually be processed:
    location (for CRS/NetCDF export) and time_step/freq_hz (for n_samples,
    dataframe assembly). Everything else is descriptive and commonly sparse
    in real records (e.g. a real site's record may have no
    date_decommissioned/time_zone/UTC_offset).
    """

    site_name: str
    latitude: float
    longitude: float
    time_step: int
    freq_hz: int

    id: str | None = None
    fluxnet_id: str | None = None
    date_commissioned: datetime | None = None
    date_decommissioned: datetime | None = None
    elevation: float | None = None
    soil: str | None = None
    tower_height: float | None = None
    vegetation: str | None = None
    canopy_height: float | None = None
    time_zone: str | None = None
    UTC_offset: float | None = None

    @field_validator("time_step", "freq_hz", mode="before")
    @classmethod
    def coerce_float_string(cls, v):
        """Accept float-string values ('30.0') as real-world records do."""
        if isinstance(v, str):
            return int(float(v))
        return v


def validate_site_metadata_structure(file: Path) -> SiteMetadataSchema:
    """Validate YAML structure and return a SiteMetadataSchema object."""
    return SiteMetadataSchema(**read_yml(file_path=file))


def build_site_metadata_from_file(file_path: Path) -> SiteMetadata:
    """Build a SiteMetadata from a single-site metadata YAML file.

    Generic-core entry point, strict validation — raises clearly on
    malformed input. See tests/test_site_metadata_builder.py for a worked
    example based on a real site's metadata record (one block of
    configs/site_metadata.yml, unwrapped to a standalone file — TERN's own
    yml_loader() never produces this shape as a standalone file, so this is
    the reference example for a non-TERN user).

    Args:
        file_path: absolute path to the single-site metadata YAML.

    Returns:
        SiteMetadata.
    """
    validated = validate_site_metadata_structure(file=file_path)
    # mode="json": datetime -> ISO string, matching the string-typed shape
    # SiteMetadata.__init__ expects (and real TERN records already carry) —
    # feeding it a native datetime object instead trips its
    # datetime.fromisoformat(value) coercion (str-only) into a needless
    # warning-and-keep-raw fallback.
    return SiteMetadata(data=validated.model_dump(mode="json"))

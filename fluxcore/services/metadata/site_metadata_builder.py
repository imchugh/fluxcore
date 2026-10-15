#!/usr/bin/env python3
"""SiteContext, and a strict file-based loader for a site's SiteMetadata.

SiteContext (SiteRuntimeConfig + SiteMetadata) is defined here so that
neither half's own module has to import the other.

build_site_metadata_from_file() is for callers without an adapter of their
own: it validates a single-site metadata YAML strictly (raising on malformed
input) before building SiteMetadata. That differs on purpose from
SiteMetadata.__init__'s tolerant coercion, which logs and keeps bad values
raw for adapters fed by messier upstream sources; it is never exercised by
this path, since validation runs first.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, field_validator

from fluxcore.domain.data_models.metadata_classes import SiteMetadata
from fluxcore.infrastructure.file_io import read_yml
from fluxcore.services.metadata.runtime_config_builder import SiteRuntimeConfig


@dataclass(frozen=True)
class SiteContext:
    """Combined runtime objects for a configured pipeline site.

    One of the contract objects (alongside SiteRuntimeConfig, VariableSpec,
    FileGroup, CanonicalQuantityMetadata) an adapter produces to drive
    fluxcore.
    """

    runtime_config: SiteRuntimeConfig
    metadata: SiteMetadata


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

    Strict validation — raises clearly on malformed input. See
    tests/test_site_metadata_builder.py for a worked example based on a real
    site's metadata record.

    Args:
        file_path: absolute path to the single-site metadata YAML.

    Returns:
        SiteMetadata.
    """
    validated = validate_site_metadata_structure(file=file_path)
    # mode="json": datetime -> ISO string, the string-typed shape
    # SiteMetadata.__init__ expects. A native datetime would trip its
    # datetime.fromisoformat(value) coercion (str-only) into a needless
    # warning-and-keep-raw fallback.
    return SiteMetadata(data=validated.model_dump(mode="json"))

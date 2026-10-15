#!/usr/bin/env python3
"""SiteMetadata, SiteContext, and a strict file-based loader for SiteMetadata.

SiteContext (SiteRuntimeConfig + SiteMetadata) is defined here, beside
SiteMetadata, so that runtime_config_builder never has to import this module.

build_site_metadata_from_file() is for callers without an adapter of their
own: it validates a single-site metadata YAML strictly (raising on malformed
input) before building SiteMetadata. That differs on purpose from
SiteMetadata.__init__'s tolerant coercion, which logs and keeps bad values
raw for adapters fed by messier upstream sources; it is never exercised by
this path, since validation runs first.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, field_validator

from fluxcore.infrastructure.file_io import read_yml
from fluxcore.services.metadata.runtime_config_builder import SiteRuntimeConfig

logger = logging.getLogger(__name__)


class SiteMetadata(dict):
    """Lightweight metadata container.

    Provides:
    - automatic type formatting
    - attribute-style access
    - dictionary behaviour
    """

    DATA_DTYPES = {
        "id": str,
        "fluxnet_id": str,
        "date_commissioned": datetime,
        "date_decommissioned": datetime,
        "latitude": float,
        "longitude": float,
        "elevation": float,
        "time_step": int,
        "freq_hz": int,
        "soil": str,
        "tower_height": float,
        "vegetation": str,
        "canopy_height": float,
        "time_zone": str,
        "UTC_offset": float,
    }

    def __init__(self, data: dict):
        """Coerce `data` values to their expected types per `DATA_DTYPES`.

        Values that are `None`, `""`, or `"nan"` become `None`. Values that
        fail coercion keep their raw, uncoerced form (logged as a warning)
        rather than raising, so one bad field doesn't block loading the rest
        of a site's metadata.
        """
        formatted = {}

        for key, value in data.items():
            if value in (None, "", "nan"):
                formatted[key] = None
                continue

            dtype = self.DATA_DTYPES.get(key)

            try:
                if dtype is float:
                    formatted[key] = float(value)

                elif dtype is int:
                    formatted[key] = int(float(value))

                elif dtype is str:
                    formatted[key] = str(value)

                elif dtype is datetime:
                    formatted[key] = datetime.fromisoformat(value)

                else:
                    formatted[key] = value

            except Exception as exc:
                logger.warning(
                    "Could not coerce %s=%r to %s for site metadata; "
                    "keeping raw value (%s)",
                    key,
                    value,
                    dtype,
                    exc,
                )
                formatted[key] = value

        super().__init__(formatted)

    def __getattr__(self, key):
        """Return `self[key]`, enabling attribute-style access to metadata fields."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        """Reject attribute assignment; SiteMetadata is immutable after construction.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError("SiteMetadata is immutable")

    def __dir__(self):
        """Include metadata keys alongside normal dict attributes in `dir()` output."""
        return sorted(set(super().__dir__()) | set(self.keys()))

    @property
    def n_samples(self) -> int | None:
        """Expected number of high-frequency samples in one averaging period.

        Calculated from time_step (minutes) and freq_hz (Hz).  Returns None
        if either field is absent or not yet populated.
        """
        time_step = self.get("time_step")
        freq_hz = self.get("freq_hz")
        if time_step is None or freq_hz is None:
            return None
        return time_step * 60 * freq_hz

    def __repr__(self):
        """Return a short repr showing the site name, or 'unknown' if absent."""
        label = self.get("site_name", "unknown")
        return f"<SiteMetadata {label}>"


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

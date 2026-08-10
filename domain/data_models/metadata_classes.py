#!/usr/bin/env python3
"""Pydantic-dataclass metadata models used across the pipeline."""

import logging
from datetime import datetime

from pydantic.dataclasses import ConfigDict, dataclass

from domain import enums

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawVariableMetadata:
    """Metadata for one raw instrument-file input that feeds a canonical variable."""

    raw_name: str
    raw_units: str
    file: str
    instrument: str | dict[str, str]
    begin: datetime | str | None = None
    end: datetime | str | None = None


@dataclass(frozen=True)
class ParsedVariableName:
    """Components parsed out of a canonical variable name string."""

    quantity: str
    qualifier: str | None = None
    vertical_location: str | None = None
    horizontal_location: str | None = None
    replicate: str | None = None
    statistic_id: str | None = (
        None  # set when suffix is a StatisticType (Av, Sd, Vr ...)
    )
    variable_type_id: str | None = None  # set when suffix is a VariableType  (QC, Ct)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CanonicalQuantityMetadata:
    """Registry metadata for one canonical quantity: units, naming, and valid range."""

    long_name: str
    standard_name: str | None
    standard_units: str
    valid_input_units: list[str]
    valid_min: float | None = None
    valid_max: float | None = None


@dataclass(frozen=True)
class VariableDefinition:
    """Complete definition of one canonical variable.

    Assembled from its raw inputs, canonical quantity metadata, and parsed
    name — the unit that `services/metadata/variable_registry.py` builds
    per raw-to-canonical mapping.
    """

    # Core identity
    variable_name: str
    quantity: str
    variable_type: enums.VariableType

    # Physical metadata
    instrument: str | dict[str, str]
    height: float

    # Structural metadata
    raw_inputs: tuple[RawVariableMetadata, ...]
    canonical: CanonicalQuantityMetadata
    parsed_name_elems: ParsedVariableName

    # Optionals
    height_range: tuple[float, float] | None = None
    statistic_type: enums.StatisticType | None = None
    diag_type: enums.DiagnosticType | None = None


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

    def to_yaml_dict(self):
        """Return a plain dict with datetime values ISO-formatted, ready for YAML."""
        return {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in self.items()
        }

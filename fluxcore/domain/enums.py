#!/usr/bin/env python3
"""Enums for statistics, variable types, loggers, file formats, and flux systems."""

from enum import Enum


def _find_by_attr(cls, attr: str, value, label: str):
    """Return the enum member whose `attr` equals `value`, or raise ValueError."""
    for item in cls:
        if getattr(item, attr) == value:
            return item
    raise ValueError(f"Unknown {label}: {value!r}")


class DiagnosticType(str, Enum):
    """Direction of a diagnostic counter: which state it counts toward."""

    VALID_COUNT = "valid_count"
    INVALID_COUNT = "invalid_count"


class StatisticType(str, Enum):
    """Statistic computed over an averaging period, with its canonical name suffix."""

    AVG = ("average", "Av")
    SUM = ("sum", "Sum")
    MIN = ("minimum", "Min")
    MAX = ("maximum", "Max")
    STDEV = ("standard_deviation", "Sd")
    VAR = ("variance", "Vr")
    COVAR = ("covariance", "Cov")
    SAMPLE = ("instantaneous", "Inst")

    def __new__(cls, value: str, suffix: str):
        """Construct a member, storing `suffix` alongside the string value."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.suffix = suffix
        return obj

    @classmethod
    def from_suffix(cls, suffix: str) -> "StatisticType":
        """Return the member whose canonical-name suffix matches `suffix`."""
        return _find_by_attr(cls, "suffix", suffix, "statistic suffix")


class VariableType(str, Enum):
    """Semantic type of a variable, with its canonical name suffix (if any)."""

    CONTINUOUS = ("continuous", None)
    QUALITY_FLAG = ("quality_flag", "QC")
    COUNTER = ("counter", "Ct")
    CATEGORICAL = ("categorical", None)
    INDEX = ("index", None)

    def __new__(cls, value: str, suffix: str | None):
        """Construct a member, storing `suffix` alongside the string value."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.suffix = suffix
        return obj

    @classmethod
    def from_suffix(cls, suffix: str) -> "VariableType":
        """Return the member whose canonical-name suffix matches `suffix`."""
        return _find_by_attr(cls, "suffix", suffix, "variable type suffix")


class LoggerType(str, Enum):
    """Logger family that wrote a file group (the site config's file_formats values).

    Not a file format: Campbell (CSI) loggers can write TOA5, TOB1, TOB3 and
    more. DEFAULT_FILE_FORMAT says which format fluxcore reads for each.
    """

    CSI = "CSI"
    LICOR = "LICOR"


class FileFormat(str, Enum):
    """Raw file format fluxcore can parse."""

    TOA5 = "TOA5"
    EDDYPRO = "EddyPro"

    @property
    def extension(self) -> str:
        """On-disk file extension, without a leading dot."""
        return {FileFormat.TOA5: "dat", FileFormat.EDDYPRO: "txt"}[self]


# The one place fluxcore assumes which format each logger family writes (a
# CSI logger's TOB3 output, for one, must be converted to TOA5 upstream).
# Letting site configs state the format per file group would replace this.
DEFAULT_FILE_FORMAT = {
    LoggerType.CSI: FileFormat.TOA5,
    LoggerType.LICOR: FileFormat.EDDYPRO,
}


class FluxSystemType(str, Enum):
    """Flux-processing software that produced a site's EddyPro-format output."""

    TERNFLUX = "TERNFLUX"
    EASYFLUX = "EASYFLUX"
    SMARTFLUX = "SMARTFLUX"
    LEGACY = "LEGACY"

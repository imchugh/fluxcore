#!/usr/bin/env python3
"""Enums for statistics, variable types, file formats, and flux systems."""

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


class FileType(str, Enum):
    """Raw instrument file format, mapped to its on-disk file extension."""

    CSI = "dat"
    LICOR = "txt"

    @property
    def extension(self) -> str:
        """File extension for this format, without a leading dot."""
        return self.value

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        """Return the member whose file extension matches `ext`."""
        return _find_by_attr(cls, "value", ext, "FileType for extension")


class FluxSystemType(str, Enum):
    """Flux-processing software that produced a site's EddyPro-format output."""

    TERNFLUX = "TERNFLUX"
    EASYFLUX = "EASYFLUX"
    SMARTFLUX = "SMARTFLUX"
    LEGACY = "LEGACY"

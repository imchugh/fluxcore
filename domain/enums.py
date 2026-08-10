#!/usr/bin/env python3
"""Created on Tue Apr 28 10:27:17 2026

@author: imchugh
"""

from enum import Enum


def _find_by_attr(cls, attr: str, value, label: str):
    """Return the enum member whose `attr` equals `value`, or raise ValueError."""
    for item in cls:
        if getattr(item, attr) == value:
            return item
    raise ValueError(f"Unknown {label}: {value!r}")


class DiagnosticType(str, Enum):
    VALID_COUNT = "valid_count"
    INVALID_COUNT = "invalid_count"


class StatisticType(str, Enum):
    """Simple statistic validation class"""

    AVG = ("average", "Av")
    SUM = ("sum", "Sum")
    MIN = ("minimum", "Min")
    MAX = ("maximum", "Max")
    STDEV = ("standard_deviation", "Sd")
    VAR = ("variance", "Vr")
    COVAR = ("covariance", "Cov")
    SAMPLE = ("instantaneous", "Inst")

    def __new__(cls, value: str, suffix: str):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.suffix = suffix
        return obj

    @classmethod
    def from_suffix(cls, suffix: str) -> "StatisticType":
        return _find_by_attr(cls, "suffix", suffix, "statistic suffix")


class VariableType(str, Enum):
    """Semantic type of variable."""

    CONTINUOUS = ("continuous", None)
    QUALITY_FLAG = ("quality_flag", "QC")
    COUNTER = ("counter", "Ct")
    CATEGORICAL = ("categorical", None)
    INDEX = ("index", None)

    def __new__(cls, value: str, suffix: str | None):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.suffix = suffix
        return obj

    @classmethod
    def from_suffix(cls, suffix: str) -> "VariableType":
        return _find_by_attr(cls, "suffix", suffix, "variable type suffix")


class FileType(str, Enum):
    """"""

    CSI = "dat"
    LICOR = "txt"

    @property
    def extension(self) -> str:
        return self.value

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        return _find_by_attr(cls, "value", ext, "FileType for extension")


class FluxSystemType(str, Enum):
    TERNFLUX = "TERNFLUX"
    EASYFLUX = "EASYFLUX"
    SMARTFLUX = "SMARTFLUX"
    LEGACY = "LEGACY"

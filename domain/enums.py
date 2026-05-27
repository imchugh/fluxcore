#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:27:17 2026

@author: imchugh
"""

from enum import Enum

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
        for item in cls:
            if item.suffix == suffix:
                return item
        raise ValueError(f"Unknown statistic suffix: {suffix}")

class VariableType(str, Enum):
    """Semantic type of variable."""

    CONTINUOUS = 'continuous'
    QUALITY_FLAG = 'quality_flag'
    COUNTER = 'counter'
    CATEGORICAL = 'categorical'
    INDEX = 'index'
    
class FileType(str, Enum):
    """"""
    
    CSI = "dat"
    LICOR = "txt"

    @property
    def extension(self) -> str:
        return self.value

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        for ft in cls:
            if ft.value == ext:
                return ft
        raise ValueError(f"No FileType for extension '{ext}'")    
        
class FluxSystemType(str, Enum):
    
    TERNFLUX = 'TERNFLUX'
    EASYFLUX = 'EASYFLUX'
    SMARTFLUX = 'SMARTFLUX'
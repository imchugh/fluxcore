#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:27:17 2026

@author: imchugh
"""

from enum import Enum

class StatisticType(str, Enum):
    """Simple statistic validation class"""
    
    AVG = 'average'
    SUM = 'sum'
    MIN = 'minimum'
    MAX = 'maximum'
    STDEV = 'standard_deviation'
    VAR = 'variance'
    COVAR = 'covariance'
    COUNT = 'count'
    SAMPLE = 'instantaneous'

class VariableType(str, Enum):
    """Semantic type of variable."""

    CONTINUOUS = 'continuous'
    QUALITY_FLAG = 'quality_flag'
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
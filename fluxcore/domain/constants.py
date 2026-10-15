#!/usr/bin/env python3
"""Physical, temporal, and formatting constants shared across the pipeline."""

# Domain constants
SITE_PLACEHOLDER = "<site>"
TIME_INDEX_NAME = "time"
DATA_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
NC_ENCODING = "seconds since 1800-01-01 00:00:00.0"

# Scientific numerical constants
# Molar masses (g/mol) as PyFluxPro uses them: IUPAC standard atomic weights
# (CO2 44.009, H2O 18.015) rounded to two decimals, so results stay
# comparable with PyFluxPro's.
CO2_MOL_MASS = 44.01
H2O_MOL_MASS = 18.02
ZERO_C_IN_K = 273.15
# Molar gas constant (J/mol/K): exact since the 2019 SI redefinition (CODATA).
R = 8.314462618

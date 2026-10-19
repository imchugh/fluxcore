#!/usr/bin/env python3
"""Physical, temporal, and formatting constants shared across the pipeline."""

from pathlib import Path

# fluxcore's packaged configs, located relative to this installed module so it
# resolves wherever fluxcore is installed (shipped via pyproject package-data).
CONFIG_PATH = Path(__file__).parents[1] / "configs"

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
# Molar mass of dry air (g/mol), as PyFluxPro uses it.
DRY_AIR_MOL_MASS = 28.97
# Specific gas constants (J/kg/K), derived from R so they stay consistent
# with it (PyFluxPro's fixed 287.04/461.5 differ by under 0.03%).
R_DRY_AIR = R / (DRY_AIR_MOL_MASS / 1000)
R_WATER_VAPOUR = R / (H2O_MOL_MASS / 1000)
CP_DRY_AIR = 1004.67  # specific heat of dry air at constant pressure, J/kg/K
GRAVITY = 9.81  # m/s^2
VON_KARMAN = 0.4
# Soil heat-storage constants (as PyFluxPro): heat capacities (J/kg/K) of the
# mineral and organic soil components and of water; density of water (kg/m^3).
SOIL_MINERAL_HEAT_CAPACITY = 840.0
SOIL_ORGANIC_HEAT_CAPACITY = 1920.0
WATER_HEAT_CAPACITY = 4180.0
WATER_DENSITY = 1000.0

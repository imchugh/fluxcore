#!/usr/bin/env python3
"""Physical calculations: derived quantities and atmospheric physics.

`@register_calculation` populates CALCULATION_REGISTRY, keyed by canonical
quantity name; `get_calculation` looks it back up. This is the home for
physical relationships between quantities -- including the L3 flux
calculations as they are added. Helpers whose inputs aren't canonical
quantities (e.g. `standard_pressure_kpa`, which takes a site elevation) are
plain unregistered functions. Unit conversions live in
services/data/unit_conversions.py.
"""

import numpy as np
from numpy.typing import ArrayLike

from domain.constants import CO2_MOL_MASS, H2O_MOL_MASS, K, R

CALCULATION_REGISTRY = {}


def register_calculation(*quantities):
    """Register the decorated function in CALCULATION_REGISTRY under each quantity."""

    def decorator(func):
        for q in quantities:
            CALCULATION_REGISTRY[q] = func
        return func

    return decorator


def get_calculation(quantity):
    """Return the registered calculation function for `quantity`, or None."""
    return CALCULATION_REGISTRY.get(quantity)


@register_calculation("AH")
def calculate_AH_from_RH(Ta: ArrayLike, RH: ArrayLike, ps: ArrayLike) -> ArrayLike:
    """Derive absolute humidity (g/m³) from air temperature, RH, and pressure."""
    es = calculate_es(Ta)
    e = es * RH / 100
    molar_density = calculate_molar_density(ps=ps, Ta=Ta)
    return e / ps * molar_density * H2O_MOL_MASS


@register_calculation("RH")
def calculate_RH_from_AH(AH: ArrayLike, Ta: ArrayLike, ps: ArrayLike) -> ArrayLike:
    """Derive relative humidity (%) from AH, air temperature, and pressure."""
    molar_density = calculate_molar_density(ps=ps, Ta=Ta)
    e = (AH / H2O_MOL_MASS) / molar_density * ps
    es = calculate_es(Ta)
    return e / es * 100


@register_calculation("es")
def calculate_es(Ta: ArrayLike) -> ArrayLike:
    """Saturation vapour pressure (kPa) from Buck (1996)."""
    return 0.61121 * np.exp((18.678 - Ta / 234.5) * (Ta / (257.14 + Ta)))


@register_calculation("CO2")
def calculate_CO2_mole_fraction(
    CO2c: ArrayLike, Ta: ArrayLike, ps: ArrayLike
) -> ArrayLike:
    """CO2 mole fraction (umol/mol) from CO2 density, air temperature, and pressure."""
    return (CO2c / CO2_MOL_MASS) / calculate_molar_density(Ta=Ta, ps=ps) * 10**3


@register_calculation("e")
def calculate_e(Ta: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """Vapour pressure (kPa) from air temperature (degC) and RH (percent)."""
    return calculate_es(Ta=Ta) * RH / 100


@register_calculation("VPD")
def calculate_vpd(Ta: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """Vapour pressure deficit (kPa) from air temperature (degC) and RH (percent)."""
    es = calculate_es(Ta=Ta)
    e = calculate_e(Ta=Ta, RH=RH)
    return es - e


@register_calculation("rho_mol")
def calculate_molar_density(ps: ArrayLike, Ta: ArrayLike) -> ArrayLike:
    """Molar density (mol/m^3) from air pressure (kPa) and air temperature (degC)."""
    return ps * 1000 / ((Ta + K) * R)


@register_calculation("Td")
def calculate_dew_point(Ta: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """Dew point temperature (degC) from air temperature (degC) and RH (percent).

    Inverts the Buck (1981) form es = 0.61121 exp(17.502 T / (240.97 + T)),
    which reproduces calculate_es (Buck 1996) to within 0.01 K over -10 to
    40 degC, so the dew point equals air temperature at RH = 100. (It was
    previously paired with Bolton's 243.5, which overshot by up to 0.4 K.)
    """
    e = calculate_e(Ta=Ta, RH=RH)
    ln_ratio = np.log(e / 0.61121)
    return (240.97 * ln_ratio) / (17.502 - ln_ratio)


def standard_pressure_kpa(elevation_m: float) -> float:
    """Standard-atmosphere (ISA) pressure in kPa at an elevation in metres.

    Valid in the troposphere (to about 11 km).
    """
    return 101.325 * (1 - 2.25577e-5 * elevation_m) ** 5.25588

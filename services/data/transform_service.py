#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import ArrayLike

from domain.constants import CO2_MOL_MASS, H2O_MOL_MASS, K, R


CONVERSION_REGISTRY = {}
CALCULATION_REGISTRY = {}


def register_conversion(*quantities):
    def decorator(func):
        for q in quantities:
            CONVERSION_REGISTRY[q] = func
        return func
    return decorator


def register_calculation(*quantities):
    def decorator(func):
        for q in quantities:
            CALCULATION_REGISTRY[q] = func
        return func
    return decorator


@register_conversion("Fco2")
def convert_CO2_flux(data, from_units='mg/m^2/s'):

    if from_units == 'mg/m^2/s':
        return data * 1000 / CO2_MOL_MASS
    raise ValueError(f"Unsupported from_units {from_units!r} for CO2 flux conversion")


@register_conversion("CO2c")
def convert_CO2_density(data, from_units='mmol/m^3'):

    if from_units == 'mmol/m^3':
        return data * CO2_MOL_MASS
    raise ValueError(f"Unsupported from_units {from_units!r} for CO2 density conversion")


@register_conversion('Sig', 'SigCO2', 'SigH2O', 'CO2Sig', 'H2OSig')
def convert_signal_strength(data, from_units='frac'):

    if from_units == 'frac':
        return data * 100
    raise ValueError(f"Unsupported from_units {from_units!r} for signal strength conversion")


@register_conversion('Diag')
def convert_diagnostic(data, n_samples, from_units='valid_count'):
    """Convert diagnostic counts from valid_count to invalid_count form.

    The pipeline standardises to invalid_count (0 = no error, n_samples = all
    errors). Sites that store valid_count are converted by subtracting from
    n_samples. Sites that store invalid_count are passed through by the caller
    without invoking this function.
    """

    if from_units == 'valid_count':
        return n_samples - data
    raise ValueError(f"Unsupported from_units {from_units!r} for diagnostic conversion")


@register_conversion('AH')
def convert_H2O_density(data, from_units='mmol/m^3'):

    if from_units == 'mmol/m^3':
        return data * H2O_MOL_MASS / 10**3
    if from_units == 'kg/m^3':
        return data * 10**3
    raise ValueError(f"Unsupported from_units {from_units!r} for H2O density conversion")


@register_conversion('Precip')
def convert_precipitation(data, from_units='pulse_0.2mm'):

    if from_units == 'pulse_0.2mm':
        return data * 0.2
    if from_units == 'pulse_0.5mm':
        return data * 0.5
    raise ValueError(f"Unsupported from_units {from_units!r} for precipitation conversion")


@register_conversion('ps')
def convert_pressure(data, from_units='Pa'):

    if from_units == 'Pa':
        return data / 10**3
    if from_units == 'hPa':
        return data / 10
    raise ValueError(f"Unsupported from_units {from_units!r} for pressure conversion")


@register_conversion('RH')
def convert_RH(data, from_units='frac'):

    if from_units == 'frac':
        return data * 100
    raise ValueError(f"Unsupported from_units {from_units!r} for RH conversion")


@register_conversion('Sws')
def convert_Sws(data, from_units='percent'):

    if from_units == 'percent':
        return data / 100
    raise ValueError(f"Unsupported from_units {from_units!r} for Sws conversion")


@register_conversion('Ta', 'Tv', 'Tbody')
def convert_temperature(data, from_units='K'):

    if from_units == 'K':
        return data - K
    raise ValueError(f"Unsupported from_units {from_units!r} for temperature conversion")


def get_unit_conversion(quantity):
    return CONVERSION_REGISTRY.get(quantity)


def get_calculation(quantity):
    return CALCULATION_REGISTRY.get(quantity)


@register_calculation('AH')
def calculate_AH_from_RH(Ta: ArrayLike, RH: ArrayLike, ps: ArrayLike) -> ArrayLike:
    """Derive absolute humidity (g/m³) from air temperature, RH, and pressure."""

    es = calculate_es(Ta)
    e = es * RH / 100
    molar_density = calculate_molar_density(ps=ps, Ta=Ta)
    return e / ps * molar_density * H2O_MOL_MASS


@register_calculation('RH')
def calculate_RH_from_AH(AH: ArrayLike, Ta: ArrayLike, ps: ArrayLike) -> ArrayLike:
    """Derive relative humidity (%) from absolute humidity, air temperature, and pressure."""

    molar_density = calculate_molar_density(ps=ps, Ta=Ta)
    e = (AH / H2O_MOL_MASS) / molar_density * ps
    es = calculate_es(Ta)
    return e / es * 100


@register_calculation('es')
def calculate_es(Ta: ArrayLike) -> ArrayLike:
    """Saturation vapour pressure (kPa) from Buck (1996)."""

    return 0.61121 * np.exp((18.678 - Ta / 234.5) * (Ta / (257.14 + Ta)))


@register_calculation('CO2')
def calculate_CO2_mole_fraction(
        CO2c: ArrayLike, Ta: ArrayLike, ps: ArrayLike
        ) -> ArrayLike:
    """CO2 mole fraction (umol/mol) from CO2 density (mg/m^3), air temperature (degC) and pressure (kPa)."""

    return (
        (CO2c / CO2_MOL_MASS) /
        calculate_molar_density(Ta=Ta, ps=ps)
        * 10**3
        )


@register_calculation('e')
def calculate_e(Ta: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """Vapour pressure (kPa) from air temperature (degC) and RH (percent)."""

    return calculate_es(Ta=Ta) * RH / 100


@register_calculation('VPD')
def calculate_vpd(Ta: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """Vapour pressure deficit (kPa) from air temperature (degC) and RH (percent)."""
    
    es = calculate_es(Ta=Ta)
    e = calculate_e(Ta=Ta, RH=RH)
    return es - e


@register_calculation('rho_mol')
def calculate_molar_density(ps: ArrayLike, Ta: ArrayLike) -> ArrayLike:
    """Molar density (mol/m^3) from air pressure (kPa) and air temperature (degC)."""

    return ps * 1000 / ((Ta + K) * R)


@register_calculation('Td')
def calculate_dew_point(Ta: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """Dew point temperature (degC) from air temperature (degC) and RH (percent)."""

    e = calculate_e(Ta=Ta, RH=RH)
    ln_ratio = np.log(e / 0.61121)
    return (243.5 * ln_ratio) / (17.502 - ln_ratio)


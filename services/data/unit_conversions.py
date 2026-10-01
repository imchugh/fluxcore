#!/usr/bin/env python3
"""Unit-conversion registry: raw logger units -> canonical units.

`@register_conversion` populates CONVERSION_REGISTRY, keyed by canonical
quantity name; `get_unit_conversion` looks it back up. Each conversion
function raises ValueError for an unrecognized `from_units` rather than
returning None. Physical calculations between quantities live in
services/data/calculations.py.
"""

import numpy as np

from domain.constants import CO2_MOL_MASS, H2O_MOL_MASS, K

CONVERSION_REGISTRY = {}


def register_conversion(*quantities):
    """Register the decorated function in CONVERSION_REGISTRY under each quantity."""

    def decorator(func):
        for q in quantities:
            CONVERSION_REGISTRY[q] = func
        return func

    return decorator


@register_conversion("Fco2")
def convert_CO2_flux(data, from_units="mg/m^2/s"):
    """Convert CO2 flux from mg/m^2/s to canonical umol/m^2/s."""
    if from_units == "mg/m^2/s":
        return data * 1000 / CO2_MOL_MASS
    raise ValueError(f"Unsupported from_units {from_units!r} for CO2 flux conversion")


@register_conversion("CO2c")
def convert_CO2_density(data, from_units="mmol/m^3"):
    """Convert CO2 density from mmol/m^3 to canonical mg/m^3."""
    if from_units == "mmol/m^3":
        return data * CO2_MOL_MASS
    raise ValueError(
        f"Unsupported from_units {from_units!r} for CO2 density conversion"
    )


@register_conversion("Sig", "SigCO2", "SigH2O")
def convert_signal_strength(data, from_units="frac"):
    """Convert an IRGA signal-strength fraction (0-1) to canonical percent."""
    if from_units == "frac":
        return data * 100
    raise ValueError(
        f"Unsupported from_units {from_units!r} for signal strength conversion"
    )


@register_conversion("Diag")
def convert_diagnostic(data, n_samples, from_units="valid_count"):
    """Convert diagnostic counts from valid_count to invalid_count form.

    The pipeline standardises to invalid_count (0 = no error, n_samples = all
    errors). Sites that store valid_count are converted by subtracting from
    n_samples. Sites that store invalid_count are passed through by the caller
    without invoking this function.
    """
    if from_units == "valid_count":
        return n_samples - data
    raise ValueError(f"Unsupported from_units {from_units!r} for diagnostic conversion")


@register_conversion("AH")
def convert_H2O_density(data, from_units="mmol/m^3"):
    """Convert H2O (absolute humidity) density to canonical g/m^3."""
    if from_units == "mmol/m^3":
        return data * H2O_MOL_MASS / 10**3
    if from_units == "kg/m^3":
        return data * 10**3
    raise ValueError(
        f"Unsupported from_units {from_units!r} for H2O density conversion"
    )


@register_conversion("Precip")
def convert_precipitation(data, from_units="pulse_0.2mm"):
    """Convert tipping-bucket rain-gauge pulse counts to canonical mm."""
    if from_units == "pulse_0.2mm":
        return data * 0.2
    if from_units == "pulse_0.5mm":
        return data * 0.5
    raise ValueError(
        f"Unsupported from_units {from_units!r} for precipitation conversion"
    )


@register_conversion("ps")
def convert_pressure(data, from_units="Pa"):
    """Convert air pressure to canonical kPa."""
    if from_units == "Pa":
        return data / 10**3
    if from_units == "hPa":
        return data / 10
    raise ValueError(f"Unsupported from_units {from_units!r} for pressure conversion")


@register_conversion("RH")
def convert_RH(data, from_units="frac"):
    """Convert relative humidity from a 0-1 fraction to canonical percent."""
    if from_units == "frac":
        return data * 100
    raise ValueError(f"Unsupported from_units {from_units!r} for RH conversion")


@register_conversion("Sws")
def convert_Sws(data, from_units="percent"):
    """Convert soil water content from percent to canonical 0-1 fraction."""
    if from_units == "percent":
        return data / 100
    raise ValueError(f"Unsupported from_units {from_units!r} for Sws conversion")


@register_conversion("Ta", "Tv")
def convert_temperature(data, from_units="K"):
    """Convert temperature from Kelvin to canonical degrees Celsius."""
    if from_units == "K":
        return data - K
    raise ValueError(
        f"Unsupported from_units {from_units!r} for temperature conversion"
    )


@register_conversion("Tbody")
def convert_tbody(data, from_units="K"):
    """Convert radiometer body temperature to canonical degrees Celsius.

    Some sites log only the raw resistance from the net radiometer's body-
    temperature sensor rather than a logger-computed temperature -- and the
    CNR4 ships with either of two different sensor types reading in ohms,
    needing two different formulas, so the two are kept as distinct
    from_units labels rather than one ambiguous "ohms":

    "ohms_ntc": the standard CNR4 NTC thermistor (resistance in the
    thousands of ohms at ambient temperature). Steinhart-Hart equation,
    coefficients from the CNR4 manual (not a generic thermistor fit --
    specific to this sensor).

    "ohms_pt100": some CNR4 units instead use a Pt100 RTD (resistance ~100
    ohms at 0 degC). Linear IEC 60751 approximation (alpha=0.00385/degC) --
    accurate enough over ordinary ambient-temperature ranges; the
    Callendar-Van Dusen quadratic correction only matters well outside that.
    """
    if from_units == "K":
        return data - K
    if from_units == "ohms_ntc":
        ln_r = np.log(data)
        return 1 / (1.0295e-3 + 2.391e-4 * ln_r + 1.568e-7 * ln_r**3) - K
    if from_units == "ohms_pt100":
        r0, alpha = 100.0, 0.00385
        return (data / r0 - 1) / alpha
    raise ValueError(f"Unsupported from_units {from_units!r} for Tbody conversion")


def get_unit_conversion(quantity):
    """Return the registered unit-conversion function for `quantity`, or None."""
    return CONVERSION_REGISTRY.get(quantity)

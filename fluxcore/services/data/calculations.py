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

from fluxcore.domain.constants import (
    CO2_MOL_MASS,
    CP_DRY_AIR,
    DRY_AIR_MOL_MASS,
    GRAVITY,
    H2O_MOL_MASS,
    R_DRY_AIR,
    R_WATER_VAPOUR,
    SOIL_MINERAL_HEAT_CAPACITY,
    SOIL_ORGANIC_HEAT_CAPACITY,
    VON_KARMAN,
    WATER_DENSITY,
    WATER_HEAT_CAPACITY,
    ZERO_C_IN_K,
    R,
)

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


@register_calculation("es", "SVP")
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


@register_calculation("Fn")
def calculate_net_radiation(
    Fsd: ArrayLike, Fsu: ArrayLike, Fld: ArrayLike, Flu: ArrayLike
) -> ArrayLike:
    """Net radiation (W/m^2) from the four shortwave/longwave components.

    Always used in preference to a logger-computed Fn, which is either
    exactly this sum (same instrument, redundant) or comes from a different,
    secondary net radiometer (e.g. "NR Lite"-family) that can disagree by
    hundreds of W/m^2. See derived_quantities.add_net_radiation.
    """
    return Fsd - Fsu + Fld - Flu


@register_calculation("rho_mol")
def calculate_molar_density(ps: ArrayLike, Ta: ArrayLike) -> ArrayLike:
    """Molar density (mol/m^3) from air pressure (kPa) and air temperature (degC)."""
    return ps * 1000 / ((Ta + ZERO_C_IN_K) * R)


@register_calculation("Td")
def calculate_dew_point(Ta: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """Dew point temperature (degC) from air temperature (degC) and RH (percent).

    Inverts the Buck (1981) form es = 0.61121 exp(17.502 T / (240.97 + T)),
    which reproduces calculate_es (Buck 1996) to within 0.01 K over -10 to
    40 degC, so the dew point equals air temperature at RH = 100. (Pairing
    it with Bolton's constant 243.5 instead overshoots by up to 0.4 K.)
    """
    e = calculate_e(Ta=Ta, RH=RH)
    ln_ratio = np.log(e / 0.61121)
    return (240.97 * ln_ratio) / (17.502 - ln_ratio)


def standard_pressure_kpa(elevation_m: float) -> float:
    """Standard-atmosphere (ISA) pressure in kPa at an elevation in metres.

    Valid in the troposphere (to about 11 km).
    """
    return 101.325 * (1 - 2.25577e-5 * elevation_m) ** 5.25588


# --- Humidity and thermodynamic properties of air (PyFluxPro L3) ------------
#
# Ports of PyFluxPro's meteorologicalfunctions as L3 uses them. Saturation
# vapour pressure is fluxcore's calculate_es (Buck 1996) throughout, not
# PyFluxPro's Tetens VPsat; the two differ by under 0.3% over 0-40 degC.

# Ratio of the molar masses of dry air and water vapour (PyFluxPro's mu).
_MU = DRY_AIR_MOL_MASS / H2O_MOL_MASS
# Campbell Scientific's coefficient relating sonic virtual temperature to air
# temperature (CSAT3 manual); the textbook virtual-temperature value is 0.61.
_SONIC_TV_COEFFICIENT = 0.51


@register_calculation("VP")
def calculate_VP_from_AH(AH: ArrayLike, Ta: ArrayLike) -> ArrayLike:
    """Vapour pressure (kPa) from AH (g/m^3) and air temperature (degC)."""
    return (AH / H2O_MOL_MASS) * R * (Ta + ZERO_C_IN_K) / 1000


def mixing_ratio(VP: ArrayLike, ps: ArrayLike) -> ArrayLike:
    """Water vapour mixing ratio (kg/kg) from vapour pressure and pressure (kPa)."""
    return (1 / _MU) * VP / (ps - VP)


@register_calculation("SH")
def calculate_SH_from_AH(AH: ArrayLike, Ta: ArrayLike, ps: ArrayLike) -> ArrayLike:
    """Specific humidity (kg/kg) from AH (g/m^3), Ta (degC) and pressure (kPa)."""
    mr = mixing_ratio(VP=calculate_VP_from_AH(AH=AH, Ta=Ta), ps=ps)
    return mr / (1 + mr)


@register_calculation("SHD")
def calculate_SHD(SH: ArrayLike, Ta: ArrayLike, ps: ArrayLike) -> ArrayLike:
    """Specific humidity deficit (kg/kg): saturation SH at Ta and ps minus SH."""
    mr_sat = mixing_ratio(VP=calculate_es(Ta), ps=ps)
    return mr_sat / (1 + mr_sat) - SH


@register_calculation("H2O")
def calculate_H2O_mole_fraction(
    AH: ArrayLike, Ta: ArrayLike, ps: ArrayLike
) -> ArrayLike:
    """H2O mole fraction (mmol/mol) from AH (g/m^3), air temperature, and pressure."""
    return (AH / H2O_MOL_MASS) / calculate_molar_density(Ta=Ta, ps=ps) * 10**3


@register_calculation("Ta")
def calculate_Ta_from_Tv(Tv: ArrayLike, SH: ArrayLike) -> ArrayLike:
    """Air temperature (degC) from sonic virtual temperature (degC) and SH (kg/kg).

    Uses Campbell Scientific's 0.51 coefficient, not the textbook 0.61. As in
    PyFluxPro's TaFromTv, SH is normally itself derived with Tv standing in
    for the unknown Ta (an approximation of order 1%).
    """
    return (Tv + ZERO_C_IN_K) / (1 + _SONIC_TV_COEFFICIENT * SH) - ZERO_C_IN_K


@register_calculation("Rhod")
def calculate_dry_air_density(Ta: ArrayLike, ps: ArrayLike, VP: ArrayLike) -> ArrayLike:
    """Partial density of dry air (kg/m^3) from Ta (degC), pressure and VP (kPa)."""
    return (ps - VP) * 1000 / ((Ta + ZERO_C_IN_K) * R_DRY_AIR)


@register_calculation("Rhoa")
def calculate_moist_air_density(
    Ta: ArrayLike, ps: ArrayLike, VP: ArrayLike
) -> ArrayLike:
    """Density of moist air (kg/m^3): dry air plus water vapour partial densities."""
    rhow = VP * 1000 / ((Ta + ZERO_C_IN_K) * R_WATER_VAPOUR)
    return calculate_dry_air_density(Ta=Ta, ps=ps, VP=VP) + rhow


def latent_heat_of_vaporisation(Ta: ArrayLike) -> ArrayLike:
    """Latent heat of vaporisation (J/kg) at air temperature Ta (degC), Stull (1988)."""
    return 2500800 - 2366.8 * Ta


def specific_heat_dry_air(Tv: ArrayLike) -> ArrayLike:
    """Specific heat of dry air at constant pressure (J/kg/K) from sonic Tv (degC).

    The EddyPro formulation.
    """
    return 1005 + (Tv + 23.12) ** 2 / 3364


def specific_heat_water_vapour(Ta: ArrayLike, RH: ArrayLike) -> ArrayLike:
    """Specific heat of water vapour at constant pressure (J/kg/K).

    The EddyPro formulation, from Ta (degC) and RH (percent).
    """
    return (
        1859 + 0.13 * RH + (0.193 + 0.0056 * RH) * Ta + (0.001 + 0.00005 * RH) * Ta**2
    )


def calculate_rho_cp(
    Ta: ArrayLike, Tv: ArrayLike, ps: ArrayLike, AH: ArrayLike, RH: ArrayLike
) -> ArrayLike:
    """Volumetric heat capacity of moist air, rho*Cp (J/m^3/K).

    Sum of the dry-air and water-vapour contributions, rhod*Cpd + rhow*Cpw
    (PyFluxPro's RhoCp). The water-vapour partial density is AH itself.
    """
    VP = calculate_VP_from_AH(AH=AH, Ta=Ta)
    rhod = calculate_dry_air_density(Ta=Ta, ps=ps, VP=VP)
    return rhod * specific_heat_dry_air(Tv) + (AH / 1000) * specific_heat_water_vapour(
        Ta, RH
    )


def potential_temperature(Ta: ArrayLike, ps: ArrayLike) -> ArrayLike:
    """Potential temperature (K) from air temperature (degC) and pressure (kPa)."""
    return (Ta + ZERO_C_IN_K) * (100 / ps) ** 0.286


def _virtual_potential_temperature(
    Ta: ArrayLike, ps: ArrayLike, VP: ArrayLike
) -> ArrayLike:
    return potential_temperature(Ta, ps) * (1 + 0.61 * mixing_ratio(VP=VP, ps=ps))


# --- Fluxes from covariances (PyFluxPro L3) ---------------------------------


@register_calculation("ustar")
def calculate_ustar(uw: ArrayLike, vw: ArrayLike) -> ArrayLike:
    """Friction velocity (m/s) from the rotated momentum covariances (m^2/s^2)."""
    return (uw**2 + vw**2) ** 0.25


@register_calculation("Fm")
def calculate_Fm(uw: ArrayLike, vw: ArrayLike, Rhoa: ArrayLike) -> ArrayLike:
    """Momentum flux (kg/m/s^2) from rotated momentum covariances and air density."""
    return Rhoa * np.sqrt(uw**2 + vw**2)


def calculate_Fhv(wT: ArrayLike, RhoCp: ArrayLike) -> ArrayLike:
    """Virtual heat flux (W/m^2) from the rotated w-Tv covariance (m.degC/s)."""
    return RhoCp * wT


@register_calculation("Fh")
def calculate_Fh_from_Fhv(
    Fhv: ArrayLike,
    Tv: ArrayLike,
    SH: ArrayLike,
    wT: ArrayLike,
    wA: ArrayLike,
    RhoCp: ArrayLike,
    Rhoa: ArrayLike,
) -> ArrayLike:
    """Sensible heat flux (W/m^2) from the virtual heat flux (PyFluxPro's FhvtoFh).

    Removes the humidity contribution to the sonic temperature flux, using
    the 0.51 coefficient of calculate_Ta_from_Tv. Tv in degC, wA (the
    rotated w-H2O covariance) in g/m^2/s, Rhoa (moist air density) in kg/m^3.
    """
    a = _SONIC_TV_COEFFICIENT
    return (
        Fhv - RhoCp * a * (Tv + ZERO_C_IN_K) * (wA / 1000) / Rhoa - RhoCp * a * SH * wT
    )


@register_calculation("Fe")
def calculate_Fe(wA: ArrayLike, Ta: ArrayLike) -> ArrayLike:
    """Latent heat flux (W/m^2) from the rotated w-H2O covariance (g/m^2/s)."""
    return latent_heat_of_vaporisation(Ta) * wA / 1000


def wpl_correct_Fe(
    Fe: ArrayLike,
    Fh: ArrayLike,
    Ta: ArrayLike,
    ps: ArrayLike,
    AH: ArrayLike,
    RhoCp: ArrayLike,
) -> ArrayLike:
    """Webb-Pearman-Leuning density correction of Fe (W/m^2), open-path IRGAs only.

    PyFluxPro's Fe_WPL (Campbell Scientific formulation). Fh must already be
    the sensible (not virtual) heat flux.
    """
    VP = calculate_VP_from_AH(AH=AH, Ta=Ta)
    rhod = calculate_dry_air_density(Ta=Ta, ps=ps, VP=VP)
    rhow = AH / 1000
    sigma = rhow / rhod
    Lv = latent_heat_of_vaporisation(Ta)
    return (
        Fe
        + _MU * sigma * Fe
        + (1 + _MU * sigma) * rhow * Lv * (Fh / RhoCp) / (Ta + ZERO_C_IN_K)
    )


def wpl_correct_Fco2(
    Fco2: ArrayLike,
    Fe: ArrayLike,
    Fh: ArrayLike,
    CO2c: ArrayLike,
    Ta: ArrayLike,
    ps: ArrayLike,
    AH: ArrayLike,
    RhoCp: ArrayLike,
) -> ArrayLike:
    """Webb-Pearman-Leuning density correction of the CO2 flux, open-path IRGAs only.

    PyFluxPro's Fco2_WPL. Fco2 (the rotated w-CO2 covariance) in mg/m^2/s
    and CO2c in mg/m^3; the result is in mg/m^2/s. Fe must already be
    WPL-corrected (wpl_correct_Fe) and Fh the sensible heat flux.
    """
    VP = calculate_VP_from_AH(AH=AH, Ta=Ta)
    rhod = calculate_dry_air_density(Ta=Ta, ps=ps, VP=VP)
    sigma = (AH / 1000) / rhod
    Lv = latent_heat_of_vaporisation(Ta)
    return (
        Fco2
        + (_MU / (1 + _MU * sigma)) * (CO2c / rhod) * (Fe / Lv)
        + (CO2c / (Ta + ZERO_C_IN_K)) * (Fh / RhoCp)
    )


@register_calculation("L")
def calculate_monin_obukhov_length(
    Ta: ArrayLike, ps: ArrayLike, AH: ArrayLike, ustar: ArrayLike, Fh: ArrayLike
) -> ArrayLike:
    """Monin-Obukhov length (m) from the sensible heat flux Fh (W/m^2)."""
    VP = calculate_VP_from_AH(AH=AH, Ta=Ta)
    rhod = calculate_dry_air_density(Ta=Ta, ps=ps, VP=VP)
    theta_v = _virtual_potential_temperature(Ta=Ta, ps=ps, VP=VP)
    return -theta_v * rhod * CP_DRY_AIR * ustar**3 / (GRAVITY * VON_KARMAN * Fh)


def monin_obukhov_length_kinematic(
    Ta: ArrayLike, ps: ArrayLike, AH: ArrayLike, ustar: ArrayLike, wT: ArrayLike
) -> ArrayLike:
    """Monin-Obukhov length (m) from the kinematic heat flux wT (m.degC/s)."""
    VP = calculate_VP_from_AH(AH=AH, Ta=Ta)
    theta_v = _virtual_potential_temperature(Ta=Ta, ps=ps, VP=VP)
    return -theta_v * np.abs(ustar) ** 3 / (GRAVITY * VON_KARMAN * wT)


@register_calculation("ET")
def calculate_ET(Fe: ArrayLike, Ta: ArrayLike, interval_seconds: float) -> ArrayLike:
    """Evapotranspiration (mm per interval) from the latent heat flux Fe (W/m^2).

    Uses the temperature-dependent Lv that calculate_Fe uses, where PyFluxPro
    divides by a constant 2.4536e6 J/kg (Lv at about 20 degC) and reports
    kg/m^2/s.
    """
    return Fe / latent_heat_of_vaporisation(Ta) * interval_seconds


@register_calculation("Fa")
def calculate_available_energy(Fn: ArrayLike, Fg: ArrayLike) -> ArrayLike:
    """Available energy (W/m^2): net radiation minus ground heat flux."""
    return Fn - Fg


# --- Storage terms (PyFluxPro L3) -------------------------------------------


@register_calculation("Sco2")
def calculate_Sco2_single_point(
    CO2: ArrayLike,
    Ta: ArrayLike,
    ps: ArrayLike,
    height: float,
    interval_seconds: float,
) -> np.ndarray:
    """CO2 storage term (umol/m^2/s) below a single CO2 measurement height (m).

    Assumes the air column beneath the sensor is well mixed: the change in
    CO2 mole fraction (umol/mol) since the previous record, converted to
    umol/m^3 with the current Ta and ps, times height over the interval.
    Inputs are a regular time series; the first record has no predecessor
    and is NaN (PyFluxPro sets it to 0). See profile_storage for profiles.
    """
    dc = np.diff(np.asarray(CO2, dtype=float), prepend=np.nan)
    return (
        height
        * dc
        * np.asarray(calculate_molar_density(ps=ps, Ta=Ta))
        / interval_seconds
    )


def calculate_soil_heat_storage(
    Ts: ArrayLike,
    Sws: ArrayLike,
    depth: float,
    bulk_density: float,
    organic_content: float,
    interval_seconds: float,
) -> np.ndarray:
    """Heat storage (W/m^2) in the soil layer above a ground heat flux plate.

    PyFluxPro's CorrectFgForStorage term, added to the plate's Fg: the soil
    volumetric heat capacity (mineral, organic and water fractions) times the
    rate of change of soil temperature Ts (degC) over the plate depth (m).
    Sws is volumetric soil water (m^3/m^3), bulk_density in kg/m^3 and
    organic_content a 0-1 mass fraction. Inputs are a regular time series;
    the first record has no predecessor and is NaN (PyFluxPro meant to mask
    it but in fact leaves its temperature change at 0).
    """
    heat_capacity = (
        (1 - organic_content) * bulk_density * SOIL_MINERAL_HEAT_CAPACITY
        + organic_content * bulk_density * SOIL_ORGANIC_HEAT_CAPACITY
        + np.asarray(Sws, dtype=float) * WATER_DENSITY * WATER_HEAT_CAPACITY
    )
    dTs = np.diff(np.asarray(Ts, dtype=float), prepend=np.nan)
    return heat_capacity * dTs / interval_seconds * abs(depth)


# --- Coordinate rotation and spectral correction (PyFluxPro L3) -------------


def rotate_2d(
    Ux,
    Uy,
    Uz,
    UxUx,
    UyUy,
    UzUz,
    UxUy,
    UxUz,
    UyUz,
    UxT,
    UyT,
    UzT,
    UxA,
    UyA,
    UzA,
    UxC,
    UyC,
    UzC,
) -> dict:
    """2D coordinate rotation of sonic-frame means and covariances to natural wind axes.

    Rotates about z to force mean v = 0, then about the new y axis to force
    mean w = 0 (Lee et al., Handbook of Micrometeorology, ch. 3); no third
    rotation. Momentum covariances use the full equations (Wesely, via
    EddyPro), as PyFluxPro's CoordRotation2D. Inputs are the sonic-frame
    means (m/s), variances, and covariances with Tv (T), H2O (A) and CO2
    (C) in their canonical units; outputs keep those units. Returns u, v, w,
    uu, vv, ww, uw, uv, vw, wT, wA, wC and the angles eta and theta (deg).
    """
    ws2d = np.sqrt(Ux**2 + Uy**2)
    ws3d = np.sqrt(Ux**2 + Uy**2 + Uz**2)
    ce, se = Ux / ws2d, Uy / ws2d  # cos/sin of eta, the rotation about z
    ct, st = ws2d / ws3d, Uz / ws3d  # cos/sin of theta, the rotation about y
    return {
        "eta": np.rad2deg(np.arctan2(se, ce)),
        "theta": np.rad2deg(np.arctan2(st, ct)),
        "u": Ux * ct * ce + Uy * ct * se + Uz * st,
        "v": Uy * ce - Ux * se,
        "w": Uz * ct - Ux * st * ce - Uy * st * se,
        "uu": UxUx * ct**2 * ce**2
        + UyUy * ct**2 * se**2
        + UzUz * st**2
        + 2 * UxUy * ct**2 * ce * se
        + 2 * UxUz * ct * st * ce
        + 2 * UyUz * ct * st * se,
        "vv": UyUy * ce**2 + UxUx * se**2 - 2 * UxUy * ce * se,
        "ww": UzUz * ct**2
        + UxUx * st**2 * ce**2
        + UyUy * st**2 * se**2
        - 2 * UxUz * ct * st * ce
        - 2 * UyUz * ct * st * se
        + 2 * UxUy * st**2 * ce * se,
        "uw": UxUz * ce * (ct**2 - st**2)
        - 2 * UxUy * ct * st * ce * se
        + UyUz * se * (ct**2 - st**2)
        - UxUx * ct * st * ce**2
        - UyUy * ct * st * se**2
        + UzUz * ct * st,
        "uv": UxUy * ct * (ce**2 - se**2)
        + UyUz * st * ce
        - UxUz * st * se
        - UxUx * ct * ce * se
        + UyUy * ct * ce * se,
        "vw": UyUz * ct * ce
        - UxUz * ct * se
        - UxUy * st * (ce**2 - se**2)
        + UxUx * st * ce * se
        - UyUy * st * ce * se,
        "wT": UzT * ct - UxT * st * ce - UyT * st * se,
        "wA": UzA * ct - UxA * st * ce - UyA * st * se,
        "wC": UzC * ct - UxC * st * ce - UyC * st * se,
    }


def _massman_stability_coefficients(zoL):
    """Massman's nx (momentum, scalar) and alpha, stable where z/L > 0."""
    zoL = np.asarray(zoL, dtype=float)
    stable = zoL > 0
    zoL = np.where(stable, zoL, 0.0)  # keep the stable branches finite elsewhere
    nx_mom = np.where(stable, 0.079 * (1 + 7.9 * zoL) ** 0.75, 0.079)
    nx_scalar = np.where(stable, 2.0 - 1.915 / (1 + 0.5 * zoL), 0.085)
    alpha = np.where(stable, 1.0, 0.925)
    return nx_mom, nx_scalar, alpha


def _massman_ratio(b, p, alpha):
    """Massman's attenuated-to-true covariance ratio from the b and p coefficients."""
    ba, pa = b**alpha, p**alpha
    return (ba / (ba + 1)) * (ba / (ba + pa)) * (1 / (pa + 1))


def massman_correction(
    uw,
    vw,
    wT,
    wA,
    wC,
    u,
    Ta,
    ps,
    AH,
    *,
    zmd: float,
    lateral_separation: float,
    longitudinal_separation: float,
    sonic_path_vertical: float,
    sonic_path_horizontal: float,
    sonic_path_tv: float,
    irga_path_diameter: float,
    irga_path_length: float,
    averaging_period: float = 1800.0,
) -> dict:
    """Massman (2000, 2001) spectral correction of the rotated covariances.

    PyFluxPro's MassmanStandard: two passes, the first estimating ustar and
    z/L from the uncorrected covariances, the second recomputing the
    correction from the first-pass-corrected ones. u is the rotated mean
    wind speed (m/s); zmd the measurement height above the displacement
    height (m); separations (sonic to IRGA) and instrument path dimensions
    in m; averaging_period in s (PyFluxPro hard-codes 1800). Returns the
    corrected uw, vw, wT, wA and wC.
    """
    tau_sonic_law = sonic_path_vertical / (8.4 * u)
    tau_sonic_lat = sonic_path_tv / (4.0 * u)
    tau_mom = np.sqrt(
        (sonic_path_vertical / (5.7 * u)) ** 2
        + (sonic_path_horizontal / (2.8 * u)) ** 2
    )
    tau_wT = np.sqrt(tau_sonic_law**2 + tau_sonic_lat**2)
    tau_irga = np.sqrt(
        tau_sonic_law**2
        + (irga_path_length / (4.0 * u)) ** 2
        + ((0.2 + 0.4 * irga_path_diameter / irga_path_length) * (irga_path_length / u))
        ** 2
        + 0.016**2  # IRGA bandwidth time constant
        + (lateral_separation / (1.1 * u)) ** 2
        + (longitudinal_separation / (1.05 * u)) ** 2
    )
    tau_block = averaging_period / 2.8

    def ratios(uw_, vw_, wT_):
        ustar = calculate_ustar(uw_, vw_)
        L = monin_obukhov_length_kinematic(Ta=Ta, ps=ps, AH=AH, ustar=ustar, wT=wT_)
        nx_mom, nx_scalar, alpha = _massman_stability_coefficients(zmd / L)
        fx_mom = nx_mom * u / zmd
        fx_scalar = nx_scalar * u / zmd
        b_mom = 2 * np.pi * fx_mom * tau_block
        b_scalar = 2 * np.pi * fx_scalar * tau_block
        return (
            _massman_ratio(b_mom, 2 * np.pi * fx_mom * tau_mom, alpha),
            _massman_ratio(b_scalar, 2 * np.pi * fx_scalar * tau_wT, alpha),
            _massman_ratio(b_scalar, 2 * np.pi * fx_scalar * tau_irga, alpha),
        )

    r_mom, r_wT, _ = ratios(uw, vw, wT)
    r_mom, r_wT, r_irga = ratios(uw / r_mom, vw / r_mom, wT / r_wT)
    return {
        "uw": uw / r_mom,
        "vw": vw / r_mom,
        "wT": wT / r_wT,
        "wA": wA / r_irga,
        "wC": wC / r_irga,
    }

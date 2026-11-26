"""Tests for services/data/unit_conversions.py."""

import unittest

import numpy as np
import yaml

from fluxcore.domain.constants import CO2_MOL_MASS, CONFIG_PATH, H2O_MOL_MASS
from fluxcore.services.data import unit_conversions as uc


class UnitConversionsTestCase(unittest.TestCase):
    def test_results_are_canonical_units(self):
        x = np.array([10.0])
        np.testing.assert_allclose(uc.convert_CO2_flux(x), 10.0 * 1000 / CO2_MOL_MASS)
        np.testing.assert_allclose(uc.convert_CO2_density(x), 10.0 * CO2_MOL_MASS)
        np.testing.assert_allclose(
            uc.convert_H2O_density(x), 10.0 * H2O_MOL_MASS / 1000
        )
        np.testing.assert_allclose(
            uc.convert_H2O_density(x, from_units="kg/m^3"), 10000.0
        )
        np.testing.assert_allclose(uc.convert_signal_strength(np.array([0.9])), 90.0)
        np.testing.assert_allclose(uc.convert_RH(np.array([0.55])), 55.0)
        np.testing.assert_allclose(uc.convert_Sws(np.array([45.0])), 0.45)
        np.testing.assert_allclose(
            uc.convert_temperature(np.array([273.15])), 0.0, atol=1e-9
        )
        np.testing.assert_allclose(uc.convert_tbody(np.array([273.15])), 0.0, atol=1e-9)
        rs = np.array([33000.0])
        ln_r = np.log(rs)
        expected_tbody_ntc = (
            1 / (1.0295e-3 + 2.391e-4 * ln_r + 1.568e-7 * ln_r**3) - 273.15
        )
        np.testing.assert_allclose(
            uc.convert_tbody(rs, from_units="ohms_ntc"), expected_tbody_ntc
        )
        np.testing.assert_allclose(
            uc.convert_tbody(np.array([110.0]), from_units="ohms_pt100"),
            (110.0 / 100.0 - 1) / 0.00385,
        )
        np.testing.assert_allclose(uc.convert_pressure(np.array([101325.0])), 101.325)
        np.testing.assert_allclose(
            uc.convert_pressure(np.array([1013.25]), from_units="hPa"), 101.325
        )
        np.testing.assert_allclose(uc.convert_precipitation(np.array([5.0])), 1.0)
        np.testing.assert_allclose(
            uc.convert_precipitation(np.array([5.0]), from_units="pulse_0.5mm"), 2.5
        )

    def test_co2_conversions(self):
        np.testing.assert_allclose(
            uc.convert_CO2_covariance(np.array([1.0])), CO2_MOL_MASS / 1000
        )
        np.testing.assert_allclose(
            uc.convert_CO2_mole_fraction(np.array([0.04])), 400.0
        )
        for quantity in ("Sco2", "NEE", "NEP", "GPP", "ER"):
            self.assertIs(uc.get_unit_conversion(quantity), uc.convert_CO2_flux)

    def test_every_valid_input_unit_converts(self):
        # A unit listed in canonical_quantities.yml passes config validation,
        # so without a conversion the build would fail at load time instead.
        known_gaps = {
            # W/m^2 -> umol/m^2/s needs a chosen photon-energy factor
            ("APAR", "W/m^2"),
            ("PAR", "W/m^2"),
            ("Fqd", "W/m^2"),
            ("Fqu", "W/m^2"),
            # a rate -> per-interval depth needs the averaging interval
            ("ET", "mmol/m^2/s"),
            # a scaling temperature is a difference: K -> degC is not an offset
            ("Tstar", "K"),
        }
        with open(CONFIG_PATH / "canonical_quantities.yml") as f:
            quantities = yaml.safe_load(f)
        for name, meta in quantities.items():
            for units in meta.get("valid_input_units") or []:
                if units == meta["standard_units"] or (name, units) in known_gaps:
                    continue
                with self.subTest(quantity=name, units=units):
                    converter = uc.get_unit_conversion(name)
                    self.assertIsNotNone(converter)
                    converter(np.array([1.0]), from_units=units)

    def test_diagnostic_valid_count_becomes_invalid_count(self):
        np.testing.assert_allclose(
            uc.convert_diagnostic(np.array([18000.0, 17990.0]), n_samples=18000),
            [0.0, 10.0],
        )

    def test_unsupported_units_raise(self):
        funcs = (
            uc.convert_CO2_flux,
            uc.convert_CO2_covariance,
            uc.convert_CO2_mole_fraction,
            uc.convert_RH,
            uc.convert_Sws,
            uc.convert_pressure,
            uc.convert_temperature,
            uc.convert_tbody,
        )
        for func in funcs:
            with self.assertRaises(ValueError):
                func(np.array([1.0]), from_units="not_a_unit")

    def test_registry_lookups(self):
        for quantity in (
            "Fco2",
            "CO2c",
            "Sig",
            "SigCO2",
            "SigH2O",
            "Diag",
            "AH",
            "Precip",
            "ps",
            "RH",
            "Sws",
            "Ta",
            "Tv",
            "Tbody",
        ):
            self.assertIsNotNone(uc.get_unit_conversion(quantity), quantity)
        self.assertIsNone(uc.get_unit_conversion("NotAQuantity"))


if __name__ == "__main__":
    unittest.main()

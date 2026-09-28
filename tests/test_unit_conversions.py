"""Tests for services/data/unit_conversions.py."""

import unittest

import numpy as np

from domain.constants import CO2_MOL_MASS, H2O_MOL_MASS
from services.data import unit_conversions as uc


class UnitConversionsTestCase(unittest.TestCase):
    def test_results_are_canonical_units(self):
        x = np.array([10.0])
        np.testing.assert_allclose(uc.convert_CO2_flux(x), 10.0 * 1000 / CO2_MOL_MASS)
        np.testing.assert_allclose(uc.convert_CO2_density(x), 10.0 * CO2_MOL_MASS)
        np.testing.assert_allclose(uc.convert_H2O_density(x), 10.0 * H2O_MOL_MASS / 1000)
        np.testing.assert_allclose(uc.convert_H2O_density(x, from_units="kg/m^3"), 10000.0)
        np.testing.assert_allclose(uc.convert_signal_strength(np.array([0.9])), 90.0)
        np.testing.assert_allclose(uc.convert_RH(np.array([0.55])), 55.0)
        np.testing.assert_allclose(uc.convert_Sws(np.array([45.0])), 0.45)
        np.testing.assert_allclose(uc.convert_temperature(np.array([273.15])), 0.0, atol=1e-9)
        np.testing.assert_allclose(uc.convert_pressure(np.array([101325.0])), 101.325)
        np.testing.assert_allclose(uc.convert_pressure(np.array([1013.25]), from_units="hPa"), 101.325)
        np.testing.assert_allclose(uc.convert_precipitation(np.array([5.0])), 1.0)
        np.testing.assert_allclose(uc.convert_precipitation(np.array([5.0]), from_units="pulse_0.5mm"), 2.5)

    def test_diagnostic_valid_count_becomes_invalid_count(self):
        np.testing.assert_allclose(uc.convert_diagnostic(np.array([18000.0, 17990.0]), n_samples=18000), [0.0, 10.0])

    def test_unsupported_units_raise(self):
        for func in (uc.convert_CO2_flux, uc.convert_RH, uc.convert_Sws, uc.convert_pressure, uc.convert_temperature):
            with self.assertRaises(ValueError):
                func(np.array([1.0]), from_units="not_a_unit")

    def test_registry_lookups(self):
        for quantity in ("Fco2", "CO2c", "Sig", "SigCO2", "SigH2O", "Diag", "AH", "Precip", "ps", "RH", "Sws", "Ta", "Tv", "Tbody"):
            self.assertIsNotNone(uc.get_unit_conversion(quantity), quantity)
        self.assertIsNone(uc.get_unit_conversion("NotAQuantity"))


if __name__ == "__main__":
    unittest.main()

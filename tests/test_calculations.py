"""Tests for services/data/calculations.py."""

import unittest

import numpy as np

from services.data import calculations


class CalculationsTestCase(unittest.TestCase):
    def test_saturation_vapour_pressure_matches_buck_reference(self):
        # Buck (1996): es(20 degC) = 2.339 kPa
        self.assertAlmostEqual(float(calculations.calculate_es(np.array(20.0))), 2.339, places=2)

    def test_dew_point_equals_air_temperature_at_saturation(self):
        # Regression guard: the dew point must invert the same saturation
        # curve calculate_es uses (a mismatched constant once made Td overshoot
        # Ta by up to 0.4 K at RH = 100).
        ta = np.array([-10.0, -5.0, 0.0, 10.0, 20.0, 30.0, 40.0])
        dew = calculations.calculate_dew_point(Ta=ta, RH=np.full(7, 100.0))
        np.testing.assert_allclose(dew, ta, atol=0.02)

    def test_dew_point_is_below_air_temperature_when_unsaturated_and_rises_with_rh(self):
        ta = np.full(4, 20.0)
        dew = calculations.calculate_dew_point(Ta=ta, RH=np.array([20.0, 40.0, 60.0, 80.0]))
        self.assertTrue(np.all(dew < ta))
        self.assertTrue(np.all(np.diff(dew) > 0))

    def test_vapour_pressure_and_deficit_are_consistent(self):
        ta, rh = np.array([25.0]), np.array([60.0])
        es = calculations.calculate_es(ta)
        np.testing.assert_allclose(calculations.calculate_e(ta, rh), es * 0.6)
        np.testing.assert_allclose(calculations.calculate_vpd(ta, rh), es * 0.4)

    def test_absolute_and_relative_humidity_round_trip(self):
        ta, rh, ps = np.array([5.0, 20.0, 35.0]), np.array([30.0, 60.0, 90.0]), np.full(3, 100.0)
        ah = calculations.calculate_AH_from_RH(Ta=ta, RH=rh, ps=ps)
        np.testing.assert_allclose(calculations.calculate_RH_from_AH(AH=ah, Ta=ta, ps=ps), rh)

    def test_molar_density_of_air_at_standard_conditions(self):
        # 101.325 kPa, 0 degC -> about 44.6 mol/m^3
        self.assertAlmostEqual(
            float(calculations.calculate_molar_density(ps=np.array(101.325), Ta=np.array(0.0))), 44.6, places=1
        )

    def test_standard_pressure_at_known_elevations(self):
        self.assertAlmostEqual(calculations.standard_pressure_kpa(0), 101.325, places=3)
        self.assertAlmostEqual(calculations.standard_pressure_kpa(1650), 83.0, delta=0.1)

    def test_registry_lookups(self):
        for quantity in ("AH", "RH", "es", "CO2", "e", "VPD", "rho_mol", "Td"):
            self.assertIsNotNone(calculations.get_calculation(quantity), quantity)
        self.assertIsNone(calculations.get_calculation("NotAQuantity"))
        # unregistered helper: its input is site elevation, not a canonical quantity
        self.assertNotIn("standard_pressure_kpa", {f.__name__ for f in calculations.CALCULATION_REGISTRY.values()})


if __name__ == "__main__":
    unittest.main()

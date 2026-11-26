"""Tests for services/data/calculations.py."""

import unittest

import numpy as np

from fluxcore.services.data import calculations


class CalculationsTestCase(unittest.TestCase):
    def test_saturation_vapour_pressure_matches_buck_reference(self):
        # Buck (1996): es(20 degC) = 2.339 kPa
        self.assertAlmostEqual(
            float(calculations.calculate_es(np.array(20.0))), 2.339, places=2
        )

    def test_dew_point_equals_air_temperature_at_saturation(self):
        # Regression guard: the dew point must invert the same saturation
        # curve calculate_es uses (a mismatched constant once made Td overshoot
        # Ta by up to 0.4 K at RH = 100).
        ta = np.array([-10.0, -5.0, 0.0, 10.0, 20.0, 30.0, 40.0])
        dew = calculations.calculate_dew_point(Ta=ta, RH=np.full(7, 100.0))
        np.testing.assert_allclose(dew, ta, atol=0.02)

    def test_dew_point_is_below_air_temperature_when_unsaturated_and_rises_with_rh(
        self,
    ):
        ta = np.full(4, 20.0)
        dew = calculations.calculate_dew_point(
            Ta=ta, RH=np.array([20.0, 40.0, 60.0, 80.0])
        )
        self.assertTrue(np.all(dew < ta))
        self.assertTrue(np.all(np.diff(dew) > 0))

    def test_vapour_pressure_and_deficit_are_consistent(self):
        ta, rh = np.array([25.0]), np.array([60.0])
        es = calculations.calculate_es(ta)
        np.testing.assert_allclose(calculations.calculate_e(ta, rh), es * 0.6)
        np.testing.assert_allclose(calculations.calculate_vpd(ta, rh), es * 0.4)

    def test_absolute_and_relative_humidity_round_trip(self):
        ta, rh = np.array([5.0, 20.0, 35.0]), np.array([30.0, 60.0, 90.0])
        ah = calculations.calculate_AH_from_RH(Ta=ta, RH=rh)
        np.testing.assert_allclose(calculations.calculate_RH_from_AH(AH=ah, Ta=ta), rh)

    def test_absolute_humidity_matches_the_pressure_form(self):
        # The old form, e/ps * molar_density(ps) * M_w: pressure cancels.
        ta, rh = np.array([5.0, 20.0, 35.0]), np.array([30.0, 60.0, 90.0])
        for ps in (np.full(3, 85.0), np.full(3, 101.325)):
            old = (
                calculations.calculate_e(ta, rh)
                / ps
                * calculations.calculate_molar_density(ps=ps, Ta=ta)
                * 18.02
            )
            np.testing.assert_allclose(
                calculations.calculate_AH_from_RH(Ta=ta, RH=rh), old, rtol=1e-12
            )
        # 20 degC, 60 %: about 10.4 g/m^3
        self.assertAlmostEqual(
            float(
                calculations.calculate_AH_from_RH(Ta=np.array(20.0), RH=np.array(60.0))
            ),
            10.4,
            places=1,
        )

    def test_molar_density_of_air_at_standard_conditions(self):
        # 101.325 kPa, 0 degC -> about 44.6 mol/m^3
        self.assertAlmostEqual(
            float(
                calculations.calculate_molar_density(
                    ps=np.array(101.325), Ta=np.array(0.0)
                )
            ),
            44.6,
            places=1,
        )

    def test_standard_pressure_at_known_elevations(self):
        self.assertAlmostEqual(calculations.standard_pressure_kpa(0), 101.325, places=3)
        self.assertAlmostEqual(
            calculations.standard_pressure_kpa(1650), 83.0, delta=0.1
        )

    def test_net_radiation_is_shortwave_plus_longwave_balance(self):
        # CumberlandPlain, 2026-09-29 audit: on-logger Rn_Avg matched this sum
        # to within 0.00 W/m^2 wherever both existed.
        fsd, fsu, fld, flu = (
            np.array(500.0),
            np.array(80.0),
            np.array(350.0),
            np.array(420.0),
        )
        self.assertAlmostEqual(
            float(
                calculations.calculate_net_radiation(Fsd=fsd, Fsu=fsu, Fld=fld, Flu=flu)
            ),
            350.0,
        )

    # --- L3 ---------------------------------------------------------------

    def _air(self):
        ta, ps = np.array([5.0, 20.0, 35.0]), np.full(3, 100.0)
        ah = calculations.calculate_AH_from_RH(Ta=ta, RH=np.full(3, 60.0))
        return ta, ps, ah

    def test_humidity_quantities_are_consistent(self):
        ta, ps, ah = self._air()
        vp = calculations.calculate_VP_from_AH(AH=ah, Ta=ta)
        np.testing.assert_allclose(vp, calculations.calculate_e(ta, np.full(3, 60.0)))
        sh = calculations.calculate_SH_from_AH(AH=ah, Ta=ta, ps=ps)
        # SH = eps e / (p - (1 - eps) e), eps = 18.02 / 28.97
        eps = 18.02 / 28.97
        np.testing.assert_allclose(sh, eps * vp / (ps - (1 - eps) * vp))
        np.testing.assert_allclose(
            calculations.calculate_H2O_mole_fraction(AH=ah, Ta=ta, ps=ps),
            vp / ps * 1000,
        )
        # saturated air has no specific humidity deficit
        ah_sat = calculations.calculate_AH_from_RH(Ta=ta, RH=np.full(3, 100.0))
        sh_sat = calculations.calculate_SH_from_AH(AH=ah_sat, Ta=ta, ps=ps)
        np.testing.assert_allclose(
            calculations.calculate_SHD(SH=sh_sat, Ta=ta, ps=ps), 0.0, atol=1e-12
        )

    def test_ta_from_tv_inverts_the_campbell_virtual_temperature(self):
        ta, sh = np.array([10.0, 30.0]), np.array([0.005, 0.02])
        tv = (ta + 273.15) * (1 + 0.51 * sh) - 273.15
        np.testing.assert_allclose(calculations.calculate_Ta_from_Tv(tv, sh), ta)

    def test_air_density_at_standard_conditions(self):
        ta, ps, vp = np.array(20.0), np.array(101.325), np.array(0.0)
        self.assertAlmostEqual(
            float(calculations.calculate_moist_air_density(ta, ps, vp)), 1.204, 3
        )
        # water vapour is lighter than the dry air it displaces
        self.assertLess(
            float(calculations.calculate_moist_air_density(ta, ps, np.array(2.0))),
            1.204,
        )

    def test_rho_cp_of_dry_air(self):
        ta, ps = np.array(20.0), np.array(101.325)
        rho_cp = calculations.calculate_rho_cp(ta, ta, ps, np.array(0.0), np.array(0.0))
        self.assertAlmostEqual(float(rho_cp), 1.204 * 1006.3, delta=1.0)

    def test_momentum_flux_and_ustar(self):
        uw, vw = np.array(-0.09), np.array(0.0)
        self.assertAlmostEqual(float(calculations.calculate_ustar(uw, vw)), 0.3)
        self.assertAlmostEqual(
            float(calculations.calculate_Fm(uw, vw, np.array(1.2))), 0.108
        )

    def test_sensible_heat_equals_virtual_heat_in_dry_air(self):
        fhv = calculations.calculate_Fhv(np.array(0.1), np.array(1200.0))
        fh = calculations.calculate_Fh_from_Fhv(
            Fhv=fhv,
            Tv=np.array(25.0),
            SH=np.array(0.0),
            wT=np.array(0.1),
            wA=np.array(0.0),
            RhoCp=np.array(1200.0),
            Rhoa=np.array(1.2),
        )
        self.assertAlmostEqual(float(fh), 120.0)

    def test_latent_heat_flux_and_et(self):
        ta = np.array(20.0)
        lv = calculations.latent_heat_of_vaporisation(ta)
        self.assertAlmostEqual(float(lv), 2.4535e6, delta=1e3)
        fe = calculations.calculate_Fe(np.array(0.1), ta)  # 0.1 g/m^2/s
        self.assertAlmostEqual(float(fe), float(lv) * 1e-4)
        # 0.1 g/m^2/s for 30 min is 0.18 kg/m^2, i.e. 0.18 mm
        self.assertAlmostEqual(float(calculations.calculate_ET(fe, ta, 1800)), 0.18)

    def test_wpl_corrections_vanish_without_fluxes(self):
        ta, ps, ah = self._air()
        zeros = np.zeros(3)
        rho_cp = np.full(3, 1200.0)
        np.testing.assert_allclose(
            calculations.wpl_correct_Fe(zeros, zeros, ta, ps, ah, rho_cp), 0.0
        )
        np.testing.assert_allclose(
            calculations.wpl_correct_Fco2(
                zeros, zeros, zeros, np.full(3, 700.0), ta, ps, ah, rho_cp
            ),
            0.0,
        )

    def test_wpl_increases_co2_flux_under_upward_heat_and_vapour_flux(self):
        ta, ps, ah = self._air()
        fh, fe, rho_cp = np.full(3, 200.0), np.full(3, 150.0), np.full(3, 1200.0)
        fco2 = np.full(3, -0.5)
        fe_wpl = calculations.wpl_correct_Fe(fe, fh, ta, ps, ah, rho_cp)
        self.assertTrue(np.all(fe_wpl > fe))
        corrected = calculations.wpl_correct_Fco2(
            fco2, fe_wpl, fh, np.full(3, 700.0), ta, ps, ah, rho_cp
        )
        self.assertTrue(np.all(corrected > fco2))

    def test_monin_obukhov_length_sign_follows_stability(self):
        ta, ps, ah = self._air()
        ustar = np.full(3, 0.3)
        unstable = calculations.calculate_monin_obukhov_length(
            ta, ps, ah, ustar, np.full(3, 100.0)
        )
        stable = calculations.calculate_monin_obukhov_length(
            ta, ps, ah, ustar, np.full(3, -20.0)
        )
        self.assertTrue(np.all(unstable < 0) and np.all(stable > 0))

    def test_available_energy(self):
        self.assertEqual(
            calculations.calculate_available_energy(np.array(400.0), np.array(50.0)),
            350.0,
        )

    def test_single_point_co2_storage(self):
        ta, ps = np.full(3, 20.0), np.full(3, 100.0)
        sco2 = calculations.calculate_Sco2_single_point(
            np.array([400.0, 400.0, 401.8]), ta, ps, height=10.0, interval_seconds=1800
        )
        self.assertTrue(np.isnan(sco2[0]))
        self.assertEqual(sco2[1], 0.0)
        # 1.8 umol/mol over a 10 m column of air in 30 min
        rho = float(calculations.calculate_molar_density(ps=ps[2], Ta=ta[2]))
        self.assertAlmostEqual(sco2[2], 10.0 * 1.8 * rho / 1800)

    def test_soil_heat_storage(self):
        s = calculations.calculate_soil_heat_storage(
            Ts=np.array([20.0, 21.8]),
            Sws=np.array([0.2, 0.2]),
            depth=0.08,
            bulk_density=1500.0,
            organic_content=0.0,
            interval_seconds=1800,
        )
        self.assertTrue(np.isnan(s[0]))
        cs = 1500.0 * 840.0 + 0.2 * 1000.0 * 4180.0
        self.assertAlmostEqual(s[1], cs * 1.8 / 1800 * 0.08)

    def test_rotate_2d_matches_the_rotation_matrix(self):
        rng = np.random.default_rng(0)
        n = 50
        ux, uy = rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)
        uz = rng.uniform(-0.3, 0.3, n)
        a = rng.normal(size=(n, 3, 3))
        cov = a @ a.transpose(0, 2, 1)
        scalars = rng.normal(size=(n, 3))
        result = calculations.rotate_2d(
            ux, uy, uz,
            cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2],
            cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 2],
            *scalars.T, *scalars.T, *scalars.T,
        )  # fmt: skip
        np.testing.assert_allclose(result["v"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["w"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["u"], np.sqrt(ux**2 + uy**2 + uz**2))
        ws2d = np.hypot(ux, uy)
        ws3d = np.sqrt(ux**2 + uy**2 + uz**2)
        ce, se, ct, st = ux / ws2d, uy / ws2d, ws2d / ws3d, uz / ws3d
        rot = np.stack(
            [
                np.stack([ct * ce, ct * se, st], -1),
                np.stack([-se, ce, np.zeros(n)], -1),
                np.stack([-st * ce, -st * se, ct], -1),
            ],
            1,
        )
        cov_rot = rot @ cov @ rot.transpose(0, 2, 1)
        for key, (i, j) in {
            "uu": (0, 0), "vv": (1, 1), "ww": (2, 2),
            "uv": (0, 1), "uw": (0, 2), "vw": (1, 2),
        }.items():  # fmt: skip
            np.testing.assert_allclose(result[key], cov_rot[:, i, j], atol=1e-12)
        w_scalar = np.einsum("nij,nj->ni", rot, scalars)[:, 2]
        for key in ("wT", "wA", "wC"):
            np.testing.assert_allclose(result[key], w_scalar, atol=1e-12)

    def _massman(self, **overrides):
        ta, ps, ah = self._air()
        kwargs = dict(
            zmd=4.0,
            lateral_separation=0.1,
            longitudinal_separation=0.05,
            sonic_path_vertical=0.115,
            sonic_path_horizontal=0.058,
            sonic_path_tv=0.115,
            irga_path_diameter=0.0095,
            irga_path_length=0.127,
        )
        kwargs.update(overrides)
        raw = dict(
            uw=np.full(3, -0.09),
            vw=np.full(3, 0.01),
            wT=np.array([0.1, -0.02, 0.15]),
            wA=np.full(3, 0.05),
            wC=np.full(3, -0.5),
        )
        u = np.array([1.0, 2.0, 4.0])
        return raw, calculations.massman_correction(
            **raw, u=u, Ta=ta, ps=ps, AH=ah, **kwargs
        )

    def test_massman_correction_only_amplifies_covariances(self):
        raw, corrected = self._massman()
        for key in raw:
            ratio = corrected[key] / raw[key]
            self.assertTrue(np.all(ratio > 1), key)
        # the separated, longer-path IRGA loses more than the sonic temperature
        self.assertTrue(
            np.all(corrected["wC"] / raw["wC"] > corrected["wT"] / raw["wT"])
        )

    def test_massman_correction_vanishes_for_ideal_instruments(self):
        tiny = 1e-9
        raw, corrected = self._massman(
            lateral_separation=0.0,
            longitudinal_separation=0.0,
            sonic_path_vertical=tiny,
            sonic_path_horizontal=tiny,
            sonic_path_tv=tiny,
            irga_path_diameter=tiny,
            irga_path_length=tiny,
            averaging_period=1e9,
        )
        for key in ("uw", "vw", "wT"):
            np.testing.assert_allclose(corrected[key], raw[key], rtol=1e-4)

    def test_registry_lookups(self):
        for quantity in (
            *("AH", "RH", "es", "SVP", "CO2", "e", "VPD", "rho_mol", "Td", "Fn"),
            *("VP", "SH", "SHD", "H2O", "Ta", "Rhod", "Rhoa"),
            *("ustar", "Fm", "Fh", "Fe", "L", "ET", "Fa", "Sco2"),
        ):
            self.assertIsNotNone(calculations.get_calculation(quantity), quantity)
        self.assertIsNone(calculations.get_calculation("NotAQuantity"))
        # unregistered helper: its input is site elevation, not a canonical quantity
        self.assertNotIn(
            "standard_pressure_kpa",
            {f.__name__ for f in calculations.CALCULATION_REGISTRY.values()},
        )


if __name__ == "__main__":
    unittest.main()

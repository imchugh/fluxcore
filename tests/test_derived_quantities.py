"""Tests for orchestration/derived_quantities.py."""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from fluxcore.orchestration.derived_quantities import (
    DatasetBuildIntermediate,
    add_day_night_indicator,
    add_net_radiation,
    derive_missing_humidity,
)
from fluxcore.services.data.calculations import calculate_AH_from_RH

LAT, LON, ELEV = -34.0027, 140.5877, 60


def _radiation_attrs(quantity):
    return {
        "quantity": quantity,
        "statistic_type": "average",
        "height": 30.0,
        "height_range": None,
        "instrument": "Kipp&Zonen CNR4",
        "instrument_history": [
            {
                "role": None,
                "periods": [
                    {"instrument": "Kipp&Zonen CNR4", "start": None, "end": None}
                ],
            }
        ],
    }


def _hmp_attrs(quantity, statistic_type="average"):
    return {
        **_radiation_attrs(quantity),
        "statistic_type": statistic_type,
        "height": 2.0,
        "instrument": "Vaisala HMP155",
        "instrument_uri": "http://example.org/hmp155",
        "instrument_history": [
            {
                "role": None,
                "periods": [
                    {"instrument": "Vaisala HMP155", "start": None, "end": None}
                ],
            }
        ],
    }


class DeriveMissingHumidityTestCase(unittest.TestCase):
    def _result(self, columns, quantities):
        idx = pd.date_range("2025-01-01", periods=2, freq="30min")
        return DatasetBuildIntermediate(
            df=pd.DataFrame(columns, index=idx),
            var_attrs={name: _hmp_attrs(q) for name, q in quantities.items()},
        )

    def test_ah_is_derived_without_any_pressure_variable(self):
        result = derive_missing_humidity(
            self._result(
                {"Ta_2m": [20.0, 10.0], "RH_2m": [60.0, 80.0]},
                {"Ta_2m": "Ta", "RH_2m": "RH"},
            )
        )
        np.testing.assert_allclose(
            result.df["AH_2m"],
            calculate_AH_from_RH(Ta=np.array([20.0, 10.0]), RH=np.array([60.0, 80.0])),
        )
        attrs = result.var_attrs["AH_2m"]
        self.assertEqual(attrs["quantity"], "AH")
        # Calculated, but it still comes from the HMP's Ta and RH.
        self.assertEqual(attrs["instrument"], "Vaisala HMP155")
        self.assertEqual(attrs["instrument_uri"], "http://example.org/hmp155")
        # Same instruments and dates as its source, each period marked calculated.
        [source_role] = result.var_attrs["RH_2m"]["instrument_history"]
        self.assertEqual(
            attrs["instrument_history"],
            [
                {
                    "role": source_role["role"],
                    "periods": [
                        {**p, "calculated_from": ["RH_2m", "Ta_2m"]}
                        for p in source_role["periods"]
                    ],
                }
            ],
        )

    def test_missing_pressure_does_not_blank_the_derived_ah(self):
        result = derive_missing_humidity(
            self._result(
                {"Ta_2m": [20.0, 10.0], "RH_2m": [60.0, 80.0], "ps": [np.nan, 100.0]},
                {"Ta_2m": "Ta", "RH_2m": "RH", "ps": "ps"},
            )
        )
        self.assertFalse(result.df["AH_2m"].isna().any())

    def test_rh_is_derived_from_ah(self):
        ah = calculate_AH_from_RH(Ta=np.array([20.0, 10.0]), RH=np.array([60.0, 80.0]))
        result = derive_missing_humidity(
            self._result(
                {"Ta_2m": [20.0, 10.0], "AH_2m": ah}, {"Ta_2m": "Ta", "AH_2m": "AH"}
            )
        )
        np.testing.assert_allclose(result.df["RH_2m"], [60.0, 80.0])


class AddNetRadiationTestCase(unittest.TestCase):
    def test_always_computes_fn_from_the_four_components(self):
        df = pd.DataFrame(
            {"Fsd": [500.0], "Fsu": [80.0], "Fld": [350.0], "Flu": [420.0]}
        )
        var_attrs = {q: _radiation_attrs(q) for q in ("Fsd", "Fsu", "Fld", "Flu")}
        result = DatasetBuildIntermediate(df=df, var_attrs=var_attrs)

        out = add_net_radiation(result)

        self.assertAlmostEqual(out.df["Fn"].iloc[0], 350.0)
        self.assertEqual(out.var_attrs["Fn"]["quantity"], "Fn")
        self.assertEqual(out.var_attrs["Fn"]["instrument"], "Kipp&Zonen CNR4")
        self.assertEqual(out.var_attrs["Fn"]["long_name"], "Net radiation")

    def test_uses_the_exactly_named_components_not_a_second_sensor(self):
        # A second shortwave sensor listed first must not become Fn's Fsd.
        df = pd.DataFrame(
            {
                "Fsd_2": [900.0],
                "Fsd": [500.0],
                "Fsu": [80.0],
                "Fld": [350.0],
                "Flu": [420.0],
            }
        )
        var_attrs = {
            q: _radiation_attrs(q.split("_")[0])
            for q in ("Fsd_2", "Fsd", "Fsu", "Fld", "Flu")
        }
        out = add_net_radiation(DatasetBuildIntermediate(df=df, var_attrs=var_attrs))
        self.assertAlmostEqual(out.df["Fn"].iloc[0], 350.0)

    def test_missing_component_returns_result_unchanged(self):
        df = pd.DataFrame({"Fsd": [500.0], "Fsu": [80.0], "Fld": [350.0]})
        var_attrs = {q: _radiation_attrs(q) for q in ("Fsd", "Fsu", "Fld")}
        result = DatasetBuildIntermediate(df=df, var_attrs=var_attrs)

        out = add_net_radiation(result)

        self.assertNotIn("Fn", out.df.columns)
        self.assertIs(out, result)

    def test_overwrites_any_preexisting_fn_column(self):
        # Deliberately unconditional (unlike the humidity/CO2 derivations): Fn
        # is never imported as of 2026-09-29, but if a stray column existed it would
        # still be replaced, not left alone.
        df = pd.DataFrame(
            {
                "Fsd": [500.0],
                "Fsu": [80.0],
                "Fld": [350.0],
                "Flu": [420.0],
                "Fn": [999.0],
            }
        )
        var_attrs = {q: _radiation_attrs(q) for q in ("Fsd", "Fsu", "Fld", "Flu")}
        result = DatasetBuildIntermediate(df=df, var_attrs=var_attrs)

        out = add_net_radiation(result)

        self.assertAlmostEqual(out.df["Fn"].iloc[0], 350.0)


def _build_dataset(with_attrs=True):
    dates = [datetime(2026, 9, 17, 0, 0) + timedelta(minutes=30 * i) for i in range(48)]
    ds = xr.Dataset(
        {"Ta_2m": ("time", np.full(len(dates), 20.0))},
        coords={"time": dates},
    )
    if with_attrs:
        ds.attrs["latitude"] = LAT
        ds.attrs["longitude"] = LON
        ds.attrs["elevation"] = ELEV
    return ds


class AddDayNightIndicatorTestCase(unittest.TestCase):
    def test_adds_correct_day_night_values(self):
        ds = _build_dataset()
        out = add_day_night_indicator(ds)

        self.assertIn("day_night", out)
        self.assertEqual(out["day_night"].dims, ("time",))

        values = out["day_night"].to_pandas()
        self.assertEqual(values.loc[datetime(2026, 9, 17, 6, 0)], 0)
        self.assertEqual(values.loc[datetime(2026, 9, 17, 6, 30)], 1)
        self.assertEqual(values.loc[datetime(2026, 9, 17, 18, 0)], 1)
        self.assertEqual(values.loc[datetime(2026, 9, 17, 18, 30)], 0)

    def test_attrs_are_cf_style_flag_metadata(self):
        ds = _build_dataset()
        out = add_day_night_indicator(ds)

        attrs = out["day_night"].attrs
        self.assertEqual(attrs["long_name"], "Day/night indicator")
        self.assertEqual(attrs["units"], "1")
        self.assertEqual(attrs["flag_values"], [0, 1])
        self.assertEqual(attrs["flag_meanings"], "night day")

    def test_missing_attrs_skips_gracefully(self):
        ds = _build_dataset(with_attrs=False)
        with self.assertLogs(
            "fluxcore.orchestration.derived_quantities", level="WARNING"
        ) as ctx:
            out = add_day_night_indicator(ds)

        self.assertNotIn("day_night", out)
        self.assertIn("Missing latitude/longitude/elevation", ctx.output[0])

    def test_empty_time_axis_adds_empty_indicator(self):
        # An incremental reload with no new records must not raise: the
        # exception would trigger incremental_zarr's full-rebuild fallback.
        ds = _build_dataset().isel(time=slice(0, 0))
        out = add_day_night_indicator(ds)

        self.assertIn("day_night", out)
        self.assertEqual(out.sizes["time"], 0)

    def test_partial_attrs_also_skips(self):
        ds = _build_dataset(with_attrs=False)
        ds.attrs["latitude"] = LAT
        ds.attrs["longitude"] = LON
        # elevation deliberately omitted
        out = add_day_night_indicator(ds)
        self.assertNotIn("day_night", out)


if __name__ == "__main__":
    unittest.main()

"""Tests for orchestration/derived_quantities.py's add_day_night_indicator."""

import logging
import unittest
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from orchestration.derived_quantities import add_day_night_indicator

LAT, LON, ELEV = -34.0027, 140.5877, 60


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
            "orchestration.derived_quantities", level="WARNING"
        ) as ctx:
            out = add_day_night_indicator(ds)

        self.assertNotIn("day_night", out)
        self.assertIn("Missing latitude/longitude/elevation", ctx.output[0])

    def test_partial_attrs_also_skips(self):
        ds = _build_dataset(with_attrs=False)
        ds.attrs["latitude"] = LAT
        ds.attrs["longitude"] = LON
        # elevation deliberately omitted
        out = add_day_night_indicator(ds)
        self.assertNotIn("day_night", out)


if __name__ == "__main__":
    unittest.main()

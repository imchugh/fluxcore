"""Integration tests for orchestration/build_L2_nc.py."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from infrastructure import file_io
from orchestration import build_L2_nc


def _build_l2_dataset(n=48 * 2, time_step=30, start="2020-01-01"):
    idx = pd.date_range(start, periods=n, freq=f"{time_step}min")
    lat, lon = [0.0], [0.0]

    ta = np.full((n, 1, 1), 10.0, dtype=float)
    ta_flag = np.zeros((n, 1, 1), dtype=int)
    ta_flag[5, 0, 0] = 2  # range_check code

    fco2 = np.full((n, 1, 1), 1.0, dtype=float)

    ds = xr.Dataset(
        {
            "Ta_Av": (
                ("time", "latitude", "longitude"),
                ta,
                {
                    "long_name": "Air temperature",
                    "units": "degC",
                    "statistic_type": "average",
                    # A real L2 store inherits this already-Zarr-safe shape
                    # from L1's own build (build_L1_zarr._json_safe_instrument_history
                    # converts a compound instrument dict to an ordered list
                    # of pairs before writing, since Zarr's attrs writer
                    # alphabetizes dict keys but preserves list order).
                    "instrument": [["sonic_anemometer", "CSAT3B"], ["irga", "LI-7500RS"]],
                },
            ),
            "Ta_Av_QCFlag": (
                ("time", "latitude", "longitude"),
                ta_flag,
                {
                    "long_name": "Ta_Av QC flag",
                    "units": "1",
                    "flag_values": [0, 2],
                    "flag_meanings": "good range_check",
                },
            ),
            "Fco2": (("time", "latitude", "longitude"), fco2, {"long_name": "CO2 flux"}),
            "crs": 0,
        },
        coords={"time": idx, "latitude": lat, "longitude": lon},
    )
    ds.attrs.update(
        {
            "site_name": "TestSite",
            "time_step": time_step,
            "nc_nrecs": n,
            "time_coverage_start": idx[0].strftime("%Y-%m-%d %H:%M:%S"),
            "time_coverage_end": idx[-1].strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return ds


class BuildL2NcTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.root = Path(self._tmp_dir.name)
        self.zarr_dir = self.root / "L2"
        self.nc_dir = self.root / "nc"

    def _write_l2(self, ds):
        file_io.write_zarr(ds=ds, store_path=self.zarr_dir / "TestSite_L2.zarr")

    def test_writes_one_file_per_year_with_expected_content(self):
        ds = _build_l2_dataset(n=48 * 400, start="2019-06-01")  # spans 2019 and 2020
        self._write_l2(ds)

        written = build_L2_nc.build_from_zarr(
            "TestSite", zarr_dir=self.zarr_dir, output_dir=self.nc_dir
        )

        names = sorted(p.name for p in written)
        self.assertEqual(names, ["TestSite_2019_L2.nc", "TestSite_2020_L2.nc"])
        for path in written:
            self.assertTrue(path.exists())

    def test_qc_flag_values_and_attrs_survive(self):
        ds = _build_l2_dataset()
        flagged_time = ds.time.values[5]
        self._write_l2(ds)

        written = build_L2_nc.build_from_zarr(
            "TestSite", zarr_dir=self.zarr_dir, output_dir=self.nc_dir, year=2020
        )
        out = xr.open_dataset(written[0])

        flag_da = out["Ta_Av_QCFlag"].squeeze(("latitude", "longitude"))
        self.assertEqual(int(flag_da.sel(time=flagged_time).values), 2)
        self.assertEqual(int(flag_da.isel(time=0).values), 0)
        self.assertEqual(list(out["Ta_Av_QCFlag"].attrs["flag_values"]), [0, 2])
        self.assertEqual(out["Ta_Av_QCFlag"].attrs["flag_meanings"], "good range_check")

    def test_compound_instrument_key_order_restored(self):
        # serialize_inst_history joins a compound `instrument` dict's values
        # into a single comma-separated string in dict key order. Zarr's
        # attrs round-trip alphabetizes dict keys (irga before
        # sonic_anemometer), so `instrument` is written as an ordered list
        # of [key, value] pairs (which Zarr's JSON writer does preserve) and
        # reconstructed into a dict in that exact order by
        # _rehydrate_instrument_history -- without that, the joined string
        # would come out as "LI-7500RS,CSAT3B" instead.
        ds = _build_l2_dataset()
        self._write_l2(ds)

        written = build_L2_nc.build_from_zarr(
            "TestSite", zarr_dir=self.zarr_dir, output_dir=self.nc_dir, year=2020
        )
        out = xr.open_dataset(written[0])
        self.assertEqual(out["Ta_Av"].attrs.get("instrument"), "CSAT3B,LI-7500RS")

    def test_single_year_filter(self):
        ds = _build_l2_dataset(n=48 * 400, start="2019-06-01")
        self._write_l2(ds)

        written = build_L2_nc.build_from_zarr(
            "TestSite", zarr_dir=self.zarr_dir, output_dir=self.nc_dir, year=2019
        )
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].name, "TestSite_2019_L2.nc")


if __name__ == "__main__":
    unittest.main()

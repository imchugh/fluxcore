"""Integration tests for orchestration/build_L2_zarr.py."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from infrastructure import file_io
from orchestration import build_L2_zarr


def _build_l1_dataset(n, time_step=30, start="2020-01-01", value=10.0):
    idx = pd.date_range(start, periods=n, freq=f"{time_step}min")
    lat, lon = [0.0], [0.0]

    ds = xr.Dataset(
        {
            "Ta_Av": (
                ("time", "latitude", "longitude"),
                np.full((n, 1, 1), value, dtype=float),
            ),
            "Ta_Av_QCFlag": (
                ("time", "latitude", "longitude"),
                np.zeros((n, 1, 1), dtype=int),
            ),
            "crs": 0,
        },
        coords={"time": idx, "latitude": lat, "longitude": lon},
    )
    ds.attrs.update(
        {
            "time_step": time_step,
            "site_name": "TestSite",
            "nc_nrecs": n,
            "time_coverage_start": idx[0].strftime("%Y-%m-%d %H:%M:%S"),
            "time_coverage_end": idx[-1].strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return ds


class BuildL2ZarrTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.root = Path(self._tmp_dir.name)
        self.l1_dir = self.root / "L1"
        self.l2_dir = self.root / "L2"
        self.qc_dir = self.root / "qc"
        self.qc_dir.mkdir(parents=True)

        (self.qc_dir / "TestSite.yml").write_text(
            "Ta_Av:\n  range_check: {lower: -10, upper: 50}\n"
        )

        patcher = mock.patch("infrastructure.paths.CONFIG_PATH", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_l1(self, ds):
        file_io.write_zarr(ds=ds, store_path=self.l1_dir / "TestSite_L1.zarr")

    def test_build_flags_out_of_range_record(self):
        ds = _build_l1_dataset(20)
        ds["Ta_Av"][5, 0, 0] = 200.0
        self._write_l1(ds)

        store_path = build_L2_zarr.build(
            "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
        )

        out = xr.open_zarr(store_path)
        flags = out["Ta_Av_QCFlag"].squeeze(("latitude", "longitude")).values
        self.assertEqual(flags[5], 2)  # range_check bit
        ta = out["Ta_Av"].squeeze(("latitude", "longitude")).values
        self.assertTrue(np.isnan(ta[5]))
        self.assertFalse(np.isnan(ta[0]))

    def test_update_appends_only_new_records(self):
        ds = _build_l1_dataset(20)
        self._write_l1(ds)
        build_L2_zarr.build("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)

        more = _build_l1_dataset(
            5,
            start=ds.time.values[-1] + pd.Timedelta(minutes=30),
            value=12.0,
        )
        for key in ("nc_nrecs", "time_coverage_start", "time_coverage_end"):
            more.attrs.pop(key, None)
        file_io.append_zarr(ds=more, store_path=self.l1_dir / "TestSite_L1.zarr")

        store_path = build_L2_zarr.update(
            "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
        )

        out = xr.open_zarr(store_path)
        self.assertEqual(out.sizes["time"], 25)
        self.assertEqual(int(out.attrs["nc_nrecs"]), 25)

    def test_update_with_no_new_records_is_a_noop(self):
        ds = _build_l1_dataset(20)
        self._write_l1(ds)
        store_path = build_L2_zarr.build(
            "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
        )
        before = xr.open_zarr(store_path).sizes["time"]

        build_L2_zarr.update("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)

        after = xr.open_zarr(store_path).sizes["time"]
        self.assertEqual(before, after)

    def test_update_falls_back_to_build_on_error(self):
        ds = _build_l1_dataset(20)
        self._write_l1(ds)
        build_L2_zarr.build("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)

        with mock.patch.object(
            build_L2_zarr, "_last_store_timestamp", side_effect=RuntimeError("boom")
        ):
            store_path = build_L2_zarr.update(
                "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
            )

        out = xr.open_zarr(store_path)
        self.assertEqual(out.sizes["time"], 20)

    def test_mad_filter_configured_update_completes(self):
        (self.qc_dir / "TestSite.yml").write_text(
            "Ta_Av:\n"
            "  mad_filter:\n"
            "    reference_var: Ta_Av\n"
            "    window_days: 1\n"
        )
        n = 48 * 5  # 5 days at 30-min steps
        idx_values = 10 + 2 * np.sin(np.linspace(0, 10 * np.pi, n))
        ds = _build_l1_dataset(n)
        ds["Ta_Av"][:] = idx_values.reshape(n, 1, 1)
        self._write_l1(ds)
        build_L2_zarr.build("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)

        more = _build_l1_dataset(
            3, start=ds.time.values[-1] + pd.Timedelta(minutes=30), value=10.0
        )
        for key in ("nc_nrecs", "time_coverage_start", "time_coverage_end"):
            more.attrs.pop(key, None)
        file_io.append_zarr(ds=more, store_path=self.l1_dir / "TestSite_L1.zarr")

        store_path = build_L2_zarr.update(
            "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
        )
        out = xr.open_zarr(store_path)
        self.assertEqual(out.sizes["time"], n + 3)


if __name__ == "__main__":
    unittest.main()

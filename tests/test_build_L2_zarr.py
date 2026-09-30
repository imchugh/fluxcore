"""Integration tests for orchestration/build_L2_zarr.py."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from infrastructure import file_io, paths
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

        # Per-site QC configs resolve via the site_config_files_L2 stream;
        # redirect just that one to the temp dir, leave other streams real.
        real_stream_path = paths.get_local_stream_path

        def _stream_path(resource, stream, site=None):
            if stream == "site_config_files_L2":
                return self.qc_dir
            return real_stream_path(resource, stream, site)

        stream_patcher = mock.patch(
            "infrastructure.paths.get_local_stream_path", side_effect=_stream_path
        )
        stream_patcher.start()
        self.addCleanup(stream_patcher.stop)

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
        self.assertEqual(flags[5], 2)  # range_check code
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

    def test_update_detects_config_change_and_forces_rebuild(self):
        ds = _build_l1_dataset(10, value=10.0)
        self._write_l1(ds)
        store_path = build_L2_zarr.build(
            "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
        )
        out = xr.open_zarr(store_path)
        self.assertTrue(bool((out["Ta_Av_QCFlag"] == 0).all()))

        # Tighten the range bound with no new L1 data arriving -- a plain
        # checkpoint-based update would see nothing new and leave the now-stale
        # flags in place. The config-hash check must catch this instead and
        # force a full rebuild that re-flags the existing records.
        (self.qc_dir / "TestSite.yml").write_text(
            "Ta_Av:\n  range_check: {lower: -10, upper: 5}\n"
        )

        build_L2_zarr.update("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)

        out = xr.open_zarr(store_path)
        self.assertTrue(bool((out["Ta_Av_QCFlag"] == 2).all()))

    def test_update_falls_back_to_build_on_error(self):
        ds = _build_l1_dataset(20)
        self._write_l1(ds)
        build_L2_zarr.build("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)

        with mock.patch(
            "orchestration.incremental_zarr.last_store_timestamp",
            side_effect=RuntimeError("boom"),
        ):
            store_path = build_L2_zarr.update(
                "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
            )

        out = xr.open_zarr(store_path)
        self.assertEqual(out.sizes["time"], 20)

    def test_dependency_default_applies_from_shared_file(self):
        # configs/qc/_dependency_defaults.yml resolves via the same
        # paths.CONFIG_PATH patch as _range_defaults.yml -- both default to
        # {CONFIG_PATH}/qc, which this fixture already points at self.qc_dir
        # (alongside the site's own TestSite.yml, a different filename in
        # the same directory).
        (self.qc_dir / "_dependency_defaults.yml").write_text(
            "Diag_SONIC:\n"
            "  range_check: {lower: 0, upper: 500}\n"
            "Fco2:\n"
            "  dependency_check: [Diag_SONIC]\n"
        )
        n = 10
        idx = pd.date_range("2020-01-01", periods=n, freq="30min")
        diag = np.zeros((n, 1, 1))
        diag[5, 0, 0] = 5000.0  # exceeds the default [0, 500] bound
        fco2 = np.ones((n, 1, 1))
        ds = xr.Dataset(
            {
                "Ta_Av": (("time", "latitude", "longitude"), np.full((n, 1, 1), 10.0)),
                "Diag_SONIC": (("time", "latitude", "longitude"), diag),
                "Fco2": (("time", "latitude", "longitude"), fco2),
                "crs": 0,
            },
            coords={"time": idx, "latitude": [0.0], "longitude": [0.0]},
        )
        ds.attrs.update(
            {
                "time_step": 30,
                "site_name": "TestSite",
                "nc_nrecs": n,
                "time_coverage_start": idx[0].strftime("%Y-%m-%d %H:%M:%S"),
                "time_coverage_end": idx[-1].strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self._write_l1(ds)

        store_path = build_L2_zarr.build(
            "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
        )
        out = xr.open_zarr(store_path)
        fco2_flags = out["Fco2_QCFlag"].squeeze(("latitude", "longitude")).values
        self.assertEqual(fco2_flags[5], 23)  # dependency_check code
        self.assertEqual(fco2_flags[0], 0)

    def test_flag_check_build_then_update_keeps_consistent_variable_set(self):
        (self.qc_dir / "TestSite.yml").write_text(
            "Fco2:\n  flag_check: {source: [Fco2_QC], reject: [8, 9]}\n"
        )

        def _make(n, start, flag):
            idx = pd.date_range(start, periods=n, freq="30min")
            dims = ("time", "latitude", "longitude")
            ds = xr.Dataset(
                {
                    "Fco2": (dims, np.ones((n, 1, 1))),
                    "Fco2_QC": (dims, np.full((n, 1, 1), float(flag))),
                    "Fco2_QC_QCFlag": (dims, np.zeros((n, 1, 1), dtype=int)),
                    "crs": 0,
                },
                coords={"time": idx, "latitude": [0.0], "longitude": [0.0]},
            )
            ds.attrs.update({"time_step": 30, "site_name": "TestSite"})
            return ds

        first = _make(10, "2020-01-01", flag=9)
        first.attrs.update(
            {
                "nc_nrecs": 10,
                "time_coverage_start": "2020-01-01 00:00:00",
                "time_coverage_end": "2020-01-01 04:30:00",
            }
        )
        self._write_l1(first)
        store_path = build_L2_zarr.build(
            "TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir
        )
        built = xr.open_zarr(store_path)
        self.assertNotIn("Fco2_QC_QCFlag", built)
        self.assertTrue(bool((built["Fco2_QCFlag"] == 9).all()))

        file_io.append_zarr(
            ds=_make(5, first.time.values[-1] + pd.Timedelta(minutes=30), flag=1),
            store_path=self.l1_dir / "TestSite_L1.zarr",
        )
        build_L2_zarr.update("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)
        out = xr.open_zarr(store_path)
        self.assertEqual(out.sizes["time"], 15)
        self.assertNotIn("Fco2_QC_QCFlag", out)
        self.assertEqual(int(out["Fco2_QCFlag"].squeeze().values[-1]), 0)

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
        # The newest L1 record is held back: the MAD filter can't test a
        # record with no successor, and an appended record is never re-tested.
        self.assertEqual(out.sizes["time"], n + 2)
        self.assertEqual(int(out["Ta_Av_QCFlag"].squeeze().values[-1]), 0)

        # A no-op update must not append the held-back record either.
        build_L2_zarr.update("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)
        self.assertEqual(xr.open_zarr(store_path).sizes["time"], n + 2)

        # Once its successor arrives, the held-back record is appended.
        successor = _build_l1_dataset(
            1, start=more.time.values[-1] + pd.Timedelta(minutes=30), value=10.0
        )
        for key in ("nc_nrecs", "time_coverage_start", "time_coverage_end"):
            successor.attrs.pop(key, None)
        file_io.append_zarr(ds=successor, store_path=self.l1_dir / "TestSite_L1.zarr")
        build_L2_zarr.update("TestSite", output_dir=self.l2_dir, l1_dir=self.l1_dir)
        out = xr.open_zarr(store_path)
        self.assertEqual(out.sizes["time"], n + 3)
        self.assertEqual(int(out["Ta_Av_QCFlag"].squeeze().values[-1]), 0)


if __name__ == "__main__":
    unittest.main()

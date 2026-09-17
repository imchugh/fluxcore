"""Tests for infrastructure/file_io.py's append_zarr."""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from infrastructure import file_io


def _write_initial_store(store_path: pathlib.Path, n=5) -> None:
    idx = pd.date_range("2026-01-01", periods=n, freq="30min")
    ds = xr.Dataset({"Ta": ("time", np.arange(float(n)))}, coords={"time": idx})
    ds.to_zarr(store_path, mode="w")


class AppendZarrTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.store_path = pathlib.Path(self._tmp_dir.name) / "test.zarr"

    def test_matching_schema_appends_normally(self):
        _write_initial_store(self.store_path, n=5)
        idx = pd.date_range("2026-01-01 02:30", periods=3, freq="30min")
        tail = xr.Dataset({"Ta": ("time", np.array([5.0, 6.0, 7.0]))}, coords={"time": idx})

        file_io.append_zarr(ds=tail, store_path=self.store_path)

        result = xr.open_zarr(self.store_path)
        self.assertEqual(result.sizes["time"], 8)

    def test_added_variable_raises_before_writing_anything(self):
        _write_initial_store(self.store_path, n=5)
        idx = pd.date_range("2026-01-01 02:30", periods=3, freq="30min")
        tail = xr.Dataset(
            {
                "Ta": ("time", np.array([5.0, 6.0, 7.0])),
                "day_night": ("time", np.array([1, 1, 0])),
            },
            coords={"time": idx},
        )

        with self.assertRaises(ValueError) as ctx:
            file_io.append_zarr(ds=tail, store_path=self.store_path)
        self.assertIn("day_night", str(ctx.exception))

        # Store must be untouched -- still openable, still the original length.
        result = xr.open_zarr(self.store_path)
        self.assertEqual(result.sizes["time"], 5)
        self.assertNotIn("day_night", result.data_vars)

    def test_removed_variable_raises_before_writing_anything(self):
        idx0 = pd.date_range("2026-01-01", periods=5, freq="30min")
        ds0 = xr.Dataset(
            {
                "Ta": ("time", np.arange(5.0)),
                "RH": ("time", np.arange(5.0)),
            },
            coords={"time": idx0},
        )
        ds0.to_zarr(self.store_path, mode="w")

        idx = pd.date_range("2026-01-01 02:30", periods=3, freq="30min")
        tail = xr.Dataset({"Ta": ("time", np.array([5.0, 6.0, 7.0]))}, coords={"time": idx})

        with self.assertRaises(ValueError) as ctx:
            file_io.append_zarr(ds=tail, store_path=self.store_path)
        self.assertIn("RH", str(ctx.exception))

        result = xr.open_zarr(self.store_path)
        self.assertEqual(result.sizes["time"], 5)


if __name__ == "__main__":
    unittest.main()

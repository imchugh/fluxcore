"""Tests for infrastructure/file_io.py's append_zarr."""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from fluxcore.infrastructure import file_io


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
        tail = xr.Dataset(
            {"Ta": ("time", np.array([5.0, 6.0, 7.0]))}, coords={"time": idx}
        )

        file_io.append_zarr(ds=tail, store_path=self.store_path)

        result = xr.open_zarr(self.store_path)
        self.assertEqual(result.sizes["time"], 8)

    def test_attrs_not_in_tail_are_preserved_after_append(self):
        # to_zarr(mode="a") replaces the store's root attrs wholesale with
        # the tail's, and also writes a consolidated .zmetadata cache that
        # xr.open_zarr reads from by default -- a regression here would
        # show up only when reading back through that default path, not by
        # inspecting the plain .zattrs file directly.
        idx0 = pd.date_range("2026-01-01", periods=5, freq="30min")
        ds0 = xr.Dataset({"Ta": ("time", np.arange(5.0))}, coords={"time": idx0})
        ds0.attrs["config_hash"] = "abc123"
        ds0.attrs["site_name"] = "TestSite"
        ds0.to_zarr(self.store_path, mode="w")

        idx = pd.date_range("2026-01-01 02:30", periods=3, freq="30min")
        tail = xr.Dataset(
            {"Ta": ("time", np.array([5.0, 6.0, 7.0]))}, coords={"time": idx}
        )
        tail.attrs["site_name"] = "TestSite"  # refreshed, but config_hash is not

        file_io.append_zarr(ds=tail, store_path=self.store_path)

        result = xr.open_zarr(self.store_path)
        self.assertEqual(result.attrs.get("config_hash"), "abc123")
        self.assertEqual(result.attrs.get("site_name"), "TestSite")

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
        tail = xr.Dataset(
            {"Ta": ("time", np.array([5.0, 6.0, 7.0]))}, coords={"time": idx}
        )

        with self.assertRaises(ValueError) as ctx:
            file_io.append_zarr(ds=tail, store_path=self.store_path)
        self.assertIn("RH", str(ctx.exception))

        result = xr.open_zarr(self.store_path)
        self.assertEqual(result.sizes["time"], 5)


if __name__ == "__main__":
    unittest.main()


class GetNumberedFilesTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.dir = pathlib.Path(self._tmp_dir.name)
        self.master = self.dir / "Site_soil.dat"

    def _touch(self, *names, directory=None):
        for name in names:
            ((directory or self.dir) / name).touch()

    def test_sorted_numerically_oldest_first(self):
        self._touch(
            "Site_soil.dat.10.backup",
            "Site_soil.dat.2.backup",
            "Site_soil.dat.backup",
            "Site_soil.dat.1.backup",
        )
        self.assertEqual(
            file_io.get_backup_files(self.master, abs_path=False),
            [
                "Site_soil.dat.backup",
                "Site_soil.dat.1.backup",
                "Site_soil.dat.2.backup",
                "Site_soil.dat.10.backup",
            ],
        )

    def test_matches_master_name_exactly(self):
        # A file whose name merely starts with the master's stem is another
        # file's backup, not this one's.
        self._touch(
            "Site_soil.dat.backup",
            "Site_soil_1min.dat.backup",
            "Site_soil.dat.backup_duped",
            "Site_soil.dat.x.backup",
            "Site_soil.dat.patch",
        )
        self.assertEqual(
            file_io.get_backup_files(self.master, abs_path=False),
            ["Site_soil.dat.backup"],
        )

    def test_patches_from_another_directory(self):
        patch_dir = self.dir / "Slow_patch"
        patch_dir.mkdir()
        self._touch("Site_soil.dat.1.patch", "Site_soil.dat.patch", directory=patch_dir)
        self._touch("Site_soil.dat.backup")
        found = file_io.get_numbered_files(
            self.master, suffix="patch", directory=patch_dir
        )
        self.assertEqual(
            [p.name for p in found], ["Site_soil.dat.patch", "Site_soil.dat.1.patch"]
        )
        self.assertTrue(all(p.parent == patch_dir for p in found))

    def test_missing_directory_is_empty(self):
        self.assertEqual(
            file_io.get_numbered_files(
                self.master, suffix="patch", directory=self.dir / "absent"
            ),
            [],
        )

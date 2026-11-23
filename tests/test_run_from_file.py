"""Tests for orchestration/run_from_file.py (standalone file-in entry points).

The builders themselves are tested elsewhere; these tests check that a
control file unpacks into the right contract objects and directories, and
that arguments win over the file. One run_L2 test runs a real L1 -> L2 store
transform end to end.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import xarray as xr
from pydantic import ValidationError
from test_build_L2_zarr import _build_l1_dataset

from fluxcore import SiteMetadata, run_L1, run_L2
from fluxcore.infrastructure import file_io
from fluxcore.orchestration import run_from_file
from fluxcore.services.metadata import qc_config_schema

L1_YAML = """\
site: TestSite
file_formats:
  flux: CSI
flux_system: TERNFLUX
flux_file: flux
variables:
  Ta_Av:
    height: 2.0
    statistic_type: average
    input_variables:
      - name: Ta_1_1_1
        instrument: Some Sensor
        file: flux
        units: degC
"""

STANDALONE_SECTIONS = """\
paths:
  input_data_path: raw
  input_patch_path: patches
  output_dir: out
site_metadata:
  latitude: -22.3
  longitude: 133.2
  time_step: 30
  freq_hz: 10
global_attrs:
  institution: Somewhere
  project: Test
"""


class RunL1TestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.root = Path(self._tmp_dir.name)
        self.control = self.root / "TestSite.yml"

        patcher = mock.patch.object(run_from_file, "build_L1_zarr")
        self.builder = patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, yaml_text, **kwargs):
        self.control.write_text(yaml_text)
        run_L1(self.control, **kwargs)
        call = self.builder.update if kwargs.get("update") else self.builder.build
        ((ctx,), call_kwargs) = call.call_args
        return ctx, call_kwargs

    def test_control_file_sections_build_the_context(self):
        ctx, kwargs = self._run(L1_YAML + STANDALONE_SECTIONS)
        rc = ctx.runtime_config
        self.assertEqual(rc.input_data_path, self.root / "raw")
        self.assertEqual(rc.input_patch_path, self.root / "patches")
        self.assertEqual(kwargs["output_dir"], self.root / "out")
        self.assertEqual(ctx.metadata.site_name, "TestSite")  # defaulted from site
        self.assertEqual(ctx.metadata.time_step, 30)
        self.assertEqual(
            kwargs["global_attrs"], {"institution": "Somewhere", "project": "Test"}
        )
        self.builder.update.assert_not_called()

    def test_arguments_win_over_the_file(self):
        meta = SiteMetadata({"site_name": "TestSite", "time_step": 60})
        ctx, kwargs = self._run(
            L1_YAML + STANDALONE_SECTIONS,
            output_dir="/elsewhere/out",
            input_data_path="/elsewhere/raw",
            site_metadata=meta,
            global_attrs={"project": "Override"},
        )
        self.assertEqual(ctx.runtime_config.input_data_path, Path("/elsewhere/raw"))
        self.assertEqual(kwargs["output_dir"], Path("/elsewhere/out"))
        self.assertIs(ctx.metadata, meta)
        # Merged key by key: the argument's project wins, the file's
        # institution stays.
        self.assertEqual(
            kwargs["global_attrs"], {"institution": "Somewhere", "project": "Override"}
        )

    def test_update_calls_the_incremental_path(self):
        self._run(L1_YAML + STANDALONE_SECTIONS, update=True)
        self.builder.update.assert_called_once()
        self.builder.build.assert_not_called()

    def test_patch_dir_is_optional(self):
        yaml_text = (L1_YAML + STANDALONE_SECTIONS).replace(
            "  input_patch_path: patches\n", ""
        )
        ctx, _ = self._run(yaml_text)
        self.assertIsNone(ctx.runtime_config.input_patch_path)

    def test_plain_site_config_needs_arguments(self):
        # A network adapter's site config has no standalone sections.
        with self.assertRaisesRegex(ValueError, "input_data_path"):
            self._run(L1_YAML)
        with self.assertRaisesRegex(ValueError, "site metadata"):
            self._run(L1_YAML, input_data_path="raw", output_dir="out")

    def test_metadata_site_name_mismatch_rejected(self):
        yaml_text = (L1_YAML + STANDALONE_SECTIONS).replace(
            "site_metadata:\n", "site_metadata:\n  site_name: Other\n"
        )
        with self.assertRaises(ValidationError):
            self._run(yaml_text)

    def test_unknown_path_key_rejected(self):
        yaml_text = (L1_YAML + STANDALONE_SECTIONS).replace(
            "output_dir: out", "outdir: out"
        )
        with self.assertRaises(ValidationError):
            self._run(yaml_text)


class RunL2TestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.root = Path(self._tmp_dir.name)
        self.control = self.root / "TestSite_L2.yml"

        # Shared QC defaults: an empty directory, so only the site's own
        # range check applies.
        defaults = self.root / "defaults"
        defaults.mkdir()
        patcher = mock.patch.object(qc_config_schema, "QC_DEFAULTS_DIR", defaults)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_runs_l1_store_to_l2_store(self):
        self.control.write_text(
            "site: TestSite\n"
            "paths: {l1_dir: stores, output_dir: stores}\n"
            "variables:\n"
            "  Ta_Av:\n"
            "    range_check: {lower: -10, upper: 5}\n"
        )
        file_io.write_zarr(
            ds=_build_l1_dataset(10), store_path=self.root / "stores/TestSite_L1.zarr"
        )

        store_path = run_L2(self.control)

        self.assertEqual(store_path, self.root / "stores/TestSite_L2.zarr")
        out = xr.open_zarr(store_path)
        self.assertEqual(out.attrs["processing_level"], "L2")
        self.assertTrue(bool((out["Ta_Av_QCFlag"] == 2).all()))  # out of range

    def test_arguments_win_and_update_is_incremental(self):
        self.control.write_text("site: TestSite\npaths: {l1_dir: a, output_dir: b}\n")
        with mock.patch.object(run_from_file, "build_L2_zarr") as builder:
            run_L2(self.control, l1_dir="/x/L1", update=True)
        ((qc_config,), kwargs) = builder.update.call_args
        self.assertEqual(qc_config.site_name, "TestSite")
        self.assertEqual(qc_config.variables, {})
        self.assertEqual(kwargs["l1_dir"], Path("/x/L1"))
        self.assertEqual(kwargs["output_dir"], self.root / "b")
        builder.build.assert_not_called()

    def test_missing_l1_dir_raises(self):
        self.control.write_text("site: TestSite\n")
        with self.assertRaisesRegex(ValueError, "l1_dir"):
            run_L2(self.control, output_dir="out")


if __name__ == "__main__":
    unittest.main()

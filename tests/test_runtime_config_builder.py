"""Tests for the generic-core config assembly path (no TERN dependency)."""

import tempfile
import unittest
from pathlib import Path

from services.metadata.runtime_config_builder import build_runtime_config_from_file

VALID_YAML = """\
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
      Ta_1_1_1:
        instrument: Some Unregistered Sensor
        file: flux
        units: degC
"""


class RuntimeConfigBuilderTestCase(unittest.TestCase):
    def test_build_from_file_has_no_tern_dependency(self):
        """A structurally-valid config assembles with no instrument_registry
        involvement at all — an arbitrary instrument name is accepted, and
        instrument_uri resolves to None since no enrichment map is supplied."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "TestSite.yml"
            config_path.write_text(VALID_YAML)

            cfg = build_runtime_config_from_file(config_path)

        self.assertEqual(cfg.site_name, "TestSite")
        var_def = cfg.variables["Ta_Av"]
        self.assertEqual(var_def.instrument, "Some Unregistered Sensor")
        self.assertIsNone(var_def.instrument_uri)


if __name__ == "__main__":
    unittest.main(verbosity=2)

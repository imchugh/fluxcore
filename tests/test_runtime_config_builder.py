"""Tests for the generic-core config assembly path (no TERN dependency)."""

import tempfile
import unittest
from pathlib import Path

from fluxcore.services.metadata.core.runtime_config_builder import (
    build_runtime_config_from_file,
)

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

    def _build(self, yaml_text):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "TestSite.yml"
            config_path.write_text(yaml_text)
            return build_runtime_config_from_file(config_path)

    def test_repeated_instrument_era_is_rejected(self):
        """A -> B -> A would overwrite A's first era in instrument history."""
        with self.assertRaisesRegex(ValueError, "returns to an earlier instrument"):
            self._build(CHANGEOVER_YAML.format(third="Sensor A"))

    def test_distinct_instrument_eras_are_accepted(self):
        cfg = self._build(CHANGEOVER_YAML.format(third="Sensor C"))
        self.assertEqual(len(cfg.variables["Ta_Av"].raw_inputs), 3)


CHANGEOVER_YAML = """\
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
      Ta_a:
        instrument: Sensor A
        file: flux
        units: degC
        end: 2020-01-01 00:00
      Ta_b:
        instrument: Sensor B
        file: flux
        units: degC
        begin: 2020-01-01 00:30
        end: 2021-01-01 00:00
      Ta_c:
        instrument: {third}
        file: flux
        units: degC
        begin: 2021-01-01 00:30
"""


if __name__ == "__main__":
    unittest.main(verbosity=2)

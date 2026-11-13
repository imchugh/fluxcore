"""Tests for the generic-core config assembly path (no TERN dependency)."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fluxcore.domain.enums import DiagnosticType, StatisticType
from fluxcore.orchestration.dataframe_builder import _apply_conversions
from fluxcore.services.metadata.file_group_builder import build_file_groups
from fluxcore.services.metadata.runtime_config_builder import (
    build_runtime_config_from_file,
)
from fluxcore.services.metadata.variable_registry import build_variable_registry

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

    def test_input_file_missing_from_file_formats_is_rejected(self):
        yaml_text = VALID_YAML.replace("file: flux", "file: met").replace(
            "flux_file: flux", "flux_file: met"
        )
        with self.assertRaisesRegex(ValueError, r"not declared in file_formats.*met"):
            self._build(yaml_text)

    def test_stitched_counter_keeps_each_inputs_diag_type(self):
        """A counter stitched across a logger change keeps each input's own
        diag_type, so the valid_count era is converted and the invalid_count
        era passes through (it used to inherit the first input's type)."""
        cfg = self._build_with_data_path(STITCHED_COUNTER_YAML)
        raw_inputs = cfg.variables["Diag_SONIC"].raw_inputs
        self.assertEqual(
            [r.diag_type for r in raw_inputs],
            [DiagnosticType.INVALID_COUNT, DiagnosticType.VALID_COUNT],
        )

        specs = build_variable_registry(
            runtime_cfg=cfg, file_groups=build_file_groups(cfg)
        )
        by_raw_name = {s.raw_name: s.diag_type for s in specs.values()}
        self.assertEqual(by_raw_name["Warn_old"], DiagnosticType.INVALID_COUNT)
        self.assertEqual(by_raw_name["samples_new"], DiagnosticType.VALID_COUNT)

    def _build_with_data_path(self, yaml_text):
        """As _build, with an input_data_path so file groups can be built."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "TestSite.yml"
            config_path.write_text(yaml_text)
            # File groups only need a base path; no data is read.
            return build_runtime_config_from_file(
                config_path, input_data_path=Path("/some/arbitrary/data/dir")
            )

    def test_stitched_variance_and_stdev_inputs_both_output_stdev(self):
        """A variance era and a standard-deviation era of one variable are
        each converted from their own raw statistic: the variance is
        square-rooted, the standard deviation passes through."""
        cfg = self._build_with_data_path(STITCHED_STATISTIC_YAML)
        self.assertEqual(
            [r.statistic_type for r in cfg.variables["Ux_SONIC_Sd"].raw_inputs],
            [StatisticType.VAR, StatisticType.STDEV],
        )

        specs = build_variable_registry(
            runtime_cfg=cfg, file_groups=build_file_groups(cfg)
        )
        by_raw_name = {s.raw_name: s for s in specs.values()}
        self.assertEqual(by_raw_name["covUxUx"].raw_statistic_type, StatisticType.VAR)
        self.assertEqual(
            by_raw_name["Ux_stdev"].raw_statistic_type, StatisticType.STDEV
        )

        df = pd.DataFrame(
            {"covUxUx_gp0": [4.0], "Ux_stdev_gp1": [2.0]},
            index=pd.DatetimeIndex(["2020-01-01"], name="time"),
        )
        out = _apply_conversions(df=df, specs_by_alias=specs)
        self.assertEqual(out["covUxUx_gp0"].iloc[0], 2.0)
        self.assertEqual(out["Ux_stdev_gp1"].iloc[0], 2.0)

    def test_input_statistic_defaults_to_the_variables(self):
        cfg = self._build(VALID_YAML)
        self.assertEqual(
            cfg.variables["Ta_Av"].raw_inputs[0].statistic_type, StatisticType.AVG
        )

    def test_input_units_are_checked_against_the_inputs_statistic(self):
        """A variance input needs squared units, even on a stdev variable."""
        yaml_text = STITCHED_STATISTIC_YAML.replace("units: m^2/s^2", "units: m/s")
        with self.assertRaisesRegex(ValueError, r"covUxUx.*units 'm/s'"):
            self._build(yaml_text)

    def test_input_statistic_other_than_variance_stdev_swap_is_rejected(self):
        yaml_text = VALID_YAML.replace(
            "        units: degC\n",
            "        units: degC\n        statistic_type: sum\n",
        )
        with self.assertRaisesRegex(ValueError, "only variance and standard_deviation"):
            self._build(yaml_text)

    def test_input_statistic_on_a_counter_is_rejected(self):
        yaml_text = STITCHED_COUNTER_YAML.replace(
            "        diag_type: valid_count\n",
            "        diag_type: valid_count\n        statistic_type: variance\n",
        )
        with self.assertRaisesRegex(
            ValueError, "cannot define an input statistic_type"
        ):
            self._build(yaml_text)

    def test_unused_file_formats_entry_is_rejected(self):
        yaml_text = VALID_YAML.replace("  flux: CSI\n", "  flux: CSI\n  spare: CSI\n")
        with self.assertRaisesRegex(ValueError, r"no input variable uses.*spare"):
            self._build(yaml_text)


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


STITCHED_COUNTER_YAML = """\
site: TestSite
file_formats:
  old: CSI
  new: CSI
flux_system: TERNFLUX
flux_file: new
variables:
  Diag_SONIC:
    variable_type: counter
    height: 15.0
    input_variables:
      Warn_old:
        instrument: Sensor A
        units: dimensionless
        diag_type: invalid_count
        file: old
        end: 2020-01-01 00:00
      samples_new:
        instrument: Sensor A
        units: dimensionless
        diag_type: valid_count
        file: new
        begin: 2020-01-01 00:30
"""


STITCHED_STATISTIC_YAML = """\
site: TestSite
file_formats:
  old: CSI
  new: CSI
flux_system: TERNFLUX
flux_file: new
variables:
  Ux_SONIC_Sd:
    statistic_type: standard_deviation
    height: 15.0
    input_variables:
      covUxUx:
        instrument: Sensor A
        units: m^2/s^2
        statistic_type: variance
        file: old
        end: 2020-01-01 00:00
      Ux_stdev:
        instrument: Sensor A
        units: m/s
        file: new
        begin: 2020-01-01 00:30
"""


if __name__ == "__main__":
    unittest.main(verbosity=2)

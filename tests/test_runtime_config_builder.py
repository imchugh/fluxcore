"""Tests for the generic-core config assembly path (no TERN dependency)."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from fluxcore.domain.enums import DiagnosticType, StatisticType
from fluxcore.orchestration import dataframe_builder
from fluxcore.orchestration.dataframe_builder import _apply_conversions
from fluxcore.services.metadata.file_group_builder import build_file_groups
from fluxcore.services.metadata.instrument_history import (
    InstrumentHistory,
    InstrumentPeriod,
)
from fluxcore.services.metadata.runtime_config_builder import (
    build_runtime_config_from_file,
)
from fluxcore.services.metadata.variable_registry import (
    build_variable_registry,
    group_by_canonical_name,
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
      - name: Ta_1_1_1
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
            {"Ux_SONIC_Sd#0": [4.0], "Ux_SONIC_Sd#1": [2.0]},
            index=pd.DatetimeIndex(["2020-01-01"], name="time"),
        )
        out = _apply_conversions(df=df, specs_by_alias=specs)
        self.assertEqual(out["Ux_SONIC_Sd#0"].iloc[0], 2.0)
        self.assertEqual(out["Ux_SONIC_Sd#1"].iloc[0], 2.0)

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

    def test_same_column_from_two_files_is_stitched(self):
        """A logger change that keeps a column name: two inputs share a name
        but read different files, and each gets its own registry alias."""
        cfg = self._build_with_data_path(SAME_COLUMN_YAML)
        raw_inputs = cfg.variables["WTD"].raw_inputs
        self.assertEqual([r.raw_name for r in raw_inputs], ["Level", "Level"])
        self.assertEqual([r.file for r in raw_inputs], ["old", "new"])

        specs = build_variable_registry(
            runtime_cfg=cfg, file_groups=build_file_groups(cfg)
        )
        self.assertEqual(
            sorted((s.raw_name, s.file_group) for s in specs.values()),
            [("Level", "new"), ("Level", "old")],
        )

    def test_same_column_from_one_file_is_stitched_across_a_units_change(self):
        """One column, one file, two periods with different units: each period
        is converted from its own units, then stitched at the dates."""
        cfg = self._build_with_data_path(UNITS_CHANGE_YAML)
        file_groups = build_file_groups(cfg)
        specs = build_variable_registry(runtime_cfg=cfg, file_groups=file_groups)
        self.assertEqual(list(specs), ["Sws_Av#0", "Sws_Av#1"])
        self.assertEqual([s.site_units for s in specs.values()], ["percent", "m^3/m^3"])

        idx = pd.date_range("2020-01-01 23:00", periods=4, freq="30min", name="time")
        raw = pd.DataFrame({"VWC": [40.0, 40.0, 0.3, 0.3]}, index=idx)
        df = dataframe_builder._build_file_group_dataframe(
            file_group=_stub_file_group("soil", "VWC"),
            loader=lambda _: raw,
            specs_by_alias=specs,
            time_step=30,
        )
        out = dataframe_builder._merge_overlapping_variables(
            df=df, specs_by_alias=specs
        )
        self.assertEqual(list(out.columns), ["Sws_Av"])
        self.assertEqual(out["Sws_Av"].tolist(), [0.4, 0.4, 0.3, 0.3])

    def test_same_column_from_one_file_over_overlapping_periods_is_rejected(self):
        yaml_text = UNITS_CHANGE_YAML.replace(
            "begin: 2020-01-02 00:00", "begin: 2020-01-01 23:30"
        )
        with self.assertRaisesRegex(
            ValueError, "'VWC' from file 'soil' is listed for overlapping periods"
        ):
            self._build(yaml_text)

    def test_one_column_can_feed_two_variables(self):
        """Inputs are keyed by variable, so one raw column can feed two."""
        yaml_text = VALID_YAML + (
            "  Ta_Sd:\n"
            "    height: 2.0\n"
            "    statistic_type: standard_deviation\n"
            "    input_variables:\n"
            "      - name: Ta_1_1_1\n"
            "        instrument: Some Unregistered Sensor\n"
            "        file: flux\n"
            "        units: degC\n"
        )
        cfg = self._build_with_data_path(yaml_text)
        file_groups = build_file_groups(cfg)
        specs = build_variable_registry(runtime_cfg=cfg, file_groups=file_groups)
        idx = pd.date_range("2020-01-01", periods=2, freq="30min", name="time")
        df = dataframe_builder._build_file_group_dataframe(
            file_group=_stub_file_group("flux", "Ta_1_1_1"),
            loader=lambda _: pd.DataFrame({"Ta_1_1_1": [1.0, 2.0]}, index=idx),
            specs_by_alias=specs,
            time_step=30,
        )
        self.assertEqual(sorted(df.columns), ["Ta_Av#0", "Ta_Sd#0"])
        self.assertEqual(df["Ta_Sd#0"].tolist(), [1.0, 2.0])

    def test_consecutive_inputs_from_one_instrument_are_one_era(self):
        """A -> A -> B is one changeover, not a return to an earlier era."""
        yaml_text = CHANGEOVER_YAML.format(third="Sensor B").replace(
            "instrument: Sensor B", "instrument: Sensor A", 1
        )
        cfg = self._build_with_data_path(yaml_text)
        self.assertEqual(
            [r.instrument for r in cfg.variables["Ta_Av"].raw_inputs],
            ["Sensor A", "Sensor A", "Sensor B"],
        )

        # Sensor A's period runs from its first input's begin to its second
        # input's end, not just over the second input.
        specs = build_variable_registry(
            runtime_cfg=cfg, file_groups=build_file_groups(cfg)
        )
        history = InstrumentHistory.from_specs(group_by_canonical_name(specs)["Ta_Av"])
        self.assertEqual(
            history.roles[0].periods,
            (
                InstrumentPeriod("Sensor A", None, datetime(2021, 1, 1, 0, 0)),
                InstrumentPeriod("Sensor B", datetime(2021, 1, 1, 0, 30), None),
            ),
        )

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
      - name: Ta_a
        instrument: Sensor A
        file: flux
        units: degC
        end: 2020-01-01 00:00
      - name: Ta_b
        instrument: Sensor B
        file: flux
        units: degC
        begin: 2020-01-01 00:30
        end: 2021-01-01 00:00
      - name: Ta_c
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
      - name: Warn_old
        instrument: Sensor A
        units: dimensionless
        diag_type: invalid_count
        file: old
        end: 2020-01-01 00:00
      - name: samples_new
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
      - name: covUxUx
        instrument: Sensor A
        units: m^2/s^2
        statistic_type: variance
        file: old
        end: 2020-01-01 00:00
      - name: Ux_stdev
        instrument: Sensor A
        units: m/s
        file: new
        begin: 2020-01-01 00:30
"""


SAME_COLUMN_YAML = """\
site: TestSite
file_formats:
  old: CSI
  new: CSI
flux_system: TERNFLUX
flux_file: new
variables:
  WTD:
    height: -13.0
    statistic_type: average
    input_variables:
      - name: Level
        instrument: Sensor A
        units: m
        file: old
        end: 2020-01-01 00:00
      - name: Level
        instrument: Sensor A
        units: m
        file: new
        begin: 2020-01-01 00:30
"""


def _stub_file_group(group, *raw_names):
    """A one-file FileGroup stand-in that doesn't read headers from disk."""
    return SimpleNamespace(
        group=group,
        variables_by_file={f"{group}.dat": None},
        expected_variables=set(raw_names),
    )


UNITS_CHANGE_YAML = """\
site: TestSite
file_formats:
  soil: CSI
flux_system: TERNFLUX
flux_file: soil
variables:
  Sws_Av:
    height: -0.05
    statistic_type: average
    input_variables:
      - name: VWC
        instrument: Sensor A
        units: percent
        file: soil
        end: 2020-01-01 23:30
      - name: VWC
        instrument: Sensor A
        units: m^3/m^3
        file: soil
        begin: 2020-01-02 00:00
"""

if __name__ == "__main__":
    unittest.main(verbosity=2)

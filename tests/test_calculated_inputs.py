"""Tests for calculated inputs (calculate_from) and their instrument history."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from fluxcore.orchestration import dataframe_builder
from fluxcore.orchestration.build_L1_nc import serialize_inst_history
from fluxcore.orchestration.derived_quantities import (
    DatasetBuildIntermediate,
    derive_missing_humidity,
)
from fluxcore.services.data.calculations import (
    calculate_AH_from_RH,
    calculate_RH_from_AH,
)
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

# An old logger logged only the AH it calculated from its HMP's RH; the new
# one logs RH itself. RH_2m's old era is calculated from AH_2m and Ta_2m.
CALCULATED_YAML = """\
site: TestSite
file_formats:
  old: CSI
  new: CSI
flux_system: TERNFLUX
flux_file: new
variables:
  Ta_2m:
    height: 2.0
    statistic_type: average
    input_variables:
      - name: t_old
        instrument: Vaisala HMP45C
        file: old
        units: degC
        end: 2020-01-01 01:00
      - name: Ta_new
        instrument: Vaisala HMP155
        file: new
        units: degC
        begin: 2020-01-01 01:30
  AH_2m:
    height: 2.0
    statistic_type: average
    input_variables:
      - name: ah_old
        instrument: Vaisala HMP45C
        file: old
        units: g/m^3
        end: 2020-01-01 01:00
      - name: AH_new
        instrument: Vaisala HMP155
        file: new
        units: g/m^3
        begin: 2020-01-01 01:30
  RH_2m:
    height: 2.0
    statistic_type: average
    input_variables:
      - calculate_from: [AH_2m, Ta_2m]
        instrument: Vaisala HMP45C
        end: 2020-01-01 01:00
      - name: RH_new
        instrument: Vaisala HMP155
        file: new
        units: percent
        begin: 2020-01-01 01:30
"""

OLD_INPUT = """\
      - calculate_from: [AH_2m, Ta_2m]
        instrument: Vaisala HMP45C
        end: 2020-01-01 01:00
"""


def _build(yaml_text, data_path=None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "TestSite.yml"
        config_path.write_text(yaml_text)
        return build_runtime_config_from_file(config_path, input_data_path=data_path)


def _stub_file_group(group, *raw_names):
    return SimpleNamespace(
        group=group,
        variables_by_file={f"{group}.dat": None},
        expected_variables=set(raw_names),
    )


class CalculatedInputSchemaTestCase(unittest.TestCase):
    def test_calculated_input_is_accepted(self):
        cfg = _build(CALCULATED_YAML)
        first = cfg.variables["RH_2m"].raw_inputs[0]
        self.assertEqual(first.calculate_from, ("AH_2m", "Ta_2m"))
        self.assertIsNone(first.raw_name)
        self.assertIsNone(first.file)

    def test_calculated_input_with_a_raw_column_is_rejected(self):
        yaml_text = CALCULATED_YAML.replace(
            OLD_INPUT, OLD_INPUT + "        name: rh_old\n"
        )
        with self.assertRaisesRegex(ValueError, "cannot define name"):
            _build(yaml_text)

    def test_measured_input_without_a_column_is_rejected(self):
        yaml_text = CALCULATED_YAML.replace(
            OLD_INPUT,
            OLD_INPUT.replace(
                "      - calculate_from: [AH_2m, Ta_2m]\n", "      - units: percent\n"
            ),
        )
        with self.assertRaisesRegex(ValueError, "must define name, file"):
            _build(yaml_text)

    def test_unknown_source_is_rejected(self):
        yaml_text = CALCULATED_YAML.replace("[AH_2m, Ta_2m]", "[AH_9m, Ta_2m]")
        with self.assertRaisesRegex(ValueError, "'AH_9m' is not a configured"):
            _build(yaml_text)

    def test_wrong_source_quantities_are_rejected(self):
        yaml_text = CALCULATED_YAML.replace("[AH_2m, Ta_2m]", "[Ta_2m]")
        with self.assertRaisesRegex(ValueError, r"needs one variable each of"):
            _build(yaml_text)

    def test_calculated_period_overlapping_a_measured_one_is_rejected(self):
        yaml_text = CALCULATED_YAML.replace(
            OLD_INPUT,
            OLD_INPUT.replace("end: 2020-01-01 01:00", "end: 2020-01-01 02:00"),
        )
        with self.assertRaisesRegex(ValueError, "must not overlap another input"):
            _build(yaml_text)

    def test_calculations_cannot_chain(self):
        # AH_2m calculated from RH_2m, which is itself partly calculated.
        yaml_text = CALCULATED_YAML.replace(
            """      - name: ah_old
        instrument: Vaisala HMP45C
        file: old
        units: g/m^3
        end: 2020-01-01 01:00
""",
            """      - calculate_from: [RH_2m, Ta_2m]
        instrument: Vaisala HMP45C
        end: 2020-01-01 01:00
""",
        )
        with self.assertRaisesRegex(ValueError, "calculations can't chain"):
            _build(yaml_text)


FLUX_YAML = """\
site: TestSite
file_formats:
  old: CSI
  new: CSI
flux_system: TERNFLUX
flux_file: new
variables:
  Fco2:
    height: 23.0
    statistic_type: average
    input_variables:
      - name: Fc_old
        instrument:
          sonic_anemometer: Campbell Scientific CSAT3
          irga: LI-COR LI-7500A
        file: old
        units: mg/m^2/s
        end: {old_end}
      - name: Fco2
        instrument:
          sonic_anemometer: Campbell Scientific CSAT3B
          irga: LI-COR LI-7500RS
        file: new
        units: mg/m^2/s
        begin: 2020-01-01 01:30
  Fe:
    height: 23.0
    statistic_type: average
    input_variables:
      - name: Fe_old
        instrument:
          sonic_anemometer: Campbell Scientific CSAT3
          irga: LI-COR LI-7500A
        file: old
        units: W/m^2
        end: 2020-01-01 01:00
      - name: Fe
        instrument:
          sonic_anemometer: Campbell Scientific CSAT3B
          irga: LI-COR LI-7500RS
        file: new
        units: W/m^2
        begin: 2020-01-01 01:30
"""


class FluxSystemChangeoverTestCase(unittest.TestCase):
    def test_a_flux_system_change_between_eras_is_accepted(self):
        cfg = _build(FLUX_YAML.format(old_end="2020-01-01 01:00"))
        self.assertEqual(len(cfg.variables["Fco2"].raw_inputs), 2)

    def test_different_pairings_in_the_same_period_are_rejected(self):
        # Fco2's old era now runs into Fe's new era.
        with self.assertRaisesRegex(ValueError, "pairings for the same period"):
            _build(FLUX_YAML.format(old_end="2020-01-01 02:00"))


class CalculatedInputDataframeTestCase(unittest.TestCase):
    """The calculated period is filled from the stitched source variables."""

    def setUp(self):
        self.cfg = _build(CALCULATED_YAML, data_path=Path("/arbitrary"))
        self.specs = build_variable_registry(
            runtime_cfg=self.cfg, file_groups=build_file_groups(self.cfg)
        )
        idx = pd.date_range("2020-01-01 00:30", periods=4, freq="30min", name="time")
        self.ah = calculate_AH_from_RH(Ta=np.array([20.0, 10.0]), RH=[60.0, 80.0])
        frames = {
            "old": pd.DataFrame(
                {"t_old": [20.0, 10.0], "ah_old": self.ah}, index=idx[:2]
            ),
            "new": pd.DataFrame(
                {"Ta_new": [15.0, 15.0], "AH_new": [9.0, 9.0], "RH_new": [70.0, 71.0]},
                index=idx[2:],
            ),
        }
        groups = {
            "old": _stub_file_group("old", "t_old", "ah_old"),
            "new": _stub_file_group("new", "Ta_new", "AH_new", "RH_new"),
        }
        df = pd.concat(
            [
                dataframe_builder._build_file_group_dataframe(
                    file_group=groups[name],
                    loader=lambda _, name=name: frames[name],
                    specs_by_alias=self.specs,
                    time_step=30,
                )
                for name in ("old", "new")
            ],
            axis=1,
        )
        df = dataframe_builder._merge_overlapping_variables(df, self.specs)
        df = dataframe_builder._rename_to_canonical(df, self.specs)
        self.df = dataframe_builder._apply_calculated_inputs(df, self.specs)

    def test_old_era_is_calculated_and_new_era_is_measured(self):
        np.testing.assert_allclose(self.df["RH_2m"].iloc[:2], [60.0, 80.0])
        self.assertEqual(self.df["RH_2m"].iloc[2:].tolist(), [70.0, 71.0])

    def test_calculation_matches_the_registered_function(self):
        np.testing.assert_allclose(
            self.df["RH_2m"].iloc[:2],
            calculate_RH_from_AH(AH=self.ah, Ta=np.array([20.0, 10.0])),
        )

    def test_variable_with_only_a_calculated_input_gets_a_column(self):
        df = pd.DataFrame(
            {"AH_2m": self.ah, "Ta_2m": [20.0, 10.0]},
            index=pd.date_range("2020-01-01 00:30", periods=2, freq="30min"),
        )
        specs = {
            a: s for a, s in self.specs.items() if s.calculate_from or "RH" not in a
        }
        out = dataframe_builder._apply_calculated_inputs(df, specs)
        np.testing.assert_allclose(out["RH_2m"], [60.0, 80.0])

    def test_quantities_filter_builds_sources_but_drops_them(self):
        groups = {
            "old": SimpleNamespace(file_format="TOA5"),
            "new": SimpleNamespace(file_format="TOA5"),
        }
        built = {}

        def fake_group_df(file_group, specs_by_alias, **_):
            built.setdefault("aliases", set()).update(specs_by_alias)
            return self.df[["RH_2m"]]

        from unittest import mock

        with (
            mock.patch.object(dataframe_builder.raw_data_loader, "get_data_loader"),
            mock.patch.object(
                dataframe_builder,
                "_build_file_group_dataframe",
                side_effect=fake_group_df,
            ),
            mock.patch.object(
                dataframe_builder,
                "_merge_overlapping_variables",
                side_effect=lambda df, **_: df,
            ),
            mock.patch.object(
                dataframe_builder,
                "_rename_to_canonical",
                side_effect=lambda df, **_: self.df.copy(),
            ),
        ):
            out = dataframe_builder.build_dataframe(
                file_groups=groups,
                specs_by_alias=self.specs,
                quantities={"RH"},
                flux_file="new",
            )
        # The sources' specs were built for the calculation...
        self.assertTrue({"AH_2m#0", "Ta_2m#0"} <= built["aliases"])
        # ...but only the requested quantity is returned.
        self.assertEqual(list(out.columns), ["RH_2m"])


class CalculatedInstrumentHistoryTestCase(unittest.TestCase):
    """A calculated period is marked in the variable's instrument_history."""

    def setUp(self):
        self.cfg = _build(CALCULATED_YAML, data_path=Path("/arbitrary"))
        specs = build_variable_registry(
            runtime_cfg=self.cfg, file_groups=build_file_groups(self.cfg)
        )
        self.by_variable = group_by_canonical_name(specs)

    def test_measured_variable_has_no_calculated_period(self):
        history = InstrumentHistory.from_specs(self.by_variable["AH_2m"])
        self.assertTrue(
            all(p.calculated_from is None for p in history.roles[0].periods)
        )
        self.assertNotIn("calculated_from", str(history.to_attr()))

    def test_history_records_the_calculated_and_measured_periods(self):
        history = InstrumentHistory.from_specs(self.by_variable["RH_2m"])
        self.assertEqual(
            history.roles[0].periods,
            (
                InstrumentPeriod(
                    instrument="Vaisala HMP45C",
                    start=None,
                    end=datetime(2020, 1, 1, 1, 0),
                    calculated_from=("AH_2m", "Ta_2m"),
                ),
                InstrumentPeriod(
                    instrument="Vaisala HMP155",
                    start=datetime(2020, 1, 1, 1, 30),
                    end=None,
                ),
            ),
        )

    def test_measured_to_calculated_on_one_instrument_is_two_periods(self):
        spec = SimpleNamespace(instrument="HMP", canonical_name="RH")
        history = InstrumentHistory.from_specs(
            [
                SimpleNamespace(
                    **vars(spec),
                    begin=None,
                    end="2020-01-01T01:00:00",
                    calculate_from=("AH", "Ta"),
                ),
                SimpleNamespace(
                    **vars(spec),
                    begin="2020-01-01T01:30:00",
                    end=None,
                    calculate_from=None,
                ),
            ]
        )
        self.assertEqual(len(history.roles[0].periods), 2)

    def test_attr_round_trips(self):
        history = InstrumentHistory.from_specs(self.by_variable["RH_2m"])
        self.assertEqual(InstrumentHistory.from_attr(history.to_attr()), history)
        self.assertEqual(
            history.to_attr()[0]["periods"][0]["calculated_from"], ["AH_2m", "Ta_2m"]
        )

    def test_netcdf_attrs_per_window(self):
        history = InstrumentHistory.from_specs(self.by_variable["RH_2m"])
        instrument, attr = history.netcdf_attrs(
            datetime(2020, 1, 1, 0, 30), datetime(2020, 1, 1, 2)
        )
        self.assertEqual(instrument, "Vaisala HMP155")
        self.assertEqual(
            attr,
            "(Vaisala HMP45C calculated from AH_2m Ta_2m,2020-01-01T00:30:00,"
            "2020-01-01T01:00:00)|(Vaisala HMP155,2020-01-01T01:30:00,"
            "2020-01-01T02:00:00)",
        )
        # A window with only the measured period: one instrument, no history.
        self.assertIsNone(
            history.netcdf_attrs(datetime(2021, 1, 1), datetime(2021, 12, 31))[1]
        )

    def test_a_wholly_calculated_window_still_gets_a_history(self):
        history = InstrumentHistory.from_specs(self.by_variable["AH_2m"]).calculated(
            ["RH_2m", "Ta_2m"]
        )
        _, attr = history.netcdf_attrs(datetime(2021, 1, 1), datetime(2022, 1, 1))
        self.assertEqual(
            attr,
            "(Vaisala HMP155 calculated from RH_2m Ta_2m,2021-01-01T00:00:00,"
            "2022-01-01T00:00:00)",
        )


class DerivedQuantityHistoryTestCase(unittest.TestCase):
    def test_derived_rh_marks_its_sources_in_the_history(self):
        idx = pd.date_range("2025-01-01", periods=2, freq="30min")
        history = [
            {
                "role": None,
                "periods": [
                    {"instrument": "Vaisala HMP155", "start": None, "end": None}
                ],
            }
        ]
        hmp = {
            "statistic_type": "average",
            "height": 2.0,
            "height_range": None,
            "instrument": "Vaisala HMP155",
            "instrument_history": history,
        }
        result = derive_missing_humidity(
            DatasetBuildIntermediate(
                df=pd.DataFrame(
                    {"Ta_2m": [20.0, 10.0], "AH_2m": [9.0, 8.0]}, index=idx
                ),
                var_attrs={
                    "Ta_2m": {**hmp, "quantity": "Ta"},
                    "AH_2m": {**hmp, "quantity": "AH"},
                },
            )
        )
        period = result.var_attrs["RH_2m"]["instrument_history"][0]["periods"][0]
        self.assertEqual(period["calculated_from"], ["AH_2m", "Ta_2m"])
        # The source's own history is untouched.
        self.assertNotIn(
            "calculated_from",
            result.var_attrs["AH_2m"]["instrument_history"][0]["periods"][0],
        )


class SerializeCalculatedHistoryTestCase(unittest.TestCase):
    def _ds(self, history_attr):
        idx = pd.date_range("2020-01-01 00:30", "2021-01-01 00:00", freq="30min")
        ds = xr.Dataset(
            {"RH_2m": ("time", np.zeros(len(idx)))},
            coords={"time": idx},
            attrs={"time_step": 30},
        )
        ds["RH_2m"].attrs["instrument_history"] = history_attr
        return ds

    def _attr(self, changeover):
        return [
            {
                "role": None,
                "periods": [
                    {
                        "instrument": "Vaisala HMP45C",
                        "start": None,
                        "end": f"{changeover}-06-30T23:30:00",
                        "calculated_from": ["AH_2m", "Ta_2m"],
                    },
                    {
                        "instrument": "Vaisala HMP155",
                        "start": f"{changeover}-07-01T00:00:00",
                        "end": None,
                    },
                ],
            }
        ]

    def test_year_with_a_calculated_period(self):
        out = serialize_inst_history(self._ds(self._attr(2020)), year=2020)
        self.assertEqual(
            out["RH_2m"].attrs["instrument_history"],
            "(Vaisala HMP45C calculated from AH_2m Ta_2m,2020-01-01T00:30:00,"
            "2020-06-30T23:30:00)|(Vaisala HMP155,2020-07-01T00:00:00,"
            "2021-01-01T00:00:00)",
        )

    def test_all_measured_year_with_one_instrument_has_no_history(self):
        out = serialize_inst_history(self._ds(self._attr(2019)), year=2020)
        self.assertNotIn("instrument_history", out["RH_2m"].attrs)
        self.assertEqual(out["RH_2m"].attrs["instrument"], "Vaisala HMP155")


if __name__ == "__main__":
    unittest.main()

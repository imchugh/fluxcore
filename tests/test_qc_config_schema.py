"""Tests for services/metadata/qc_config_schema.py."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import ValidationError

from services.metadata import qc_config_schema

VALID_YAML = """\
Ta_Av:
  range_check: {lower: -10, upper: 50}
Fco2:
  range_check: {lower: -50, upper: 50}
  dependency_check: [Ux, Uy, Uz]
  mad_filter:
    reference_var: Fsd_Av
    window_days: 13
"""


class QCConfigStructureTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.config_dir = Path(self._tmp_dir.name)

    def _write(self, name, content):
        path = self.config_dir / name
        path.write_text(content)
        return path

    def test_valid_yaml_round_trips(self):
        self._write("TestSite.yml", VALID_YAML)
        cfg = qc_config_schema.load_qc_config("TestSite", config_dir=self.config_dir)
        self.assertEqual(cfg.site_name, "TestSite")
        self.assertIn("Ta_Av", cfg.variables)
        self.assertEqual(cfg.variables["Ta_Av"].range_check.lower, -10)
        self.assertEqual(cfg.variables["Fco2"].mad_filter.reference_var, "Fsd_Av")

    def test_missing_file_returns_empty_config(self):
        cfg = qc_config_schema.load_qc_config("NoSuchSite", config_dir=self.config_dir)
        self.assertEqual(cfg.variables, {})

    def test_bad_range_order_raises(self):
        path = self._write("Bad.yml", "Ta_Av:\n  range_check: {lower: 50, upper: -10}\n")
        with self.assertRaises(ValidationError):
            qc_config_schema.validate_qc_config_structure(path)

    def test_unknown_key_rejected(self):
        path = self._write(
            "Bad2.yml", "Ta_Av:\n  rangecheck: {lower: -10, upper: 50}\n"
        )
        with self.assertRaises(ValidationError):
            qc_config_schema.validate_qc_config_structure(path)


class RangeDefaultsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.config_dir = Path(self._tmp_dir.name)

    def test_missing_file_returns_empty_dict(self):
        result = qc_config_schema.load_range_defaults(config_dir=self.config_dir)
        self.assertEqual(result, {})

    def test_flat_and_nested_entries_load(self):
        (self.config_dir / "_range_defaults.yml").write_text(
            "Fco2: [-50, 30]\nTa:\n  Av: [-10, 50]\n  Sd: [0, 5]\n"
        )
        result = qc_config_schema.load_range_defaults(config_dir=self.config_dir)
        self.assertEqual(result["Fco2"], [-50, 30])
        self.assertEqual(result["Ta"]["Av"], [-10, 50])

    def test_real_repo_range_defaults_file_loads(self):
        # Regression guard for the actual hand-maintained
        # configs/qc/_range_defaults.yml -- catches YAML/duplicate-key
        # errors without needing to know its exact contents.
        result = qc_config_schema.load_range_defaults()
        self.assertIn("Fco2", result)
        self.assertIn("Diag", result)
        self.assertIsInstance(result["Diag"], dict)


class ValidateVariablesTestCase(unittest.TestCase):
    def test_raises_for_all_unresolved_names(self):
        cfg = qc_config_schema.SiteQCConfig(
            site_name="TestSite",
            variables={
                "Fco2": qc_config_schema.VariableQCSpec(
                    dependency_check=["Ux", "MissingDep"],
                    mad_filter=qc_config_schema.MADFilterSpec(
                        reference_var="MissingRef"
                    ),
                ),
            },
        )
        with self.assertRaises(ValueError) as ctx:
            qc_config_schema.validate_qc_config_variables(
                cfg, available_variables={"Ux"}
            )
        msg = str(ctx.exception)
        self.assertIn("Fco2", msg)
        self.assertIn("MissingDep", msg)
        self.assertIn("MissingRef", msg)

    def test_passes_when_all_present(self):
        cfg = qc_config_schema.SiteQCConfig(
            site_name="TestSite",
            variables={
                "Ta_Av": qc_config_schema.VariableQCSpec(
                    range_check=qc_config_schema.RangeCheckSpec(lower=-10, upper=50)
                ),
            },
        )
        qc_config_schema.validate_qc_config_variables(
            cfg, available_variables={"Ta_Av"}
        )


class ResolveDefaultRangeTestCase(unittest.TestCase):
    """resolve_default_range: the Dataset-free lookup shared with the monitors."""

    def setUp(self):
        from services.metadata.core.variable_name_parser import NameParser

        self.parser = NameParser()
        self.defaults = {
            "Fco2": [-50, 30],
            "Ta": {"Av": [-10, 50], "Sd": [0, 5]},
            "Diag": {"IRGA": [0, 500], "SONIC": [0, 400]},
        }

    def _resolve(self, name, suffix=None):
        return qc_config_schema.resolve_default_range(
            name, suffix, self.defaults, self.parser
        )

    def test_flat_quantity(self):
        self.assertEqual(self._resolve("Fco2"), (-50, 30))

    def test_statistic_keyed_uses_suffix(self):
        self.assertEqual(self._resolve("Ta_Av", "Av"), (-10, 50))
        self.assertEqual(self._resolve("Ta_Sd", "Sd"), (0, 5))

    def test_statistic_keyed_without_suffix_is_none(self):
        self.assertIsNone(self._resolve("Ta_Av"))

    def test_qualifier_keyed(self):
        self.assertEqual(self._resolve("Diag_SONIC"), (0, 400))

    def test_unknown_quantity_and_unparseable_name_are_none(self):
        self.assertIsNone(self._resolve("Nope"))
        self.assertIsNone(self._resolve("not a name"))

    def test_station_pressure_default_follows_elevation(self):
        defaults = {"ps": {"standard_pressure_pm": 10}}
        sea = qc_config_schema.resolve_default_range("ps", None, defaults, self.parser, elevation=0)
        self.assertAlmostEqual(sea[0], 91.325, places=3)
        self.assertAlmostEqual(sea[1], 111.325, places=3)
        high = qc_config_schema.resolve_default_range("ps", None, defaults, self.parser, elevation=1650)
        # ISA pressure at 1650 m is 83.0 kPa
        self.assertAlmostEqual((high[0] + high[1]) / 2, 83.0, delta=0.1)
        self.assertAlmostEqual(high[1] - high[0], 20.0)

    def test_station_pressure_default_needs_an_elevation(self):
        defaults = {"ps": {"standard_pressure_pm": 10}}
        self.assertIsNone(qc_config_schema.resolve_default_range("ps", None, defaults, self.parser))

    def test_lookup_wrapper_reads_elevation_off_dataset_attrs(self):
        defaults = {"ps": {"standard_pressure_pm": 10}}
        ds = _resolve_ds(["ps"])
        ds.attrs["elevation"] = 1650.0
        lo, hi = qc_config_schema.lookup_default_range(ds, "ps", defaults, self.parser)
        self.assertAlmostEqual((lo + hi) / 2, 83.0, delta=0.1)
        del ds.attrs["elevation"]
        self.assertIsNone(qc_config_schema.lookup_default_range(ds, "ps", defaults, self.parser))

    def test_real_ps_default_is_elevation_derived(self):
        real = qc_config_schema.load_range_defaults()
        self.assertEqual(real["ps"], {"standard_pressure_pm": 10})

    def test_lookup_wrapper_reads_statistic_off_dataset_attrs(self):
        ds = _resolve_ds(["Ta_Av"])
        ds["Ta_Av"].attrs["statistic_type"] = "average"
        self.assertEqual(
            qc_config_schema.lookup_default_range(ds, "Ta_Av", self.defaults, self.parser),
            (-10, 50),
        )


class FlagCheckSchemaTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.path = Path(self._tmp_dir.name) / "Site.yml"

    def test_loads(self):
        self.path.write_text(
            "Fco2:\n  flag_check: {source: [Fco2_QC], reject: [8, 9]}\n"
        )
        cfg = qc_config_schema.validate_qc_config_structure(self.path)
        spec = cfg.root["Fco2"].flag_check
        self.assertEqual(spec.source, ["Fco2_QC"])
        self.assertEqual(spec.reject, [8.0, 9.0])

    def test_empty_lists_rejected(self):
        for body in ("source: [], reject: [9]", "source: [Fco2_QC], reject: []"):
            self.path.write_text(f"Fco2:\n  flag_check: {{{body}}}\n")
            with self.assertRaises(ValidationError):
                qc_config_schema.validate_qc_config_structure(self.path)

    def test_missing_source_variable_fails_strict_validation(self):
        cfg = qc_config_schema.SiteQCConfig(
            site_name="TestSite",
            variables={
                "Fco2": qc_config_schema.VariableQCSpec(
                    flag_check=qc_config_schema.FlagCheckSpec(
                        source=["Fco2_QC"], reject=[9]
                    )
                ),
            },
        )
        with self.assertRaises(ValueError) as ctx:
            qc_config_schema.validate_qc_config_variables(cfg, {"Fco2"})
        self.assertIn("Fco2_QC", str(ctx.exception))


class DependencyGraphOrderTestCase(unittest.TestCase):
    def test_topological_order(self):
        cfg = qc_config_schema.SiteQCConfig(
            site_name="TestSite",
            variables={
                "A": qc_config_schema.VariableQCSpec(dependency_check=["B"]),
                "B": qc_config_schema.VariableQCSpec(dependency_check=["C"]),
                "C": qc_config_schema.VariableQCSpec(),
            },
        )
        order = cfg.dependency_graph_order()
        self.assertLess(order.index("C"), order.index("B"))
        self.assertLess(order.index("B"), order.index("A"))

    def test_cycle_raises(self):
        cfg = qc_config_schema.SiteQCConfig(
            site_name="TestSite",
            variables={
                "A": qc_config_schema.VariableQCSpec(dependency_check=["B"]),
                "B": qc_config_schema.VariableQCSpec(dependency_check=["A"]),
            },
        )
        with self.assertRaises(ValueError):
            cfg.dependency_graph_order()


class DependencyDefaultsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.config_dir = Path(self._tmp_dir.name)

    def test_missing_file_returns_empty_dict(self):
        result = qc_config_schema.load_dependency_defaults(config_dir=self.config_dir)
        self.assertEqual(result, {})

    def test_valid_file_loads(self):
        (self.config_dir / "_dependency_defaults.yml").write_text(
            "Diag_SONIC:\n  range_check: {lower: 0, upper: 500}\n"
            "UxA:\n  dependency_check: [Diag_SONIC]\n"
        )
        result = qc_config_schema.load_dependency_defaults(config_dir=self.config_dir)
        self.assertEqual(result["Diag_SONIC"].range_check.upper, 500)
        self.assertEqual(result["UxA"].dependency_check, ["Diag_SONIC"])

    def test_malformed_file_raises_same_as_a_site_file(self):
        (self.config_dir / "_dependency_defaults.yml").write_text(
            "UxA:\n  bogus_check: [1, 2]\n"
        )
        with self.assertRaises(ValidationError):
            qc_config_schema.load_dependency_defaults(config_dir=self.config_dir)

    def test_real_repo_dependency_defaults_file_loads(self):
        # Regression guard for the actual hand-maintained
        # configs/qc/_dependency_defaults.yml.
        result = qc_config_schema.load_dependency_defaults()
        self.assertIn("UxA", result)
        self.assertIn("AH_IRGA_Av", result)
        # Only dependency checks live here: no numeric bounds, and no
        # per-flux-system logger QC flag defaults (those are per-site).
        for spec in result.values():
            self.assertIsNone(spec.range_check)
            self.assertIsNone(spec.flag_check)
        self.assertNotIn("Fco2", result)


def _resolve_ds(var_names):
    """Minimal xr.Dataset with the given data_vars, no real values needed
    for resolve_qc_config's merge logic beyond variable presence."""
    idx = pd.date_range("2020-01-01", periods=3, freq="30min")
    data_vars = {name: (("time",), np.array([1.0, 2.0, 3.0])) for name in var_names}
    return xr.Dataset(data_vars, coords={"time": idx})


class ResolveQCConfigTestCase(unittest.TestCase):
    def test_default_dependency_check_added_when_site_silent(self):
        ds = _resolve_ds(["Fco2", "Diag_SONIC"])
        site_config = qc_config_schema.SiteQCConfig(site_name="Test", variables={})
        defaults = {
            "Fco2": qc_config_schema.VariableQCSpec(dependency_check=["Diag_SONIC"]),
        }
        merged = qc_config_schema.resolve_qc_config(site_config, defaults, {}, ds)
        self.assertEqual(merged.variables["Fco2"].dependency_check, ["Diag_SONIC"])

    def test_site_range_check_kept_default_dependency_check_still_added(self):
        ds = _resolve_ds(["Fco2", "Diag_SONIC"])
        site_config = qc_config_schema.SiteQCConfig(
            site_name="Test",
            variables={
                "Fco2": qc_config_schema.VariableQCSpec(
                    range_check=qc_config_schema.RangeCheckSpec(lower=-10, upper=10)
                ),
            },
        )
        defaults = {
            "Fco2": qc_config_schema.VariableQCSpec(dependency_check=["Diag_SONIC"]),
        }
        merged = qc_config_schema.resolve_qc_config(site_config, defaults, {}, ds)
        self.assertEqual(merged.variables["Fco2"].range_check.lower, -10)
        self.assertEqual(merged.variables["Fco2"].dependency_check, ["Diag_SONIC"])

    def test_site_dependency_check_kept_range_default_still_added(self):
        ds = _resolve_ds(["Fco2"])
        site_config = qc_config_schema.SiteQCConfig(
            site_name="Test",
            variables={
                "Fco2": qc_config_schema.VariableQCSpec(dependency_check=["Something"]),
            },
        )
        range_defaults = {"Fco2": [-50, 30]}
        merged = qc_config_schema.resolve_qc_config(site_config, {}, range_defaults, ds)
        self.assertEqual(merged.variables["Fco2"].dependency_check, ["Something"])
        self.assertEqual(merged.variables["Fco2"].range_check.lower, -50)
        self.assertEqual(merged.variables["Fco2"].range_check.upper, 30)

    def test_default_entry_dropped_when_own_variable_absent(self):
        ds = _resolve_ds(["Ta_Av"])
        site_config = qc_config_schema.SiteQCConfig(site_name="Test", variables={})
        defaults = {
            "Fco2": qc_config_schema.VariableQCSpec(dependency_check=["Diag_SONIC"]),
        }
        merged = qc_config_schema.resolve_qc_config(site_config, defaults, {}, ds)
        self.assertNotIn("Fco2", merged.variables)

    def test_unavailable_dependency_source_dropped_individually(self):
        ds = _resolve_ds(["Fco2", "SigCO2_IRGA"])
        site_config = qc_config_schema.SiteQCConfig(site_name="Test", variables={})
        defaults = {
            "Fco2": qc_config_schema.VariableQCSpec(
                dependency_check=["Diag_SONIC", "SigCO2_IRGA"]
            ),
        }
        merged = qc_config_schema.resolve_qc_config(site_config, defaults, {}, ds)
        self.assertEqual(merged.variables["Fco2"].dependency_check, ["SigCO2_IRGA"])

    def test_all_dependency_sources_unavailable_collapses_whole_entry(self):
        # The CumberlandPlain / EddyPro-only shape: the dependent exists but
        # every one of its default conditionals is absent, and there's no
        # range default either -- the variable should end up untouched.
        ds = _resolve_ds(["Fco2"])
        site_config = qc_config_schema.SiteQCConfig(site_name="Test", variables={})
        defaults = {
            "Fco2": qc_config_schema.VariableQCSpec(
                dependency_check=["Diag_SONIC", "Uz_SONIC_Sd"]
            ),
        }
        merged = qc_config_schema.resolve_qc_config(site_config, defaults, {}, ds)
        self.assertNotIn("Fco2", merged.variables)

    def test_all_sources_unavailable_but_range_default_still_applies(self):
        ds = _resolve_ds(["Fco2"])
        site_config = qc_config_schema.SiteQCConfig(site_name="Test", variables={})
        defaults = {
            "Fco2": qc_config_schema.VariableQCSpec(dependency_check=["Diag_SONIC"]),
        }
        range_defaults = {"Fco2": [-50, 30]}
        merged = qc_config_schema.resolve_qc_config(
            site_config, defaults, range_defaults, ds
        )
        self.assertIn("Fco2", merged.variables)
        self.assertIsNone(merged.variables["Fco2"].dependency_check)
        self.assertEqual(merged.variables["Fco2"].range_check.lower, -50)

    def test_default_range_check_used_for_quality_flag_variable(self):
        # e.g. Diag_SONIC/Fco2_QC -- no _range_defaults.yml quantity entry;
        # the bound comes from the dependency_defaults entry itself.
        ds = _resolve_ds(["Fco2_QC"])
        site_config = qc_config_schema.SiteQCConfig(site_name="Test", variables={})
        defaults = {
            "Fco2_QC": qc_config_schema.VariableQCSpec(
                range_check=qc_config_schema.RangeCheckSpec(lower=-0.5, upper=1.5)
            ),
        }
        merged = qc_config_schema.resolve_qc_config(site_config, defaults, {}, ds)
        self.assertEqual(merged.variables["Fco2_QC"].range_check.lower, -0.5)

    def test_qcflag_and_crs_excluded_from_merge(self):
        ds = _resolve_ds(["Fco2", "Fco2_QCFlag", "crs"])
        site_config = qc_config_schema.SiteQCConfig(site_name="Test", variables={})
        merged = qc_config_schema.resolve_qc_config(site_config, {}, {}, ds)
        self.assertNotIn("Fco2_QCFlag", merged.variables)
        self.assertNotIn("crs", merged.variables)

    def test_flag_check_kept_and_source_gets_no_default_range(self):
        # Fco2_QC parses as quantity Fco2, so without gate-only handling it
        # would wrongly receive the flux range default.
        ds = _resolve_ds(["Fco2", "Fco2_QC"])
        site_config = qc_config_schema.SiteQCConfig(
            site_name="Test",
            variables={
                "Fco2": qc_config_schema.VariableQCSpec(
                    flag_check=qc_config_schema.FlagCheckSpec(
                        source=["Fco2_QC"], reject=[8, 9]
                    )
                ),
            },
        )
        merged = qc_config_schema.resolve_qc_config(
            site_config, {}, {"Fco2": [-50, 30]}, ds
        )
        self.assertEqual(merged.variables["Fco2"].flag_check.reject, [8.0, 9.0])
        self.assertEqual(merged.variables["Fco2"].range_check.lower, -50)
        self.assertNotIn("Fco2_QC", merged.variables)

    def test_site_range_still_overrides_elevation_derived_default(self):
        ds = _resolve_ds(["ps"])
        ds.attrs["elevation"] = 1650.0
        site_config = qc_config_schema.SiteQCConfig(
            site_name="Test",
            variables={
                "ps": qc_config_schema.VariableQCSpec(
                    range_check=qc_config_schema.RangeCheckSpec(lower=60, upper=70)
                ),
            },
        )
        merged = qc_config_schema.resolve_qc_config(
            site_config, {}, {"ps": {"standard_pressure_pm": 10}}, ds
        )
        self.assertEqual(merged.variables["ps"].range_check.lower, 60)

    def test_elevation_derived_default_applies_when_site_is_silent(self):
        ds = _resolve_ds(["ps"])
        ds.attrs["elevation"] = 1650.0
        merged = qc_config_schema.resolve_qc_config(
            qc_config_schema.SiteQCConfig(site_name="Test", variables={}),
            {},
            {"ps": {"standard_pressure_pm": 10}},
            ds,
        )
        self.assertAlmostEqual(merged.variables["ps"].range_check.lower, 73.0, delta=0.1)

    def test_untouched_variable_omitted_from_result(self):
        ds = _resolve_ds(["SomeRandomVar"])
        site_config = qc_config_schema.SiteQCConfig(site_name="Test", variables={})
        merged = qc_config_schema.resolve_qc_config(site_config, {}, {}, ds)
        self.assertEqual(merged.variables, {})


if __name__ == "__main__":
    unittest.main()

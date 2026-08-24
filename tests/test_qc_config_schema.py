"""Tests for services/metadata/qc_config_schema.py."""

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

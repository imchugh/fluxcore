"""Tests for FileGroup.validate_or_raise and build_file_groups.

Header discovery (variables_by_file) is bypassed by pre-populating the
cache directly, so no real raw data files are needed on disk.
"""

import tempfile
import unittest
from pathlib import Path

from services.metadata.file_group_builder import FileGroup, build_file_groups
from services.metadata.runtime_config_builder import (
    build_runtime_config,
    build_runtime_config_from_file,
)
from services.metadata.site_config_schema import validate_L1_config_structure

MASTER = Path("/store/Raw_data/HowardSprings/Flux/Slow/HowardSprings_flux.dat")

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


def _make_group(expected_variables, found_variables):
    group = FileGroup(
        group="flux",
        master=MASTER,
        backups=[],
        file_format="TOA5",
        expected_variables=expected_variables,
    )
    group._variables_by_file_cache = {MASTER: found_variables}
    return group


class TestValidateOrRaise(unittest.TestCase):
    def test_raises_on_missing_variable(self):
        group = _make_group(
            expected_variables={"Ta_1_1_1", "Fake_Var"},
            found_variables={"Ta_1_1_1"},
        )
        with self.assertRaises(ValueError) as ctx:
            group.validate_or_raise()
        self.assertIn("Fake_Var", str(ctx.exception))
        self.assertIn("flux", str(ctx.exception))

    def test_no_raise_when_all_expected_found(self):
        group = _make_group(
            expected_variables={"Ta_1_1_1"},
            found_variables={"Ta_1_1_1", "Extra_Var"},
        )
        group.validate_or_raise()

    def test_no_raise_when_no_expected_variables(self):
        group = _make_group(expected_variables=set(), found_variables=set())
        group.validate_or_raise()


class TestBuildFileGroups(unittest.TestCase):
    def test_uses_explicit_input_data_path_with_no_tern_dependency(self):
        """Given an explicit input_data_path, build_file_groups resolves
        FileGroup.master under it with no TERN/paths.py involvement."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "TestSite.yml"
            config_path.write_text(VALID_YAML)
            validated_config = validate_L1_config_structure(file=config_path)

            input_data_path = Path("/some/arbitrary/data/dir")
            runtime_cfg = build_runtime_config(
                validated_config, input_data_path=input_data_path
            )

            groups = build_file_groups(runtime_cfg)

        self.assertEqual(groups["flux"].master, input_data_path / "flux.dat")

    def test_from_file_wrapper_accepts_input_data_path_directly(self):
        """build_runtime_config_from_file itself takes input_data_path, so a
        standalone caller doesn't need to drop down to validate_L1_config_structure
        + build_runtime_config just to get a runnable config from one call."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "TestSite.yml"
            config_path.write_text(VALID_YAML)

            input_data_path = Path("/some/arbitrary/data/dir")
            runtime_cfg = build_runtime_config_from_file(
                config_path, input_data_path=input_data_path
            )

            groups = build_file_groups(runtime_cfg)

        self.assertEqual(groups["flux"].master, input_data_path / "flux.dat")

    def test_raises_when_input_data_path_not_set(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "TestSite.yml"
            config_path.write_text(VALID_YAML)

            runtime_cfg = build_runtime_config_from_file(config_path)

        with self.assertRaises(ValueError) as ctx:
            build_file_groups(runtime_cfg)
        self.assertIn("TestSite", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Tests for FileGroup.validate_or_raise.

Header discovery (variables_by_file) is bypassed by pre-populating the
cache directly, so no real raw data files are needed on disk.
"""

import unittest
from pathlib import Path

from services.metadata.file_group_builder import FileGroup

MASTER = Path('/store/Raw_data/HowardSprings/Flux/Slow/HowardSprings_flux.dat')


def _make_group(expected_variables, found_variables):
    group = FileGroup(
        group='flux',
        master=MASTER,
        backups=[],
        file_format='TOA5',
        expected_variables=expected_variables,
        )
    group._variables_by_file_cache = {MASTER: found_variables}
    return group


class TestValidateOrRaise(unittest.TestCase):

    def test_raises_on_missing_variable(self):
        group = _make_group(
            expected_variables={'Ta_1_1_1', 'Fake_Var'},
            found_variables={'Ta_1_1_1'},
            )
        with self.assertRaises(ValueError) as ctx:
            group.validate_or_raise()
        self.assertIn('Fake_Var', str(ctx.exception))
        self.assertIn('flux', str(ctx.exception))

    def test_no_raise_when_all_expected_found(self):
        group = _make_group(
            expected_variables={'Ta_1_1_1'},
            found_variables={'Ta_1_1_1', 'Extra_Var'},
            )
        group.validate_or_raise()

    def test_no_raise_when_no_expected_variables(self):
        group = _make_group(expected_variables=set(), found_variables=set())
        group.validate_or_raise()


if __name__ == '__main__':
    unittest.main(verbosity=2)
